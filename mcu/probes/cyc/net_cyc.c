/*
 * CP 10.1 DoD: GVSOC cycle counts for a generated AutoTiler graph.
 *
 * ONE harness for every probe model. The CP 10.1 pair (graft raw-head vs
 * yolo11n-pose raw-head) exports an identical entry signature -- 1 int8 input,
 * 6 int8 outputs, matched shapes -- so the same object code drives both and any
 * cycle delta is the backbone alone. Everything model-specific arrives as a -D
 * from the Makefile (driven by mcu/probes/cyc_probe.sh).
 *
 * Numbers printed here are SIM cycles and are RANKING-ONLY, same discipline as
 * the Jetson LUT (procedure.md, Stage 0).
 */

#include "pmsis.h"
#include AT_MODEL_KERNELS_H

/* AT_MODEL_PREFIX is a bare token (e.g. yolo11n_pose_224_raw); paste it onto
 * the AutoTiler symbol names. The extra level of indirection is what forces the
 * argument to expand before ## sees it. */
#define _CAT(a, b) a##b
#define CAT(a, b) _CAT(a, b)

#define CNN_RUN CAT(AT_MODEL_PREFIX, CNN)
#define CNN_CONSTRUCT CAT(AT_MODEL_PREFIX, CNN_Construct)
#define CNN_DESTRUCT CAT(AT_MODEL_PREFIX, CNN_Destruct)
#define CNN_L3_FLASH CAT(AT_MODEL_PREFIX, _L3_Flash)
#define CNN_L3_MEMORY CAT(AT_MODEL_PREFIX, _L3_Memory)

#ifndef STACK_SIZE
#define STACK_SIZE 1024
#endif

/* The generated kernels reference this but never define it -- the app owns it
 * (mnist example does the same). 0 = base of the READFS-flashed tensor file. */
AT_HYPERFLASH_FS_EXT_ADDR_TYPE CNN_L3_FLASH = 0;

/* Base of the arena CNN_Construct() allocates from its own (file-static)
 * HyperRAM device. Valid only after Construct. */
extern AT_HYPERRAM_POINTER CNN_L3_MEMORY;

#ifndef AT_L3_ARENA
#error "AT_L3_ARENA (bytes CNN_Construct allocates in HyperRAM) must be -D'd"
#endif
#ifndef AT_L3_TOTAL
#define AT_L3_TOTAL 8000000
#endif

/* HyperRAM DMA wants aligned addresses, and the raw-head tensors do not oblige
 * (scores is 1029 bytes -- odd), so packing them back-to-back would hand
 * AutoTiler a misaligned buffer from the second output on. Pad every buffer up. */
#define L3_ALIGN 16u
#define ALIGN_UP(x) (((x) + (L3_ALIGN - 1u)) & ~(L3_ALIGN - 1u))

/* Graph IO all lives in HyperRAM, and that is forced, not preferred.
 *
 * MEASURED on this pair: the generated graph's .text is ~329 KB of GAP8's
 * 512 KB L2 (214 node functions), leaving only ~180 KB of L2 heap -- less than
 * AutoTiler's own working set needs, before any IO buffer exists. So neither the
 * 150528-byte input nor the 179389 bytes of raw-head outputs can sit in L2; both
 * are placed in HyperRAM at codegen via default_{input,output}_{home,exec}_location.
 *
 * AutoTiler passes these args to AT_HYPERRAM_CL_COPY as uint32 HyperRAM
 * *addresses* and never dereferences them on the FC, so the app only has to hand
 * it non-overlapping addresses. It does not -- and cannot -- allocate them: the
 * graph's HyperRAM device is file-static, and pi_ram_alloc is per-device, so a
 * second device would be a second allocator over the same 8 MB, handing back
 * addresses that collide with the arena. Placing the buffers directly above the
 * arena side-steps the allocator entirely; the bound check below is what keeps
 * that honest. */
static AT_HYPERRAM_POINTER In_L3;
static AT_HYPERRAM_POINTER Out_L3[6];
static const unsigned int OutBytes[6] = {AT_OUT1_BYTES, AT_OUT2_BYTES,
                                         AT_OUT3_BYTES, AT_OUT4_BYTES,
                                         AT_OUT5_BYTES, AT_OUT6_BYTES};

static int cnn_err;

#if CYC_SMOKE == 5
/* Same cluster entry/exit path as run_cnn, minus the graph: isolates cluster
 * dispatch + the perf timers from the CNN itself. */
static void run_noop(void *arg)
{
    (void)arg;
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    cnn_err = 0;
}
#endif

static void run_cnn(void *arg)
{
    (void)arg;
    gap_cl_starttimer();
    gap_cl_resethwtimer();
    cnn_err = CNN_RUN((signed char *)In_L3, (signed char *)Out_L3[0],
                      (signed char *)Out_L3[1], (signed char *)Out_L3[2],
                      (signed char *)Out_L3[3], (signed char *)Out_L3[4],
                      (signed char *)Out_L3[5]);
}

static int cyc_probe(void)
{
    struct pi_device cluster_dev;
    struct pi_cluster_conf cl_conf;
    struct pi_cluster_task task;
    unsigned int i, total_cyc = 0, total_op = 0, out_span = 0;

    printf("CYC_MODEL %s\n", AT_MODEL_PREFIX_STR);

    /* Bisect aid (`make ... CYC_SMOKE=<n>`): stop after stage n. Stdout is
     * buffered and lost on SIGABRT, so a crash reports as silence -- staging is
     * the only way to see how far the run actually got.
     *   1 = binary boots + printf reaches host   2 = cluster opens
     *   3 = graph constructs (opens HyperFlash/HyperRAM, allocs L3/L2/L1)
     *   4 = IO addresses placed under the arena
     *   5 = a no-op cluster task dispatches (isolates dispatch from the graph) */
#if CYC_SMOKE == 1
    printf("CYC_SMOKE 1 boot ok\n");
    pmsis_exit(0);
    return 0;
#endif

    pi_cluster_conf_init(&cl_conf);
    cl_conf.id = 0;
    cl_conf.cc_stack_size = STACK_SIZE;
    pi_open_from_conf(&cluster_dev, (void *)&cl_conf);
    if (pi_cluster_open(&cluster_dev)) {
        printf("CYC_ERROR cluster open failed\n");
        pmsis_exit(-2);
    }

#if CYC_SMOKE == 2
    printf("CYC_SMOKE 2 cluster ok\n");
    pmsis_exit(0);
    return 0;
#endif

    /* MUST follow pi_cluster_open, and must precede reading CNN_L3_MEMORY. */
    cnn_err = CNN_CONSTRUCT();
    if (cnn_err) {
        printf("CYC_ERROR construct rc=%d (1=flash open, 2=L3 alloc, 3=L2 alloc,"
               " 4=L1 alloc)\n", cnn_err);
        pmsis_exit(-3);
    }

    /* Pack the IO into the free HyperRAM immediately BELOW AutoTiler's arena.
     * Below, not above: PMSIS's extern_alloc serves from the top of the device,
     * so Construct's arena ends at the last byte and [0, CNN_L3_MEMORY) is what
     * is left. Deriving the base from the observed arena address (rather than
     * assuming it starts at 0) is what makes this independent of both the
     * allocator's direction and the true device size -- the first run proved the
     * point by reporting arena_base=5248064 on a device the tiler was told was
     * 8000000 bytes.
     *
     * The input is deliberately left uninitialised: cycles through AutoTiler's
     * int8 kernels are data-independent (fixed tile loops, no sparsity path, no
     * data-dependent branching), so its contents cannot move the number this
     * probe exists to produce, and writing it would need a HyperRAM device the
     * app cannot open. Accuracy is not this probe's job -- CP 10.2 owns that. */
#if CYC_SMOKE == 3
    printf("CYC_SMOKE 3 construct ok, arena_base=%u\n",
           (unsigned int)CNN_L3_MEMORY);
    CNN_DESTRUCT();
    pmsis_exit(0);
    return 0;
#endif

    out_span = ALIGN_UP(AT_INPUT_BYTES);
    for (i = 0; i < 6; i++) out_span += ALIGN_UP(OutBytes[i]);
    if (out_span > (unsigned int)CNN_L3_MEMORY) {
        printf("CYC_ERROR IO does not fit under the arena: arena_base=%u"
               " arena=%u io=%u\n", (unsigned int)CNN_L3_MEMORY,
               (unsigned int)AT_L3_ARENA, out_span);
        pmsis_exit(-4);
    }
    In_L3 = (CNN_L3_MEMORY - out_span) & ~(L3_ALIGN - 1u);
    Out_L3[0] = In_L3 + ALIGN_UP(AT_INPUT_BYTES);
    for (i = 1; i < 6; i++) Out_L3[i] = Out_L3[i - 1] + ALIGN_UP(OutBytes[i - 1]);

    printf("CYC_MEM arena_base=%u arena_bytes=%u in_base=%u in_bytes=%d"
           " out_base=%u io_bytes=%u\n", (unsigned int)CNN_L3_MEMORY,
           (unsigned int)AT_L3_ARENA, (unsigned int)In_L3, AT_INPUT_BYTES,
           (unsigned int)Out_L3[0], out_span);

#if CYC_SMOKE == 4
    printf("CYC_SMOKE 4 io placed ok\n");
    CNN_DESTRUCT();
    pmsis_exit(0);
    return 0;
#endif

#if CYC_SMOKE == 5
    pi_cluster_task(&task, run_noop, NULL);
#else
    pi_cluster_task(&task, run_cnn, NULL);
#endif
    /* pi_cluster_task_stacks() sets ONLY the slave stack (task->slave_stack_size);
     * pi_cluster_task() leaves the master's task->stack_size at 0, which the
     * runtime silently resolves to a 2048-byte default -- and cc_stack_size in
     * the cluster conf does NOT govern it. The generated <prefix>CNN() frame
     * alone is ~2 KB (it declares one HyperRAM event struct per DMA channel),
     * so on PE0 it overran that default by ~900 B and GVSOC aborted with
     * "SP-based access outside stack". mnist never trips this: its CNN frame is
     * tiny. Set the master stack explicitly; STACK_SIZE == CLUSTER_STACK_SIZE,
     * the same value model_decl.mk already deducts from AutoTiler's L1 budget,
     * so the two stay consistent. */
    task.stack_size = (uint32_t)STACK_SIZE;
    pi_cluster_task_stacks(&task, NULL, SLAVE_STACK_SIZE);
    pi_cluster_send_task(&cluster_dev, &task);

#if CYC_SMOKE == 5
    printf("CYC_SMOKE 5 cluster dispatch ok\n");
    CNN_DESTRUCT();
    pmsis_exit(0);
    return 0;
#endif

    if (cnn_err) {
        printf("CYC_ERROR CNN returned %d\n", cnn_err);
        pmsis_exit(-5);
    }

    for (i = 0; i < (sizeof(AT_GraphPerf) / sizeof(unsigned int)); i++) {
        printf("CYC_NODE %u %s %u %u\n", i, AT_GraphNodeNames[i],
               AT_GraphPerf[i], AT_GraphOperInfosNames[i]);
        total_cyc += AT_GraphPerf[i];
        total_op += AT_GraphOperInfosNames[i];
    }
    /* AT_GraphPerf_CNN_Total is AutoTiler's own whole-graph counter; it brackets
     * the full run, so it also catches anything the per-node sum misses. */
    printf("CYC_TOTAL sum_nodes=%u at_total=%u operations=%u nodes=%u\n",
           total_cyc, AT_GraphPerf_CNN_Total, total_op,
           (unsigned int)(sizeof(AT_GraphPerf) / sizeof(unsigned int)));

    CNN_DESTRUCT();
    pi_cluster_close(&cluster_dev);
    printf("CYC_DONE\n");
    pmsis_exit(0);
    return 0;
}

int main(void)
{
    return pmsis_kickoff((void *)cyc_probe);
}
