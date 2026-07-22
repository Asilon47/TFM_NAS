/*
 * mcu/board/net_bench/net_bench.c
 *
 * On-silicon cycle + FPS bench for one AutoTiler int8 graph on the GAP8 AI-deck.
 * Composition (see docs/superpowers/specs/2026-07-22-mcu-board-net-bench-design.md):
 *   - board shell (freq/uart/cpx/cluster/FreeRTOS) from Bitcraze classification.c
 *   - graph-run + HyperRAM-IO + AT_GraphPerf core from mcu/probes/cyc/net_cyc.c
 *   - a timing loop over N runs -> wall-clock FPS
 * No camera: the input is a fixed, uninitialised HyperRAM buffer. int8 AutoTiler
 * cycles are data-independent (fixed tile loops, no sparsity/branch path), so its
 * contents cannot move the number this bench produces (net_cyc.c argues the same).
 *
 * Numbers are SIM-discipline: AT_GraphPerf cycles are RANKING-ONLY; the wall-clock
 * ms/FPS is the honest absolute. Output goes over CPX (LOG_TO_CRTP) -> readable in
 * cfclient's Console tab over radio, or over WiFi.
 */

#include "pmsis.h"
#include "bsp/bsp.h"
#include "cpx.h"
#include <string.h>
#include AT_MODEL_KERNELS_H

/* Paste the bare AT_MODEL_PREFIX token onto the AutoTiler symbol names. The extra
 * indirection forces the argument to expand before ## sees it (net_cyc.c trick). */
#define _CAT(a, b) a##b
#define CAT(a, b) _CAT(a, b)
#define CNN_RUN CAT(AT_MODEL_PREFIX, CNN)
#define CNN_CONSTRUCT CAT(AT_MODEL_PREFIX, CNN_Construct)
#define CNN_DESTRUCT CAT(AT_MODEL_PREFIX, CNN_Destruct)
#define CNN_L3_FLASH CAT(AT_MODEL_PREFIX, _L3_Flash)
#define CNN_L3_MEMORY CAT(AT_MODEL_PREFIX, _L3_Memory)

/* Stringify the prefix token for the report line (no reliance on an external -D). */
#define BENCH_XSTR(s) #s
#define BENCH_STR(s) BENCH_XSTR(s)

#ifndef STACK_SIZE
#define STACK_SIZE 8192
#endif
#ifndef SLAVE_STACK_SIZE
#define SLAVE_STACK_SIZE 1024
#endif
#ifndef BENCH_ITERS
#define BENCH_ITERS 20
#endif
#ifndef BENCH_WARMUP
#define BENCH_WARMUP 2
#endif
#ifndef FREQ_CL
#define FREQ_CL 175
#endif
#ifndef FREQ_FC
#define FREQ_FC 250
#endif
#ifndef AT_MODEL_RES
#define AT_MODEL_RES 0
#endif

/* The app owns _L3_Flash (AutoTiler's Kernels.h references it); 0 = base of the
 * READFS-flashed tensor file. Same contract as classification.c / net_cyc.c. */
AT_HYPERFLASH_FS_EXT_ADDR_TYPE CNN_L3_FLASH = 0;
/* Base of the arena CNN_Construct() allocated in its own HyperRAM device. Valid
 * only after Construct(). */
extern AT_HYPERRAM_POINTER CNN_L3_MEMORY;

#define L3_ALIGN 16u
#define ALIGN_UP(x) (((x) + (L3_ALIGN - 1u)) & ~(L3_ALIGN - 1u))

/* Graph IO lives in HyperRAM (forced): the generated .text is ~330-405 KB of the
 * 512 KB L2, so neither the input nor the ~180 KB of raw-head outputs fit in L2.
 * The nntool script sets default_{input,output}_location = HRAM; the app packs the
 * buffers into the free HyperRAM immediately BELOW the arena (extern_alloc serves
 * from the top, so [0, CNN_L3_MEMORY) is free). Derived from the observed arena
 * base, so it is independent of the true device size (net_cyc.c). */
static AT_HYPERRAM_POINTER In_L3;
static AT_HYPERRAM_POINTER Out_L3[6];
static const unsigned int OutBytes[6] = {AT_OUT1_BYTES, AT_OUT2_BYTES, AT_OUT3_BYTES,
                                         AT_OUT4_BYTES, AT_OUT5_BYTES, AT_OUT6_BYTES};

static int cnn_err;
static struct pi_device cluster_dev;
static struct pi_cluster_task *cl_task;

/* Cluster entry: reset the hw timer, run the 7-arg raw-head graph. */
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

static void net_bench(void)
{
    /* Clocks: CL=175 to match the sim's basis; report FREQ_CL in the line so
     * cyc<->ms is reconstructable on the host. Voltage first, as classification. */
    __pi_pmu_voltage_set(PI_PMU_DOMAIN_FC, 1200);
    pi_freq_set(PI_FREQ_DOMAIN_FC, FREQ_FC * 1000 * 1000);
    pi_freq_set(PI_FREQ_DOMAIN_CL, FREQ_CL * 1000 * 1000);

    struct pi_uart_conf uart_conf;
    struct pi_device uart_dev;
    pi_uart_conf_init(&uart_conf);
    uart_conf.baudrate_bps = 115200;
    pi_open_from_conf(&uart_dev, &uart_conf);
    if (pi_uart_open(&uart_dev)) {
        pmsis_exit(-1);
    }

    cpxInit();
    cpxEnableFunction(CPX_F_WIFI_CTRL);
    cpxPrintToConsole(LOG_TO_CRTP, "*** net_bench %s res=%d iters=%d ***\n",
                      BENCH_STR(AT_MODEL_PREFIX), AT_MODEL_RES, BENCH_ITERS);

    /* Staged bring-up (ported from net_cyc.c CYC_SMOKE): a GAP8 crash reports as
     * SILENCE (buffered output lost on abort), so staging is the only way to see
     * how far an L2-fit failure got. 1 boot 2 cluster 3 construct 4 io 5 dispatch. */
#if BENCH_SMOKE == 1
    cpxPrintToConsole(LOG_TO_CRTP, "SMOKE 1 boot ok\n");
    pmsis_exit(0);
    return;
#endif

    struct pi_cluster_conf cl_conf;
    pi_cluster_conf_init(&cl_conf);
    cl_conf.id = 0;
    pi_open_from_conf(&cluster_dev, &cl_conf);
    if (pi_cluster_open(&cluster_dev)) {
        cpxPrintToConsole(LOG_TO_CRTP, "ERR cluster open\n");
        pmsis_exit(-2);
    }

#if BENCH_SMOKE == 2
    cpxPrintToConsole(LOG_TO_CRTP, "SMOKE 2 cluster ok\n");
    pmsis_exit(0);
    return;
#endif

    cnn_err = CNN_CONSTRUCT();
    if (cnn_err) {
        cpxPrintToConsole(LOG_TO_CRTP,
                          "ERR construct rc=%d (1 flash,2 L3,3 L2,4 L1)\n", cnn_err);
        pmsis_exit(-3);
    }

#if BENCH_SMOKE == 3
    cpxPrintToConsole(LOG_TO_CRTP, "SMOKE 3 construct ok arena_base=%u\n",
                      (unsigned int)CNN_L3_MEMORY);
    CNN_DESTRUCT();
    pmsis_exit(0);
    return;
#endif

    /* Pack IO into the free HyperRAM below the arena. Bound-checked against the
     * runtime arena base (CNN_L3_MEMORY) -- AT_L3_ARENA is NOT needed (net_cyc.c
     * uses it only in a diagnostic), which keeps this build single-pass. */
    unsigned int out_span = ALIGN_UP(AT_INPUT_BYTES);
    for (int i = 0; i < 6; i++) out_span += ALIGN_UP(OutBytes[i]);
    if (out_span > (unsigned int)CNN_L3_MEMORY) {
        cpxPrintToConsole(LOG_TO_CRTP, "ERR IO !fit arena_base=%u io=%u\n",
                          (unsigned int)CNN_L3_MEMORY, out_span);
        pmsis_exit(-4);
    }
    In_L3 = (CNN_L3_MEMORY - out_span) & ~(L3_ALIGN - 1u);
    Out_L3[0] = In_L3 + ALIGN_UP(AT_INPUT_BYTES);
    for (int i = 1; i < 6; i++) Out_L3[i] = Out_L3[i - 1] + ALIGN_UP(OutBytes[i - 1]);

#if BENCH_SMOKE == 4
    cpxPrintToConsole(LOG_TO_CRTP, "SMOKE 4 io placed base=%u span=%u\n",
                      (unsigned int)In_L3, out_span);
    CNN_DESTRUCT();
    pmsis_exit(0);
    return;
#endif

    /* Heap cluster task; set BOTH stacks (the grafts overran the 2 KB master
     * default in sim -> "SP-based access outside stack"). classification idiom. */
    cl_task = pmsis_l2_malloc(sizeof(struct pi_cluster_task));
    if (!cl_task) {
        cpxPrintToConsole(LOG_TO_CRTP, "ERR task alloc\n");
        pmsis_exit(-6);
    }
    memset(cl_task, 0, sizeof(struct pi_cluster_task));
    cl_task->entry = &run_cnn;
    cl_task->stack_size = STACK_SIZE;
    cl_task->slave_stack_size = SLAVE_STACK_SIZE;
    cl_task->arg = NULL;

#if BENCH_SMOKE == 5
    pi_cluster_send_task_to_cl(&cluster_dev, cl_task);
    cpxPrintToConsole(LOG_TO_CRTP, "SMOKE 5 dispatch ok rc=%d\n", cnn_err);
    CNN_DESTRUCT();
    pmsis_exit(0);
    return;
#endif

    for (int i = 0; i < BENCH_WARMUP; i++) {
        pi_cluster_send_task_to_cl(&cluster_dev, cl_task);
    }

    /* Forever: N timed runs -> mean us/inference; AT_GraphPerf cluster cycles. */
    while (1) {
        unsigned int t0 = pi_time_get_us();
        for (int i = 0; i < BENCH_ITERS; i++) {
            pi_cluster_send_task_to_cl(&cluster_dev, cl_task);
            if (cnn_err) {
                cpxPrintToConsole(LOG_TO_CRTP, "ERR CNN rc=%d\n", cnn_err);
                pmsis_exit(-5);
            }
        }
        unsigned int t1 = pi_time_get_us();

        unsigned int n_nodes = sizeof(AT_GraphPerf) / sizeof(unsigned int);
        unsigned int sum_nodes = 0;
        for (unsigned int i = 0; i < n_nodes; i++) sum_nodes += AT_GraphPerf[i];
        unsigned int at_total = AT_GraphPerf_CNN_Total;
        unsigned int us_per = (t1 - t0) / BENCH_ITERS;

        cpxPrintToConsole(LOG_TO_CRTP,
                          "BENCH model=%s res=%d cyc=%u nodes=%u clk_us=%u n=%d fcl=%d\n",
                          BENCH_STR(AT_MODEL_PREFIX), AT_MODEL_RES, at_total,
                          sum_nodes, us_per, BENCH_ITERS, FREQ_CL);
    }
}

int main(void)
{
    pi_bsp_init();
    return pmsis_kickoff((void *)net_bench);
}
