# CP 10.1 DoD step: build the cycle-probe APP around AutoTiler-generated kernels
# and run it on GVSOC.
#
# The sibling ../at/Makefile is app-less on purpose (it only proves codegen).
# This one adds net_cyc.c as the main so there is something to actually run, and
# turns on the graph cycle monitor. Driven per model by mcu/probes/cyc_probe.sh,
# which overrides MODEL_PREFIX / MODEL_BUILD / NNTOOL_SCRIPT / AT_L3_ARENA /
# MODEL_L2_MEMORY on the command line. Reuses the SDK's own model_decl.mk +
# model_rules.mk + pmsis_rules.mk so the flow stays identical to a shipped GAP8
# project.
ifndef GAP_SDK_HOME
  $(error Run inside the tfm-gap8 image with sourceme.sh sourced)
endif

MODEL_PREFIX ?= net
TRAINED_MODEL ?= /workspace/TFM_NAS/models/res224/$(MODEL_PREFIX).onnx
MODEL_SQ8 = 1
NNTOOL_SCRIPT ?= nntool_script

# Calibration sample already exists; overriding IMAGES stops model_rules.mk from
# trying to synthesize mnist samples.
IMAGES = /workspace/TFM_NAS/data/mcu/probes/aq_sample

# Target-side generated elementwise kernels (PULP builtins): these compile INTO
# the app, never into the host GenTile -- see procedure.md CP 10.1 fact 3.
MODEL_EXPRESSIONS = $(MODEL_BUILD)/Expression_Kernels.c

# The AI-deck 1.1 carries HyperRAM + HyperFlash, so pin those rather than
# inheriting DEFAULT (this also matches the AT_MEM_L3_HRAM / AT_MEM_L3_HFLASH
# the nntool script sets, keeping the C types consistent).
FLASH_TYPE = HYPER
RAM_TYPE = HYPER

# Route printf through the host bridge rather than the UART model. Both SDK
# references that run clean on GVSOC (basic/helloworld, nn/nntool/mnist) set
# this; without it the very first printf hits an unbound UART port and GVSOC
# dies with SIGABRT before emitting a single line (the abort surfaces as
# "Magick: abort due to signal 6" only because the board's camera model links
# ImageMagick, whose signal handler catches it -- the camera is a red herring).
io=host

include $(GAP_SDK_HOME)/examples/gap8/nn/nntool/common/model_decl.mk

# Graph IO byte counts. Defaults are the CP 10.1 raw-head pair @224 (nc=1,
# 8 kpts) -- IDENTICAL for the graft and the yolo11n twin, which is what makes
# the two runs comparable. Order matches RAW_HEAD_OUTPUTS in
# detect/export_grafted_onnx.py: boxes, scores, feats_p3, feats_p4, feats_p5, kpts.
AT_INPUT_BYTES ?= 150528
AT_OUT1_BYTES ?= 65856
AT_OUT2_BYTES ?= 1029
AT_OUT3_BYTES ?= 50176
AT_OUT4_BYTES ?= 25088
AT_OUT5_BYTES ?= 12544
AT_OUT6_BYTES ?= 24696

# Bytes CNN_Construct() allocates in HyperRAM; only known after codegen, so
# cyc_probe.sh greps it out of the generated kernels and passes it back in.
AT_L3_ARENA ?= 0

APP = net_cyc
MAIN = /workspace/TFM_NAS/mcu/probes/cyc/net_cyc.c
APP_SRCS += $(MAIN) $(MODEL_GEN_C) $(MODEL_EXPRESSIONS) $(CNN_LIB)

APP_CFLAGS += -gdwarf-2 -gstrict-dwarf -O3 -mno-memcpy -fno-tree-loop-distribute-patterns
APP_CFLAGS += -I. -I$(MODEL_COMMON_INC) -I$(TILER_EMU_INC) -I$(TILER_INC) $(CNN_LIB_INCLUDE) -I$(MODEL_BUILD)
APP_CFLAGS += -DPERF -DAT_MODEL_PREFIX=$(MODEL_PREFIX) $(MODEL_SIZE_CFLAGS)
APP_CFLAGS += -DAT_MODEL_PREFIX_STR='"$(MODEL_PREFIX)"'
APP_CFLAGS += -DAT_MODEL_KERNELS_H='"$(MODEL_PREFIX)Kernels.h"'
APP_CFLAGS += -DAT_L3_ARENA=$(AT_L3_ARENA) -DAT_L3_TOTAL=$(TARGET_L3_SIZE)
# CYC_SMOKE=1: print and exit before any AutoTiler/cluster/HyperRAM call.
CYC_SMOKE ?= 0
APP_CFLAGS += -DCYC_SMOKE=$(CYC_SMOKE)
APP_CFLAGS += -DAT_INPUT_BYTES=$(AT_INPUT_BYTES)
APP_CFLAGS += -DAT_OUT1_BYTES=$(AT_OUT1_BYTES) -DAT_OUT2_BYTES=$(AT_OUT2_BYTES)
APP_CFLAGS += -DAT_OUT3_BYTES=$(AT_OUT3_BYTES) -DAT_OUT4_BYTES=$(AT_OUT4_BYTES)
APP_CFLAGS += -DAT_OUT5_BYTES=$(AT_OUT5_BYTES) -DAT_OUT6_BYTES=$(AT_OUT6_BYTES)
APP_CFLAGS += -DSTACK_SIZE=$(CLUSTER_STACK_SIZE) -DSLAVE_STACK_SIZE=$(CLUSTER_SLAVE_STACK_SIZE)

# The weights live in the flash image, not the binary -- READFS puts them there.
READFS_FILES=$(abspath $(MODEL_TENSORS))

all:: model

clean:: clean_model

include $(GAP_SDK_HOME)/examples/gap8/nn/nntool/common/model_rules.mk
include $(RULES_DIR)/pmsis_rules.mk
