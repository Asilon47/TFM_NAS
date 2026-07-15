# CP 10.1 step 3: nntool state -> AutoTiler model -> GenTile (links the
# recovered LibTile.a) -> generated GAP8 kernels + memory plan.
#
# App-less on purpose: `make model` exercises every codegen gate without a
# main.c. Driven per model by mcu/probes/gen_probe.sh, which overrides
# MODEL_PREFIX / TRAINED_MODEL / NNTOOL_SCRIPT / MODEL_BUILD on the command
# line. Reuses the SDK's own example machinery (model_decl.mk/model_rules.mk)
# so the flow stays identical to a shipped GAP8 project.
ifndef GAP_SDK_HOME
  $(error Run inside the tfm-gap8 image with sourceme.sh sourced)
endif

MODEL_PREFIX ?= net
TRAINED_MODEL ?= /workspace/TFM_NAS/models/res224/$(MODEL_PREFIX).onnx
MODEL_SQ8 = 1
NNTOOL_SCRIPT ?= nntool_script_sq8
# Calibration sample already exists (quant_probe.sh creates it); overriding
# IMAGES stops model_rules.mk from trying to synthesize mnist samples.
IMAGES = /workspace/TFM_NAS/data/mcu/probes/aq_sample
# The expression matcher emits generated elementwise kernels as a second C
# file. It is a TARGET-side kernel source (PULP builtins — compiles into the
# app, not into the host GenTile); declaring it here only registers it as a
# `nntool -g` output for make's grouped target.
MODEL_EXPRESSIONS = $(MODEL_BUILD)/Expression_Kernels.c

include $(GAP_SDK_HOME)/examples/gap8/nn/nntool/common/model_decl.mk
include $(GAP_SDK_HOME)/examples/gap8/nn/nntool/common/model_rules.mk
