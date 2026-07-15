# CP 10.1 — GAP8 toolchain image (NNTool + AutoTiler + GVSOC), x86-64 host.
#
# The vendor (GreenWaves Technologies) is defunct; this build reconstructs the
# toolchain from surviving public sources, all pinned:
#   - gap_sdk @ upstream master tip (frozen since 2024-01-16)
#   - prebuilt GAP RISC-V toolchain repo (frozen 2024-03)
#   - LibTile.a recovered + SHA-verified by mcu/fetch_tiler.sh (run it before
#     `docker build`; the blob is proprietary and lives only in mcu/vendor/)
# Recipe derived from the community-proven cbezaitis/gap_sdk Dockerfile
# (2026-03, verified "Test success !" on GVSOC); we build from upstream
# instead of the fork and bake GVSOC at image-build time.
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND=noninteractive

# OS deps per gap_sdk README "OS Requirements installation" (ubuntu 20.04).
RUN apt-get update && apt-get install -y --no-install-recommends \
        autoconf automake bison build-essential cmake curl doxygen flex git \
        gtkwave libftdi-dev libftdi1 libjpeg-dev libsdl2-dev libsdl2-ttf-dev \
        libsndfile1-dev graphicsmagick-libmagick-dev-compat libtool \
        libusb-1.0-0-dev pkg-config python3-pip rsync scons texinfo wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 10

# Prebuilt GAP RISC-V toolchain. install.sh needs sudo; as root we replicate
# its rsync into the path the SDK expects.
ARG TOOLCHAIN_SHA=e6b5fbbfbf57a0ad1092efb280d0417cf3ee7e51
RUN git clone --depth 1 \
        https://github.com/GreenWaves-Technologies/gap_riscv_toolchain_ubuntu.git \
        /opt/toolchain \
    && cd /opt/toolchain \
    && test "$(git rev-parse HEAD)" = "${TOOLCHAIN_SHA}" \
        || { echo "FATAL: toolchain master moved off the pinned SHA (vendor repo should be frozen — investigate before trusting it)"; exit 1; } \
    && mkdir -p /usr/lib/gap_riscv_toolchain \
    && rsync -a --delete --exclude ".git*" /opt/toolchain/ /usr/lib/gap_riscv_toolchain \
    && rm -rf /opt/toolchain

# gap_sdk at the frozen master tip (submodules pin the matching gvsoc etc.).
ARG GAP_SDK_SHA=a23026507efe57410b98c5945544cff3150eb996
RUN git clone --depth 1 --recurse-submodules --jobs 4 \
        https://github.com/GreenWaves-Technologies/gap_sdk.git /opt/gap_sdk \
    && cd /opt/gap_sdk \
    && test "$(git rev-parse HEAD)" = "${GAP_SDK_SHA}" \
        || { echo "FATAL: gap_sdk master moved off the pinned SHA (vendor repo should be frozen — investigate before trusting it)"; exit 1; }

# The recovered AutoTiler core (libtile.4.3.5.a — matches the Makefile's
# TILER_VER pin). Placing it here makes `make autotiler` a no-op, so no build
# step ever hits the dead GreenWaves download server.
COPY vendor/LibTile.a /opt/gap_sdk/tools/autotiler_v3/Autotiler/LibTile.a

WORKDIR /opt/gap_sdk

RUN pip3 install --no-cache-dir --upgrade pip \
    && pip3 install --no-cache-dir -r requirements.txt

# Bake the GVSOC simulator (the cycle oracle) at build time so containers are
# immediately runnable; `--board gapuino` selects GAP8 non-interactively.
RUN bash -lc "source sourceme.sh --board gapuino && make -f gap8/gap8.mk gvsoc"

# NNTool deps last: the flakiest layer (old pins, C extensions) must not cost
# the GVSOC layer cache on failure.
RUN pip3 install --no-cache-dir -r tools/nntool/requirements.txt

RUN printf 'gap_sdk=%s\ntoolchain=%s\ntiler=libtile.4.3.5.a sha256=541f4978f3e55c6650207b3f8eebcfe0d311acd3320f81eb0a59890ec928c728\n' \
        "${GAP_SDK_SHA}" "${TOOLCHAIN_SHA}" > /IMAGE_PROVENANCE

CMD ["/bin/bash"]
