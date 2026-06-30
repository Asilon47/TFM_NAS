# CP 3.3 warm-head Bayesian-Optimization search — AGX Jetson (L4T / aarch64) image.
#
# A single-GPU continuation of the Kaggle run: the board is a COMPUTE node (it reads the
# Orin-Nano-measured LUT as a static file, it never measures latency), so the search still
# optimizes for Orin Nano latency. Built natively on the Jetson by jetson/deploy.sh, which
# auto-detects the L4T release and passes the matching base image via --build-arg.
#
# The base must carry torch>=2.0 (botorch's MixedSingleTaskGP / qLogExpectedImprovement need
# it) -> JetPack 5.1.x (r35) or 6.x (r36). dustynv/ultralytics already bundles torch +
# torchvision + ultralytics + opencv for aarch64, so only the BO stack is pip-added.
ARG L4T_BASE=dustynv/ultralytics:r36.2.0
FROM ${L4T_BASE}

# Fail the build early (not 5 days into a run) if the base torch is too old for botorch.
RUN python3 -c "import sys,torch; v=tuple(int(x) for x in torch.__version__.split('.')[:2]); sys.exit(0 if v>=(2,0) else 'FATAL: torch '+torch.__version__+' <2.0 - botorch needs torch>=2; use a JetPack 5.1+/6 base')"

# Add the NAS + BO stack under a torch constraint so botorch / gpytorch / ofa never upgrade
# the Jetson's torch (the same pin discipline kaggle/run.py uses). scipy / pyyaml / numpy /
# Pillow / cv2 already come from the ultralytics base.
RUN python3 -c "import torch,pathlib; pathlib.Path('/tmp/constraints.txt').write_text('torch=='+torch.__version__+'\n')" \
 && pip install --no-cache-dir -c /tmp/constraints.txt \
      ofa==0.1.0.post202307202001 botorch gpytorch gdown

WORKDIR /workspace/TFM_NAS
COPY . /workspace/TFM_NAS

# Bake the SHA-pinned OFA supernet checkpoint into the image so runs need no network.
RUN python3 -m supernet.download_ofa

# Entry: locate the mounted /data, resume from the cache shards, run the 5-seed search.
ENTRYPOINT ["python3", "jetson/run_search.py"]
