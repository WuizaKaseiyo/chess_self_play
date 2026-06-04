FROM vllm/vllm-openai:v0.11.2

# Populated automatically by BuildKit / buildx.
ARG TARGETARCH

# Keep pip quiet and deterministic-ish, and make it easy to run this repo when
# it is bind-mounted into the container at /workspace/chess_rl.
ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/workspace/chess_rl \
    PATH=/workspace/chess_rl:${PATH}

WORKDIR /workspace/chess_rl

# Chess reward function dependency.
RUN pip3 install --no-cache-dir python-chess

# Transformers models in this repo default to FlashAttention2 via
# `attn_implementation=flash_attention_2`.
#
# Building flash-attn from source under QEMU (x86 host -> arm64 image) is flaky
# and can segfault inside nvcc. Use prebuilt wheels instead.
#
# Note: these wheels are built for cu128 + torch2.9, but work with the cu129
# torch bundled in `vllm/vllm-openai:v0.11.2` (CUDA minor/soname compatible).
RUN case "${TARGETARCH}" in \
        arm64) \
            pip3 install --no-cache-dir --no-deps --force-reinstall \
              https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.4/flash_attn-2.8.3%2Bcu128torch2.9-cp312-cp312-linux_aarch64.whl \
            ;; \
        amd64) \
            pip3 install --no-cache-dir --no-deps --force-reinstall \
              https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.8/flash_attn-2.8.3%2Bcu128torch2.9-cp312-cp312-linux_x86_64.whl \
            ;; \
        *) \
            echo "Unsupported TARGETARCH='${TARGETARCH}' for flash-attn install" >&2; \
            exit 1 \
            ;; \
    esac

# VERL runtime deps needed by `train_chess.sh` (kept minimal to reduce ARM64 build pain).
RUN pip3 install --no-cache-dir \
    codetiming \
    datasets \
    dill \
    hydra-core \
    pandas \
    peft \
    'pyarrow>=19.0.0' \
    pybind11 \
    pylatexenc \
    'tensordict>=0.8.0,<=0.10.0,!=0.9.0' \
    torchdata \
    wandb \
    'packaging>=20.0' \
    tensorboard \
    huggingface_hub

# The base image uses a non-shell entrypoint. Make this image usable for
# training workflows by default.
ENTRYPOINT ["bash", "-lc"]
CMD ["python3 -V"]
