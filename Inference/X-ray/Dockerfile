FROM --platform=linux/amd64 pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)

# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

# Ensures that NVIDIA runtime is used
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user models /opt/app/models
COPY --chown=user:user packages /opt/app/packages

COPY --chown=user:user requirements.txt /opt/app/
COPY --chown=user:user utils/utils.py /opt/app/utils/
COPY --chown=user:user inference.py /opt/app/
COPY --chown=user:user Xray_inference_nnunet.py /opt/app/

RUN pip install --no-index --find-links=/opt/app/packages --user --no-cache-dir --no-color --requirement /opt/app/requirements.txt
ENTRYPOINT ["python", "inference.py"]
