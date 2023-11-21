ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:23.10-py3
FROM ${FROM_IMAGE_NAME}

WORKDIR /workspace/lora

RUN git clone https://github.com/NVIDIA/NeMo && \
    cd NeMo && \
    pip install --no-build-isolation -e ".[nlp]"

ENV PYTHONPATH /workspace/lora/NeMo/.:${PYTHONPATH}    
ADD . /workspace/lora

