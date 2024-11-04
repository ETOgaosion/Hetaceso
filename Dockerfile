FROM nvcr.io/nvidia/pytorch:24.04-py3

COPY ./ /workspace/Hetaceso/

RUN pip install -e /workspace/Hetaceso/external/transformer_engine/

WORKDIR /workspace