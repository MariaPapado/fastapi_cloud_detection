FROM python:3.12-slim
WORKDIR /api

ARG MODELS_PORT
ENV MODELS_PORT=${MODELS_PORT}

COPY . /api

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN yes | ln -s /usr/bin/python3 /usr/bin/python && python -m pip install poetry

#ARG CODEARTIFACT_TOKEN
#RUN poetry config http-basic.aws aws ${CODEARTIFACT_TOKEN} > /dev/null 2>&1
RUN poetry config virtualenvs.create false && poetry install --no-root

# CUDA
#RUN yes | pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113 --default-timeout=3600
#RUN yes | pip install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0 segmentation_models_pytorch pytorch_lightning --extra-index-url https://download.pytorch.org/whl/cpu

CMD uvicorn app:app --host 0.0.0.0 --port ${MODELS_PORT}
