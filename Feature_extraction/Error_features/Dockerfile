ARG MODEL_IMAGE="model-dl"

FROM python:3.10 as env

# Install system dependencies
RUN apt-get update && \
    apt-get install -y \
    g++ \
    libffi-dev \
    musl-dev \
    git \
    git-lfs \
    locales && \
    git lfs install && \
    echo "LC_ALL=en_US.UTF-8" >> /etc/environment && \
    echo "en_US.UTF-8 UTF-8" >> /etc/locale.gen && \
    echo "LANG=en_US.UTF-8" > /etc/locale.conf && \
    locale-gen en_US.UTF-8

ENV PYTHONIOENCODING=utf-8
ENV MKL_NUM_THREADS=""

WORKDIR /app

RUN adduser --system --group app --home /home/app && chown -R app:app /app
USER app

ENV PATH="/home/app/.local/bin:${PATH}"

COPY --chown=app:app requirements.txt .
RUN pip install --user swig~=3.0.12 && \
    pip install --user -r requirements.txt && \
    rm requirements.txt && \
    python -c "import nltk; nltk.download(\"punkt\")"

FROM busybox as model-cp

ARG MODEL_DIR=./models

COPY ${MODEL_DIR} /models

FROM alpine as model-dl
RUN apk update && \
    apk add git git-lfs yq && \
    git lfs install

ARG GEC_CONFIG=""
ARG SPELLER_CONFIG=""

COPY models models

RUN if [ -n "$GEC_CONFIG" ]; then \
    GEC_MODEL=$(yq '.huggingface' ${GEC_CONFIG}) && \
    git lfs clone --progress https://huggingface.co/$GEC_MODEL models/$GEC_MODEL; \
    fi && \
    if [ -n "$SPELLER_CONFIG" ]; then \
    SPELLER_MODEL=$(yq '.huggingface' ${SPELLER_CONFIG}) && \
    git lfs clone --progress https://huggingface.co/$SPELLER_MODEL models/$SPELLER_MODEL; \
    fi

FROM $MODEL_IMAGE as model

FROM env

COPY --chown=app:app --from=model /models /app/models
COPY --chown=app:app . .

ENTRYPOINT ["python", "main.py"]