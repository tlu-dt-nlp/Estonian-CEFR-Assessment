# Grammatical Error Correction

This repository contains code for running Estonian spell-checking and grammatical error correction (GEC) models to
process incoming requests. These error-correction models can be called individually or combined in a sequence. The
worker is compatible with our [GEC API](https://ghcr.io/tartunlp/grammar-api) and can be used to process requests from
RabbitMQ. You can find the example for offline usage in both the Colab file named `GEC_and_spell_demo.ipynb` and the
Python script named `example.py`.

The GEC implementation uses Transformer-based machine translation models to normalize the input text, the models are
trained using custom [modular NMT implementation of FairSeq](https://github.com/TartuNLP/fairseq). Statistical spelling correction relies on the Jamspell algorithm that analyzes word contexts based on a trigram language
model. The toolkit also contains a part-of-speech n-gram based error detection tool (see the `posgram_finder` directory).

The project is developed by the [NLP research group](https://tartunlp.ai/) at
the [University of Tartu](https://ut.ee/) and the [language technology research group](https://elle.tlu.ee/about/people)
at the [Tallinn University](https://tlu.ee).

## Setup

The Estonian GEC worker can be used by running the prebuilt images.

There are two separate images:

- [`grammar-worker`](https://ghcr.io/tartunlp/grammar-worker) (documented below)
- [`grammar-model`](https://ghcr.io/tartunlp/grammar-model) [DEPRECATED]

The worker can be set up using the [`grammar-worker`](https://ghcr.io/tartunlp/grammar-worker)
image. The base image contains only the environment setup and code to run the models, and is designed to be used in a
CPU
environment. The container should be configured using the following parameters:

- Environment variables:
    - Variables that configure the connection to a [RabbitMQ message broker](https://www.rabbitmq.com/):
        - `MQ_USERNAME` - RabbitMQ username
        - `MQ_PASSWORD` - RabbitMQ user password
        - `MQ_HOST` - RabbitMQ host
        - `MQ_PORT` (optional) - RabbitMQ port (`5672` by default)
        - `MQ_EXCHANGE` (optional) - RabbitMQ exchange name (`grammar` by default)
        - `MQ_CONNECTION_NAME` (optional) - friendly connection name (`GEC worker` by default)
        - `MQ_HEARTBEAT` (optional) - heartbeat interval (`60` seconds by default)
    - PyTorch-related variables:
        - `MKL_NUM_THREADS` (optional) - number of threads used for intra-op parallelism by PyTorch. This defaults to
          the number of CPU cores which may cause computational overhead when deployed on larger nodes. Alternatively,
          the `docker run` flag `--cpuset-cpus` can be used to control this. For more details, refer to
          the [performance and hardware requirements](#performance-and-hardware-requirements) section below.
    - Worker-related variables:
        - `WORKER_MAX_INPUT_LENGTH` (optional) - the number of characters allowed per request (`10000` by default).
          Longer requests will return validation errors.

- Optional runtime flags (the `COMMAND` option):
    - `--gec-model-config` - path to the GEC model config file. Compatible files are included in the `models/`
      directory
      and the format is described in `models/README.md`.
    - `--spell-model-config` - path to the spell-checking model config file. Compatible files are included in the
      `models/` directory and the format is described in `models/README.md`.
    - `--correction-list-config` - path to the correction list config file. Compatible files are included in the
      `models/` directory and the format is described in `models/README.md
    - `--log-config` - path to logging config files (`logging/logging.ini` by default), `logging/debug.ini` can be used
      for debug-level logging
    - `--port` - port of the healthcheck probes (`8000` by default):

- Endpoints for healthcheck probes:
    - `/health/startup` - NB! When Speller is used, the startup probe may take up to 10 minutes to complete.
    - `/health/readiness`
    - `/health/liveness`

The main image contains no models. There are additional images with suffix `et` that contain the default Estonian
models. Alternatively models can be mounted to the container at `/app/models` using a volume or they will be downloaded
automatically from HuggingFace upon startup.

### Sample configuration

As the worker is designed to be used with the [GEC API](https://ghcr.io/tartunlp/grammar-api), the following docker
compose sample configuration can be used to set up the worker and the API:

```yaml
version: '3'
services:
  rabbitmq:
    image: 'rabbitmq'
  gec_api:
    image: ghcr.io/tartunlp/grammar-api:latest
    environment:
      - MQ_HOST=rabbitmq
      - MQ_PORT=5672
    ports:
      - '80:8000'
    depends_on:
      - rabbitmq
    restart: always
  gec_worker:
    image: ghcr.io/tartunlp/grammar-worker:et
    environment:
      - MQ_HOST=rabbitmq
      - MQ_PORT=5672
      - MKL_NUM_THREADS=16
    depends_on:
      - rabbitmq
    restart: always
```

### Building new images

The image can be built with and without including models. The following build-time arguments can be used to configure
the default build:

- `MODEL_IMAGE` - the image name where the model is copied from. By default, uses the `model-dl` stage (described
  below).
- `GEC_CONFIG` - the path to the GEC model config file. If specified, will download the model files from HuggingFace
  and include them in the image. Only used if `MODEL_IMAGE` equals `model-dl`.
- `SPELL_CONFIG` - the path to the spell-checking model config file. If specified, will download the model files from
  HuggingFace and include them in the image. Only used if `MODEL_IMAGE` equals `model-dl`.
- `MODEL_DIR` - the directory where the model files are located. By default, uses the `models/` directory. Only used if
  `MODEL_IMAGE` equals `model-cp`.

Intermediate build targets:

- `env` - an intermediate build stage with all packages installed, but no code.
- `model-cp` - an optional stage to copy models from a local directory.
- `model` - an alias for the model image, the value of `MODEL_IMAGE` or `model-dl` by default.

To skip unnecessary stages, BuildKit should be enabled.

## Manual / development setup

For a manual setup, use a python environment with Git LFS, Python 3.10, install Swig 3.0 for JamSpell and requirements
from `requirements.txt`. To use a GPU, make sure [CUDA](https://developer.nvidia.com/cuda-downloads) is installed.

```shell
sudo apt-get install git-lfs
git lfs install
conda create -n grammar-worker python=3.10
conda activate grammar-worker
pip install swig~=3.0.10
pip install -r requirements.txt
```

Model files will be downloaded automatically from HuggingFace upon startup. Alternatively, you can download model files
manually. For more information about models, please refer to `models/README.md`.

To initialize the sentence splitting functionality, the following command should be run before starting the application:

```shell
python -c "import nltk; nltk.download(\"punkt\")"
```

RabbitMQ and PyTorch parameters should be configured with environment variables as described above or in an `.env` file
in the root folder of the repository. The worker can be started with:

```shell
python main.py [--gec-model-config models/gec_model_config.yaml] [--spell-model-config models/spell_model_config.yaml] [--correction-list-config models/correction_list_min3.yaml] [--log-config logging/logging.ini]
```

Or you can run the test script which does not require RabbitMQ:

```shell
python -m unittest test.py
```

Alternatively, you may refer to the example usage:

```shell
python example.py
```

### Performance and Hardware Requirements

The exact RAM usage depends on the model and should always be tested, but a conservative estimate is to have **12 GB of
memory** available.

The performance depends on the available CPU resources, however, this should be fine-tuned for the deployment
infrastructure. By default, PyTorch will try to utilize all CPU cores to 100% and run as many threads as there are
cores. This can cause major computational overhead if the worker is deployed on large nodes. The **number of threads
used should be limited** using the `MKL_NUM_THREADS` environment variable or the `docker run` flag `--cpuset-cpus`.

Limiting CPU usage by docker configuration which only limits CPU shares is not sufficient (e.g. `docker run` flag
`--cpus` or the CPU limit in K8s, unless the non-default
[static CPU Manager policy](https://kubernetes.io/docs/tasks/administer-cluster/cpu-management-policies/) is used). For
example, on a node with 128 cores, setting the CPU limit at `16.0` results in 128 parallel threads running with each one
utilizing only 1/8 of each core's computational potential. This amplifies the effect of multithreading overhead and can
result in inference speeds up to 20x slower than expected.

Although the optimal number of threads depends on the exact model and infrastructure used, a good starting point is
around `16`. With optimal configuration and modern hardware, the worker should be able to process ~7 sentences per
second. For more information, please refer to
[PyTorch documentation](https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html).

### Request Format

The worker consumes GEC requests from a RabbitMQ message broker and responds with the replacement suggestions. The
following format is compatible with the [GEC API](https://ghcr.io/tartunlp/grammar-api).

Requests should be published with the following parameters:

- Exchange name: `grammar` (exchange type is `direct`)
- Routing key: `grammar.<language>` where `<language>` refers to 2-letter ISO language code of the input text. For
  example `grammar.et`.
- Message properties:
    - Correlation ID - a UID for each request that can be used to correlate requests and responses.
    - Reply To - name of the callback queue where the response should be posted.
    - Content Type - `application/json`
- JSON-formatted message content with the following keys:
    - `text` – input text, either a string or a list of strings which are allowed to contain multiple sentences or
      paragraphs.
    - `language` – 2-letter ISO language code

The worker will return a response with the following parameters:

- Exchange name: (empty string)
- Routing key: the Reply To property value from the request
- Message properties:
    - Correlation ID - the Correlation ID value of the request
    - Content Type - `application/json`
- JSON-formatted message content with the following keys:
    - `status` - a human-readable status message, `OK` by default
    - `status_code` – (integer) a HTTP status code, `200` by default
    - `corrections` - A list of corrections formatted as the POST request output defined in
      the [API](https://github.com/tartunlp/grammar-api). May be `null` in case `status_code!=200`


