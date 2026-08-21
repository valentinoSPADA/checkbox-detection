# Detection sidecar: OpenCV proposals + ONNX classification.
#
# Single stage. A multi-stage build buys little here because the runtime genuinely needs the
# same wheels the "build" would install -- there is no compiled artifact to hand over -- and
# the size win comes instead from never installing the training dependencies at all: torch
# and its transitive graph are roughly a gigabyte, and requirements.txt deliberately excludes
# them (see requirements-train.txt).
FROM python:3.12-slim AS runtime

# Keep Python from writing .pyc into the layer and from buffering logs, which would delay
# every log line until the container exits -- exactly when it is least useful.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # onnxruntime and OpenCV each spawn a thread pool sized to the host's core count. Two
    # pools competing for the same cores makes latency worse under concurrency, not better,
    # so both are pinned to one thread and parallelism is left to the number of workers.
    OMP_NUM_THREADS=1 \
    ORT_THREADS=1

WORKDIR /app

# Requirements are copied first so the dependency layer is cached independently of source
# changes; without this split every edit to engine/ reinstalls OpenCV.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY engine/ ./engine/
COPY app/ ./app/
COPY models/ ./models/

# Run as an unprivileged user. The service accepts untrusted uploads and decodes them through
# native image codecs, which is precisely the code path where running as root turns a memory
# bug into a container compromise.
RUN useradd --create-home --uid 10001 detector && chown -R detector:detector /app
USER detector

EXPOSE 8000

# The healthcheck asks the app, not the port: a process that is listening but failed to load
# its model is not healthy, and only /health knows the difference.
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys,json; \
r=json.load(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)); \
sys.exit(0 if r.get('model_loaded') else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
