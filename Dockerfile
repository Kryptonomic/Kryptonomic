# fibbot: Fibonacci-retracement PAPER-trading bot. No live trading code exists in this image.
#   docker build -t fibbot .
#   docker run -d --init --restart unless-stopped -e ANTHROPIC_API_KEY -v fibbot-data:/data fibbot
# All state (candles, SQLite DB, brain notebook, REPORT.md) lives in the /data volume.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

# Non-root user; /data is created and owned by it BEFORE the VOLUME line so a new named volume
# inherits that ownership.
RUN useradd --create-home --uid 10001 --user-group fibbot \
    && mkdir /data && chown fibbot:fibbot /data

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY fibbot/ fibbot/

ENV FIBBOT_HOME=/data
VOLUME /data
USER fibbot

CMD ["python", "-m", "fibbot", "run"]
