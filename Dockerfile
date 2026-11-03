# fibbot: Fibonacci-retracement PAPER-trading bot. No live trading code exists in this image.
#   docker build -t fibbot .
#   docker run -d --init --stop-timeout 300 --restart unless-stopped -e ANTHROPIC_API_KEY \
#     -v fibbot-data:/data fibbot
# All state (candles, SQLite DB, brain notebook, REPORT.md) lives in the /data volume.
# --init forwards docker stop's SIGTERM to the bot, which then finishes its current step (a
# Claude cycle wraps up at its next turn) and exits. Docker's default grace period is only 10 s
# before SIGKILL, which would throw away a paid cycle in progress: --stop-timeout 300 (compose:
# stop_grace_period: 5m) gives it time. A SIGKILL never corrupts the state (SQLite + atomic
# writes); it only loses the unfinished step.
# Do not set "home" in a mounted /app/fibbot.json: the state belongs in the volume
# (FIBBOT_HOME=/data, which takes precedence over a config file's "home" anyway).
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
