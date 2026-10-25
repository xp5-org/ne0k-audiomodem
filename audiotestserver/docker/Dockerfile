# Audio Test Server / Client node — Debian 13 (trixie)
FROM debian:13-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    # ---- node runtime defaults (override with -e) ----
    PORT=8081 \
    APP_DIR=/testsrc \
    PULSE_SERVER=tcp:host.docker.internal:4713

# ---- System deps -----------------------------------------------------------
#  libportaudio2 / libsndfile1 : runtime libs for sounddevice + soundfile
#  pulseaudio-utils            : `pactl`/`paplay` + PulseAudio client that talks
#                                to the host's PulseAudio server over TCP
#  alsa-utils                  : optional, for --device /dev/snd passthrough
#  openssh-server              : LLM shell access into the container
#  curl / procps               : reload calls + process inspection
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-dev \
        libportaudio2 libsndfile1 \
        pulseaudio-utils alsa-utils \
        openssh-server curl procps ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# ---- Python venv (PEP 668: don't touch the system interpreter) -------------
RUN python3 -m venv "$VIRTUAL_ENV"
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# ---- SSH server ------------------------------------------------------------
#  Key-only root login. authorized_keys is bind-mounted at runtime from ./ssh.
RUN mkdir -p /run/sshd /root/.ssh \
    && sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config \
    && sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/'  /etc/ssh/sshd_config \
    && chmod 700 /root/.ssh

# ---- PulseAudio client config ----------------------------------------------
# PULSE_SERVER (set via compose) points sounddevice/PortAudio at the host.
COPY docker/pulse-client.conf /etc/pulse/client.conf

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

# The project itself is bind-mounted here at runtime (live code editing).
WORKDIR /testsrc
EXPOSE 8081 22

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
