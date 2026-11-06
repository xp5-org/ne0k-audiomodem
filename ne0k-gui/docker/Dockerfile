# modemgui dev/runtime image — Debian 13, same base as ../../docker/Dockerfile.
#
# BUILD CONTEXT IS THE REPO ROOT (/audiotestserver), not this directory, because
# the C cores live in ../tests/{OFDM,CSS}/c and are compiled, not copied:
#
#     docker build -f modemgui/docker/Dockerfile -t modemgui .                  headless
#     docker build -f modemgui/docker/Dockerfile --target gui -t modemgui:gui . + Qt GUI
#
# Stages: `engine` (C cores, audio, headless tests) -> `gui` (+PySide6, offscreen
# tests, X11/Wayland run).  The LAST stage is an alias of `engine`, so a plain
# `docker build` stays small.
#
# At run time compose bind-mounts the tree over /audiotestserver for live edits;
# the entrypoint then runs an incremental `make`, so editing a C file and
# restarting the container is the whole rebuild loop.
FROM debian:13-slim AS engine

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

#  gcc / make / libc6-dev : build the C cores
#  libportaudio2          : `sd:` connector (system audio devices, ALSA)
#  pulseaudio-utils       : `pulse:` connector -- pactl / parec / pacat, which
#                           honour PULSE_SERVER (TCP to the host, or a mounted socket)
#  alsa-utils             : aplay -l etc., for diagnosing --device /dev/snd passthrough
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv \
        gcc make libc6-dev \
        libportaudio2 pulseaudio-utils alsa-utils \
        procps ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VIRTUAL_ENV"
COPY modemgui/requirements.txt /tmp/requirements.txt
# GUI-only lines (PySide6, pyqtgraph, pytest-qt) are filtered out of the engine stage.
RUN grep -viE '^(PySide6|pyqtgraph|pytest-qt)' /tmp/requirements.txt > /tmp/req-engine.txt \
    && pip install --upgrade pip && pip install -r /tmp/req-engine.txt

# No local daemon in the container: never autospawn one, and no shared memory
# (it cannot cross the TCP / container boundary).
COPY modemgui/docker/pulse-client.conf /etc/pulse/client.conf

# Baked copy so `docker run` works without the bind mount; compose overrides it.
COPY tests/c        /audiotestserver/tests/c
COPY tests/OFDM/c   /audiotestserver/tests/OFDM/c
COPY tests/CSS/c    /audiotestserver/tests/CSS/c
COPY modemgui       /audiotestserver/modemgui
WORKDIR /audiotestserver/modemgui
RUN make

COPY modemgui/docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
CMD ["python", "-m", "modemgui", "devices"]

# ---------------------------------------------------------------------------
FROM engine AS gui
#  libegl1 libgl1          : Qt's GL stack (PySide6 wheels bundle Qt, not the system GL)
#  libxkbcommon-x11-0 libxcb-cursor0 : the xcb platform plugin (X11 / xrdp / XWayland)
#  libdbus-1-3 libfontconfig1 fonts-dejavu-core : plugin deps, and text that renders
RUN apt-get update && apt-get install -y --no-install-recommends \
        libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0 libxcb-cursor0 \
        libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 \
        libxcb-shape0 libxcb-xinerama0 libxcb-xkb1 libdbus-1-3 libfontconfig1 \
        fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/*
RUN pip install -r /tmp/requirements.txt
# Tests and CI run headless; override with QT_QPA_PLATFORM=xcb to show the window.
ENV QT_QPA_PLATFORM=offscreen
CMD ["python", "-m", "pytest", "-q"]

# Default target for a plain `docker build`.
FROM engine AS default
