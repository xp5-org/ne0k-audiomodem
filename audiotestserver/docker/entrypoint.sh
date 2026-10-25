#!/usr/bin/env bash
# Entrypoint for an audio test node.
#
#   ROLE         server | client            (default: server)
#   NODE_ID      node identity              (default: role name)
#   PORT         Flask/SocketIO bind port   (default: 8001)
#   SERVER_IP    server address, client role only (enables auto-connect)
#   APP_DIR      dir containing app.py      (default: auto-detect)
#   USERPASSWORD if set, enable SSH password login for root with this password
#   SSH_KEY_DIR  dir holding a public key / authorized_keys to install
#
# Python is supervised in a restart loop so that POST /api/reload (which exits
# the interpreter) picks up freshly bind-mounted code. sshd runs independently,
# so an LLM's SSH session survives the reload.
set -u

ROLE="${ROLE:-server}"
NODE_ID="${NODE_ID:-$ROLE}"
PORT="${PORT:-8001}"

# --- locate app.py ----------------------------------------------------------
# Honor APP_DIR if given, else probe the usual mount points.
if [ -n "${APP_DIR:-}" ] && [ -f "${APP_DIR}/app.py" ]; then
    :
else
    APP_DIR=""
    for d in /testsrc /audiotestserver /app; do
        if [ -f "$d/app.py" ]; then APP_DIR="$d"; break; fi
    done
fi
if [ -z "$APP_DIR" ] || [ ! -f "$APP_DIR/app.py" ]; then
    echo "[entrypoint] FATAL: app.py not found. Mount the source and/or set APP_DIR." >&2
    echo "[entrypoint] looked in: \$APP_DIR, /testsrc, /audiotestserver, /app" >&2
    exit 1
fi
echo "[entrypoint] APP_DIR=$APP_DIR"

# --- SSH auth: public key -------------------------------------------
mkdir -p /root/.ssh && chmod 700 /root/.ssh
install_key() {
    local src="$1"
    [ -f "$src" ] && cat "$src" >> /root/.ssh/authorized_keys
}
install_key /root/.ssh/authorized_keys.mounted     # compose bind-mount
if [ -n "${SSH_KEY_DIR:-}" ] && [ -d "$SSH_KEY_DIR" ]; then
    install_key "$SSH_KEY_DIR/authorized_keys"
    for pub in "$SSH_KEY_DIR"/*.pub; do install_key "$pub"; done
fi
[ -f /root/.ssh/authorized_keys ] && chmod 600 /root/.ssh/authorized_keys

# --- SSH auth: password (opt-in via USERPASSWORD) ---------------------------
if [ -n "${USERPASSWORD:-}" ]; then
    echo "root:${USERPASSWORD}" | chpasswd
    sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config
    sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
    echo "[entrypoint] SSH password auth ENABLED for root"
fi

# --- start sshd -------------------------------------------------------------
ssh-keygen -A >/dev/null 2>&1
/usr/sbin/sshd
echo "[entrypoint] sshd started (port 22)"

# --- build node arguments ---------------------------------------------------
ARGS=("$ROLE" --id "$NODE_ID" --port "$PORT")
if [ "$ROLE" = "client" ] && [ -n "${SERVER_IP:-}" ]; then
    ARGS+=(--host "$SERVER_IP")
fi

echo "[entrypoint] node role=$ROLE id=$NODE_ID port=$PORT server=${SERVER_IP:-n/a}"
echo "[entrypoint] PULSE_SERVER=${PULSE_SERVER:-unset}"

# --- supervised restart loop ------------------------------------------------
trap 'echo "[entrypoint] stopping"; exit 0' TERM INT
while true; do
    python3 "$APP_DIR/app.py" "${ARGS[@]}"
    code=$?
    echo "[entrypoint] app.py exited (code=$code) — relaunching in 1s"
    sleep 1
done
