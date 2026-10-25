# Dockerized audio test bench

One image (`docker/Dockerfile`, Debian 13) runs either role of `app.py`.
`docker-compose.yml` brings up a **server** plus one **client**; copy the
`client1` block for more clients.

```
docker compose up --build
```

- Server UI/API: http://localhost:8001
- Client1 UI:    http://localhost:8011
- Server SSH:    `ssh -i ssh/id_ed25519 -p 2222 root@localhost`

The project directory is bind-mounted at `/audiotestserver`, so edits on the
host are visible in the containers immediately (no rebuild).

---

## Audio: connecting host mic + speaker

Containers can't touch host audio hardware directly (on macOS Docker runs in a
VM). The portable answer for **both macOS and Linux** is to run a PulseAudio
server on the host over TCP; each container's `PULSE_SERVER` env var
(`tcp:host.docker.internal:4713`, already set in compose) points at it, and
`sounddevice`/PortAudio inside the container then sees the host's devices.

### macOS host
```sh
brew install pulseaudio
pulseaudio --load="module-native-protocol-tcp auth-ip-acl=127.0.0.1;172.16.0.0/12 port=4713" \
           --load=module-coreaudio-detect \
           --exit-idle-time=-1 --daemonize
```
Grant microphone permission to the terminal/`pulseaudio` process in
System Settings → Privacy & Security → Microphone.

### Linux host (PulseAudio or PipeWire-pulse)
```sh
pactl load-module module-native-protocol-tcp auth-ip-acl=127.0.0.1;172.16.0.0/12 port=4713
```
The `172.16.0.0/12` ACL covers Docker's default bridge subnets. Tighten it to
your actual `docker network inspect` subnet if you prefer.

> Linux-only alternative: instead of TCP PulseAudio you can pass ALSA hardware
> straight through by adding to a service:
> ```yaml
>     devices: ["/dev/snd:/dev/snd"]
>     group_add: ["audio"]
> ```
> and dropping `PULSE_SERVER`. This does **not** work on macOS.

### Verify audio from inside a container
```sh
ssh -i ssh/id_ed25519 -p 2222 root@localhost
pactl info                 # should print the host server
python3 -c "import sounddevice as sd; print(sd.query_devices())"
```

---

## The `/reload` workflow (live code updates)

`app.py` is supervised by `docker/entrypoint.sh` in a restart loop, with `sshd`
running separately. To apply host-side code edits without a container rebuild:

```sh
ssh -i ssh/id_ed25519 -p 2222 root@localhost
curl -X POST http://127.0.0.1:8001/api/reload
```

`POST /api/reload` stops any running test subprocesses, then exits the Python
interpreter; the entrypoint loop relaunches it against the fresh bind-mounted
code. Because `sshd` is independent of the Python process, the SSH session
stays alive across the reload.

To reload a **client** node, SSH to the server container and curl the client by
its compose service name, e.g. `curl -X POST http://client1:8001/api/reload`
(all nodes share the compose network).

---

## SSH keys

`ssh/id_ed25519{,.pub}` generated in the project root. The **public** key
is bind-mounted into the server container as `authorized_keys`; the **private**
key stays on the host for the LLM to use. Keep the private key out of version
control (it's covered by `.gitignore`).
