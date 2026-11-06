"""Find an 86Box process and the PTYs behind its "Virtual Console (COM)" ports.

86Box's stdio COM device (mode 3 / CMD) creates one PTY per serial port and
holds the MASTER end open for as long as the VM runs; nothing labels which
master is which port. Everything here reads the kernel's own fd table, so it
needs no 86Box rebuild and no log line (char_stdio.c's "Bound to /dev/pts/N" is
compiled out of the regular build).

Linux: /proc/<pid>/cmdline to find the `86box -P <vm_path>` process, then
/proc/<pid>/fd + fdinfo ("tty-index:") for each ptmx master.
macOS: `ps -ww` for the process, `lsof` for the masters (the DEVICE minor number
is the /dev/ttysNNN slave index). The macOS path is written from the approach
serial_bridge.py (in the 86box_null_modem container) verified by hand; it has
not been exercised from this container.

POSIX only -- Windows has no PTYs to find.
"""
import os
import re
import subprocess
import sys
import time

IS_MACOS = sys.platform == "darwin"
TTY_INDEX_RE = re.compile(r"^tty-index:\s*(\d+)", re.MULTILINE)


def pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _find_pid_linux(vm_path):
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                args = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
        except OSError:
            continue
        if "-P" in args:
            i = args.index("-P")
            if i + 1 < len(args) and os.path.abspath(args[i + 1]) == vm_path:
                return int(entry)
    return None


def _find_pid_macos(vm_path):
    try:
        out = subprocess.run(["ps", "-ww", "-Ao", "pid=,command="],
                             capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    for line in out.splitlines():
        pid_str, _, cmd = line.strip().partition(" ")
        args = cmd.split()
        if "-P" in args:
            i = args.index("-P")
            if i + 1 < len(args) and os.path.abspath(args[i + 1]) == vm_path:
                try:
                    return int(pid_str)
                except ValueError:
                    continue
    return None


def find_86box_pid(vm_path):
    """Pid of the process launched as `86box -P <vm_path> ...`, or None."""
    vm_path = os.path.abspath(vm_path)
    return _find_pid_macos(vm_path) if IS_MACOS else _find_pid_linux(vm_path)


def list_master_ptys(pid):
    """[(fd, slave_path)] for every ptmx master the process holds, by fd."""
    found = []
    if IS_MACOS:
        try:
            out = subprocess.run(["lsof", "-a", "-p", str(pid), "-d", "0-1024"],
                                 capture_output=True, text=True, timeout=10).stdout
        except (OSError, subprocess.SubprocessError):
            return []
        for line in out.splitlines():
            parts = line.split()
            if not parts or parts[-1] != "/dev/ptmx":
                continue
            fdm = re.match(r"\d+", parts[3]) if len(parts) > 3 else None
            dev = next((p for p in parts if "," in p and p.replace(",", "").isdigit()), None)
            if fdm and dev:
                found.append((int(fdm.group()), f"/dev/ttys{int(dev.split(',')[1]):03d}"))
        return sorted(found)
    fd_dir = f"/proc/{pid}/fd"
    try:
        fds = os.listdir(fd_dir)
    except OSError:
        return []
    for fd in fds:
        try:
            if os.path.basename(os.readlink(os.path.join(fd_dir, fd))) != "ptmx":
                continue
            with open(f"/proc/{pid}/fdinfo/{fd}") as f:
                m = TTY_INDEX_RE.search(f.read())
        except OSError:
            continue
        if m:
            found.append((int(fd), f"/dev/pts/{m.group(1)}"))
    return sorted(found)


def discover_ptys(vm_path, count, timeout):
    """Wait for the VM's process, then for `count` PTY masters; return the
    slave paths in ascending master-fd order. RuntimeError if the process never
    appears or holds a different number of masters than asked for."""
    vm_path = os.path.abspath(vm_path)
    deadline = time.time() + timeout
    pid, masters = None, []
    while time.time() < deadline:
        if pid is None:
            pid = find_86box_pid(vm_path)
        if pid is not None:
            masters = list_master_ptys(pid)
            if len(masters) >= count:
                break
            if not pid_alive(pid):
                pid = None
        time.sleep(0.5)
    if pid is None:
        raise RuntimeError(f"no 86Box process with -P {vm_path}")
    if len(masters) != count:
        raise RuntimeError(f"86Box pid {pid} holds {len(masters)} PTY master(s) "
                           f"{[p for _, p in masters]} but {count} modem(s) need one; "
                           "set an explicit \"pty\" per modem if this is expected")
    return [p for _, p in masters]


def open_raw(path):
    """Open a PTY slave as a raw byte pipe: no echo, no line discipline."""
    import termios
    fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    return fd
