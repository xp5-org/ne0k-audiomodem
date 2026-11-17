"""audio/hostmixer.py: address routing and the pactl / amixer parsers (no OS mixer needed)."""
from modemgui.audio import SINK, SOURCE, hostmixer as hm


def test_route_picks_the_backend_from_the_address():
    assert hm.route("pulse:alsa_input.x", SOURCE) == ("pulse", "alsa_input.x")
    assert hm.route("pulse:", SINK) == ("pulse", "")
    assert hm.route("sd:0", SOURCE, "MacBook Air Microphone", "darwin") == ("coreaudio", "MacBook Air Microphone")
    assert hm.route("sd:12", SOURCE, "pulse", "linux") == ("pulse", "")
    assert hm.route("sd:16", SINK, "default", "linux") == ("pulse", "")
    assert hm.route("sd:0", SOURCE, "HDA Intel PCH: 92HD90BXX Analog (hw:0,0)", "linux") == ("alsa", "0")
    assert hm.route("sd:3", SINK, "HDA NVidia: HDMI 0 (hw:1,8)", "linux") == ("alsa", "1")
    assert hm.route("sd:1", SOURCE, "Microphone", "win32")[0] == "none"
    assert hm.route("loop:a", SOURCE)[0] == "none"
    assert hm.route(None, SOURCE)[0] == "none"


def test_pactl_volume_and_mute():
    assert hm.parse_pactl_volume("Volume: mono: 49152 /  75% / -7.50 dB\n") == (0.75, -7.5)
    v, db = hm.parse_pactl_volume("Volume: front-left: 65536 / 100% / 0.00 dB,   front-right: 32768 /  50% / "
                                  "-18.06 dB\n        balance -1.00\n")
    assert v == 0.75 and abs(db - -9.03) < 1e-6
    v, db = hm.parse_pactl_volume("Volume: mono: 0 /   0% / -inf dB")
    assert v == 0.0 and db == float("-inf")
    assert hm.parse_pactl_mute("Mute: yes") is True and hm.parse_pactl_mute("Mute: no") is False
    assert hm.parse_pactl_mute("") is None


def test_pulse_agc_from_echo_cancel():
    mods = "7\tmodule-alsa-card\tdevice_id=0\n22\tmodule-echo-cancel\taec_method=webrtc source_name=ec_src\n"
    assert hm.pulse_agc("alsa_input.pci.analog-stereo", "", None) == ("off", [])
    agc, notes = hm.pulse_agc("alsa_input.pci.analog-stereo", mods, None)
    assert agc == "off" and "not its source" in notes[0]
    assert hm.pulse_agc("echo-cancel-source", mods, None)[0] == "on"
    off = mods.replace("aec_method=webrtc", "aec_method=webrtc aec_args=analog_gain_control=0 agc=0")
    assert hm.pulse_agc("echo-cancel-source", off, None)[0] == "off"
    js = '[{"name": "ec_src", "driver": "module-echo-cancel.c", "properties": {}}]'
    assert hm.pulse_agc("ec_src", mods, js)[0] == "on"


def test_amixer_scontrols_and_rows():
    txt = "Simple mixer control 'Master',0\nSimple mixer control 'Capture',0\nSimple mixer control 'Mic Boost',0\n"
    assert hm.parse_amixer_scontrols(txt) == ["Master", "Capture", "Mic Boost"]
    st = hm.MixerState(SOURCE, "pulse", device="x", volume=0.75, volume_db=-7.5, mute=False, settable=True,
                       agc="off", notes=["n1"])
    rows = dict(st.rows("audio.host.in"))
    assert rows["audio.host.in.volume"] == "0.75" and rows["audio.host.in.agc"] == "off"
    assert rows["audio.host.in.mute"] == "no" and rows["audio.host.in.notes"] == "n1"
    assert "audio.host.out.agc" not in dict(hm.MixerState(SINK, "pulse").rows("audio.host.out"))
    assert dict(hm.MixerState(SOURCE, "none", notes=["loop: is not a hardware device"]).rows("p")) == {
        "p.backend": "none", "p.notes": "loop: is not a hardware device"}


def test_get_never_raises_without_a_mixer():
    st = hm.get("loop:a", SOURCE)
    assert st.backend == "none" and st.volume is None


def test_notes_stay_on_one_line():
    st = hm.MixerState(SOURCE, "pulse", notes=["pactl x: Connection refused\npa_context_connect() failed"])
    assert "\n" not in dict(st.rows("p"))["p.notes"]


def test_host_env_restores_the_users_library_path_when_frozen():
    from modemgui.audio.sysenv import host_env
    base = {"LD_LIBRARY_PATH": "/app/_internal", "LD_LIBRARY_PATH_ORIG": "/opt/lib", "DYLD_LIBRARY_PATH": "/app",
            "PATH": "/bin"}
    env = host_env(base, frozen=True)
    assert env["LD_LIBRARY_PATH"] == "/opt/lib" and "DYLD_LIBRARY_PATH" not in env and env["PATH"] == "/bin"
    assert "LD_LIBRARY_PATH_ORIG" not in env
    assert host_env(base, frozen=False) == base


def test_alsa_notes_pick_the_right_side():
    assert hm.alsa_wanted("Internal Mic Boost", SOURCE) and hm.alsa_wanted("Capture", SOURCE)
    assert not hm.alsa_wanted("Capture", SINK) and not hm.alsa_wanted("Mic Boost", SINK)
    assert hm.alsa_wanted("Master", SINK) and hm.alsa_wanted("Speaker", SINK) and not hm.alsa_wanted("Master", SOURCE)


def test_name_address_resolves_to_the_device_the_stream_opens():
    import pytest
    pytest.importorskip("PySide6")
    from modemgui.audio import DeviceInfo
    from modemgui.ui.control_host import device_name
    devs = [DeviceInfo("sd", "0", "ONN43", SINK), DeviceInfo("sd", "1", "BlackHole 2ch", SINK),
            DeviceInfo("sd", "3", "MacBook Air Speakers", SINK, default=True)]
    assert device_name("sd:name=macbook", devs) == "MacBook Air Speakers"
    assert device_name("sd:1", devs) == "BlackHole 2ch"
    assert device_name("sd:name=nothing", devs) == ""
