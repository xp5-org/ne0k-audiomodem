# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG
"""modemgui — desktop/web front end for the OFDM and CSS modems.

Layering (bottom up):

    audio.base / audio.*   connectors: system audio, PulseAudio, file, SDR plugins
    audio.pipe             AudioPipe: one RX source + one TX sink, fan-out taps
    native                 ctypes loaders for the C cores in ../tests/{OFDM,CSS}/c
    (future) dsp, ui

Nothing above the connector layer knows what a "device" physically is.
"""
__version__ = "0.1.0"
APP_NAME = "NE0K Modem"          # what users see; the python package keeps the name modemgui
