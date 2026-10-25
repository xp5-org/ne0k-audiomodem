# SPDX-License-Identifier: MIT
# Copyright (c) 2026 XP5-ORG

import sys
import os
import json
import queue
import pygame
import pyaudio
import numpy as np

CONFIG_FILE = "spectrograph.json"

cfg = {
    "min_db": 20.0,
    "max_db": 100.0,
    "max_hz": 22050.0,
    "color": 1.0,
    "scroll": 1.0,
    "width": 800,
    "height": 672
}

if os.path.exists(CONFIG_FILE):
    try:
        with open(CONFIG_FILE, "r") as f:
            loaded = json.load(f)
            for k, v in loaded.items():
                if k in cfg:
                    cfg[k] = v
    except:
        pass

WIDTH = int(cfg["width"])
HEIGHT = int(cfg["height"])
UI_HEIGHT = 160
SPEC_HEIGHT = max(100, HEIGHT - UI_HEIGHT)

CHANNELS = 1
RATE = 44100
FPS = 30
chunk_size = RATE // FPS

audio_queue = queue.Queue()

def audio_callback(in_data, frame_count, time_info, status):
    audio_queue.put(in_data)
    return (None, pyaudio.paContinue)

p = pyaudio.PyAudio()

try:
    stream = p.open(
        format=pyaudio.paInt16,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=chunk_size,
        stream_callback=audio_callback
    )
    stream.start_stream()
except Exception as e:
    print(e)
    p.terminate()
    sys.exit(1)

pygame.init()

screen = pygame.display.set_mode(
    (WIDTH, HEIGHT),
    pygame.RESIZABLE | pygame.DOUBLEBUF,
    vsync=1
)

pygame.display.set_caption("Live Audio Spectrograph")

clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

spec_surface = pygame.Surface((WIDTH, SPEC_HEIGHT))
spec_surface.fill((0, 0, 0))

sliders = [
    {
        "key": "min_db",
        "label": "Min dB",
        "min": 0.0,
        "max": 140.0,
        "row": 0,
        "col": 0
    },
    {
        "key": "max_db",
        "label": "Max dB",
        "min": 0.0,
        "max": 140.0,
        "row": 0,
        "col": 1
    },
    {
        "key": "max_hz",
        "label": "Max Hz",
        "min": 500.0,
        "max": RATE / 2,
        "row": 1,
        "col": 0
    },
    {
        "key": "color",
        "label": "Color",
        "min": 0.0,
        "max": 2.0,
        "row": 1,
        "col": 1
    },
    {
        "key": "scroll",
        "label": "Scroll Width",
        "min": 1.0,
        "max": 8.0,
        "row": 2,
        "col": 0
    }
]

dragging = None

def get_rect(row, col, w, sh):
    margin_x = 50
    margin_y = 30
    col_w = w // 2

    return pygame.Rect(
        margin_x + col * col_w,
        sh + margin_y + row * 40,
        col_w - 2 * margin_x,
        10
    )

running = True

while running:

    if not pygame.mouse.get_pressed()[0]:
        dragging = None

    for event in pygame.event.get():

        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False

        elif event.type == pygame.VIDEORESIZE:

            WIDTH = event.w
            HEIGHT = event.h

            SPEC_HEIGHT = max(100, HEIGHT - UI_HEIGHT)

            screen = pygame.display.set_mode(
                (WIDTH, HEIGHT),
                pygame.RESIZABLE | pygame.DOUBLEBUF,
                vsync=1
            )

            new_spec = pygame.Surface((WIDTH, SPEC_HEIGHT))

            new_spec.blit(
                spec_surface,
                (
                    WIDTH - spec_surface.get_width(),
                    SPEC_HEIGHT - spec_surface.get_height()
                )
            )

            spec_surface = new_spec

            cfg["width"] = WIDTH
            cfg["height"] = HEIGHT

        elif event.type == pygame.MOUSEBUTTONDOWN:

            if event.button == 1:

                for s in sliders:

                    rect = get_rect(
                        s["row"],
                        s["col"],
                        WIDTH,
                        SPEC_HEIGHT
                    )

                    if rect.collidepoint(event.pos):
                        dragging = s["key"]

        elif event.type == pygame.MOUSEBUTTONUP:

            if event.button == 1:
                dragging = None

        elif event.type == pygame.MOUSEMOTION:

            if dragging:

                for s in sliders:

                    if s["key"] == dragging:

                        rect = get_rect(
                            s["row"],
                            s["col"],
                            WIDTH,
                            SPEC_HEIGHT
                        )

                        mx = max(
                            rect.left,
                            min(event.pos[0], rect.right)
                        )

                        val = float(mx - rect.left) / float(rect.width)

                        cfg[s["key"]] = (
                            s["min"] +
                            val * (s["max"] - s["min"])
                        )

                        if dragging == "min_db":
                            if cfg["min_db"] >= cfg["max_db"]:
                                cfg["min_db"] = cfg["max_db"] - 1.0

                        if dragging == "max_db":
                            if cfg["max_db"] <= cfg["min_db"]:
                                cfg["max_db"] = cfg["min_db"] + 1.0

    scroll_width = max(1, int(cfg["scroll"]))

    pending_audio = []
    while not audio_queue.empty():
        try:
            pending_audio.append(audio_queue.get_nowait())
        except queue.Empty:
            break

    for data in pending_audio:
        audio_data = np.frombuffer(
            data,
            dtype=np.int16
        )

        new_column = pygame.Surface(
            (scroll_width, SPEC_HEIGHT)
        )

        if len(audio_data) > 0:

            fft_data = np.abs(
                np.fft.rfft(audio_data)
            )

            if len(fft_data) > 0:

                db_data = 20.0 * np.log10(
                    fft_data + 1e-6
                )

                max_bin = max(
                    1,
                    int(
                        (
                            cfg["max_hz"] /
                            (RATE / 2.0)
                        ) * len(db_data)
                    )
                )

                db_data = db_data[:max_bin]

                if len(db_data) != SPEC_HEIGHT:

                    xp = np.linspace(
                        0.0,
                        1.0,
                        len(db_data)
                    )

                    x = np.linspace(
                        0.0,
                        1.0,
                        SPEC_HEIGHT
                    )

                    db_data = np.interp(
                        x,
                        xp,
                        db_data
                    )

                c_val = cfg["color"]

                if c_val <= 1.0:

                    base_r = 255.0 * (1.0 - c_val)
                    base_g = 255.0 * c_val
                    base_b = 0.0

                else:

                    base_r = 0.0
                    base_g = 255.0 * (2.0 - c_val)
                    base_b = 255.0 * (c_val - 1.0)

                for y in range(SPEC_HEIGHT):

                    db_val = db_data[SPEC_HEIGHT - 1 - y]

                    intensity = (
                        (db_val - cfg["min_db"]) /
                        (cfg["max_db"] - cfg["min_db"])
                    )

                    intensity = max(
                        0.0,
                        min(1.0, intensity)
                    )

                    rv = int(intensity * base_r)
                    gv = int(intensity * base_g)
                    bv = int(intensity * base_b)

                    for x in range(scroll_width):
                        new_column.set_at(
                            (x, y),
                            (rv, gv, bv)
                        )

        spec_surface.blit(
            spec_surface,
            (-scroll_width, 0)
        )

        spec_surface.blit(
            new_column,
            (WIDTH - scroll_width, 0)
        )

    screen.blit(spec_surface, (0, 0))

    pygame.draw.rect(
        screen,
        (30, 30, 30),
        (0, SPEC_HEIGHT, WIDTH, UI_HEIGHT)
    )

    for s in sliders:

        rect = get_rect(
            s["row"],
            s["col"],
            WIDTH,
            SPEC_HEIGHT
        )

        pygame.draw.rect(
            screen,
            (100, 100, 100),
            rect
        )

        val_pct = (
            (cfg[s["key"]] - s["min"]) /
            (s["max"] - s["min"])
        )

        knob_x = rect.left + int(
            val_pct * rect.width
        )

        pygame.draw.circle(
            screen,
            (200, 200, 255),
            (knob_x, rect.centery),
            8
        )

        if s["key"] == "color":

            txt = font.render(
                "Color (R -> G -> B)",
                True,
                (255, 255, 255)
            )

        else:

            txt = font.render(
                "%s: %d" % (
                    s["label"],
                    int(cfg[s["key"]])
                ),
                True,
                (255, 255, 255)
            )

        screen.blit(
            txt,
            (rect.left, rect.top - 20)
        )

    pygame.display.flip()
    clock.tick(FPS)

try:
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=4)

except:
    pass

stream.stop_stream()
stream.close()

p.terminate()

pygame.quit()

sys.exit()