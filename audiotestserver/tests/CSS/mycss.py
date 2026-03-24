import numpy as np
import wave

def generate_chirp(f0, f1, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    k = (f1 - f0) / duration
    phi = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    return np.sin(phi)

def get_bits(byte_val):
    return [(byte_val >> i) & 1 for i in range(7, -1, -1)]

def apply_ecc(bits):
    encoded = []
    for b in bits:
        encoded.extend([b, b, b])
    return encoded

def create_audio_css(text, filename="css_output.wav"):
    fs = 44100
    f_min, f_max = 400, 6000
    symbol_dur = 0.04 
    
    up_chirp = generate_chirp(f_min, f_max, symbol_dur, fs)
    down_chirp = generate_chirp(f_max, f_min, symbol_dur, fs)
    
    preamble = [1, 1, 1, 1, 0, 0, 1, 1]
    
    raw_data = text.encode('utf-8', 'ignore')
    chunk_size = 12 
    
    audio_buffer = []
    
    for b in preamble:
        audio_buffer.extend(up_chirp if b == 1 else down_chirp)
        
    for i in range(0, len(raw_data), chunk_size):
        chunk = raw_data[i:i+chunk_size]
        seq_num = (i // chunk_size) % 256
        
        frame_payload = bytearray([seq_num]) + chunk
        
        frame_bits = []
        for byte in frame_payload:
            frame_bits.extend(get_bits(byte))
            
        protected_bits = apply_ecc(frame_bits)
        
        for bit in protected_bits:
            audio_buffer.extend(up_chirp if bit == 1 else down_chirp)
            
    audio_signal = np.array(audio_buffer)
    audio_signal = (audio_signal * 32767).astype(np.int16)
    
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(fs)
        wf.writeframes(audio_signal.tobytes())

lorem_ipsum = """
Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor 
incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis 
nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.
"""

create_audio_css(lorem_ipsum)

import json

settings = {
    "fs": 44100,
    "f_min": 400,
    "f_max": 6000,
    "symbol_dur": 0.04,
    "chunk_size": 12,
    "preamble": [1, 1, 1, 1, 0, 0, 1, 1]
}

with open("info.json", "w") as f:
    json.dump(settings, f)