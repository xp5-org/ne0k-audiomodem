import numpy as np
import sounddevice as sd
import json

def generate_reference_chirp(f0, f1, duration, fs):
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    k = (f1 - f0) / duration
    phi = 2 * np.pi * (f0 * t + 0.5 * k * t**2)
    return np.sin(phi).astype(np.float32)

with open("info.json", "r") as f:
    conf = json.load(f)

FS = conf["fs"]
DUR = conf["symbol_dur"]
SAMPLES_PER_SYMBOL = int(FS * DUR)

up_ref = generate_reference_chirp(conf["f_min"], conf["f_max"], DUR, FS)
down_ref = generate_reference_chirp(conf["f_max"], conf["f_min"], DUR, FS)

def get_bytes_from_bits(bit_list):
    out = []
    for i in range(0, len(bit_list), 8):
        byte_bits = bit_list[i:i+8]
        if len(byte_bits) < 8: break
        val = 0
        for bit in byte_bits:
            val = (val << 1) | bit
        out.append(val)
    return bytes(out)

print("Listening for CSS data...")

bit_stream = []
sync_found = False
preamble = conf["preamble"]

with sd.InputStream(samplerate=FS, channels=1, dtype='float32') as stream:
    try:
        while True:
            chunk, overflowed = stream.read(SAMPLES_PER_SYMBOL)
            chunk = chunk.flatten()
            
            peak = np.max(np.abs(chunk))
            if peak > 0.001:
                chunk /= peak
            
            up_corr = np.max(np.correlate(chunk, up_ref, mode='same'))
            down_corr = np.max(np.correlate(chunk, down_ref, mode='same'))
            
            bit = 1 if up_corr > down_corr else 0
            bit_stream.append(bit)
            
            if not sync_found:
                if bit_stream[-len(preamble):] == preamble:
                    print("\n[Sync Acquired] Decoding...")
                    sync_found = True
                    bit_stream = []
            else:
                bits_per_frame = (conf["chunk_size"] + 1) * 8 * 3
                if len(bit_stream) >= bits_per_frame:
                    votes = np.array(bit_stream).reshape(-1, 3)
                    corrected_bits = (np.sum(votes, axis=1) > 1).astype(int)
                    
                    payload_bytes = get_bytes_from_bits(corrected_bits)
                    seq_num = payload_bytes[0]
                    try:
                        message = payload_bytes[1:].decode('utf-8')
                        print(f"Frame {seq_num:03d}: {message}", flush=True)
                    except UnicodeDecodeError:
                        print(f"Frame {seq_num:03d}: [Decoding Error]")
                    
                    bit_stream = []
                    
    except KeyboardInterrupt:
        print("\nExiting.")