# test_sd_beep.py  ← ejecútalo
import numpy as np, sounddevice as sd, time
sr = 48000
sd.default.samplerate = sr
sd.default.blocksize  = 512
sd.play(0.3*np.sin(2*np.pi*440*np.linspace(0, 2, sr*2)), sr)
print("♫ Beep 2 s…")
time.sleep(2)
sd.stop()
