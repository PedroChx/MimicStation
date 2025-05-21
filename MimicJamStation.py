"""
dual_theremin_sd_akwf.py
────────────────────────
• 2 manos, 6 gestos
    Open_Palm    → lead_sawte
    Closed_Fist  → sub_bass
    Thumb_Up     → soft_pad
    Victory      → bell
    Pointing_Up  → pluck
    ILoveYou     → choir
• X = filtro  | Y = nota (±1 oct) | Z = volumen (mano cerca = +dB)
"""

# ---------------- audio ----------------------------------------------
import numpy as np, sounddevice as sd, soundfile as sf, time, pathlib

SR = 48000
sd.default.samplerate, sd.default.blocksize = SR, 256

def load_table(wav_path, size=2048):
    """Lee cualquier WAV mono y devuelve un wavetable normalizado de `size` muestras."""
    sig, _ = sf.read(str(wav_path), dtype="float32")
    if sig.ndim > 1: sig = sig[:,0]
    sig = sig / max(0.001, np.abs(sig).max())
    x_old = np.linspace(0,1,len(sig), False)
    x_new = np.linspace(0,1,size, False)
    return np.interp(x_new, x_old, sig).astype(np.float32)

AKWF = pathlib.Path("AKWF")

VOICES = {
    "lead_supersaw": dict(file="akwf_sawtooth/akwf_0064.wav",  vol=.35, q=.25),
    "sub_pulse"   : dict(file="akwf_pulse/akwf_pulse_0012.wav", vol=.45, q=.10),
    "dream_pad"   : dict(file="akwf_blended/akwf_blend_0034.wav", vol=.30, q=.20),
    "glass_bell"  : dict(file="akwf_epiano/akwf_epiano_0008.wav", vol=.28, q=.60),
    "vox_ooo"     : dict(file="akwf_vowel/akwf_ooo_0030.wav",   vol=.25, q=.35),
    "noise_swirl" : dict(file="akwf_noise/akwf_noise_0012.wav", vol=.20, q=.05),
}


for v in VOICES.values():
    f = AKWF / v["file"]
    v["table"] = load_table(f) if f.exists() else np.sin(2*np.pi*np.linspace(0,1,2048))

# runtime state
phase  = {k:0.0       for k in VOICES}  # fase actual
freq   = {k:0.0       for k in VOICES}  # Hz
amp    = {k:0.0       for k in VOICES}  # 0-1
cut    = {k:1000.0    for k in VOICES}  # Hz
lp_val = {k:0.0       for k in VOICES}  # filtro RC

def audio_cb(outdata, frames, t, status):
    global phase, lp_val
    out = np.zeros(frames, dtype=np.float32)
    for name, v in VOICES.items():
        if amp[name] < 1e-4: continue
        # wavetable lookup
        incr = freq[name] / SR * len(v["table"])
        idx  = (phase[name] + incr*np.arange(frames)) % len(v["table"])
        sig  = v["table"][idx.astype(np.int32)] * amp[name] * v["vol"]
        phase[name] = idx[-1]  # save last phase

        # 1er orden low-pass
        alpha = np.exp(-2*np.pi*cut[name]/SR)
        y = lp_val[name]
        for i,x in enumerate(sig):
            y = (1-alpha)*x + alpha*y
            sig[i] = y
        lp_val[name] = y

        out += sig
    outdata[:] = out.reshape(-1,1)

stream = sd.OutputStream(channels=1, callback=audio_cb)
stream.start()

# ---------------- vision ---------------------------------------------
import cv2, mediapipe as mp
from mediapipe.tasks import python as mp_task
from mediapipe.framework.formats import landmark_pb2

MODEL = "gesture_recognizer.task"
options = mp_task.vision.GestureRecognizerOptions(
    base_options=mp_task.BaseOptions(model_asset_path=MODEL),
    running_mode=mp_task.vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=.6,
    min_tracking_confidence=.5)
rec = mp_task.vision.GestureRecognizer.create_from_options(options)
mpd = mp.solutions.drawing_utils
CONN= mp.solutions.hands.HAND_CONNECTIONS

MAP = {
    "Open_Palm"   : "lead_supersaw",
    "Closed_Fist" : "sub_pulse",
    "Thumb_Up"    : "dream_pad",
    "Victory"     : "glass_bell",
    "Pointing_Up" : "noise_swirl",
    "ILoveYou"    : "vox_ooo",
}


def lerp(a,b,t): return a + (b-a)*t

# ---------------- cámara ---------------------------------------------
cap, t0 = cv2.VideoCapture(0), time.time()
while cap.isOpened():
    ok, frame = cap.read()
    if not ok: break
    h,w = frame.shape[:2]
    img = mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    res = rec.recognize_for_video(img, int((time.time()-t0)*1000))

    # reset amplitudes cada frame
    for k in amp: amp[k] *= 0.85  # pequeño release

    if res.hand_landmarks:
        for idx, hand in enumerate(res.hand_landmarks):
            xs=[p.x for p in hand]; ys=[p.y for p in hand]; zs=[p.z for p in hand]
            nx=np.clip(sum(xs)/21,0,1); ny=np.clip(sum(ys)/21,0,1)
            nz=-sum(zs)/21                       # mano cerca = valor positivo

            g = res.gestures[idx][0] if res.gestures and res.gestures[idx] else None
            if not (g and g.score>.5 and g.category_name in MAP): continue
            voice = MAP[g.category_name]

            # parámetros
            freq[voice] = 440*2**((lerp(-12,12,1-ny))/12)   # ±1 oct
            cut[voice]  = lerp(400, 4000, nx)
            amp[voice]  = np.clip(nz*15, 0, 0.9)            # volumen con Z

            # draw
            proto = landmark_pb2.NormalizedLandmarkList()
            proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=p.x,y=p.y,z=p.z) for p in hand])
            mpd.draw_landmarks(frame, proto, CONN)
            cx=int(sum(xs)/21*w); cy=int(sum(ys)/21*h)
            cv2.putText(frame,f"{g.category_name}->{voice}",(cx-80,cy-25),
                        cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)

    cv2.imshow("Dual Theremin – sounddevice", frame)
    if cv2.waitKey(1)&0xFF==27: break  # Esc

cap.release(); stream.stop(); stream.close()
