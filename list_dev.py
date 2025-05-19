"""
dual_theremin_sd_akwf.py   –   v2
────────────────────────────────────────────
• Dos manos, 6 gestos, 6 timbres muy diferenciados.
• Controles:
      Y (↑↓)  → ±1 octava
      X (←→)  → filtro 400–4 000 Hz
      Z (distancia) → volumen 0–0.9
"""

# ─────────── AUDIO ───────────────────────────────────────────
import numpy as np, sounddevice as sd, soundfile as sf, time, pathlib
SR = 48000
sd.default.samplerate = SR
sd.default.blocksize  = 256

def load_table(path, size=2048):
    sig,_ = sf.read(str(path), dtype="float32")
    if sig.ndim>1: sig=sig[:,0]
    sig /= max(0.001, np.abs(sig).max())
    return np.interp(np.linspace(0,1,size,False),
                     np.linspace(0,1,len(sig),False), sig).astype(np.float32)

AKWF = pathlib.Path("AKWF")

VOICES = {
    "lead_supersaw": dict(file="akwf_sawtooth/akwf_0064.wav",   vol=.35, q=.25),
    "sub_pulse"   : dict(file="akwf_pulse/akwf_pulse_0012.wav", vol=.45, q=.10),
    "dream_pad"   : dict(file="akwf_blended/akwf_blend_0034.wav",vol=.30, q=.20),
    "glass_bell"  : dict(file="akwf_epiano/akwf_epiano_0008.wav", vol=.28, q=.60),
    "vox_ooo"     : dict(file="akwf_vowel/akwf_ooo_0030.wav",    vol=.25, q=.35),
    "noise_swirl" : dict(file="akwf_noise/akwf_noise_0012.wav",  vol=.20, q=.05),
}

for v in VOICES.values():
    wav = AKWF / v["file"]
    v["table"] = load_table(wav) if wav.exists() else np.sin(2*np.pi*np.linspace(0,1,2048))

phase = {k:0.0 for k in VOICES};  freq = {k:0.0 for k in VOICES}
amp   = {k:0.0 for k in VOICES};  cut  = {k:1000.0 for k in VOICES}
lp    = {k:0.0 for k in VOICES}

def audio_cb(out, frames, t, status):
    global phase, lp
    out.fill(0.0)
    ts = np.arange(frames)/SR
    for name,v in VOICES.items():
        if amp[name] < 1e-4: continue
        # vibrato ±4 cent @5 Hz
        vib = np.sin(2*np.pi*5*(t+ts)) * 4/1200
        f_inst = freq[name] * 2**vib
        incr = f_inst / SR * len(v["table"])
        idx  = (phase[name] + np.cumsum(incr)) % len(v["table"])
        sig  = v["table"][idx.astype(np.int32)] * amp[name] * v["vol"]

        # chorus para dream_pad
        if name=="dream_pad":
            sig = 0.5*(sig + np.roll(sig, 5))

        # filtro RC
        alpha = np.exp(-2*np.pi*cut[name]/SR)
        y = lp[name]
        for i,x in enumerate(sig):
            y = (1-alpha)*x + alpha*y
            sig[i] = y
        lp[name]=y
        out[:,0]+=sig
        phase[name]=idx[-1]
sd.OutputStream(channels=1, callback=audio_cb).start()

# ─────────── VISION ─────────────────────────────────────────
import cv2, mediapipe as mp
from mediapipe.tasks import python as mp_task
from mediapipe.framework.formats import landmark_pb2

MODEL="gesture_recognizer.task"
opts=mp_task.vision.GestureRecognizerOptions(
    base_options=mp_task.BaseOptions(model_asset_path=MODEL),
    running_mode=mp_task.vision.RunningMode.VIDEO,
    num_hands=2, min_hand_detection_confidence=.6, min_tracking_confidence=.5)
rec=mp_task.vision.GestureRecognizer.create_from_options(opts)
mpd=mp.solutions.drawing_utils; CONN=mp.solutions.hands.HAND_CONNECTIONS
lerp=lambda a,b,t: a+(b-a)*t

# gestos ↔ voces
MAP = {
  "Open_Palm"   : "lead_supersaw",
  "Closed_Fist" : "sub_pulse",
  "Thumb_Up"    : "dream_pad",
  "Victory"     : "glass_bell",
  "Pointing_Up" : "noise_swirl",
  "ILoveYou"    : "vox_ooo",
}
ACCEPT=set(MAP); MIN=.45

# ─────────── CAM ────────────────────────────────────────────
cap,t0=cv2.VideoCapture(0),time.time()
while cap.isOpened():
    ok,fr=cap.read()
    if not ok: break
    h,w=fr.shape[:2]
    img=mp.Image(mp.ImageFormat.SRGB, cv2.cvtColor(fr,cv2.COLOR_BGR2RGB))
    res=rec.recognize_for_video(img, int((time.time()-t0)*1000))

    # release suave
    for k in amp: amp[k]*=0.85

    if res.hand_landmarks:
        for idx,hand in enumerate(res.hand_landmarks):
            xs=[p.x for p in hand]; ys=[p.y for p in hand]; zs=[p.z for p in hand]
            nx=np.clip(sum(xs)/21,0,1); ny=np.clip(sum(ys)/21,0,1); nz=-sum(zs)/21
            g = res.gestures[idx][0] if res.gestures and res.gestures[idx] else None
            if not (g and g.category_name in ACCEPT and g.score>MIN): continue
            v = MAP[g.category_name]
            freq[v]=440*2**((lerp(-12,12,1-ny))/12)
            cut[v]=lerp(400,4000,nx)
            amp[v]=np.clip(nz*15,0,0.9)

            proto=landmark_pb2.NormalizedLandmarkList()
            proto.landmark.extend([landmark_pb2.NormalizedLandmark(x=p.x,y=p.y,z=p.z) for p in hand])
            mpd.draw_landmarks(fr, proto, CONN)
            cx=int(sum(xs)/21*w); cy=int(sum(ys)/21*h)
            cv2.putText(fr,f"{g.category_name}->{v}",(cx-80,cy-25),
                        cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)

    cv2.imshow("Dual Theremin AKWF (sd)", fr)
    if cv2.waitKey(1)&0xFF==27: break

cap.release(); sd.stop()
