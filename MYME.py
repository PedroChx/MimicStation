"""
mimicStation.py
────────────────────────────────────────────────────────────
Instrumento gestual con 2 manos – 6 gestos.

Flujo de uso
────────────
1.  Aparece el **Selector de instrumentos** (Tkinter).
    • Elige un preset o asigna WAV por gesto.
    • Pulsa **Start**.

2.  Se abre la **ventana de cámara** (OpenCV) con una barra lateral que
    muestra el mapeo Gesto → WAV (stem).
    •  Mueve las manos para tocar.
    •  Pulsa **R** para regresar al selector (sin cerrar el programa).
    •  Pulsa **Esc** para salir completamente.

Gestos ↔ Parámetros
    X (←→) = filtro 400-4000 Hz
    Y (↑↓) = nota ±1 octava
    Z (dist) = volumen 0-0.9
"""

# ───────────────────── Imports ─────────────────────
import numpy as np, sounddevice as sd, soundfile as sf
import cv2, pathlib, time, threading, tkinter as tk
from tkinter import ttk, messagebox
from mediapipe.tasks import python as mp_task
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp

# ─────────────────── Constantes ────────────────────
SR = 48_000
sd.default.samplerate, sd.default.blocksize = SR, 256
AKWF   = pathlib.Path("AKWF")
GESTOS = ["Open_Palm", "Closed_Fist", "Thumb_Up",
          "Victory", "Pointing_Up", "Rock"]

# ────────────── Wavetable helper ───────────────
def load_table(path: str | pathlib.Path, size=2048) -> np.ndarray:
    p = pathlib.Path(path)
    if not p.is_file():
        return np.sin(2*np.pi*np.linspace(0, 1, size, False))
    sig, _ = sf.read(str(p), dtype="float32")
    if sig.ndim > 1:
        sig = sig[:, 0]
    sig /= max(0.001, np.abs(sig).max())
    x_old = np.linspace(0, 1, len(sig), False)
    x_new = np.linspace(0, 1, size, False)
    return np.interp(x_new, x_old, sig).astype(np.float32)

# ─────────────────── Presets ─────────────────────
presets = {
    "—": {},  # sin preset (selección manual)

    # ───────────────────── 1. TRANCE (rápido, brillante)
    "Trance": {
        "Open_Palm" : "AKWF/akwf_sawtooth/akwf_0064.wav",   # supersaw lead
        "Closed_Fist": "AKWF/akwf_pulse/akwf_pulse_0004.wav", # hard sub-bass
        "Thumb_Up"  : "AKWF/akwf_blended/akwf_blend_0012.wav", # pluck agresivo
        "Victory"   : "AKWF/akwf_bell/akwf_bell_0002.wav",   # campana metálica
        "Pointing_Up": "AKWF/akwf_noise/akwf_noise_burst_0002.wav", # white-burst fx
        "Rock"  : "AKWF/akwf_vowel/akwf_aaa_0022.wav",   # coro “aa”
    },

    # ───────────────────── 2. AMBIENT (texturas suaves)
    "Ambient": {
        "Open_Palm" : "AKWF/akwf_triangle/akwf_tri_0006.wav",  # pad cristal
        "Closed_Fist": "AKWF/akwf_pulse/akwf_pulse_0032.wav",  # bajo redondo
        "Thumb_Up"  : "AKWF/akwf_blended/akwf_blend_soft_0018.wav", # pad dulce
        "Victory"   : "AKWF/akwf_epiano/akwf_epiano_0006.wav", # e-piano aireado
        "Pointing_Up": "AKWF/akwf_noise/akwf_noise_wind_0003.wav", # viento
        "Rock"  : "AKWF/akwf_vowel/akwf_ooo_0020.wav",    # coro “oo”
    },

    # ───────────────────── 3. ORCHESTRAL (cuerdas + percusión)
    "Orchestral": {
        "Open_Palm" : "AKWF/akwf_vowel/akwf_ih_0026.wav",    # string-pad tibio
        "Closed_Fist": "AKWF/akwf_square/akwf_sqwav_0003.wav", # contrabajo square
        "Thumb_Up"  : "AKWF/akwf_epiano/akwf_epiano_0002.wav", # cello-pad
        "Victory"   : "AKWF/akwf_epiano/akwf_epiano_0034.wav", # pizzicato-bell
        "Pointing_Up": "AKWF/akwf_noise/akwf_noise_swish_0006.wav", # timpani-swish
        "Rock"  : "AKWF/akwf_vowel/akwf_eh_0018.wav",    # coro “eh”
    },

    # ───────────────────── 4. DREAM POP (etérico, choruseado)
    "Dream Pop": {
        "Open_Palm" : "AKWF/akwf_blended/akwf_blend_0020.wav",  # pad dream-saw
        "Closed_Fist": "AKWF/akwf_pulse/akwf_pulse_0008.wav",   # bajo warm-pulse
        "Thumb_Up"  : "AKWF/akwf_triangle/akwf_tri_0006.wav",   # lead tri-chorus
        "Victory"   : "AKWF/akwf_bell/akwf_bell_soft_0004.wav", # bell dream
        "Pointing_Up": "AKWF/akwf_noise/akwf_noise_wind_0002.wav", # shimmer
        "Rock"  : "AKWF/akwf_vowel/akwf_ooo_0016.wav",      # vocal “oo”
    },

    # ───────────────────── 5. LO-FI CHILL (cálido / granuloso)
    "Lo-Fi Chill": {
        "Open_Palm" : "AKWF/akwf_sawtooth/akwf_0032.wav",      # saw envejecida
        "Closed_Fist": "AKWF/akwf_pulse/akwf_pulse_0018.wav",  # bass dusty
        "Thumb_Up"  : "AKWF/akwf_blended/akwf_blend_0062.wav", # pad ribbon
        "Victory"   : "AKWF/akwf_epiano/akwf_epiano_0004.wav", # Rhodes saturado
        "Pointing_Up": "AKWF/akwf_noise/akwf_noise_swish_0004.wav", # vinyl hiss
        "Rock"  : "AKWF/akwf_vowel/akwf_ah_0012.wav",      # coro “ah”
    },

    # ───────────────────── 6. CINEMATIC SOFT (textura soundtrack)
    "Cinematic Soft": {
        "Open_Palm" : "AKWF/akwf_vowel/akwf_ih_0020.wav",      # pad voz-cuerda
        "Closed_Fist": "AKWF/akwf_square/akwf_sqwav_0003.wav", # bajo square-warm
        "Thumb_Up"  : "AKWF/akwf_blended/akwf_blend_0046.wav", # pad air
        "Victory"   : "AKWF/akwf_epiano/akwf_epiano_0014.wav", # glock-epiano
        "Pointing_Up": "AKWF/akwf_noise/akwf_noise_wind_0006.wav", # noise FX
        "Rock"  : "AKWF/akwf_vowel/akwf_eh_0018.wav",      # coro “eh”
    },
}

# ───────────── Audio runtime state ──────────────
voices, phase, freq, amp, cut, lp_val = {}, {}, {}, {}, {}, {}
state_lock = threading.Lock()

def rebuild_voices(mapping: dict[str, str]):
    """Carga wavetables según el mapeo elegido en el selector."""
    with state_lock:
        voices.clear()
        for gesto, wav in mapping.items():
            voices[gesto] = {"file": wav,
                             "table": load_table(wav),
                             "vol" : 0.35}
        for k in voices:
            phase[k] = freq[k] = lp_val[k] = 0.0
            amp[k]   = 0.0
            cut[k]   = 1000.0

def audio_cb(outdata, frames, t, status):
    out = np.zeros(frames, dtype=np.float32)
    with state_lock:
        for name, v in voices.items():
            if amp.get(name, 0.0) < 1e-4:
                continue
            incr = freq[name] / SR * len(v["table"])
            idx  = (phase[name] + incr*np.arange(frames)) % len(v["table"])
            sig  = v["table"][idx.astype(np.int32)] * amp[name] * v["vol"]
            phase[name] = idx[-1]
            alpha = np.exp(-2*np.pi*cut[name] / SR)
            y = lp_val[name]
            for i, x in enumerate(sig):
                y = (1-alpha)*x + alpha*y
                sig[i] = y
            lp_val[name] = y
            out += sig
    outdata[:] = out.reshape(-1, 1)

stream = sd.OutputStream(channels=1, callback=audio_cb)
stream.start()

# ─────────────── Tkinter selector ───────────────
def selector_window(initial="Trance") -> dict[str,str] | None:
    """Muestra selector; devuelve mapping válido o None si cancela."""
    root = tk.Tk(); root.title("Mimic – Selector"); root.resizable(False,False)

    # preset
    tk.Label(root, text="Preset").grid(row=0,column=0,pady=4,sticky="e")
    preset = tk.StringVar(value=initial)
    preset_box = ttk.Combobox(root, values=list(presets),
                              textvariable=preset, state="readonly", width=20)
    preset_box.grid(row=0,column=1,pady=4,sticky="w")

    # combobox por gesto
    wav_list = ["—"] + sorted(str(p) for p in AKWF.rglob("*.wav"))
    boxes={}
    for i,g in enumerate(GESTOS, start=1):
        tk.Label(root, text=g, width=12, anchor="e")\
          .grid(row=i,column=0,padx=4,sticky="e")
        cb=ttk.Combobox(root, values=wav_list, state="readonly", width=60)
        cb.grid(row=i,column=1,padx=4,pady=1,sticky="w")
        boxes[g]=cb

    def load_preset(*_):
        m = presets[preset.get()]
        for g in GESTOS: boxes[g].set(m[g])
    preset_box.bind("<<ComboboxSelected>>", load_preset)
    load_preset()

    mapping={}
    def start():
        for g,cb in boxes.items():
            w=cb.get()
            if w=="—":
                messagebox.showerror("Falta WAV", f"Selecciona WAV para {g}")
                return
            mapping[g]=w
        root.destroy()
    ttk.Button(root, text="Start", command=start, width=12)\
        .grid(row=len(GESTOS)+1,column=1,pady=10,sticky="e")
    root.mainloop()
    return mapping if mapping else None

# ─────────────── MediaPipe setup helpers ───────────────
mp_draw = mp.solutions.drawing_utils
CONN    = mp.solutions.hands.HAND_CONNECTIONS
lerp    = lambda a,b,t: a+(b-a)*t
MAP_G2V = {g:g for g in GESTOS}

def sidebar(frame, mapping, width=260):
    """Barra gris a la derecha con los stems activos."""
    h,w0 = frame.shape[:2]
    bar = np.full((h,width,3), (40,40,40), dtype=np.uint8)
    y=30
    for g in GESTOS:
        name = pathlib.Path(mapping[g]).stem
        cv2.putText(bar, f"{g:<11} → {name}", (10,y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 1, cv2.LINE_AA)
        y += 32
    return np.hstack([frame, bar])

# ─────────────────────── Main loop ──────────────────────
if __name__ == "__main__":
    current_preset = "Trance"

    while True:   # ← program loop: Selector → Cámara → Selector …
        mapping = selector_window(current_preset)
        if mapping is None:
            break                      # usuario cerró
        # memoriza preset si fue exacto
        current_preset = next((k for k,v in presets.items() if v==mapping),
                              current_preset)
        rebuild_voices(mapping)

        # recrea recognizer para evitar error de timestamp
        options = mp_task.vision.GestureRecognizerOptions(
            base_options = mp_task.BaseOptions(model_asset_path="gesture_recognizer.task"),
            running_mode = mp_task.vision.RunningMode.VIDEO,
            num_hands=2, min_hand_detection_confidence=.6, min_tracking_confidence=.5)
        recognizer = mp_task.vision.GestureRecognizer.create_from_options(options)

        cap, t0 = cv2.VideoCapture(0), time.time()
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break

            img = mp.Image(mp.ImageFormat.SRGB,
                           cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            ts  = int((time.time()-t0)*1000)
            res = recognizer.recognize_for_video(img, ts)

            for k in amp: amp[k] *= 0.85      # release

            if res.hand_landmarks:
                for idx, hand in enumerate(res.hand_landmarks):
                    # comprueba que haya lista de gestos
                    if idx >= len(res.gestures) or not res.gestures[idx]:
                        continue
                    g = res.gestures[idx][0]
                    if not g.category_name or g.category_name not in MAP_G2V:
                        continue
                    key = MAP_G2V[g.category_name]

                    xs=[p.x for p in hand]; ys=[p.y for p in hand]; zs=[p.z for p in hand]
                    nx=np.clip(sum(xs)/21,0,1); ny=np.clip(sum(ys)/21,0,1); nz=-sum(zs)/21
                    freq[key] = 440*2**((lerp(-12,12,1-ny))/12)
                    cut[key]  = lerp(400,4000,nx)
                    amp[key]  = np.clip(nz*15, 0, .9)

                    proto=landmark_pb2.NormalizedLandmarkList()
                    proto.landmark.extend([landmark_pb2.NormalizedLandmark(
                        x=p.x,y=p.y,z=p.z) for p in hand])
                    mp_draw.draw_landmarks(frame, proto, CONN)

            frame_disp = sidebar(frame, mapping)
            cv2.imshow("Mimic – Live  (R=menú  Esc=salir)", frame_disp)
            k = cv2.waitKey(1) & 0xFF
            if k == 27:                       # Esc → salir
                cap.release(); cv2.destroyAllWindows()
                stream.stop(); stream.close()
                quit()
            if k in (ord('r'), ord('R')):     # volver al selector
                cap.release(); cv2.destroyAllWindows()
                break
