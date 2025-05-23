"""
mimicStation.py
─────────────────────────────────────────────────────────
Instrumento gestual con 2 manos – 6 gestos.
Ahora incluye una GUI (Tkinter) que permite:

1. Seleccionar un WAV distinto para cada gesto mediante combobox.
2. Cargar presets temáticos que rellenan las combobox de golpe.
3. Iniciar la demo solo cuando todos los gestos tienen asignado un WAV.

Gestos          –   Timbres por defecto del preset “Trance”
------------------------------------------------------------
Open_Palm       →   lead_supersaw
Closed_Fist     →   sub_pulse
Thumb_Up        →   dream_pad
Victory         →   glass_bell
Pointing_Up     →   noise_swirl
ILoveYou        →   vox_ooo

Controles:
    X (←→)   = filtro 400-4000 Hz
    Y (↑↓)   = nota ±1 octava
    Z (dist) = volumen 0-0.9
Esc = Salir
"""

# ───────────────────────────────── AUDIO ──────────────────────────────
import numpy as np, sounddevice as sd, soundfile as sf, pathlib, time
SR = 48_000
sd.default.samplerate, sd.default.blocksize = SR, 256

def load_table(path: str | pathlib.Path, size=2048) -> np.ndarray:
    """
    Carga un WAV mono y lo normaliza a `size` muestras.
    Si la ruta no existe o no es WAV, devuelve una onda seno de respaldo.
    """
    p = pathlib.Path(path)
    if not p.is_file():                        # → respaldo seguro
        return np.sin(2*np.pi*np.linspace(0, 1, size, False))
    sig, _ = sf.read(str(p), dtype="float32")
    if sig.ndim > 1: sig = sig[:, 0]           # solo canal L si es estéreo
    sig /= max(0.001, np.abs(sig).max())       # normaliza
    x_old = np.linspace(0, 1, len(sig), False)
    x_new = np.linspace(0, 1, size, False)
    return np.interp(x_new, x_old, sig).astype(np.float32)

# Diccionario dinámico que llenaremos desde la GUI
voices: dict[str, dict] = {}

# Estados en tiempo de ejecución (se llenarán tras la GUI)
phase, freq, amp, cut, lp_val = {}, {}, {}, {}, {}

def audio_cb(outdata, frames, t, status):
    """Callback PortAudio: genera el mix mono en `outdata`."""
    out = np.zeros(frames, dtype=np.float32)
    for name, v in voices.items():
        if amp[name] < 1e-4:            # silencio → ahorra CPU
            continue
        incr = freq[name] / SR * len(v["table"])
        idx = (phase[name] + incr*np.arange(frames)) % len(v["table"])
        sig = v["table"][idx.astype(np.int32)] * amp[name] * v["vol"]
        phase[name] = idx[-1]
        # filtro RC 1º orden
        alpha = np.exp(-2*np.pi*cut[name] / SR)
        y = lp_val[name]
        for i, x in enumerate(sig):
            y = (1 - alpha) * x + alpha * y
            sig[i] = y
        lp_val[name] = y
        out += sig
    outdata[:] = out.reshape(-1, 1)

stream = sd.OutputStream(channels=1, callback=audio_cb)

# ───────────────────────────────── GUI ────────────────────────────────
import tkinter as tk
from tkinter import ttk, messagebox

AKWF = pathlib.Path("AKWF")                     # carpeta de WAVs
GESTOS = ["Open_Palm", "Closed_Fist", "Thumb_Up",
          "Victory", "Pointing_Up", "ILoveYou"]

# Lista de WAV disponibles (relativos) + opción nula "—"
wav_files = ["—"] + sorted(str(p) for p in AKWF.rglob("*.wav"))

# Presets: mapa gesto → archivo WAV (relativo a carpeta AKWF)
presets = {
    "—": {},                       # sin preset
    "Trance": {
        "Open_Palm":  "AKWF/akwf_sawtooth/akwf_0064.wav",
        "Closed_Fist": "AKWF/akwf_pulse/akwf_pulse_0012.wav",
        "Thumb_Up":    "AKWF/akwf_blended/akwf_blend_0034.wav",
        "Victory":     "AKWF/akwf_epiano/akwf_epiano_0008.wav",
        "Pointing_Up": "AKWF/akwf_noise/akwf_noise_0012.wav",
        "ILoveYou":    "AKWF/akwf_vowel/akwf_ooo_0030.wav",
    },
    "Ambient": {
        "Open_Palm":  "AKWF/akwf_triangle/akwf_tri_0006.wav",
        "Closed_Fist":"AKWF/akwf_pulse/akwf_pulse_0020.wav",
        "Thumb_Up":   "AKWF/akwf_blended/akwf_blend_0088.wav",
        "Victory":    "AKWF/akwf_sine/akwf_sin_0040.wav",
        "Pointing_Up":"AKWF/akwf_noise/akwf_noise_0008.wav",
        "ILoveYou":   "AKWF/akwf_vowel/akwf_eh_0024.wav",
    },
    "Orchestral": {
        "Open_Palm":  "AKWF/akwf_vowel/akwf_ah_0026.wav",
        "Closed_Fist":"AKWF/akwf_square/akwf_sqwav_0005.wav",
        "Thumb_Up":   "AKWF/akwf_epiano/akwf_epiano_0002.wav",
        "Victory":    "AKWF/akwf_epiano/akwf_epiano_0032.wav",
        "Pointing_Up":"AKWF/akwf_noise/akwf_noise_0010.wav",
        "ILoveYou":   "AKWF/akwf_vowel/akwf_ih_0030.wav",
    },
}

root = tk.Tk()
root.title("Mimic Theremin – Selector de instrumentos")
boxes: dict[str, ttk.Combobox] = {}

# Preset selector
tk.Label(root, text="Preset").grid(row=0, column=0, pady=4, sticky="e")
preset_var = tk.StringVar(value="—")
preset_box = ttk.Combobox(root, values=list(presets), textvariable=preset_var,
                          state="readonly", width=18)
preset_box.grid(row=0, column=1, pady=4, sticky="w")

def apply_preset(*_):
    sel = preset_var.get()
    mapping = presets.get(sel, {})
    for g, cb in boxes.items():
        cb.set(mapping.get(g, "—"))
preset_box.bind("<<ComboboxSelected>>", apply_preset)

# Combobox por gesto
for i, g in enumerate(GESTOS, start=1):
    tk.Label(root, text=g, width=12, anchor="e").grid(row=i, column=0, padx=4, sticky="e")
    cb = ttk.Combobox(root, values=wav_files, state="readonly", width=55)
    cb.set("—")
    cb.grid(row=i, column=1, padx=4, pady=2, sticky="w")
    boxes[g] = cb

def start_demo():
    # Construir diccionario voices a partir de la GUI
    voices.clear()
    for g, cb in boxes.items():
        wav_path = cb.get()
        if wav_path == "—":
            messagebox.showerror("Falta selección",
                                 f"Selecciona un archivo WAV para el gesto: {g}")
            return
        voices[g] = {"table": load_table(wav_path), "vol": 0.35}
    # Inicializar estados
    for k in voices:
        phase[k] = freq[k] = lp_val[k] = 0.0
        amp[k] = 0.0
        cut[k] = 1_000.0
    root.destroy()          # cerrar GUI y seguir al main loop

ttk.Button(root, text="Start", command=start_demo, width=12).grid(
    row=len(GESTOS)+1, column=1, pady=10)
root.mainloop()

# ───────────────────────────────── VISIÓN ─────────────────────────────
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
CONN = mp.solutions.hands.HAND_CONNECTIONS

# Mapeo gesto → voz según lo cargado por la GUI
MAP = {
    "Open_Palm":   "Open_Palm",
    "Closed_Fist": "Closed_Fist",
    "Thumb_Up":    "Thumb_Up",
    "Victory":     "Victory",
    "Pointing_Up": "Pointing_Up",
    "ILoveYou":    "ILoveYou",
}
lerp = lambda a, b, t: a + (b - a) * t

# ───────────────────────────────── CÁMARA ─────────────────────────────
cap, t0 = cv2.VideoCapture(0), time.time()
stream.start()                         # inicia audio después del GUI
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break
    h, w = frame.shape[:2]
    img = mp.Image(mp.ImageFormat.SRGB,
                   cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    res = rec.recognize_for_video(img, int((time.time() - t0) * 1000))

    # Release suave cada frame
    for k in amp:
        amp[k] *= 0.85

    if res.hand_landmarks:
        for idx, hand in enumerate(res.hand_landmarks):
            xs = [p.x for p in hand]; ys = [p.y for p in hand]; zs = [p.z for p in hand]
            nx = np.clip(sum(xs) / 21, 0, 1)
            ny = np.clip(sum(ys) / 21, 0, 1)
            nz = -sum(zs) / 21              # cerca = positivo

            g = res.gestures[idx][0] if res.gestures and res.gestures[idx] else None
            if not (g and g.score > .5):
                continue
            voz_key = MAP.get(g.category_name)
            if voz_key not in voices:
                continue                    # por seguridad

            # Parámetros de la voz
            freq[voz_key] = 440 * 2 ** ((lerp(-12, 12, 1 - ny)) / 12)
            cut[voz_key]  = lerp(400, 4_000, nx)
            amp[voz_key]  = np.clip(nz * 15, 0, .9)

            # Dibujo de ayuda
            proto = landmark_pb2.NormalizedLandmarkList()
            proto.landmark.extend(
                [landmark_pb2.NormalizedLandmark(x=p.x, y=p.y, z=p.z) for p in hand])
            mpd.draw_landmarks(frame, proto, CONN)
            cx = int(sum(xs) / 21 * w); cy = int(sum(ys) / 21 * h)
            cv2.putText(frame, f"{g.category_name}",
                        (cx - 60, cy - 25),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (0, 255, 0), 2)

    cv2.imshow("Mimic Theremin – GUI Edition", frame)
    if cv2.waitKey(1) & 0xFF == 27:      # Esc
        break

cap.release()
stream.stop(); stream.close()
cv2.destroyAllWindows()
