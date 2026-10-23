import os
import re
import json
import time
import glob
import wave
import uuid
import asyncio
import shutil
import hashlib
import subprocess
import queue
import tempfile
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, features

st.set_page_config(page_title="Recap → Multi-Language", page_icon="🎙️", layout="centered")

TARGET, MAXD = 4.2, 5.6

VOX_SPACES = [
    {"space": "openbmb/VoxCPM-Demo", "type": "demo"},
    {"space": "hgghfhjfhjguyjf/Voxcpm-Burmese-Tts", "type": "burmese"},
]

TRIM = (
    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,apad=pad_dur=0.2"
)

ENGINES = ["Edge TTS", "VoxCPM (Edge TTS မသုံးဘူး)"]
STATE_DIR = "/tmp/recap_state"
os.makedirs(STATE_DIR, exist_ok=True)

# ====================================================================
# ဘာသာစကား အချက်အလက် (အသံ + စာတန်း ခွဲပုံ)
# ====================================================================
LANG_CONF = {
    "မြန်မာ": {
        "edge_voices": [
            ("my-MM-ThihaNeural", "Thiha (ယောက်ျား)"),
            ("my-MM-NilarNeural", "Nilar (မိန်းမ)"),
        ],
        "sentence_end": r"(?<=။)",
        "prompt": (
            "You are a Burmese (Myanmar) recap-video narrator. Translate every numbered line "
            "below into natural spoken Burmese narration (storyteller style). Rules: do NOT "
            "summarize, shorten or skip; do NOT move content between lines; keep character "
            "names consistent across all lines; use ONLY Burmese script (Myanmar Unicode), "
            "no English/Chinese letters. Output exactly one line per input line in the format "
            "[number] Burmese text -- same numbers, no extra comments.\n\n"
        ),
        "font_keys": ("padauk", "myanmar", "pyidaung", "notosansmyanmar", "myanmarpadaung"),
        "clean": r"[^\u1000-\u109F\uAA60-\uAA7F\uA9E0-\uA9FF\s0-9.,!?\"'%\-]",
        "is_mm": True,
    },
    "ဂျာမနီ": {
        "edge_voices": [
            ("de-DE-KatjaNeural", "Katja (မိန်းမ)"),
            ("de-DE-ConradNeural", "Conrad (ယောက်ျား)"),
            ("de-DE-AmalaNeural", "Amala (မိန်းမ)"),
            ("de-DE-BerndNeural", "Bernd (ယောက်ျား)"),
        ],
        "sentence_end": r"(?<=[.!?])\s+",
        "prompt": (
            "You are a German (Deutsch) recap-video narrator. Translate every numbered line "
            "below into natural spoken German narration (storyteller style). Rules: do NOT "
            "summarize, shorten or skip; do NOT move content between lines; keep character "
            "names consistent across all lines; use ONLY German (Latin script), no Chinese "
            "letters. Output exactly one line per input line in the format "
            "[number] German text -- same numbers, no extra comments.\n\n"
        ),
        "font_keys": ("dejavusans", "dejavu", "liberation", "notosans", "arial", "freesans"),
        "clean": r"[^A-Za-zÄÖÜäöüß0-9\s.,!?\"'%\-]",
        "is_mm": False,
    },
    "ဂျပန်": {
        "edge_voices": [
            ("ja-JP-NanamiNeural", "Nanami (မိန်းမ)"),
            ("ja-JP-KeitaNeural", "Keita (ယောက်ျား)"),
            ("ja-JP-AoiNeural", "Aoi (မိန်းမ)"),
            ("ja-JP-DaichiNeural", "Daichi (ယောက်ျား)"),
            ("ja-JP-MayuNeural", "Mayu (မိန်းမ)"),
            ("ja-JP-NaokiNeural", "Naoki (ယောက်ျား)"),
            ("ja-JP-ShioriNeural", "Shiori (မိန်းမ)"),
        ],
        "sentence_end": r"(?<=[。！？!?])",
        "prompt": (
            "You are a Japanese (日本語) recap-video narrator. Translate every numbered line "
            "below into natural spoken Japanese narration (storyteller style). Rules: do NOT "
            "summarize, shorten or skip; do NOT move content between lines; keep character "
            "names consistent across all lines; use ONLY Japanese (Kanji, Hiragana, Katakana), "
            "no English/Chinese letters except kanji. Output exactly one line per input line "
            "in the format [number] Japanese text -- same numbers, no extra comments.\n\n"
        ),
        "font_keys": ("notosanscjk", "notosansjp", "notosansjapanese", "takao", "ipag", "vl-gothic", "msgothic"),
        "clean": r"[^\u3000-\u303F\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\uFF00-\uFFEF0-9\s.,!?\"'%\-]",
        "is_mm": False,
    },
}

# ====================================================================
# Settings (သိမ်းထားပြီး session ပြန်စရင်လည်း ပြန်ယူမယ်)
# ====================================================================
DEFAULTS = {
    "mode": "🔗 Link",
    "link_url": "",
    "src_lang": "auto",
    "wsize": "base",
    "fast_whisper": True,
    "height": 720,
    "max_speed": 1.25,
    "max_min": 15,
    "preset": "ultrafast",
    "fps_out": 24,
    "voice_lang": "မြန်မာ",
    "engine": ENGINES[0],
    "rate_n": 45,
    "tts_workers": 8,
    "vox_retry": 4,
    "use_sub": True,
    "sub_size": 7.0,
    "sub_pos": 90,
    "use_neon": True,
    "neon_th": 3,       # ပုံသေ ထားပြီး
    "neon_speed": 0.4,  # ပုံသေ ထားပြီး
    "zoom": 1.0,
}
for _lg, _c in LANG_CONF.items():
    DEFAULTS[f"voice_{_lg}"] = _c["edge_voices"][0][0]

OPTS = {
    "mode": ["🔗 Link", "📁 Upload"],
    "src_lang": ["auto", "en", "zh", "ko", "ja", "hi", "th", "vi", "id"],
    "wsize": ["tiny", "base", "small"],
    "height": [360, 480, 720, 1080],
    "preset": ["ultrafast", "superfast", "veryfast"],
    "fps_out": [20, 24, 25, 30],
    "voice_lang": list(LANG_CONF.keys()),
    "engine": ENGINES,
}
RANGES = {
    "max_speed": (1.0, 2.0, float), "max_min": (1, 120, int), "rate_n": (-50, 100, int),
    "tts_workers": (1, 10, int), "vox_retry": (1, 8, int), "sub_size": (3.0, 12.0, float),
    "sub_pos": (0, 100, int), "zoom": (1.0, 1.25, float),
}
BOOLS = {"fast_whisper", "use_sub", "use_neon"}


def _coerce(k, v):
    if k in OPTS:
        return v if v in OPTS[k] else None
    if k in RANGES:
        lo, hi, typ = RANGES[k]
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        return typ(v) if lo <= v <= hi else None
    if k in BOOLS:
        return v if isinstance(v, bool) else None
    if k.startswith("voice_"):
        lg = k[6:]
        if lg in LANG_CONF and v in [x for x, _ in LANG_CONF[lg]["edge_voices"]]:
            return v
        return None
    if k == "link_url":
        return v if isinstance(v, str) else None
    return None


# ====================================================================
# Generic helpers
# ====================================================================
class Cancelled(Exception):
    pass


def check_cancel(job):
    if job.get("cancel"):
        raise Cancelled()


class JobBar:
    def __init__(self, job, lo=0.0, hi=1.0):
        self.job, self.lo, self.hi = job, lo, hi

    def progress(self, f):
        try:
            self.job["frac"] = self.lo + (self.hi - self.lo) * max(0.0, min(float(f), 1.0))
        except Exception:
            pass

    def set_label(self, text):
        self.job["label"] = text


def secret(name, default=""):
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def sh(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-1200:])
    return r.stdout


def dur(path):
    return float(
        sh(["ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "csv=p=0", path]).strip()
    )


def fmt_t(sec):
    sec = float(sec)
    return f"{sec:.1f} စက္ကန့်" if sec < 60 else f"{int(sec // 60)} မိနစ် {sec % 60:.0f} စက္ကန့်"


def wav_dur(path):
    with wave.open(path) as w:
        return w.getnframes() / float(w.getframerate())


def timing_table(d):
    rows = "\n".join(f"| {k} | **{fmt_t(v)}** |" for k, v in d.items())
    return "| အဆင့် | ကြာချိန် |\n|---|---|\n" + rows


def clean_text(text, lang="မြန်မာ"):
    conf = LANG_CONF.get(lang, LANG_CONF["မြန်မာ"])
    return re.sub(conf["clean"], "", text).strip()


# ====================================================================
# Background job registry
# ====================================================================
@st.cache_resource
def get_registry():
    return {"jobs": {}, "models": {}, "lock": threading.Lock()}


def get_whisper(reg, size):
    with reg["lock"]:
        if size not in reg["models"]:
            from faster_whisper import WhisperModel
            reg["models"][size] = WhisperModel(
                size, device="cpu", compute_type="int8",
                cpu_threads=os.cpu_count() or 4)
        return reg["models"][size]


def start_job(reg, jid, kind, target, *args):
    job = {"kind": kind, "state": "running", "frac": 0.0, "label": "စတင်နေတယ်...",
           "error": "", "out": None, "cancel": False, "t0": time.time(), "consumed": False}
    reg["jobs"][jid] = job

    def runner():
        try:
            target(job, *args)
            if job["state"] == "running":
                job["state"] = "done"
        except Cancelled:
            job["state"] = "cancelled"
        except Exception as ex:
            job["error"] = f"{ex}\n\n{traceback.format_exc()[-900:]}"
            job["state"] = "error"

    threading.Thread(target=runner, daemon=True).start()
    return job


# ====================================================================
# TTS
# ====================================================================
def tts_edge(text, voice, rate, mp3, proxy=None):
    import edge_tts

    async def run():
        kw = {"proxy": proxy} if proxy else {}
        await edge_tts.Communicate(text, voice, rate=rate, **kw).save(mp3)

    asyncio.run(run())


def _gclient(space, token):
    from gradio_client import Client
    if token:
        try:
            return Client(space, token=token)
        except TypeError:
            return Client(space, hf_token=token)
    return Client(space)


class VoxEngine:
    def __init__(self, ref_path=None, token=None):
        self.ref = ref_path
        self.token = token
        self.clients = {}
        self.fails = {i: 0 for i in range(len(VOX_SPACES))}
        self.dead = set()
        self.active = None
        self.last_err = ""
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.dead.clear()
            self.fails = {i: 0 for i in range(len(VOX_SPACES))}
            self.clients = {}

    def synth(self, text, out_wav):
        from gradio_client import handle_file
        order = [i for i in range(len(VOX_SPACES)) if i not in self.dead]
        if self.active in order:
            order.remove(self.active)
            order.insert(0, self.active)
        for i in order:
            sp = VOX_SPACES[i]
            if sp["type"] == "burmese" and not self.ref:
                self.dead.add(i)
                self.last_err = f"{sp['space']}: reference audio လိုအပ်တယ်"
                continue
            try:
                cl = self.clients.get(i)
                if cl is None:
                    cl = _gclient(sp["space"], self.token)
                    self.clients[i] = cl
                rf = handle_file(self.ref) if self.ref else None
                if sp["type"] == "demo":
                    res = cl.predict(
                        text_input=text,
                        control_instruction="A warm young woman, calm and expressive",
                        reference_wav_path_input=rf,
                        use_prompt_text=False, prompt_text_input="",
                        cfg_value_input=2.0, do_normalize=True, denoise=False,
                        api_name="/generate")
                else:
                    res = cl.predict(
                        target_text=text, ref_audio=rf,
                        ref_text="မြန်မာ အသံနမူနာ", cfg_value=2.0,
                        inference_timesteps=10, api_name="/tts")
                p = res[0] if isinstance(res, (tuple, list)) else res
                shutil.copy(p, out_wav)
                self.active = i
                return sp["space"]
            except Exception as ex:
                self.last_err = f"{sp['space']}: {str(ex)[:120]}"
                self.fails[i] += 1
                if self.fails[i] >= 2:
                    self.dead.add(i)
        return None


# ====================================================================
# Font helpers
# ====================================================================
def _prefer_regular(paths):
    def score(p):
        b = os.path.basename(p).lower()
        s = 0
        if "regular" in b:
            s -= 2
        if any(x in b for x in ("bold", "italic", "light", "thin", "black", "medium", "mono", "serif")):
            s += 2
        return (s, len(b))
    return sorted(paths, key=score)


def find_font_for(lang="မြန်မာ"):
    here = os.path.dirname(os.path.abspath(__file__))
    conf = LANG_CONF.get(lang, LANG_CONF["မြန်မာ"])

    repo = (glob.glob(os.path.join(here, "*.ttf")) + glob.glob(os.path.join(here, "*.otf")) +
            glob.glob(os.path.join(here, "*.ttc")) +
            glob.glob(os.path.join(here, "fonts", "*.ttf")) +
            glob.glob(os.path.join(here, "fonts", "*.otf")) +
            glob.glob(os.path.join(here, "fonts", "*.ttc")))

    if conf["is_mm"]:
        for name in ("MyanmarPadaung.ttf", "fonts/MyanmarPadaung.ttf"):
            pref = os.path.join(here, name)
            if os.path.exists(pref):
                return pref
        if repo:
            return repo[0]
    else:
        for key in conf["font_keys"]:
            for f in _prefer_regular(repo):
                if key in os.path.basename(f).lower():
                    return f

    sysf = (glob.glob("/usr/share/fonts/**/*.ttf", recursive=True) +
            glob.glob("/usr/share/fonts/**/*.otf", recursive=True) +
            glob.glob("/usr/share/fonts/**/*.ttc", recursive=True))
    sysf = _prefer_regular(sysf)
    for key in conf["font_keys"]:
        for f in sysf:
            if key in os.path.basename(f).lower():
                return f
    return None


_FONT_CACHE = {}


def get_font(path, size, layout="raqm"):
    key = (path, size, layout)
    if key not in _FONT_CACHE:
        try:
            use_raqm = layout == "raqm" and features.check("raqm")
            eng = ImageFont.Layout.RAQM if use_raqm else ImageFont.Layout.BASIC
            _FONT_CACHE[key] = ImageFont.truetype(path, size, layout_engine=eng)
        except Exception:
            try:
                _FONT_CACHE[key] = ImageFont.truetype(path, size)
            except Exception:
                _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _is_start(c):
    return (0x1000 <= c <= 0x1021) or (0x1023 <= c <= 0x1027) or c in (0x1029, 0x102A, 0x103F)


def split_syll(word):
    idx = [0]
    n = len(word)
    for i in range(1, n):
        c = ord(word[i])
        if not _is_start(c):
            continue
        prev = ord(word[i - 1])
        nxt = ord(word[i + 1]) if i + 1 < n else 0
        if prev == 0x1039 or nxt in (0x103A, 0x1039):
            continue
        idx.append(i)
    idx.append(n)
    return [word[idx[k]:idx[k + 1]] for k in range(len(idx) - 1)]


def wrap_text(text, font, maxw, is_mm=False):
    lines, cur = [], ""
    if not is_mm and " " not in text and any(0x3040 <= ord(c) <= 0x9FFF for c in text):
        for ch in text:
            if not cur or font.getlength(cur + ch) <= maxw:
                cur += ch
            else:
                lines.append(cur)
                cur = ch
        if cur:
            lines.append(cur)
        return lines

    for word in text.split():
        pieces = [word] if font.getlength(word) <= maxw else (split_syll(word) if is_mm else list(word))
        for pi, p in enumerate(pieces):
            sep = " " if (pi == 0 and cur) else ""
            cand = cur + sep + p
            if not cur or font.getlength(cand) <= maxw:
                cur = cand
            else:
                lines.append(cur)
                cur = p
    if cur:
        lines.append(cur)
    return lines


def split_for_sub(text, mc=36, lang="မြန်မာ"):
    conf = LANG_CONF.get(lang, LANG_CONF["မြန်မာ"])
    is_mm = conf["is_mm"]
    sents = [s.strip() for s in re.split(conf["sentence_end"], text) if s.strip()]
    parts = []
    for s in sents:
        if len(s) <= mc:
            parts.append(s)
            continue
        if lang == "ဂျပန်":
            cur = ""
            for ch in s:
                if not cur or len(cur + ch) <= mc:
                    cur += ch
                else:
                    parts.append(cur)
                    cur = ch
            if cur:
                parts.append(cur)
            continue
        cur = ""
        for word in s.split():
            pieces = [word] if len(word) <= mc else (split_syll(word) if is_mm else list(word))
            for pi, p in enumerate(pieces):
                sep = " " if (pi == 0 and cur) else ""
                if not cur or len(cur + sep + p) <= mc:
                    cur = cur + sep + p
                else:
                    parts.append(cur)
                    cur = p
        if cur:
            parts.append(cur)
    return parts or [text]


def build_overlay(text, W, H, font_path, fs, pos_y, alpha=200, color=(255, 255, 255),
                  bold=0, layout="raqm", is_mm=True):
    font = get_font(font_path, fs, layout)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    pad_x, pad_y = int(fs * 0.6), int(fs * 0.5)
    lines = wrap_text(text, font, W - 2 * pad_x, is_mm)
    if not lines:
        return None
    lh = int(fs * 1.4)
    box_w = W
    box_h = len(lines) * lh + 2 * pad_y
    box_x = 0
    max_y = max(0, H - box_h)
    box_y = max(0, min(int(pos_y / 100 * max_y), max_y))

    d.rounded_rectangle([box_x, box_y, box_x + box_w, box_y + box_h],
                        radius=int(fs * 0.66), fill=(0, 0, 0, alpha))
    ink = (int(color[0]), int(color[1]), int(color[2]), 255)
    sw = max(1, round(fs / 15))
    ty = box_y + pad_y
    for ln in lines:
        lw = font.getlength(ln)
        lx = box_x + (box_w - lw) / 2
        d.text((lx, ty), ln, font=font, fill=(0, 0, 0, 255),
               stroke_width=sw, stroke_fill=(0, 0, 0, 255))
        d.text((lx, ty), ln, font=font, fill=ink,
               stroke_width=1 if bold else 0, stroke_fill=ink)
        ty += lh

    arr = np.array(img)
    rows = np.where(arr[:, :, 3].any(axis=1))[0]
    if len(rows) == 0:
        return None
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    crop = arr[y0:y1]
    a = crop[:, :, 3:4].astype(np.float32) / 255.0
    bgr = crop[:, :, [2, 1, 0]].astype(np.float32)
    return {"y0": y0, "y1": y1, "inv": 1.0 - a, "pre": bgr * a}


def blend_overlay(fr, ov):
    reg = fr[ov["y0"]:ov["y1"]].astype(np.float32) * ov["inv"] + ov["pre"]
    fr[ov["y0"]:ov["y1"]] = np.clip(reg, 0, 255).astype(np.uint8)


# ====================================================================
# Neon border (Video Aspect Ratio အတိုင်း လိုက်လည်စေရန် ပြင်ထားသည်)
# ====================================================================
class Neon:
    BASE = np.array([12, 8, 10], np.float32)
    CYAN = np.array([255, 255, 0], np.float32)
    MAGENTA = np.array([110, 30, 255], np.float32)
    IDLE = 0.14
    NBINS = 2048

    def __init__(self, W, H, th=10, speed=0.4, tail=0.5):
        self.W, self.H, self.speed, self.tail = W, H, speed, tail
        th = max(2, min(th, min(W, H) // 4))
        self.P = 2 * (W + H)
        yy, xx = np.mgrid[0:H, 0:W]
        ring = (xx < th) | (xx >= W - th) | (yy < th) | (yy >= H - th)
        ys, xs = np.nonzero(ring)
        dt, db, dl, dr = ys, H - 1 - ys, xs, W - 1 - xs
        m = np.minimum.reduce([dt, db, dl, dr])
        
        S = np.where(m == dt, xs,
            np.where(m == dr, W + ys,
            np.where(m == db, 2 * W + H + (W - xs),
                     2 * W + 2 * H - ys))).astype(np.float64)
                     
        self.ys, self.xs = ys, xs
        self.Sf = ((S % self.P) / float(self.P)).astype(np.float32)
        self.L = np.arange(self.NBINS, dtype=np.float32) / float(self.NBINS)
        self.idx = np.clip((self.Sf * self.NBINS).astype(np.int32), 0, self.NBINS - 1)

    def _layer(self, h, color):
        dist = (h - self.L) % 1.0
        a = np.clip(1.0 - dist / self.tail, 0, 1)
        glow = self.IDLE + (1.0 - self.IDLE) * (a ** 0.55)
        core = np.clip(1.0 - dist / 0.06, 0, 1) ** 2
        return color * glow[:, None] + 255.0 * 0.9 * core[:, None]

    def apply(self, fr, t):
        head = (t * self.speed) % 1.0
        c = self.BASE + self._layer(head, self.CYAN) + self._layer(head + 0.5, self.MAGENTA)
        strip = np.clip(c, 0, 255).astype(np.uint8)
        fr[self.ys, self.xs] = strip[self.idx]


def zoom_flip(fr, zoom, flip):
    zoomed = zoom > 1.001
    if zoomed:
        H, W = fr.shape[:2]
        cw, ch = int(W / zoom), int(H / zoom)
        x0, y0 = (W - cw) // 2, (H - ch) // 2
        fr = cv2.resize(fr[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
    if flip:
        return cv2.flip(fr, 1)
    return fr if zoomed else fr.copy()


# မူရင်း Video ရဲ့ အနက်ရောင်ဘောင် (Letterbox) ကို ဖြတ်ထုတ်ဖို့ Helper
def auto_crop_video(src, dst, target_h):
    cap = cv2.VideoCapture(src)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return w, h, 0, 0
    
    # ဘောင် ဖြတ်ဖို့ အနက်ရောင်ကို ရှာမယ်
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    rows = np.where(edges.any(axis=1))[0]
    if len(rows) > 0:
        y0, y1 = rows[0], rows[-1] + 1
        # အနည်းဆုံး 5px ကနေ စဖြတ်မယ်
        if y0 < 5: y0 = 0
        if h - y1 < 5: y1 = h
        cw = w
        ch = y1 - y0
        if ch < h * 0.95: # သိသိသာသာ ဖြတ်ရမယ်ဆိုရင်
            scale = target_h / ch
            new_w = int(round(cw * scale / 2) * 2) # စုံဂဏန်းဖြစ်ရမယ်
            new_h = int(target_h)
            vf = f"crop={cw}:{ch}:0:{y0},scale={new_w}:{new_h}"
            subprocess.run(["ffmpeg", "-y", "-i", src, "-vf", vf, "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-c:a", "copy", dst], capture_output=True)
            return new_w, new_h, cw, ch
    # မဖြတ်ရင် ရိုးရိုး resize
    new_w = int(round(w * target_h / h / 2) * 2)
    new_h = int(target_h)
    subprocess.run(["ffmpeg", "-y", "-i", src, "-vf", f"scale={new_w}:{new_h}", "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-c:a", "copy", dst], capture_output=True)
    return new_w, new_h, w, h


def render_final(src, audio, out, plan, total, height, subs, font_path, fs_pct, pos_y,
                 neon_on, neon_th, neon_speed, flip, zoom, style=None, preset="ultrafast",
                 progress=None, fps_out=30, is_mm=True, cancel=None):
    style = style or {}
    
    # ဗီဒီယိုကို အနက်ရောင်ဘောင် ဖြတ်ပြီး resize လုပ်မယ်
    temp_vid = out + "_cropped.mp4"
    W, H, _, _ = auto_crop_video(src, temp_vid, height)
    
    cap = cv2.VideoCapture(temp_vid)
    sfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9
    nfr = max(1, int(round(total * fps_out)))
    fs = max(16, int(W * fs_pct / 100))
    neon = Neon(W, H, neon_th, neon_speed) if neon_on else None

    q = queue.Queue(maxsize=32)
    stop = threading.Event()

    def put(x):
        while not stop.is_set():
            try:
                q.put(x, timeout=0.5)
                return
            except queue.Full:
                continue

    def producer():
        cur, last, ui = -1, None, 0
        try:
            for j in range(nfr):
                if stop.is_set():
                    return
                T = j / fps_out
                while ui < len(plan) - 1 and T >= plan[ui + 1][0]:
                    ui += 1
                cs, _d, s0, spd = plan[ui]
                idx = min(int((s0 + (T - cs) * spd) * sfps + 0.5), stotal - 1)
                if idx > cur + 90:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    cur = idx - 1
                new = False
                while cur < idx:
                    if not cap.grab():
                        break
                    cur += 1
                    new = True
                if new:
                    ok, f = cap.retrieve()
                    if ok:
                        if f.shape[1] != W or f.shape[0] != H:
                            f = cv2.resize(f, (W, H))
                        last = f
                if last is None:
                    return
                put(last)
        finally:
            put(None)

    log_path = out + ".log"
    log = open(log_path, "wb")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
           "-s", f"{W}x{H}", "-r", str(fps_out), "-i", "-", "-i", audio,
           "-map", "0:v", "-map", "1:a:0",
           "-c:v", "libx264", "-preset", preset, "-crf", "27", "-pix_fmt", "yuv420p",
           "-threads", "0", "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
           "-t", f"{total:.3f}", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
    th = threading.Thread(target=producer, daemon=True)
    th.start()

    k, cur_k, cur_ov, j = 0, -1, None, 0
    try:
        while True:
            fr = q.get()
            if fr is None:
                break
            t = j / fps_out
            fr = zoom_flip(fr, zoom, flip)
            if neon:
                neon.apply(fr, t)
            if subs and font_path:
                while k < len(subs) and subs[k][1] < t:
                    k += 1
                if k < len(subs) and subs[k][0] <= t:
                    if cur_k != k:
                        cur_ov = build_overlay(subs[k][2], W, H, font_path, fs, pos_y,
                                               is_mm=is_mm, **style)
                        cur_k = k
                    if cur_ov:
                        blend_overlay(fr, cur_ov)
            proc.stdin.write(fr.data)
            j += 1
            if j % 30 == 0:
                if progress:
                    progress.progress(min(j / nfr, 1.0))
                if cancel:
                    cancel()
    except BrokenPipeError:
        pass
    finally:
        stop.set()
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
        th.join(timeout=5)
        cap.release()
        try:
            proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=None)
        except Exception:
            pass
        log.close()
        
    if os.path.exists(temp_vid):
        os.remove(temp_vid)
        
    if proc.returncode != 0:
        with open(log_path, "rb") as f:
            tail = f.read()[-800:].decode("utf-8", "ignore")
        raise RuntimeError("render: ffmpeg fail\n" + tail)
    if progress:
        progress.progress(1.0)
    return out


def make_preview(path, height, text, font_path, fs_pct, pos_y, neon_on, neon_th,
                 neon_speed, flip, zoom, style=None, is_mm=True):
    temp_vid = path + "_pv.mp4"
    W, H, _, _ = auto_crop_video(path, temp_vid, height)
    
    cap = cv2.VideoCapture(temp_vid)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    ok, fr = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, fr = cap.read()
    cap.release()
    if os.path.exists(temp_vid):
        os.remove(temp_vid)
        
    if not ok:
        return None
        
    fr = cv2.resize(fr, (W, H))
    fr = zoom_flip(fr, zoom, flip)
    if neon_on:
        Neon(W, H, neon_th, neon_speed).apply(fr, 1.0)
    if font_path:
        ov = build_overlay(text, W, H, font_path, max(16, int(W * fs_pct / 100)), pos_y,
                           is_mm=is_mm, **(style or {}))
        if ov:
            blend_overlay(fr, ov)
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


@st.cache_data(show_spinner=False, max_entries=6)
def cached_preview(path, fsize, height, text, font_path, fs_pct, pos_y, neon_on,
                   neon_th, neon_speed, flip, zoom, alpha, is_mm):
    style = {"alpha": alpha, "color": (255, 255, 255), "bold": 0, "layout": "raqm"}
    return make_preview(path, height, text, font_path, fs_pct, pos_y, neon_on, neon_th,
                        neon_speed, flip, zoom, style, is_mm)


# ====================================================================
# Manual translate helpers
# ====================================================================
def build_prompt(units, lang="မြန်မာ"):
    lines = "\n".join(f"[{i}] {u['src']}" for i, u in enumerate(units))
    conf = LANG_CONF.get(lang, LANG_CONF["မြန်မာ"])
    return conf["prompt"] + lines


def parse_numbered(text, n):
    got = {}
    for line in text.splitlines():
        line = line.strip().strip("*").strip()
        m = re.match(r"^[\[\(#]?\s*(\d+)\s*[\]\)\.:\-–]?\s*(.+)$", line)
        if m and int(m.group(1)) < n:
            got[int(m.group(1))] = m.group(2).strip().strip("*").strip()
    if not got:
        plain = [l.strip() for l in text.splitlines() if l.strip()]
        if len(plain) == n:
            got = {i: plain[i] for i in range(n)}
    return got


# ====================================================================
# Link download
# ====================================================================
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".ts", ".flv")


def _fix_dropbox(url):
    u = urlparse(url)
    q = parse_qs(u.query)
    q["dl"] = ["1"]
    return urlunparse(u._replace(query=urlencode(q, doseq=True)))


def _dl_direct(url, out, bar=None):
    import requests
    with requests.get(url, stream=True, timeout=30, allow_redirects=True,
                      headers={"User-Agent": "Mozilla/5.0"}) as r:
        r.raise_for_status()
        ctype = r.headers.get("content-type", "").lower()
        if "text/html" in ctype:
            raise RuntimeError("Link က video ဖိုင် မဟုတ်ဘဲ web page ဖြစ်နေတယ်")
        total = int(r.headers.get("content-length") or 0)
        got = 0
        with open(out, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                got += len(chunk)
                if bar is not None and total:
                    bar.progress(min(got / total, 1.0))
    if not os.path.exists(out) or os.path.getsize(out) < 1024:
        raise RuntimeError("ဖိုင် မရပါ")
    return out


def _dl_gdrive(url, out):
    import gdown
    gdown.download(url, out, quiet=True, fuzzy=True)
    if not os.path.exists(out) or os.path.getsize(out) < 1024:
        raise RuntimeError("Google Drive ဖိုင် မရပါ (Share ကို 'Anyone with the link' ထားပါ)")
    return out


def _dl_ytdlp(url, out_noext, height, bar=None, cookies_text=None):
    import yt_dlp

    def hook(d):
        if bar is not None and d.get("status") == "downloading":
            tot = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
            if tot:
                bar.progress(min(d.get("downloaded_bytes", 0) / tot, 1.0))

    opts = {
        "format": (f"bv*[height<={height}][vcodec^=avc1]+ba[ext=m4a]/"
                   f"b[height<={height}][vcodec^=avc1]/"
                   f"bv*[height<={height}]+ba/b[height<={height}]/bv*+ba/b"),
        "outtmpl": out_noext + ".%(ext)s",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 5,
        "fragment_retries": 5,
        "concurrent_fragment_downloads": 4,
        "progress_hooks": [hook],
    }
    if cookies_text:
        cf = out_noext + "_cookies.txt"
        with open(cf, "w") as f:
            f.write(cookies_text)
        opts["cookiefile"] = cf
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    files = [p for p in glob.glob(out_noext + ".*")
             if not p.endswith((".part", ".ytdl", "_cookies.txt"))]
    if not files:
        raise RuntimeError("yt-dlp ဖိုင် မထွက်ပါ")
    return max(files, key=os.path.getsize)


def _ensure_mp4(raw, final, height):
    try:
        codec = sh(["ffprobe", "-v", "error", "-select_streams", "v:0",
                    "-show_entries", "stream=codec_name", "-of", "csv=p=0", raw]).strip()
    except Exception:
        codec = ""
    if not codec:
        raise RuntimeError("Video stream မတွေ့ပါ (ဖိုင်က video မဟုတ်နိုင်)")
    if codec == "h264":
        try:
            sh(["ffmpeg", "-y", "-i", raw, "-c", "copy", "-movflags", "+faststart", final])
            return final
        except Exception:
            pass
    sh(["ffmpeg", "-y", "-i", raw, "-vf", f"scale=-2:'min(ih,{int(height)})'",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", final])
    return final


def fetch_link(url, height, bar=None, cookies_text=None):
    url = url.strip()
    if not re.match(r"^https?://", url, re.I):
        raise ValueError("Link က http:// ဒါမှမဟုတ် https:// နဲ့ စရမယ်")
    key = hashlib.md5(url.encode()).hexdigest()[:12]
    final = f"/tmp/recap_up_{key}.mp4"
    if os.path.exists(final) and os.path.getsize(final) > 1024:
        return final
    work = tempfile.mkdtemp(prefix="dl_")
    try:
        u = urlparse(url)
        host, path_l = u.netloc.lower(), u.path.lower()
        raw, errs = None, []
        if "drive.google.com" in host or "docs.google.com" in host:
            raw = os.path.join(work, "gd.bin")
            _dl_gdrive(url, raw)
        else:
            if "dropbox.com" in host:
                url = _fix_dropbox(url)
                methods = ["direct", "ytdlp"]
            elif path_l.endswith(VIDEO_EXT):
                methods = ["direct", "ytdlp"]
            else:
                methods = ["ytdlp", "direct"]
            for m in methods:
                try:
                    if bar is not None:
                        bar.progress(0.0)
                    if m == "direct":
                        raw = _dl_direct(url, os.path.join(work, "direct.bin"), bar)
                    else:
                        raw = _dl_ytdlp(url, os.path.join(work, "yt"), height, bar, cookies_text)
                    break
                except Exception as ex:
                    errs.append(f"{m}: {str(ex)[:200]}")
                    raw = None
            if not raw:
                raise RuntimeError("Link ကနေ မဆွဲနိုင်ပါ\n" + "\n".join(errs))
        if bar is not None:
            bar.progress(1.0)
            if hasattr(bar, "set_label"):
                bar.set_label("🔄 MP4 ပြောင်းနေတယ် (ဖိုင်ကြီးရင် ခဏကြာမယ်)...")
        _ensure_mp4(raw, final, height)
        return final
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ====================================================================
# Background job targets (st.* မခေါ်ရ)
# ====================================================================
def job_download(job, url, height, cookies):
    job["label"] = "⬇️ Video ဆွဲနေတယ်..."
    p = fetch_link(url, height, JobBar(job), cookies)
    job["out"] = {"path": p, "src": url}


def run_whisper(job, wm, audio, lang, fast, vdur, bar):
    raw = []
    if fast:
        try:
            from faster_whisper import BatchedInferencePipeline
            pipe = BatchedInferencePipeline(model=wm)
            segs, info = pipe.transcribe(audio, language=lang, beam_size=1,
                                         batch_size=8, without_timestamps=False)
            for s in segs:
                check_cancel(job)
                if s.text.strip():
                    raw.append((s.start, s.end, s.text.strip()))
                bar.progress(min(s.end / vdur, 1.0))
            return raw, info
        except Cancelled:
            raise
        except Exception:
            raw = []
    segs, info = wm.transcribe(audio, language=lang, vad_filter=True, beam_size=1,
                               temperature=0.0, condition_on_previous_text=False)
    for s in segs:
        check_cancel(job)
        if s.text.strip():
            raw.append((s.start, s.end, s.text.strip()))
        bar.progress(min(s.end / vdur, 1.0))
    return raw, info


def job_whisper(job, reg, wd, vid_path, P):
    T1 = {}
    src = os.path.join(wd, "src.mp4")
    try:
        os.link(vid_path, src)
    except Exception:
        shutil.copy(vid_path, src)

    vdur = dur(src)
    if vdur > P["max_min"] * 60:
        raise RuntimeError(f"Video က {vdur/60:.1f} မိနစ်ရှိတယ်။ {P['max_min']} မိနစ်အောက်ပဲ လက်ခံတယ် "
                           "(Settings မှာ ပြောင်းလို့ရ)")

    job["label"] = "🎞️ Audio ထုတ်နေတယ်..."
    t_a = time.time()
    wav = os.path.join(wd, "audio16k.wav")
    sh(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", wav])
    with wave.open(wav) as wf:
        audio = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    T1["🎞️ Audio ထုတ်"] = time.time() - t_a
    job["frac"] = 0.08
    check_cancel(job)

    job["label"] = "📦 Whisper model ဖွင့်နေတယ် (ပထမအကြိမ် download ကြာမယ်)..."
    t_a = time.time()
    wm = get_whisper(reg, P["wsize"])
    T1["📦 Whisper model ဖွင့်"] = time.time() - t_a
    job["frac"] = 0.15
    check_cancel(job)

    job["label"] = "🎧 Whisper စာသားပြောင်းနေတယ်..."
    t_a = time.time()
    lang = None if P["src_lang"] == "auto" else P["src_lang"]
    raw, info = run_whisper(job, wm, audio, lang, P["fast"], vdur, JobBar(job, 0.15, 0.97))
    T1["🎧 Whisper စာသားပြောင်း"] = time.time() - t_a

    units = []
    for s, e, t in raw:
        if units and (units[-1][1] - units[-1][0]) < TARGET and (e - units[-1][0]) <= MAXD:
            units[-1] = (units[-1][0], e, units[-1][2] + " " + t)
        else:
            units.append((s, e, t))
    if not units:
        raise RuntimeError("စကားပြောသံ မတွေ့ပါ")
    job["out"] = {
        "units": [{"start": u[0], "end": u[1], "src": u[2], "my": ""} for u in units],
        "vdur": vdur, "T1": T1,
        "info": f"ဘာသာစကား: {info.language} ({info.language_probability:.0%}) · အပိုင်း {len(units)} ခု",
    }


def job_render(job, wd, src, units, my_list, vdur, P):
    n = len(units)
    T = {}
    t_all = time.time()

    vox = VoxEngine(P["ref_path"], P["hf_token"]) if P["is_vox"] else None
    sig_base = f"{P['engine']}|{P['voice']}|{P['rate']}|{P['ref_sig']}"

    def do_line(i):
        check_cancel(job)
        wav = f"{wd}/a{i:04d}.wav"
        keyf = f"{wd}/a{i:04d}.key"
        key = hashlib.md5(f"{sig_base}|{my_list[i]}".encode()).hexdigest()
        if os.path.exists(wav) and os.path.exists(keyf):
            try:
                with open(keyf) as kf:
                    if kf.read().strip() == key:
                        return i, wav_dur(wav), vox is not None
            except Exception:
                pass
        if vox is not None:
            raw_audio = f"{wd}/a{i:04d}_raw.wav"
            used = None
            for attempt in range(P["vox_retry"]):
                check_cancel(job)
                used = vox.synth(my_list[i], raw_audio)
                if used:
                    break
                vox.reset()
                time.sleep(3)
            if not used:
                raise RuntimeError(
                    f"VoxCPM မအောင်မြင်ပါ (စာကြောင်း #{i}): {vox.last_err}\n\n"
                    "Edge TTS ကို မသုံးထားပါ။ Space sleep/queue ဖြစ်နေနိုင်လို့ "
                    "ခဏနေပြီး ပြန်စမ်းပါ။")
        else:
            raw_audio = f"{wd}/a{i:04d}_raw.mp3"
            for attempt in range(4):
                try:
                    tts_edge(my_list[i], P["voice"], P["rate"], raw_audio, P["proxy"])
                    break
                except Exception as ex:
                    if attempt == 3:
                        raise RuntimeError(
                            f"Edge TTS မအောင်မြင်ပါ (စာကြောင်း #{i}): {ex}\n\n{my_list[i]}")
                    time.sleep(1.5)
        sh(["ffmpeg", "-y", "-i", raw_audio, "-af", TRIM, "-ar", "44100", "-ac", "1", wav])
        with open(keyf, "w") as kf:
            kf.write(key)
        return i, wav_dur(wav), vox is not None

    durs = [0.0] * n
    vox_ok, done = 0, 0
    workers = 1 if vox is not None else P["tts_workers"]
    t_a = time.time()
    job["label"] = f"🔊 အသံ ထုတ်နေတယ် 0/{n}"
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(do_line, i) for i in range(n)]
        try:
            for fu in as_completed(futs):
                i_, d_, u_ = fu.result()
                durs[i_] = d_
                vox_ok += 1 if u_ else 0
                done += 1
                job["frac"] = 0.3 * done / n
                job["label"] = f"🔊 အသံ ထုတ်နေတယ် {done}/{n}"
        except BaseException:
            for f_ in futs:
                f_.cancel()
            raise
    T["🔊 အသံထုတ် (TTS)"] = time.time() - t_a

    check_cancel(job)
    t_a = time.time()
    job["label"] = "🔗 အသံ ပေါင်းနေတယ်..."
    lst = f"{wd}/alist.txt"
    with open(lst, "w") as f:
        for i in range(n):
            f.write(f"file '{wd}/a{i:04d}.wav'\n")
    allwav = f"{wd}/all.wav"
    sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", allwav])
    T["🔗 အသံ ပေါင်း"] = time.time() - t_a

    bounds = [0.0] + [(units[i]["end"] + units[i + 1]["start"]) / 2
                      for i in range(n - 1)] + [vdur]
    plan, cum = [], 0.0
    for i in range(n):
        s_, e_ = bounds[i], bounds[i + 1]
        w = max(e_ - s_, 0.2)
        d = max(durs[i], 0.05)
        w_use = d * P["max_speed"] if w / d > P["max_speed"] else w
        plan.append((cum, d, s_, w_use / d))
        cum += d

    subs, t0 = [], 0.0
    if P["use_sub"]:
        for i in range(n):
            parts = split_for_sub(my_list[i], lang=P["voice_lang"])
            tot = sum(len(p) for p in parts) or 1
            cur = t0
            for p in parts:
                dd = durs[i] * len(p) / tot
                subs.append((cur, cur + dd, p))
                cur += dd
            t0 += durs[i]

    t_a = time.time()
    job["label"] = "🎬 Video render (Smart Sync + စာတန်း + Neon)..."
    job["frac"] = 0.3
    final = f"{wd}/final.mp4"
    render_final(src, allwav, final, plan, cum, P["height"], subs,
                 P["font_path"] if P["use_sub"] else None, P["sub_size"], P["sub_pos"],
                 P["use_neon"], P["neon_th"], P["neon_speed"], P["flip"], P["zoom"],
                 P["sub_style"], preset=P["preset"], progress=JobBar(job, 0.3, 1.0),
                 fps_out=P["fps_out"], is_mm=P["is_mm"], cancel=lambda: check_cancel(job))
    T["🎬 Video render"] = time.time() - t_a
    T["⏱️ စုစုပေါင်း (အဆင့် 2)"] = time.time() - t_all
    msg = f"🎬 Render ပြီး ({cum:.1f}s / မူရင်း {vdur:.1f}s)"
    if vox is not None:
        msg += f" · VoxCPM {vox_ok}/{n}"
    job["out"] = {"final": final, "T2": T, "info": msg}


# ====================================================================
# Session / persistence helpers
# ====================================================================
def get_jid():
    jid = st.query_params.get("jid")
    if not jid or not re.fullmatch(r"[0-9a-f]{6,16}", str(jid)):
        jid = uuid.uuid4().hex[:10]
        st.query_params["jid"] = jid
    return str(jid)


def _state_path(jid):
    return os.path.join(STATE_DIR, f"{jid}.json")


def persist(jid):
    try:
        data = {
            "cfg": st.session_state.get("_cfg", {}),
            "wd": st.session_state.get("wd"),
            "units": st.session_state.get("units"),
            "vdur": st.session_state.get("vdur"),
            "t1": st.session_state.get("t1"),
            "t2": st.session_state.get("t2"),
            "manual": st.session_state.get("manual"),
            "apply_msg": st.session_state.get("apply_msg"),
            "info1": st.session_state.get("info1"),
            "info2": st.session_state.get("info2"),
            "link_path": st.session_state.get("link_path"),
            "link_src": st.session_state.get("link_src"),
            "up_paths": st.session_state.get("up_paths", []),
            "has_result": bool(st.session_state.get("result")),
        }
        tmp = _state_path(jid) + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, _state_path(jid))
    except Exception:
        pass


def restore(jid):
    p = _state_path(jid)
    if not os.path.exists(p):
        return
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    cfg = {}
    for k, v in (data.get("cfg") or {}).items():
        cv = _coerce(k, v)
        if cv is not None:
            cfg[k] = cv
    st.session_state["_cfg"] = cfg

    wd = data.get("wd")
    if wd and os.path.isdir(wd):
        st.session_state["wd"] = wd
        for k in ("units", "vdur", "t1", "t2", "manual", "apply_msg", "info1", "info2"):
            if data.get(k) is not None:
                st.session_state[k] = data[k]
        final = os.path.join(wd, "final.mp4")
        if data.get("has_result") and os.path.exists(final):
            try:
                with open(final, "rb") as f:
                    st.session_state["result"] = f.read()
            except Exception:
                pass
    lp = data.get("link_path")
    if lp and os.path.exists(lp):
        st.session_state["link_path"] = lp
        st.session_state["link_src"] = data.get("link_src")
    ups = [x for x in (data.get("up_paths") or []) if os.path.exists(x)]
    if ups:
        st.session_state["up_paths"] = ups


def reset_job():
    wd = st.session_state.get("wd")
    if wd and os.path.isdir(wd):
        shutil.rmtree(wd, ignore_errors=True)
    for k in ("wd", "units", "vdur", "result", "manual", "apply_msg", "gem_paste",
              "t1", "t2", "info1", "info2", "job_err"):
        st.session_state.pop(k, None)


def cleanup_uploads():
    for p in st.session_state.get("up_paths", []):
        try:
            os.remove(p)
        except Exception:
            pass
    st.session_state["up_paths"] = []
    st.session_state.pop("link_path", None)
    st.session_state.pop("link_src", None)
    st.session_state["up_key"] = st.session_state.get("up_key", 0) + 1


def _clear_link_input():
    st.session_state["link_url"] = ""
    st.session_state.setdefault("_cfg", {})["link_url"] = ""


def cleanup_after_download():
    wd = st.session_state.get("wd")
    if wd and os.path.isdir(wd):
        shutil.rmtree(wd, ignore_errors=True)
    for k in ("wd", "units", "vdur", "manual", "apply_msg", "gem_paste", "info1", "info2"):
        st.session_state.pop(k, None)
    cleanup_uploads()
    _clear_link_input()
    st.session_state["cleaned"] = True


def full_reset():
    reset_job()
    cleanup_uploads()
    _clear_link_input()
    st.session_state.pop("cleaned", None)
    st.session_state.pop("show_vid", None)
    try:
        get_registry()["jobs"].pop(st.session_state.get("_jid"), None)
    except Exception:
        pass


def save_upload(up):
    key = hashlib.md5(f"{up.name}-{up.size}".encode()).hexdigest()[:12]
    path = f"/tmp/recap_up_{key}.mp4"
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(up.getbuffer())
    paths = st.session_state.setdefault("up_paths", [])
    if path not in paths:
        paths.append(path)
    return path


def consume_job(reg, jid):
    job = reg["jobs"].get(jid)
    if not job or job["state"] == "running" or job.get("consumed"):
        return
    job["consumed"] = True
    kind, state = job["kind"], job["state"]
    if state in ("error", "cancelled"):
        st.session_state["job_err"] = job["error"] if state == "error" else "⛔ ရပ်လိုက်ပါပြီ"
        if kind == "whisper":
            wd = st.session_state.pop("wd", None)
            if wd:
                shutil.rmtree(wd, ignore_errors=True)
        return
    out = job["out"] or {}
    if kind == "download":
        p = out["path"]
        paths = st.session_state.setdefault("up_paths", [])
        if p not in paths:
            paths.append(p)
        st.session_state["link_path"] = p
        st.session_state["link_src"] = out["src"]
    elif kind == "whisper":
        st.session_state["units"] = out["units"]
        st.session_state["vdur"] = out["vdur"]
        st.session_state["t1"] = out["T1"]
        st.session_state["info1"] = out["info"]
        st.session_state["manual"] = True
    elif kind == "render":
        try:
            with open(out["final"], "rb") as f:
                st.session_state["result"] = f.read()
            st.session_state["t2"] = out["T2"]
            st.session_state["info2"] = out["info"]
        except Exception as ex:
            st.session_state["job_err"] = f"ရလဒ်ဖိုင် ဖတ်မရ: {ex}"


@st.fragment(run_every=1.0)
def job_panel(jid):
    job = get_registry()["jobs"].get(jid)
    if not job or job["state"] != "running":
        st.rerun()
        return
    st.progress(max(0.0, min(float(job["frac"]), 1.0)),
                text=f"{job['label']} · {fmt_t(time.time() - job['t0'])}")

    def _cancel():
        job["cancel"] = True

    st.button("⛔ ရပ်မယ်", key="btn_cancel", on_click=_cancel)
    st.caption("💡 Setting တွေ ပြောင်းလို့ရတယ်၊ ဒီ အလုပ်က နောက်ကွယ်မှာ ဆက်လုပ်နေမယ်။ "
               "Browser ပိတ်ပြီး ပြန်ဖွင့်ရင်လည်း (ဒီ URL အတိုင်း) ဆက်ကြည့်လို့ရတယ်။")


# ====================================================================
# UI
# ====================================================================
reg = get_registry()
JID = get_jid()
st.session_state["_jid"] = JID

if not st.session_state.get("_restored"):
    st.session_state["_restored"] = True
    restore(JID)

CFG = st.session_state.setdefault("_cfg", {})
for _k, _dv in DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = CFG.get(_k, _dv)

consume_job(reg, JID)
_job = reg["jobs"].get(JID)
busy = bool(_job and _job["state"] == "running")

st.title("🎙️ Recap → Multi-Language")
st.caption("Whisper → Gemini → Edge TTS / VoxCPM → Smart Sync → PNG စာတန်း + Neon border")

with st.expander("⚙️ Settings", expanded=False):
    c1, c2 = st.columns(2)
    src_lang = c1.selectbox("မူရင်းဘာသာ", OPTS["src_lang"], key="src_lang")
    wsize = c2.selectbox("Whisper model", OPTS["wsize"], key="wsize",
                         help="small = တိကျ/နှေး၊ tiny = မြန်/အမှားများ")
    fast_whisper = st.toggle("⚡ Whisper batch mode (ပိုမြန်)", key="fast_whisper",
                             help="faster-whisper 1.1+ လိုတယ်။ မရရင် အလိုအလျောက် ပုံမှန်နည်းကို ပြန်သုံးမယ်")
    c3, c4 = st.columns(2)
    height = c3.select_slider("Output height", OPTS["height"], key="height",
                              help="480 သုံးရင် render အရမ်းမြန်တယ်")
    max_speed = c4.slider("ရုပ် အမြန်ဆုံး ဆ", 1.0, 2.0, step=0.05, key="max_speed")
    c7, c8 = st.columns(2)
    fps_out = c7.selectbox("Output FPS", OPTS["fps_out"], key="fps_out",
                           help="နည်းလေ render မြန်လေ (24 ကောင်းတယ်)")
    preset = c8.selectbox("Render speed", OPTS["preset"], key="preset",
                          help="ultrafast = အမြန်ဆုံး (file အနည်းငယ်ကြီး)")
    max_min = st.slider("Video အရှည် ကန့်သတ် (မိနစ်)", 1, 120, key="max_min")

with st.expander("🔊 ဘာသာစကား + အသံ (Voice)", expanded=True):
    voice_lang = st.radio("ဘာသာစကား (အသံ + Prompt)", OPTS["voice_lang"],
                          horizontal=True, key="voice_lang")
    lang_conf = LANG_CONF[voice_lang]

    engine = st.radio("TTS engine", ENGINES, horizontal=False, key="engine")
    is_vox = engine.startswith("Vox")

    voice = lang_conf["edge_voices"][0][0]
    rate_n = st.session_state["rate_n"]
    tts_workers = 1 if is_vox else st.session_state["tts_workers"]
    if not is_vox:
        vlabels = {v: lbl for v, lbl in lang_conf["edge_voices"]}
        voice = st.selectbox("Edge voice", list(vlabels.keys()),
                             format_func=lambda v: vlabels.get(v, v),
                             key=f"voice_{voice_lang}")
        rate_n = st.slider("Edge speed %", -50, 100, step=5, key="rate_n")
        tts_workers = st.slider("TTS တစ်ပြိုင်နက် အရေအတွက်", 1, 10, key="tts_workers",
                                help="များရင်မြန်တယ်။ Edge က error ပေးရင် လျှော့ပါ")

    ref_up = None
    vox_retry = st.session_state["vox_retry"]
    if is_vox:
        ref_up = st.file_uploader("🎤 Reference audio (အသံတစ်သံတည်း ဖြစ်ချင်ရင် တင်ပါ)",
                                  type=["wav", "mp3", "m4a"])
        vox_retry = st.slider("VoxCPM ပြန်စမ်းအကြိမ် (စာကြောင်းတစ်ခုချင်း)", 1, 8, key="vox_retry",
                              help="Space sleep/queue ဖြစ်ရင် ပြန်စမ်းမယ်။ ကုန်ရင် Edge မသုံးဘဲ error ပြမယ်")
        st.caption("⚠️ VoxCPM က မြန်မာအသံသာ ကောင်းကောင်း ထောက်ပံ့တယ်။ "
                   "ဂျာမနီ/ဂျပန် အတွက် Edge TTS ကို ပြောင်းသုံးပါ။")

rate = f"{int(rate_n):+d}%"
is_mm = lang_conf["is_mm"]

with st.expander("🎨 စာတန်း (PNG) + Neon", expanded=False):
    use_sub = st.toggle("PNG စာတန်းထည့်", key="use_sub")
    c5, c6 = st.columns(2)
    sub_size = c5.slider("စာအရွယ် (% of width)", 3.0, 12.0, step=0.5, key="sub_size")
    sub_pos = c6.slider("နေရာ (0=အပေါ် 100=အောက်)", 0, 100, step=1, key="sub_pos")
    use_neon = st.toggle("Neon border", key="use_neon")
    # ✅ Neon အထူ / Speed ကို ပုံသေ ထားတယ် (slider မပြ)
    neon_th = 3
    neon_speed = 0.4
    zoom = st.slider("Zoom", 1.0, 1.25, step=0.01, key="zoom")
    font_path = find_font_for(voice_lang)
    if not font_path:
        st.warning(f"{voice_lang} font မတွေ့ပါ။ repo သို့မဟုတ် system font ထည့်ပါ။")
    else:
        st.caption(f"🔤 Font: `{os.path.basename(font_path)}`")
    if not features.check("raqm"):
        st.warning("raqm မရှိလို့ မြန်မာစာ ပုံစံ မမှန်နိုင်ပါ (packages.txt ထဲ libraqm0 ပါစေ)")

flip = True
sub_style = {
    "alpha": int(0.8 * 255),
    "color": (255, 255, 255),
    "bold": 0,
    "layout": "raqm",
}
proxy = secret("EDGE_TTS_PROXY") or None
hf_token = secret("HF_TOKEN") or None

mode = st.radio("Video ရင်းမြစ်", OPTS["mode"], horizontal=True, key="mode")
vid_path = None
url_in = st.session_state.get("link_url", "")
if mode.startswith("🔗"):
    url_in = st.text_input("Video link (direct mp4 / Google Drive / Dropbox / YouTube / Facebook / TikTok ...)",
                           placeholder="https://...", key="link_url")
    if st.button("⬇️ Link ကနေ ဆွဲမယ်", use_container_width=True, key="btn_dl",
                 disabled=busy or not url_in.strip()):
        st.session_state.pop("job_err", None)
        st.session_state.pop("link_path", None)
        st.session_state.pop("link_src", None)
        start_job(reg, JID, "download", job_download, url_in.strip(), height,
                  secret("YT_COOKIES") or None)
        st.rerun()
    lp = st.session_state.get("link_path")
    if lp and os.path.exists(lp):
        vid_path = lp
        st.caption(f"📎 ဆွဲထားတဲ့ video: {os.path.getsize(lp)/1e6:.1f} MB")
        if st.session_state.get("link_src") != url_in.strip():
            st.warning("Link ပြောင်းထားတယ်။ ⬇️ ကို ပြန်နှိပ်မှ video အသစ်ရမယ် (အခုက အရင် link ကိုပဲ သုံးနေတယ်)")
else:
    up = st.file_uploader("Video ရွေးပါ", type=["mp4", "mov", "mkv", "webm", "m4v"],
                          key=f"vid_{st.session_state.get('up_key', 0)}")
    if up:
        vid_path = save_upload(up)

# settings ကို မှတ်ထား (widget မပေါ်တဲ့ key တွေပါ ပြန်ရအောင်)
for _k in DEFAULTS:
    if _k in st.session_state:
        CFG[_k] = st.session_state[_k]
persist(JID)

if vid_path:
    try:
        sample = {
            "မြန်မာ": "စာတန်းထိုး နမူနာ စာသား ဖြစ်ပါတယ်။",
            "ဂျာမနီ": "Dies ist ein Beispiel-Untertitel.",
            "ဂျပန်": "これはサンプルの字幕です。",
        }[voice_lang]
        pv = cached_preview(vid_path, os.path.getsize(vid_path), height, sample, font_path,
                            sub_size, sub_pos, use_neon, neon_th, neon_speed, flip, zoom,
                            sub_style["alpha"], is_mm)
        if pv is not None:
            st.image(pv, caption=f"Preview ({voice_lang})", use_container_width=True)
    except Exception as ex:
        st.caption(f"preview မရ: {str(ex)[:100]}")

# ====================================================================
# Stage 1 : Whisper (background)
# ====================================================================
if st.button("1️⃣ စာသားထုတ်", type="primary", use_container_width=True, key="btn_s1",
             disabled=busy or not vid_path):
    reset_job()
    st.session_state.pop("cleaned", None)
    wd = tempfile.mkdtemp(prefix="recap_")
    st.session_state["wd"] = wd
    start_job(reg, JID, "whisper", job_whisper, reg, wd, vid_path,
              {"src_lang": src_lang, "wsize": wsize, "max_min": max_min, "fast": fast_whisper})
    persist(JID)
    st.rerun()

# ====================================================================
# Edit + Stage 2
# ====================================================================
if st.session_state.get("units"):
    units = st.session_state["units"]
    if st.session_state.get("info1"):
        st.caption("ℹ️ " + st.session_state["info1"])
    if st.session_state.get("t1"):
        st.caption("⏱️ " + " · ".join(f"{k} {fmt_t(v)}" for k, v in st.session_state["t1"].items()))

    if st.session_state.get("manual"):
        st.subheader(f"📋 Gemini မှာ ဘာသာပြန်ရန် ({voice_lang})")
        st.caption("① အောက်က စာသားကို Copy (ညာထောင့် icon) → ② Gemini app မှာ paste → "
                   "③ ရလဒ်ကို အောက်က ကွက်ထဲ paste → ④ Apply")
        with st.expander("📄 Prompt (ဖွင့်ပြီး Copy ယူပါ)", expanded=False):
            st.code(build_prompt(units, voice_lang), language=None)
        pasted = st.text_area("Gemini ရလဒ် ဒီမှာ paste", height=200, key="gem_paste",
                              placeholder="[0] ဘာသာပြန်စာသား...\n[1] ဘာသာပြန်စာသား...")
        if st.button("✅ Apply (စာသား ဖြည့်)", use_container_width=True, key="btn_apply"):
            got = parse_numbered(pasted, len(units))
            for i, t in got.items():
                units[i]["my"] = clean_text(t, voice_lang)
            missing = [i for i in range(len(units)) if not units[i]["my"]]
            st.session_state["apply_msg"] = (
                f"ဖြည့်ပြီး {len(got)}/{len(units)} ကြောင်း"
                + (f" · မပါသေးတာ: {missing[:15]}" if missing else " ✅"))
            persist(JID)
            st.rerun()
        if st.session_state.get("apply_msg"):
            st.info(st.session_state["apply_msg"])

    if st.button("2️⃣ အသံထုတ် + Video ပေါင်း", type="primary", use_container_width=True,
                 key="btn_s2", disabled=busy):
        wd, vdur = st.session_state["wd"], st.session_state["vdur"]
        src = os.path.join(wd, "src.mp4")
        my_list = [clean_text(u["my"], voice_lang) for u in units]
        empty = [i for i, t in enumerate(my_list) if not t.strip()]
        if empty:
            st.error(f"စာသား မဖြည့်ရသေးတဲ့ စာကြောင်း: {empty[:20]}")
        else:
            ref_path, ref_sig = None, "noref"
            if is_vox and ref_up:
                ref_path = os.path.join(wd, "ref.wav")
                with open(ref_path, "wb") as f:
                    f.write(ref_up.getbuffer())
                ref_sig = f"{ref_up.name}-{ref_up.size}"
            P = {
                "is_vox": is_vox, "engine": engine, "voice": voice, "rate": rate,
                "ref_path": ref_path, "ref_sig": ref_sig, "hf_token": hf_token, "proxy": proxy,
                "tts_workers": int(tts_workers), "vox_retry": int(vox_retry),
                "max_speed": float(max_speed), "height": int(height), "preset": preset,
                "fps_out": int(fps_out), "use_sub": use_sub, "voice_lang": voice_lang,
                "font_path": font_path, "sub_size": sub_size, "sub_pos": sub_pos,
                "use_neon": use_neon, "neon_th": neon_th, "neon_speed": neon_speed,
                "flip": flip, "zoom": zoom, "sub_style": dict(sub_style), "is_mm": is_mm,
            }
            st.session_state.pop("result", None)
            st.session_state.pop("t2", None)
            st.session_state.pop("job_err", None)
            start_job(reg, JID, "render", job_render, wd, src,
                      [dict(u) for u in units], my_list, vdur, P)
            persist(JID)
            st.rerun()

# ====================================================================
# Processing panel (ခလုတ်တွေရဲ့ အောက်မှာ ပြ)
# ====================================================================
if busy:
    job_panel(JID)
if st.session_state.get("job_err"):
    st.error(st.session_state["job_err"][:1500])

# ====================================================================
# Result
# ====================================================================
if st.session_state.get("result"):
    st.subheader("✅ ရလဒ်")
    if st.session_state.get("info2"):
        st.caption(st.session_state["info2"])
    mb = len(st.session_state["result"]) / 1e6
    st.caption(f"📦 File အရွယ်အစား: {mb:.1f} MB")
    if st.toggle("▶️ ဒီမှာ video ကြည့်မယ် (ဖိုင်ကြီးရင်/net နှေးရင် ကြာတတ်)", value=False, key="show_vid"):
        st.video(st.session_state["result"])
    _tt = {}
    _tt.update(st.session_state.get("t1") or {})
    _tt.update(st.session_state.get("t2") or {})
    if _tt:
        st.markdown("**⏱️ ဘယ်အပိုင်း ဘယ်လောက်ကြာလဲ**\n\n" + timing_table(_tt))
    fname = {"မြန်မာ": "final_myanmar.mp4", "ဂျာမနီ": "final_german.mp4",
             "ဂျပန်": "final_japanese.mp4"}.get(voice_lang, "final.mp4")
    st.download_button("⬇️ Download MP4", st.session_state["result"],
                       file_name=fname, mime="video/mp4",
                       use_container_width=True, on_click=cleanup_after_download)
    if st.session_state.get("cleaned"):
        st.caption("🧹 Script နဲ့ တင်ထားတဲ့ video ကို ဖျက်ပြီးပါပြီ")

if vid_path or st.session_state.get("units") or st.session_state.get("result") \
        or st.session_state.get("link_path"):
    st.button("🗑️ အသစ်ပြန်စ (video + link ရှင်း)", use_container_width=True,
              key="btn_reset", on_click=full_reset, disabled=busy)

persist(JID)