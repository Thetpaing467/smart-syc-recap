import os
import re
import json
import time
import glob
import wave
import asyncio
import shutil
import hashlib
import subprocess
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, features

st.set_page_config(page_title="Recap → Myanmar", page_icon="🎙️", layout="centered")

DEFAULT_MODELS = (
    "gemini-3.5-flash, gemini-3.5-flash-lite, gemini-3.6-flash, "
    "gemini-3.7-flash, gemini-3.1-flash-lite"
)
TARGET, MAXD, BATCH = 4.2, 5.6, 40

VOX_SPACES = [
    {"space": "openbmb/VoxCPM-Demo", "type": "demo"},
    {"space": "hgghfhjfhjguyjf/Voxcpm-Burmese-Tts", "type": "burmese"},
]

TRIM = (
    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,apad=pad_dur=0.2"
)


# ====================================================================
# Generic helpers
# ====================================================================
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


def clean_my(text):
    """မြန်မာစာ၊ ဂဏန်း၊ ပုဒ်ဖြတ်သာ ကျန်ခဲ့"""
    return re.sub(
        r"[^\u1000-\u109F\uAA60-\uAA7F\uA9E0-\uA9FF\s0-9.,!?\"'%\-]", "", text
    ).strip()


@st.cache_resource(show_spinner="Whisper model ကို ဖွင့်နေတယ် (ပထမအကြိမ် download လုပ်မယ်)...")
def load_whisper(size):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8")


def make_gen(api_key, models):
    from google import genai
    client = genai.Client(api_key=api_key)

    def gen(prompt, json_out=True):
        last = None
        for attempt in range(3):
            for m in models:
                try:
                    cfg = {"response_mime_type": "application/json"} if json_out else None
                    r = client.models.generate_content(model=m, contents=prompt, config=cfg)
                    if r.text:
                        return r.text
                except Exception as ex:  # noqa: BLE001
                    last = ex
            time.sleep(4 * (attempt + 1))
        raise RuntimeError(f"Gemini failed: {last}")

    return gen


# ====================================================================
# TTS : Edge TTS + VoxCPM (gradio_client)
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
    """VoxCPM Space များကို အစဉ်လိုက် စမ်းမယ်။ မရရင် None ပြန် (Edge TTS ကို ခေါ်ရန်)။"""

    def __init__(self, ref_path=None, token=None):
        self.ref = ref_path
        self.token = token
        self.clients = {}
        self.fails = {i: 0 for i in range(len(VOX_SPACES))}
        self.dead = set()
        self.active = None
        self.last_err = ""

    def synth(self, text, out_wav):
        from gradio_client import handle_file
        order = [i for i in range(len(VOX_SPACES)) if i not in self.dead]
        if self.active in order:
            order.remove(self.active)
            order.insert(0, self.active)
        for i in order:
            sp = VOX_SPACES[i]
            if sp["type"] == "burmese" and not self.ref:
                self.dead.add(i)          # reference audio မပါရင် ဒီ space မသုံးနိုင်
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
            except Exception as ex:  # noqa: BLE001
                self.last_err = f"{sp['space']}: {str(ex)[:120]}"
                self.fails[i] += 1
                if self.fails[i] >= 2:
                    self.dead.add(i)
        return None


# ====================================================================
# Subtitle PNG (Pillow) helpers
# ====================================================================
def find_font():
    """Myanmar font ရှာ: repo ထဲက MyanmarPadaung.ttf → တခြား .ttf → system (Padauk/Noto Myanmar)"""
    here = os.path.dirname(os.path.abspath(__file__))
    for name in ("MyanmarPadaung.ttf", "fonts/MyanmarPadaung.ttf"):
        pref = os.path.join(here, name)
        if os.path.exists(pref):
            return pref
    repo = glob.glob(os.path.join(here, "*.ttf")) + glob.glob(os.path.join(here, "*.otf")) + \
        glob.glob(os.path.join(here, "fonts", "*.ttf")) + glob.glob(os.path.join(here, "fonts", "*.otf"))
    if repo:
        return repo[0]
    sysf = glob.glob("/usr/share/fonts/**/*.ttf", recursive=True) + \
        glob.glob("/usr/share/fonts/**/*.otf", recursive=True)
    for key in ("padauk", "myanmar", "pyidaung", "notosansmyanmar"):
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
        except Exception:  # noqa: BLE001
            try:
                _FONT_CACHE[key] = ImageFont.truetype(path, size)
            except Exception:  # noqa: BLE001
                _FONT_CACHE[key] = ImageFont.load_default()
    return _FONT_CACHE[key]


def _is_start(c):
    return (0x1000 <= c <= 0x1021) or (0x1023 <= c <= 0x1027) or c in (0x1029, 0x102A, 0x103F)


def split_syll(word):
    """မြန်မာစာလုံးကို syllable အလိုက်ခွဲ (အတိအကျမဟုတ်၊ ဖြတ်ဖို့လုံလောက်)"""
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


def wrap_text(text, font, maxw):
    lines, cur = [], ""
    for word in text.split():
        pieces = [word] if font.getlength(word) <= maxw else split_syll(word)
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


def split_for_sub(text, mc=36):
    """စာတန်းတစ်ခုချင်းစီအတွက် စာသားခွဲ (။ နဲ့ နေရာလွတ် အလိုက်)"""
    sents = [s.strip() for s in re.split(r"(?<=။)", text) if s.strip()]
    parts = []
    for s in sents:
        if len(s) <= mc:
            parts.append(s)
            continue
        cur = ""
        for word in s.split():
            pieces = [word] if len(word) <= mc else split_syll(word)
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
                  bold=0, layout="raqm"):
    """စာတန်း PNG (RGBA): အနက်ရောင် rounded box (အကျယ်အပြည့်) + ဖြူစာ + အနက်အနားသတ်"""
    font = get_font(font_path, fs, layout)
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)

    pad_x, pad_y = int(fs * 0.6), int(fs * 0.5)
    lines = wrap_text(text, font, W - 2 * pad_x)
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
# Neon border
# ====================================================================
class Neon:
    BASE = np.array([12, 8, 10], np.float32)         # BGR
    CYAN = np.array([255, 255, 0], np.float32)
    MAGENTA = np.array([110, 30, 255], np.float32)
    IDLE = 0.14

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
                     2 * W + 2 * H - ys))).astype(np.int64)
        self.ys, self.xs = ys, xs
        self.Si = S % self.P
        self.P_arr = np.arange(self.P, dtype=np.float32)

    def _layer(self, h, color):
        P = self.P
        dist = (h - self.P_arr) % P
        a = np.clip(1.0 - dist / (self.tail * P), 0, 1)
        glow = self.IDLE + (1.0 - self.IDLE) * (a ** 0.55)
        core = np.clip(1.0 - dist / (0.06 * P), 0, 1) ** 2
        return color * glow[:, None] + 255.0 * 0.9 * core[:, None]

    def apply(self, fr, t):
        head = (t * self.speed * self.P / 4.0) % self.P
        c = self.BASE + self._layer(head, self.CYAN) + self._layer(head + self.P / 2, self.MAGENTA)
        strip = np.clip(c, 0, 255).astype(np.uint8)
        fr[self.ys, self.xs] = strip[self.Si]


def zoom_flip(fr, zoom, flip):
    if zoom > 1.001:
        H, W = fr.shape[:2]
        cw, ch = int(W / zoom), int(H / zoom)
        x0, y0 = (W - cw) // 2, (H - ch) // 2
        fr = cv2.resize(fr[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR)
    if flip:
        fr = fr[:, ::-1]
    return np.ascontiguousarray(fr)


def overlay_pass(src, out, subs, font_path, fs_pct, pos_y, neon_on, neon_th,
                 neon_speed, flip, zoom, style=None, progress=None):
    """Smart-Sync video ကို frame တစ်ခုချင်းဖတ်ပြီး neon + PNG subtitle ထည့်၊ audio ပြန်ပေါင်း"""
    style = style or {}
    cap = cv2.VideoCapture(src)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    fs = max(16, int(W * fs_pct / 100))
    neon = Neon(W, H, neon_th, neon_speed) if neon_on else None

    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", f"{fps}", "-i", "-", "-i", src, "-map", "0:v", "-map", "1:a:0",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p",
           "-c:a", "copy", "-shortest", out]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    k, cur_k, cur_ov, j = 0, -1, None, 0
    try:
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            t = j / fps
            fr = zoom_flip(fr, zoom, flip)
            if neon:
                neon.apply(fr, t)
            if subs and font_path:
                while k < len(subs) and subs[k][1] < t:
                    k += 1
                if k < len(subs) and subs[k][0] <= t:
                    if cur_k != k:
                        cur_ov = build_overlay(subs[k][2], W, H, font_path, fs, pos_y, **style)
                        cur_k = k
                    if cur_ov:
                        blend_overlay(fr, cur_ov)
            proc.stdin.write(fr.tobytes())
            j += 1
            if progress and j % 15 == 0:
                progress.progress(min(j / total, 1.0))
    finally:
        cap.release()
        try:
            proc.stdin.close()
        except Exception:  # noqa: BLE001
            pass
        proc.wait()
    if proc.returncode != 0:
        raise RuntimeError("overlay render: ffmpeg fail")
    if progress:
        progress.progress(1.0)
    return out


def make_preview(path, height, text, font_path, fs_pct, pos_y, neon_on, neon_th,
                 neon_speed, flip, zoom, style=None):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_MSEC, 1000)
    ok, fr = cap.read()
    if not ok:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, fr = cap.read()
    cap.release()
    if not ok:
        return None
    h0, w0 = fr.shape[:2]
    W = int(round(w0 * height / h0 / 2) * 2)
    H = int(height)
    fr = cv2.resize(fr, (W, H))
    fr = zoom_flip(fr, zoom, flip)
    if neon_on:
        Neon(W, H, neon_th, neon_speed).apply(fr, 1.0)
    if font_path:
        ov = build_overlay(text, W, H, font_path, max(16, int(W * fs_pct / 100)), pos_y, **(style or {}))
        if ov:
            blend_overlay(fr, ov)
    return cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)


# ====================================================================
# Manual translate helpers (Gemini app မှာ copy-paste)
# ====================================================================
def build_prompt(units):
    lines = "\n".join(f"[{i}] {u['src']}" for i, u in enumerate(units))
    return (
        "You are a Burmese (Myanmar) recap-video narrator. Translate every numbered line "
        "below into natural spoken Burmese narration (storyteller style). Rules: do NOT "
        "summarize, shorten or skip; do NOT move content between lines; keep character "
        "names consistent across all lines; use ONLY Burmese script (Myanmar Unicode), "
        "no English/Chinese letters. Output exactly one line per input line in the format "
        "[number] Burmese text -- same numbers, no extra comments.\n\n" + lines
    )


def parse_numbered(text, n):
    """'[3] စာသား' / '3. စာသား' / '3) စာသား' ပုံစံကို {id: text} ပြောင်း"""
    got = {}
    for line in text.splitlines():
        line = line.strip().strip("*").strip()
        m = re.match(r"^[\[\(#]?\s*(\d+)\s*[\]\)\.:\-–]?\s*(.+)$", line)
        if m and int(m.group(1)) < n:
            got[int(m.group(1))] = m.group(2).strip().strip("*").strip()
    if not got:  # နံပါတ်မပါရင် စာကြောင်းအရေအတွက်ညီမှ အစဉ်လိုက်ယူ
        plain = [l.strip() for l in text.splitlines() if l.strip()]
        if len(plain) == n:
            got = {i: plain[i] for i in range(n)}
    return got


# ====================================================================
# Session helpers
# ====================================================================
def reset_job():
    wd = st.session_state.get("wd")
    if wd and os.path.isdir(wd):
        shutil.rmtree(wd, ignore_errors=True)
    for k in ("wd", "units", "vdur", "result", "manual", "apply_msg", "gem_paste"):
        try:
            st.session_state.pop(k, None)
        except Exception:  # noqa: BLE001
            pass


def save_upload(up):
    """Upload ကို /tmp ထဲ တစ်ခါပဲ သိမ်း (rerun တိုင်း ပြန်မရေး)"""
    key = hashlib.md5(f"{up.name}-{up.size}".encode()).hexdigest()[:12]
    path = f"/tmp/recap_up_{key}.mp4"
    if not os.path.exists(path):
        with open(path, "wb") as f:
            f.write(up.getbuffer())
    return path


# ====================================================================
# UI
# ====================================================================
st.title("🎙️ Recap → Myanmar")
st.caption("Whisper → Gemini → Edge TTS / VoxCPM → Smart Sync → PNG စာတန်း + Neon border")

# ---- ဘာသာပြန်နည်း (Manual / Auto) : ပင်မစာမျက်နှာမှာ ထင်ရှားစွာ ----
mode = st.radio(
    "ဘာသာပြန်နည်း",
    ["✋ Manual (Gemini app မှာ copy-paste · key မလို)", "🤖 Auto (Gemini API key)"],
    index=1 if secret("GEMINI_API_KEY") else 0,
    horizontal=True)
use_auto = mode.startswith("🤖")
api_key = ""
if use_auto:
    api_key = st.text_input("Gemini API key", type="password",
                            value=secret("GEMINI_API_KEY"))

with st.expander("⚙️ Settings", expanded=False):
    c1, c2 = st.columns(2)
    src_lang = c1.selectbox("မူရင်းဘာသာ", ["auto", "en", "zh", "ko", "ja", "hi", "th", "vi", "id"])
    wsize = c2.selectbox("Whisper model", ["tiny", "base", "small"], index=1,
                         help="small = တိကျ/နှေး၊ tiny = မြန်/အမှားများ")
    c3, c4 = st.columns(2)
    height = c3.select_slider("Output height", [360, 480, 720, 1080], value=720)
    max_speed = c4.slider("ရုပ် အမြန်ဆုံး ဆ", 1.0, 2.0, 1.25, 0.05)
    models_txt = st.text_input("Gemini models (Auto မှာသာ)", DEFAULT_MODELS)
    max_min = st.slider("Video အရှည် ကန့်သတ် (မိနစ်)", 1, 15, 6)

with st.expander("🔊 အသံ (Voice)", expanded=False):
    engine = st.radio("TTS engine",
                      ["Edge TTS", "VoxCPM (မရရင် Edge TTS auto)"], horizontal=False)
    voice = st.selectbox("Edge voice", ["my-MM-ThihaNeural", "my-MM-NilarNeural"],
                         format_func=lambda v: "Thiha (ယောက်ျား)" if "Thiha" in v else "Nilar (မိန်းမ)")
    rate_n = st.slider("Edge speed %", -50, 100, 0, 5)
    ref_up = None
    if engine.startswith("Vox"):
        ref_up = st.file_uploader("🎤 Reference audio (optional)", type=["wav", "mp3", "m4a"])
        st.caption("VoxCPM Space တွေက queue/sleep ဖြစ်တတ်တယ်။ မရရင် စာကြောင်းတစ်ခုချင်း Edge TTS နဲ့ အစားထိုးမယ်။")

with st.expander("🎨 စာတန်း (PNG) + Neon", expanded=False):
    use_sub = st.toggle("PNG စာတန်းထည့်", value=True)
    c5, c6 = st.columns(2)
    sub_size = c5.slider("စာအရွယ် (% of width)", 3.0, 12.0, 7.0, 0.5)
    sub_pos = c6.slider("နေရာ (0=အပေါ် 100=အောက်)", 0, 100, 90, 1)
    use_neon = st.toggle("Neon border", value=True)
    c7, c8 = st.columns(2)
    neon_th = c7.slider("Neon ထူ (px)", 4, 24, 10)
    neon_speed = c8.slider("Neon မြန်နှုန်း", 0.1, 1.0, 0.4, 0.05)
    c9, c10 = st.columns(2)
    flip = c9.toggle("Flip (ဘယ်ညာပြောင်း)", value=False)
    zoom = c10.slider("Zoom", 1.0, 1.25, 1.0, 0.01)
    c11, c12, c13 = st.columns([2, 2, 2])
    sub_color = c11.color_picker("စာအရောင်", "#FFFFFF")
    box_op = c12.slider("နောက်ခံ မည်း", 0.0, 1.0, 0.8, 0.05)
    sub_bold = c13.toggle("စာထူ", value=False)
    layout_sel = st.radio("မြန်မာစာ layout", ["Raqm (shaping)", "Basic"], horizontal=True,
                          help="စာလုံး ပုံစံမမှန်ရင် Basic ကို ပြောင်းကြည့်ပါ")
    font_path = find_font()
    if font_path:
        st.caption(f"Font: {os.path.basename(font_path)}")
    else:
        st.warning("Myanmar font မတွေ့ပါ။ MyanmarPadaung.ttf ကို repo root မှာ ထည့်ပါ။")
    if not features.check("raqm"):
        st.warning("raqm မရှိလို့ မြန်မာစာ ပုံစံ မမှန်နိုင်ပါ (packages.txt ထဲ libraqm0 ပါစေ)")

rate = f"{rate_n:+d}%"
sub_style = {
    "alpha": int(box_op * 255),
    "color": tuple(int(sub_color.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)),
    "bold": 1 if sub_bold else 0,
    "layout": "raqm" if layout_sel.startswith("Raqm") else "basic",
}
models = [m.strip() for m in models_txt.split(",") if m.strip()]
proxy = secret("EDGE_TTS_PROXY") or None
hf_token = secret("HF_TOKEN") or None

up = st.file_uploader("Video ရွေးပါ", type=["mp4", "mov", "mkv", "webm", "m4v"])

if up:
    try:
        up_path = save_upload(up)
        pv = make_preview(up_path, height, "စာတန်းထိုး နမူနာ စာသား ဖြစ်ပါတယ်။", font_path,
                          sub_size, sub_pos, use_neon, neon_th, neon_speed, flip, zoom, sub_style)
        if pv is not None:
            st.image(pv, caption="Preview", use_container_width=True)
    except Exception as ex:  # noqa: BLE001
        st.caption(f"preview မရ: {str(ex)[:100]}")

# ====================================================================
# Stage 1 : Whisper + Gemini
# ====================================================================
if st.button("1️⃣ စာသားထုတ်" + (" + မြန်မာပြန်" if use_auto else ""), type="primary",
             disabled=not (up and (api_key or not use_auto)), use_container_width=True):
    reset_job()
    wd = tempfile.mkdtemp(prefix="recap_")
    st.session_state["wd"] = wd
    src = os.path.join(wd, "src.mp4")
    shutil.copy(save_upload(up), src)

    try:
        vdur = dur(src)
        if vdur > max_min * 60:
            st.error(f"Video က {vdur/60:.1f} မိနစ်ရှိတယ်။ {max_min} မိနစ်အောက်ပဲ လက်ခံတယ် "
                     "(Settings မှာ ပြောင်းလို့ရ)")
            st.stop()
        st.session_state["vdur"] = vdur

        with st.status("🎧 Whisper စာသားပြောင်းနေတယ်...", expanded=True) as stt:
            wav = os.path.join(wd, "audio16k.wav")
            sh(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", wav])
            wm = load_whisper(wsize)
            with wave.open(wav) as wf:
                audio = np.frombuffer(
                    wf.readframes(wf.getnframes()), dtype=np.int16
                ).astype(np.float32) / 32768.0
            segs, info = wm.transcribe(
                audio, language=None if src_lang == "auto" else src_lang,
                vad_filter=True, beam_size=5)
            st.write(f"ဘာသာစကား: **{info.language}** ({info.language_probability:.0%})")
            bar = st.progress(0.0)
            raw = []
            for s in segs:
                if s.text.strip():
                    raw.append((s.start, s.end, s.text.strip()))
                bar.progress(min(s.end / vdur, 1.0))
            units = []
            for s, e, t in raw:
                if units and (units[-1][1] - units[-1][0]) < TARGET and (e - units[-1][0]) <= MAXD:
                    units[-1] = (units[-1][0], e, units[-1][2] + " " + t)
                else:
                    units.append((s, e, t))
            if not units:
                stt.update(label="စကားပြောသံ မတွေ့ပါ", state="error")
                st.stop()
            stt.update(label=f"🎧 အပိုင်း {len(units)} ခု ရပြီ", state="complete")

        my = {}
        if use_auto:
            gen = make_gen(api_key, models)
            with st.status("🌐 Gemini ဘာသာပြန်နေတယ်...", expanded=True) as gst:
                full = " ".join(u[2] for u in units)
                context = gen(
                    "Read this story transcript (any language). List main character names, "
                    "relationships, places and key terms in a short English bullet list.\n\n"
                    + full[:60000], json_out=False)
                sysmsg = (
                    "You are a Burmese (Myanmar) recap-video narrator. Input units may be in "
                    "Chinese, English or another language. Translate each unit into natural "
                    "spoken Burmese narration. Rules: do NOT summarize or shorten, do NOT skip, "
                    "do NOT move content between units, keep character names consistent, use "
                    "ONLY Burmese script (no other scripts).\nStory notes:\n" + context +
                    '\nInput is a JSON list of {"id":n,"text":"..."}. Return ONLY a JSON list '
                    'of {"id":n,"my":"..."} with the same ids.'
                )
                my = {}
                bar = st.progress(0.0)
                for b in range(0, len(units), BATCH):
                    items = [{"id": i, "text": units[i][2]}
                             for i in range(b, min(b + BATCH, len(units)))]
                    for _ in range(3):
                        try:
                            out = json.loads(gen(sysmsg + "\n\n" + json.dumps(items, ensure_ascii=False)))
                            for o in out:
                                my[int(o["id"])] = clean_my(o["my"])
                            if all(my.get(it["id"]) for it in items):
                                break
                        except Exception:  # noqa: BLE001
                            pass
                    bar.progress(min((b + BATCH) / len(units), 1.0))
                for i in range(len(units)):
                    if not my.get(i):
                        my[i] = clean_my(gen(
                            "Translate to natural spoken Burmese narration only, no notes:\n"
                            + units[i][2], json_out=False))
                gst.update(label="🌐 ဘာသာပြန်ပြီး", state="complete")

        st.session_state["units"] = [
            {"start": units[i][0], "end": units[i][1], "src": units[i][2], "my": my.get(i, "")}
            for i in range(len(units))
        ]
        st.session_state["manual"] = not use_auto
        st.session_state["ver"] = st.session_state.get("ver", 0) + 1
    except Exception as ex:  # noqa: BLE001
        st.error(f"Error: {ex}")

# ====================================================================
# Edit + Stage 2 : TTS → Smart Sync → Overlay
# ====================================================================
if st.session_state.get("units"):
    units = st.session_state["units"]

    if st.session_state.get("manual"):
        st.subheader("📋 Gemini မှာ ဘာသာပြန်ရန်")
        st.caption("① အောက်က စာသားကို Copy (ညာထောင့် icon) → ② Gemini app မှာ paste → "
                   "③ ရလဒ်ကို အောက်က ကွက်ထဲ paste → ④ Apply")
        st.code(build_prompt(units), language=None)
        pasted = st.text_area("Gemini ရလဒ် ဒီမှာ paste", height=200, key="gem_paste",
                              placeholder="[0] မြန်မာစာ...\n[1] မြန်မာစာ...")
        if st.button("✅ Apply (စာသား ဖြည့်)", use_container_width=True):
            got = parse_numbered(pasted, len(units))
            for i, t in got.items():
                units[i]["my"] = clean_my(t)
            missing = [i for i in range(len(units)) if not units[i]["my"]]
            st.session_state["apply_msg"] = (
                f"ဖြည့်ပြီး {len(got)}/{len(units)} ကြောင်း"
                + (f" · မပါသေးတာ: {missing[:15]}" if missing else " ✅"))
            st.session_state["ver"] = st.session_state.get("ver", 0) + 1
            st.rerun()
        if st.session_state.get("apply_msg"):
            st.info(st.session_state["apply_msg"])

    st.subheader("✏️ မြန်မာစာသား စစ်/ပြင်")
    df = pd.DataFrame({"#": range(len(units)),
                       "မူရင်း": [u["src"] for u in units],
                       "မြန်မာ": [u["my"] for u in units]})
    edited = st.data_editor(df, disabled=["#", "မူရင်း"], hide_index=True,
                            use_container_width=True, key=f"editor_{st.session_state.get('ver', 0)}")

    if st.button("2️⃣ အသံထုတ် + Video ပေါင်း", type="primary", use_container_width=True):
        wd, vdur = st.session_state["wd"], st.session_state["vdur"]
        src = os.path.join(wd, "src.mp4")
        my_list = [clean_my(str(t)) or units[i]["my"] for i, t in enumerate(edited["မြန်မာ"])]
        n = len(units)
        empty = [i for i, t in enumerate(my_list) if not t.strip()]
        if empty:
            st.error(f"မြန်မာစာသား မဖြည့်ရသေးတဲ့ စာကြောင်း: {empty[:20]}")
            st.stop()
        try:
            # ---------------- TTS
            vox = None
            if engine.startswith("Vox"):
                ref_path = None
                if ref_up:
                    ref_path = os.path.join(wd, "ref.wav")
                    with open(ref_path, "wb") as f:
                        f.write(ref_up.getbuffer())
                vox = VoxEngine(ref_path, hf_token)

            fallback = 0
            durs = []
            with st.status("🔊 မြန်မာအသံ ထုတ်နေတယ်...", expanded=True) as tst:
                bar = st.progress(0.0)
                for i in range(n):
                    raw_audio = f"{wd}/a{i:04d}_raw"
                    wav = f"{wd}/a{i:04d}.wav"
                    used = None
                    if vox is not None:
                        used = vox.synth(my_list[i], raw_audio + ".wav")
                        if used:
                            raw_audio += ".wav"
                    if not used:
                        if vox is not None:
                            fallback += 1
                        raw_audio += ".mp3"
                        for attempt in range(4):
                            try:
                                tts_edge(my_list[i], voice, rate, raw_audio, proxy)
                                break
                            except Exception as ex:  # noqa: BLE001
                                if attempt == 3:
                                    raise RuntimeError(
                                        f"Edge TTS မအောင်မြင်ပါ (စာကြောင်း #{i}): {ex}\n\n{my_list[i]}")
                                time.sleep(2)
                    sh(["ffmpeg", "-y", "-i", raw_audio, "-af", TRIM, "-ar", "44100", wav])
                    durs.append(dur(wav))
                    bar.progress((i + 1) / n)
                msg = "🔊 အသံ ပြီးပြီ"
                if vox is not None:
                    msg += f" (VoxCPM: {n - fallback}/{n})"
                    if fallback and vox.last_err:
                        st.caption(f"Vox error: {vox.last_err}")
                tst.update(label=msg, state="complete")

            # ---------------- Smart Sync
            with st.status("🎬 Smart Sync render...", expanded=True) as rst:
                bounds = [0.0] + [(units[i]["end"] + units[i + 1]["start"]) / 2
                                  for i in range(n - 1)] + [vdur]
                clips = []
                bar = st.progress(0.0)
                for i in range(n):
                    s, e = bounds[i], bounds[i + 1]
                    w = max(e - s, 0.2)
                    wav = f"{wd}/a{i:04d}.wav"
                    d = durs[i]
                    if w / d > max_speed:
                        w_use, factor = d * max_speed, 1 / max_speed
                    else:
                        w_use, factor = w, d / w
                    out = f"{wd}/c{i:04d}.mp4"
                    sh(["ffmpeg", "-y", "-ss", f"{s:.3f}", "-t", f"{w_use:.3f}", "-i", src,
                        "-i", wav,
                        "-vf", f"setpts=(PTS-STARTPTS)*{factor:.6f},fps=30,"
                               f"scale=-2:{height},setsar=1,tpad=stop_mode=clone:stop_duration=2",
                        "-map", "0:v:0", "-map", "1:a:0", "-t", f"{d:.3f}",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "22",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-ac", "2", out])
                    clips.append(out)
                    bar.progress((i + 1) / n)

                lst = f"{wd}/list.txt"
                with open(lst, "w") as f:
                    for c in clips:
                        f.write(f"file '{c}'\n")
                synced = f"{wd}/synced.mp4"
                sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", synced])
                rst.update(label=f"🎬 Smart Sync ပြီး ({dur(synced):.1f}s / မူရင်း {vdur:.1f}s)",
                           state="complete")

            # ---------------- PNG subtitle + Neon
            final = synced
            if use_sub or use_neon or flip or zoom > 1.001:
                subs, t0 = [], 0.0
                if use_sub:
                    for i in range(n):
                        parts = split_for_sub(my_list[i])
                        tot = sum(len(p) for p in parts) or 1
                        cur = t0
                        for p in parts:
                            dd = durs[i] * len(p) / tot
                            subs.append((cur, cur + dd, p))
                            cur += dd
                        t0 += durs[i]
                with st.status("🖼️ PNG စာတန်း + Neon ထည့်နေတယ်...", expanded=True) as ost:
                    bar = st.progress(0.0)
                    final = f"{wd}/final.mp4"
                    overlay_pass(synced, final, subs, font_path if use_sub else None,
                                 sub_size, sub_pos, use_neon, neon_th, neon_speed,
                                 flip, zoom, sub_style, progress=bar)
                    ost.update(label="🖼️ စာတန်း/Neon ပြီး", state="complete")

            with open(final, "rb") as f:
                st.session_state["result"] = f.read()
        except Exception as ex:  # noqa: BLE001
            st.error(f"Error: {ex}")

# ====================================================================
# Result
# ====================================================================
if st.session_state.get("result"):
    st.subheader("✅ ရလဒ်")
    st.video(st.session_state["result"])
    st.download_button("⬇️ Download MP4", st.session_state["result"],
                       file_name="final_myanmar.mp4", mime="video/mp4",
                       use_container_width=True)
    if st.button("🗑️ အသစ်ပြန်စ", use_container_width=True):
        reset_job()
        st.rerun()
