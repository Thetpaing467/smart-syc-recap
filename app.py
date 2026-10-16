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
import queue
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFont, features

st.set_page_config(page_title="Recap → Myanmar", page_icon="🎙️", layout="centered")

TARGET, MAXD = 4.2, 5.6

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


def fmt_t(sec):
    sec = float(sec)
    return f"{sec:.1f} စက္ကန့်" if sec < 60 else f"{int(sec // 60)} မိနစ် {sec % 60:.0f} စက္ကန့်"


def wav_dur(path):
    with wave.open(path) as w:
        return w.getnframes() / float(w.getframerate())


def timing_table(d):
    rows = "\n".join(f"| {k} | **{fmt_t(v)}** |" for k, v in d.items())
    return "| အဆင့် | ကြာချိန် |\n|---|---|\n" + rows


def clean_my(text):
    """မြန်မာစာ၊ ဂဏန်း၊ ပုဒ်ဖြတ်သာ ကျန်ခဲ့"""
    return re.sub(
        r"[^\u1000-\u109F\uAA60-\uAA7F\uA9E0-\uA9FF\s0-9.,!?\"'%\-]", "", text
    ).strip()


@st.cache_resource(show_spinner="Whisper model ကို ဖွင့်နေတယ် (ပထမအကြိမ် download လုပ်မယ်)...")
def load_whisper(size):
    from faster_whisper import WhisperModel
    return WhisperModel(size, device="cpu", compute_type="int8",
                        cpu_threads=os.cpu_count() or 4)


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


def render_final(src, audio, out, plan, total, height, subs, font_path, fs_pct, pos_y,
                 neon_on, neon_th, neon_speed, flip, zoom, style=None, preset="ultrafast",
                 progress=None, fps_out=30):
    """Single pass: source frame တစ်ခါဖတ် → smart-sync retime + zoom/flip + neon + စာတန်း → encode တစ်ခါ။
    decode thread / python processing / ffmpeg encode သုံးခု တစ်ပြိုင်နက် အလုပ်လုပ်။"""
    style = style or {}
    cap = cv2.VideoCapture(src)
    sfps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w0, h0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    stotal = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 10 ** 9
    H = int(height)
    W = int(round(w0 * H / h0 / 2) * 2)
    nfr = max(1, int(round(total * fps_out)))
    fs = max(16, int(W * fs_pct / 100))
    neon = Neon(W, H, neon_th, neon_speed) if neon_on else None

    q = queue.Queue(maxsize=24)
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
                if idx > cur + 90:                      # ကြီးကြီးကျော်ရင် seek
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
                put(last.copy())
        finally:
            put(None)

    log_path = out + ".log"
    log = open(log_path, "wb")
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(fps_out), "-i", "-", "-i", audio, "-map", "0:v", "-map", "1:a:0",
           "-c:v", "libx264", "-preset", preset, "-crf", "23", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart",
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
                        cur_ov = build_overlay(subs[k][2], W, H, font_path, fs, pos_y, **style)
                        cur_k = k
                    if cur_ov:
                        blend_overlay(fr, cur_ov)
            proc.stdin.write(fr.tobytes())
            j += 1
            if progress and j % 30 == 0:
                progress.progress(min(j / nfr, 1.0))
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
        except Exception:  # noqa: BLE001
            pass
        proc.wait()
        log.close()
    if proc.returncode != 0:
        with open(log_path, "rb") as f:
            tail = f.read()[-800:].decode("utf-8", "ignore")
        raise RuntimeError("render: ffmpeg fail\n" + tail)
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
    for k in ("wd", "units", "vdur", "result", "manual", "apply_msg", "gem_paste", "t1", "t2"):
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

with st.expander("⚙️ Settings", expanded=False):
    c1, c2 = st.columns(2)
    src_lang = c1.selectbox("မူရင်းဘာသာ", ["auto", "en", "zh", "ko", "ja", "hi", "th", "vi", "id"])
    wsize = c2.selectbox("Whisper model", ["tiny", "base", "small"], index=1,
                         help="small = တိကျ/နှေး၊ tiny = မြန်/အမှားများ")
    c3, c4 = st.columns(2)
    height = c3.select_slider("Output height", [360, 480, 720, 1080], value=720)
    max_speed = c4.slider("ရုပ် အမြန်ဆုံး ဆ", 1.0, 2.0, 1.25, 0.05)
    max_min = st.slider("Video အရှည် ကန့်သတ် (မိနစ်)", 1, 15, 6)
    preset = st.selectbox("Render speed", ["ultrafast", "superfast", "veryfast"], index=0,
                          help="ultrafast = အမြန်ဆုံး (file အနည်းငယ်ကြီး)၊ veryfast = file သေး/နှေးနည်းနည်း")

with st.expander("🔊 အသံ (Voice)", expanded=False):
    engine = st.radio("TTS engine",
                      ["Edge TTS", "VoxCPM (မရရင် Edge TTS auto)"], horizontal=False)
    voice = st.selectbox("Edge voice", ["my-MM-ThihaNeural", "my-MM-NilarNeural"],
                         format_func=lambda v: "Thiha (ယောက်ျား)" if "Thiha" in v else "Nilar (မိန်းမ)")
    rate_n = st.slider("Edge speed %", -50, 100, 0, 5)
    tts_workers = st.slider("TTS တစ်ပြိုင်နက် အရေအတွက်", 1, 10, 6,
                            help="များရင်မြန်တယ်။ Edge က error ပေးရင် လျှော့ပါ (VoxCPM မှာ 1 ပဲသုံး)")
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
if st.button("1️⃣ စာသားထုတ်", type="primary", disabled=not up, use_container_width=True):
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

        T1 = {}
        with st.status("🎧 Whisper စာသားပြောင်းနေတယ်...", expanded=True) as stt:
            t_a = time.time()
            wav = os.path.join(wd, "audio16k.wav")
            sh(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", wav])
            with wave.open(wav) as wf:
                audio = np.frombuffer(
                    wf.readframes(wf.getnframes()), dtype=np.int16
                ).astype(np.float32) / 32768.0
            T1["🎞️ Audio ထုတ်"] = time.time() - t_a
            t_a = time.time()
            wm = load_whisper(wsize)
            T1["📦 Whisper model ဖွင့်"] = time.time() - t_a
            t_a = time.time()
            segs, info = wm.transcribe(
                audio, language=None if src_lang == "auto" else src_lang,
                vad_filter=True, beam_size=1, condition_on_previous_text=False)
            st.write(f"ဘာသာစကား: **{info.language}** ({info.language_probability:.0%})")
            bar = st.progress(0.0)
            raw = []
            for s in segs:
                if s.text.strip():
                    raw.append((s.start, s.end, s.text.strip()))
                bar.progress(min(s.end / vdur, 1.0))
            T1["🎧 Whisper စာသားပြောင်း"] = time.time() - t_a
            units = []
            for s, e, t in raw:
                if units and (units[-1][1] - units[-1][0]) < TARGET and (e - units[-1][0]) <= MAXD:
                    units[-1] = (units[-1][0], e, units[-1][2] + " " + t)
                else:
                    units.append((s, e, t))
            if not units:
                stt.update(label="စကားပြောသံ မတွေ့ပါ", state="error")
                st.stop()
            stt.update(label=f"🎧 အပိုင်း {len(units)} ခု ရပြီ · {fmt_t(sum(T1.values()))}", state="complete")
            st.session_state["t1"] = T1

        st.session_state["units"] = [
            {"start": units[i][0], "end": units[i][1], "src": units[i][2], "my": ""}
            for i in range(len(units))
        ]
        st.session_state["manual"] = True
        st.session_state["ver"] = st.session_state.get("ver", 0) + 1
    except Exception as ex:  # noqa: BLE001
        st.error(f"Error: {ex}")

# ====================================================================
# Edit + Stage 2 : TTS → Smart Sync → Overlay
# ====================================================================
if st.session_state.get("units"):
    units = st.session_state["units"]
    if st.session_state.get("t1"):
        st.caption("⏱️ " + " · ".join(f"{k} {fmt_t(v)}" for k, v in st.session_state["t1"].items()))

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

    if st.button("2️⃣ အသံထုတ် + Video ပေါင်း", type="primary", use_container_width=True):
        wd, vdur = st.session_state["wd"], st.session_state["vdur"]
        src = os.path.join(wd, "src.mp4")
        my_list = [clean_my(u["my"]) for u in units]
        n = len(units)
        empty = [i for i, t in enumerate(my_list) if not t.strip()]
        if empty:
            st.error(f"မြန်မာစာသား မဖြည့်ရသေးတဲ့ စာကြောင်း: {empty[:20]}")
            st.stop()
        try:
            T = {}
            t_all = time.time()

            # ---------------- TTS (တစ်ပြိုင်နက်)
            vox = None
            if engine.startswith("Vox"):
                ref_path = None
                if ref_up:
                    ref_path = os.path.join(wd, "ref.wav")
                    with open(ref_path, "wb") as f:
                        f.write(ref_up.getbuffer())
                vox = VoxEngine(ref_path, hf_token)

            def do_line(i):
                raw_audio = f"{wd}/a{i:04d}_raw"
                wav = f"{wd}/a{i:04d}.wav"
                used = None
                if vox is not None:
                    used = vox.synth(my_list[i], raw_audio + ".wav")
                    if used:
                        raw_audio += ".wav"
                if not used:
                    raw_audio += ".mp3"
                    for attempt in range(4):
                        try:
                            tts_edge(my_list[i], voice, rate, raw_audio, proxy)
                            break
                        except Exception as ex:  # noqa: BLE001
                            if attempt == 3:
                                raise RuntimeError(
                                    f"Edge TTS မအောင်မြင်ပါ (စာကြောင်း #{i}): {ex}\n\n{my_list[i]}")
                            time.sleep(1.5)
                sh(["ffmpeg", "-y", "-i", raw_audio, "-af", TRIM, "-ar", "44100", "-ac", "1", wav])
                return i, wav_dur(wav), bool(used)

            durs = [0.0] * n
            vox_ok = 0
            workers = 1 if vox is not None else tts_workers
            t_a = time.time()
            with st.status("🔊 မြန်မာအသံ ထုတ်နေတယ်...", expanded=True) as tst:
                bar = st.progress(0.0)
                done = 0
                with ThreadPoolExecutor(max_workers=workers) as pool:
                    futs = [pool.submit(do_line, i) for i in range(n)]
                    try:
                        for fu in as_completed(futs):
                            i_, d_, u_ = fu.result()
                            durs[i_] = d_
                            vox_ok += 1 if u_ else 0
                            done += 1
                            bar.progress(done / n)
                    except Exception:
                        for f_ in futs:
                            f_.cancel()
                        raise
                T["🔊 အသံထုတ် (TTS)"] = time.time() - t_a
                msg = f"🔊 အသံ ပြီးပြီ · {fmt_t(T['🔊 အသံထုတ် (TTS)'])}"
                if vox is not None:
                    msg += f" (VoxCPM: {vox_ok}/{n})"
                    if vox_ok < n and vox.last_err:
                        st.caption(f"Vox error: {vox.last_err}")
                tst.update(label=msg, state="complete")

            # ---------------- အသံ ပေါင်း
            t_a = time.time()
            lst = f"{wd}/alist.txt"
            with open(lst, "w") as f:
                for i in range(n):
                    f.write(f"file '{wd}/a{i:04d}.wav'\n")
            allwav = f"{wd}/all.wav"
            sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", allwav])
            T["🔗 အသံ ပေါင်း"] = time.time() - t_a

            # ---------------- Smart Sync plan + Render (single pass)
            bounds = [0.0] + [(units[i]["end"] + units[i + 1]["start"]) / 2
                              for i in range(n - 1)] + [vdur]
            plan, cum = [], 0.0
            for i in range(n):
                s_, e_ = bounds[i], bounds[i + 1]
                w = max(e_ - s_, 0.2)
                d = max(durs[i], 0.05)
                w_use = d * max_speed if w / d > max_speed else w
                plan.append((cum, d, s_, w_use / d))
                cum += d

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

            t_a = time.time()
            with st.status("🎬 Video render (Smart Sync + စာတန်း + Neon)...", expanded=True) as rst:
                bar = st.progress(0.0)
                final = f"{wd}/final.mp4"
                render_final(src, allwav, final, plan, cum, height, subs,
                             font_path if use_sub else None, sub_size, sub_pos,
                             use_neon, neon_th, neon_speed, flip, zoom, sub_style,
                             preset=preset, progress=bar)
                T["🎬 Video render"] = time.time() - t_a
                rst.update(label=f"🎬 Render ပြီး ({cum:.1f}s / မူရင်း {vdur:.1f}s) · "
                                 f"{fmt_t(T['🎬 Video render'])}", state="complete")
            T["⏱️ စုစုပေါင်း (အဆင့် 2)"] = time.time() - t_all
            st.session_state["t2"] = T

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
    _tt = {}
    _tt.update(st.session_state.get("t1") or {})
    _tt.update(st.session_state.get("t2") or {})
    if _tt:
        st.markdown("**⏱️ ဘယ်အပိုင်း ဘယ်လောက်ကြာလဲ**\n\n" + timing_table(_tt))
    st.download_button("⬇️ Download MP4", st.session_state["result"],
                       file_name="final_myanmar.mp4", mime="video/mp4",
                       use_container_width=True)
    if st.button("🗑️ အသစ်ပြန်စ", use_container_width=True):
        reset_job()
        st.rerun()
