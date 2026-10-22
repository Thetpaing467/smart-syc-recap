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
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

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


def clean_text(text, lang="မြန်မာ"):
    """ဘာသာစကားအလိုက် မလိုတဲ့ စာလုံးဖျက်"""
    if lang == "ဂျာမနီ":
        return re.sub(r"[^A-Za-zÄÖÜäöüß0-9\s.,!?\"'%\-]", "", text).strip()
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
    """VoxCPM Space များကို အစဉ်လိုက် စမ်းမယ်။ မရရင် None ပြန် (Edge TTS ကို မခေါ်တော့ - caller က retry/error ပြမယ်)။"""

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
    BASE = np.array([12, 8, 10], np.float32)
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
                put(last.copy())
        finally:
            put(None)

    log_path = out + ".log"
    log = open(log_path, "wb")
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
           "-r", str(fps_out), "-i", "-", "-i", audio, "-map", "0:v", "-map", "1:a:0",
           "-c:v", "libx264", "-preset", preset, "-crf", "27", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart",
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
PROMPT_LANGS = {
    "မြန်မာ": (
        "You are a Burmese (Myanmar) recap-video narrator. Translate every numbered line "
        "below into natural spoken Burmese narration (storyteller style). Rules: do NOT "
        "summarize, shorten or skip; do NOT move content between lines; keep character "
        "names consistent across all lines; use ONLY Burmese script (Myanmar Unicode), "
        "no English/Chinese letters. Output exactly one line per input line in the format "
        "[number] Burmese text -- same numbers, no extra comments.\n\n"
    ),
    "ဂျာမနီ": (
        "You are a German (Deutsch) recap-video narrator. Translate every numbered line "
        "below into natural spoken German narration (storyteller style). Rules: do NOT "
        "summarize, shorten or skip; do NOT move content between lines; keep character "
        "names consistent across all lines; use ONLY German (Latin script), no Chinese "
        "letters. Output exactly one line per input line in the format "
        "[number] German text -- same numbers, no extra comments.\n\n"
    ),
}


def build_prompt(units, lang="မြန်မာ"):
    lines = "\n".join(f"[{i}] {u['src']}" for i, u in enumerate(units))
    return PROMPT_LANGS.get(lang, PROMPT_LANGS["မြန်မာ"]) + lines


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


def cleanup_uploads():
    for p in st.session_state.get("up_paths", []):
        try:
            os.remove(p)
        except Exception:  # noqa: BLE001
            pass
    st.session_state["up_paths"] = []
    st.session_state.pop("link_path", None)
    st.session_state.pop("link_src", None)
    st.session_state["up_key"] = st.session_state.get("up_key", 0) + 1


def cleanup_after_download():
    wd = st.session_state.get("wd")
    if wd and os.path.isdir(wd):
        shutil.rmtree(wd, ignore_errors=True)
    for k in ("wd", "units", "vdur", "manual", "apply_msg", "gem_paste"):
        st.session_state.pop(k, None)
    cleanup_uploads()
    st.session_state["cleaned"] = True


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
        "format": f"bv*[height<={height}]+ba/b[height<={height}]/bv*+ba/b",
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
    except Exception:  # noqa: BLE001
        codec = ""
    if not codec:
        raise RuntimeError("Video stream မတွေ့ပါ (ဖိုင်က video မဟုတ်နိုင်)")
    if codec == "h264":
        try:
            sh(["ffmpeg", "-y", "-i", raw, "-c", "copy", "-movflags", "+faststart", final])
            return final
        except Exception:  # noqa: BLE001
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
                except Exception as ex:  # noqa: BLE001
                    errs.append(f"{m}: {str(ex)[:200]}")
                    raw = None
            if not raw:
                raise RuntimeError("Link ကနေ မဆွဲနိုင်ပါ\n" + "\n".join(errs))
        if bar is not None:
            bar.progress(1.0)
        _ensure_mp4(raw, final, height)
        return final
    finally:
        shutil.rmtree(work, ignore_errors=True)


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
    max_min = st.slider("Video အရှည် ကန့်သတ် (မိနစ်)", 1, 120, 15)
    preset = st.selectbox("Render speed", ["ultrafast", "superfast", "veryfast"], index=0,
                          help="ultrafast = အမြန်ဆုံး (file အနည်းငယ်ကြီး)၊ veryfast = file သေး/နှေးနည်းနည်း")

with st.expander("🔊 အသံ (Voice)", expanded=False):
    engine = st.radio("TTS engine",
                      ["Edge TTS", "VoxCPM (Edge TTS / Thiha မသုံးဘူး)"], horizontal=False)
    is_vox = engine.startswith("Vox")

    voice, rate_n = "my-MM-ThihaNeural", 0
    if not is_vox:
        voice = st.selectbox("Edge voice", ["my-MM-ThihaNeural", "my-MM-NilarNeural"],
                             format_func=lambda v: "Thiha (ယောက်ျား)" if "Thiha" in v else "Nilar (မိန်းမ)")
        # ---- Speed default 45 (သင့်တောင်းဆိုချက်)
        rate_n = st.slider("Edge speed %", -50, 100, 45, 5)
        tts_workers = st.slider("TTS တစ်ပြိုင်နက် အရေအတွက်", 1, 10, 6,
                                help="များရင်မြန်တယ်။ Edge က error ပေးရင် လျှော့ပါ")
    else:
        tts_workers = 1

    ref_up = None
    vox_retry = 4
    if is_vox:
        ref_up = st.file_uploader("🎤 Reference audio (အသံတစ်သံတည်း ဖြစ်ချင်ရင် တင်ပါ)",
                                  type=["wav", "mp3", "m4a"])
        vox_retry = st.slider("VoxCPM ပြန်စမ်းအကြိမ် (စာကြောင်းတစ်ခုချင်း)", 1, 8, 4,
                              help="Space sleep/queue ဖြစ်ရင် ပြန်စမ်းမယ်။ ကုန်ရင် Edge မသုံးဘဲ error ပြမယ်")
        st.caption("VoxCPM ရွေးထားရင် Edge TTS (Thiha/Nilar) လုံးဝ မဝင်ပါ။ "
                   "VoxCPM မရရင် error ပြပြီး ရပ်မယ်၊ ခဏနေ ပြန်စမ်းပါ။ "
                   "Reference audio တင်ရင် စာကြောင်းတိုင်း အသံတူမယ်။")

with st.expander("🎨 စာတန်း (PNG) + Neon", expanded=False):
    use_sub = st.toggle("PNG စာတန်းထည့်", value=True)
    c5, c6 = st.columns(2)
    sub_size = c5.slider("စာအရွယ် (% of width)", 3.0, 12.0, 7.0, 0.5)
    sub_pos = c6.slider("နေရာ (0=အပေါ် 100=အောက်)", 0, 100, 90, 1)
    use_neon = st.toggle("Neon border", value=True)
    zoom = st.slider("Zoom", 1.0, 1.25, 1.0, 0.01)
    font_path = find_font()
    if not font_path:
        st.warning("Myanmar font မတွေ့ပါ။ MyanmarPadaung.ttf ကို repo root မှာ ထည့်ပါ။")
    if not features.check("raqm"):
        st.warning("raqm မရှိလို့ မြန်မာစာ ပုံစံ မမှန်နိုင်ပါ (packages.txt ထဲ libraqm0 ပါစေ)")

rate = f"{rate_n:+d}%"

neon_th = 10
neon_speed = 0.4
flip = True
sub_style = {
    "alpha": int(0.8 * 255),
    "color": (255, 255, 255),
    "bold": 0,
    "layout": "raqm",
}
proxy = secret("EDGE_TTS_PROXY") or None
hf_token = secret("HF_TOKEN") or None

mode = st.radio("Video ရင်းမြစ်", ["🔗 Link", "📁 Upload"], horizontal=True)
vid_path = None
if mode.startswith("🔗"):
    url_in = st.text_input("Video link (direct mp4 / Google Drive / Dropbox / YouTube / Facebook / TikTok ...)",
                           placeholder="https://...", key="link_url")
    if st.button("⬇️ Link ကနေ ဆွဲမယ်", use_container_width=True, disabled=not url_in.strip()):
        with st.status("⬇️ Video ဆွဲနေတယ်...", expanded=True) as ds:
            dbar = st.progress(0.0)
            t_dl = time.time()
            try:
                p = fetch_link(url_in, height, dbar, secret("YT_COOKIES") or None)
                paths = st.session_state.setdefault("up_paths", [])
                if p not in paths:
                    paths.append(p)
                st.session_state["link_path"] = p
                st.session_state["link_src"] = url_in.strip()
                ds.update(label=f"✅ ဆွဲပြီး · {os.path.getsize(p)/1e6:.1f} MB · {fmt_t(time.time() - t_dl)}",
                          state="complete")
            except Exception as ex:  # noqa: BLE001
                st.session_state.pop("link_path", None)
                st.session_state.pop("link_src", None)
                ds.update(label="❌ မဆွဲနိုင်ပါ", state="error")
                st.error(str(ex)[:900])
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

if vid_path:
    try:
        pv = make_preview(vid_path, height, "စာတန်းထိုး နမူနာ စာသား ဖြစ်ပါတယ်။", font_path,
                          sub_size, sub_pos, use_neon, neon_th, neon_speed, flip, zoom, sub_style)
        if pv is not None:
            st.image(pv, caption="Preview", use_container_width=True)
    except Exception as ex:  # noqa: BLE001
        st.caption(f"preview မရ: {str(ex)[:100]}")

# ====================================================================
# Stage 1 : Whisper + Gemini
# ====================================================================
if st.button("1️⃣ စာသားထုတ်", type="primary", disabled=not vid_path, use_container_width=True):
    reset_job()
    st.session_state.pop("cleaned", None)
    wd = tempfile.mkdtemp(prefix="recap_")
    st.session_state["wd"] = wd
    src = os.path.join(wd, "src.mp4")
    shutil.copy(vid_path, src)

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
        # ---- ဘာသာစကား ရွေးချယ်မှု (မြန်မာ / ဂျာမနီ) ----
        prompt_lang = st.radio("ဘာသာစကား", ["မြန်မာ", "ဂျာမနီ"], horizontal=True,
                               key="prompt_lang",
                               help="Gemini ကို ဘယ်ဘာသာနဲ့ ဘာသာပြန်ခိုင်းမလဲ")
        st.caption("① အောက်က စာသားကို Copy (ညာထောင့် icon) → ② Gemini app မှာ paste → "
                   "③ ရလဒ်ကို အောက်က ကွက်ထဲ paste → ④ Apply")
        st.code(build_prompt(units, prompt_lang), language=None)
        pasted = st.text_area("Gemini ရလဒ် ဒီမှာ paste", height=200, key="gem_paste",
                              placeholder="[0] ဘာသာပြန်စာသား...\n[1] ဘာသာပြန်စာသား...")
        if st.button("✅ Apply (စာသား ဖြည့်)", use_container_width=True):
            got = parse_numbered(pasted, len(units))
            for i, t in got.items():
                units[i]["my"] = clean_text(t, prompt_lang)
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
        my_list = [clean_text(u["my"], st.session_state.get("prompt_lang", "မြန်မာ")) for u in units]
        n = len(units)
        empty = [i for i, t in enumerate(my_list) if not t.strip()]
        if empty:
            st.error(f"စာသား မဖြည့်ရသေးတဲ့ စာကြောင်း: {empty[:20]}")
            st.stop()
        try:
            T = {}
            t_all = time.time()

            vox = None
            if is_vox:
                ref_path = None
                if ref_up:
                    ref_path = os.path.join(wd, "ref.wav")
                    with open(ref_path, "wb") as f:
                        f.write(ref_up.getbuffer())
                vox = VoxEngine(ref_path, hf_token)

            sig_base = f"{engine}|{voice}|{rate}|" + \
                (f"{ref_up.name}-{ref_up.size}" if ref_up else "noref")

            def do_line(i):
                wav = f"{wd}/a{i:04d}.wav"
                keyf = f"{wd}/a{i:04d}.key"
                key = hashlib.md5(f"{sig_base}|{my_list[i]}".encode()).hexdigest()
                if os.path.exists(wav) and os.path.exists(keyf):
                    try:
                        with open(keyf) as kf:
                            if kf.read().strip() == key:
                                return i, wav_dur(wav), vox is not None
                    except Exception:  # noqa: BLE001
                        pass
                if vox is not None:
                    raw_audio = f"{wd}/a{i:04d}_raw.wav"
                    used = None
                    for attempt in range(vox_retry):
                        used = vox.synth(my_list[i], raw_audio)
                        if used:
                            break
                        vox.reset()
                        time.sleep(3)
                    if not used:
                        raise RuntimeError(
                            f"VoxCPM မအောင်မြင်ပါ (စာကြောင်း #{i}): {vox.last_err}\n\n"
                            "Edge TTS (Thiha) ကို မသုံးထားပါ။ Space sleep/queue ဖြစ်နေနိုင်လို့ "
                            "ခဏနေပြီး ပြန်စမ်းပါ။")
                else:
                    raw_audio = f"{wd}/a{i:04d}_raw.mp3"
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
                with open(keyf, "w") as kf:
                    kf.write(key)
                return i, wav_dur(wav), vox is not None

            durs = [0.0] * n
            vox_ok = 0
            workers = 1 if vox is not None else tts_workers
            t_a = time.time()
            with st.status("🔊 အသံ ထုတ်နေတယ်...", expanded=True) as tst:
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
                        tst.update(label="🔊 အသံထုတ် မအောင်မြင်ပါ", state="error")
                        raise
                T["🔊 အသံထုတ် (TTS)"] = time.time() - t_a
                msg = f"🔊 အသံ ပြီးပြီ · {fmt_t(T['🔊 အသံထုတ် (TTS)'])}"
                if vox is not None:
                    msg += f" (VoxCPM: {vox_ok}/{n})"
                tst.update(label=msg, state="complete")

            t_a = time.time()
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
    mb = len(st.session_state["result"]) / 1e6
    st.caption(f"📦 File အရွယ်အစား: {mb:.1f} MB")
    if st.toggle("▶️ ဒီမှာ video ကြည့်မယ် (ဖိုင်ကြီးရင်/net နှေးရင် ကြာတတ်)", value=False, key="show_vid"):
        st.video(st.session_state["result"])
    _tt = {}
    _tt.update(st.session_state.get("t1") or {})
    _tt.update(st.session_state.get("t2") or {})
    if _tt:
        st.markdown("**⏱️ ဘယ်အပိုင်း ဘယ်လောက်ကြာလဲ**\n\n" + timing_table(_tt))
    st.download_button("⬇️ Download MP4", st.session_state["result"],
                       file_name="final_myanmar.mp4", mime="video/mp4",
                       use_container_width=True, on_click=cleanup_after_download)
    if st.session_state.get("cleaned"):
        st.caption("🧹 Script နဲ့ တင်ထားတဲ့ video ကို ဖျက်ပြီးပါပြီ")
    if st.button("🗑️ အသစ်ပြန်စ", use_container_width=True):
        reset_job()
        cleanup_uploads()
        st.session_state.pop("cleaned", None)
        st.rerun()