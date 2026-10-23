import streamlit as st
import streamlit.components.v1 as components
import os, re, hashlib, ffmpeg, shutil, subprocess, asyncio, time
import concurrent.futures
import numpy as np
import edge_tts
from PIL import Image, ImageDraw, ImageFont
import cv2
from gradio_client import Client, handle_file

os.environ["HF_HOME"] = "/tmp/hf_cache"

SPACES = [
    {"space": "openbmb/VoxCPM-Demo", "type": "demo"},
    {"space": "hgghfhjfhjguyjf/Voxcpm-Burmese-Tts", "type": "burmese"},
]

PASSWORD = "voxcpm2026"
FONT_FILE = "MyanmarPadaung.ttf"

FS, BH, BA = 30, 100, 200
ENC_PRESET = "fast"
ENC_CRF = 18
FINAL_PRESET = "ultrafast"
FINAL_CRF = 20
USE_FAST_VAD = True   # Whisper မသုံးဘဲ VAD နဲ့ speech ရှာ (အမြန်ဆုံး)
AUDIO_BITRATE = "128k"
TTS_CHUNK = 600
EDGE_CHUNK = 400   # chunk ကြီးလေ request နည်းလေ (rate limit လျော့)
TTS_WORKERS = 2
PNG_WORKERS = 4

CANVAS_URL = "https://gemini.google.com/share/a96d9ba3e76e"   # Gemini Canvas
WHISPER_MODEL = "tiny"
WHISPER_LANG = "my"

BOX_WIDTH_RATIO = 1.0
PADDING_Y = 15
CORNER_RADIUS = 20

# TikTok Logo Colors
TIKTOK_CYAN = "#25F4EE"
TIKTOK_MAGENTA = "#FE2C55"
TIKTOK_BLACK = "#000000"

EDGE_VOICES = {
    "female": "my-MM-NilarNeural",
    "male":   "my-MM-ThihaNeural",
}
EDGE_VOICE_FIXED = "male"

st.set_page_config(page_title="Myanmar TTS Recap", page_icon="🎬", layout="centered")

st.markdown("""
<style>
.stApp{background:linear-gradient(160deg,#0f0f23,#1a1a35,#0f0f23);color:#e8e8f0}
#MainMenu,footer,header{visibility:hidden}
.main-title{text-align:center;font-size:2.2rem;font-weight:900;
 background:linear-gradient(90deg,#ff6b9d,#c66bff,#6ba8ff);
 -webkit-background-clip:text;-webkit-text-fill-color:transparent;
 background-clip:text;margin-bottom:4px}
.main-sub{text-align:center;color:#8888aa;font-size:.9rem;margin-bottom:20px}
.stButton>button{background:linear-gradient(135deg,#667eea,#764ba2)!important;
 color:#fff!important;border:none!important;border-radius:10px!important;
 padding:12px 20px!important;font-weight:600!important;
 box-shadow:0 4px 15px rgba(102,126,234,.3)!important}
.stTextArea textarea{background:rgba(255,255,255,.04)!important;
 border:1px solid rgba(255,255,255,.1)!important;color:#fff!important;
 border-radius:10px!important}
.stFileUploader{background:rgba(255,255,255,.02);border-radius:10px;padding:8px}
.stAlert{border-radius:10px!important;border:none!important}
hr{border-color:rgba(255,255,255,.08);margin:24px 0}
.timer-box{background:linear-gradient(135deg,#667eea,#764ba2);
 border-radius:16px;padding:24px;text-align:center;margin:15px 0;
 box-shadow:0 8px 30px rgba(102,126,234,.4)}
.timer-title{color:#fff;font-size:.9rem;font-weight:600;letter-spacing:1px;margin-bottom:8px}
.timer-value{color:#fff;font-size:3.2rem;font-weight:900;line-height:1}
.timer-unit{font-size:1.5rem;font-weight:700;margin-left:8px}
.step-timer{background:rgba(255,255,255,.05);border-left:4px solid #667eea;
 border-radius:10px;padding:12px 18px;margin:8px 0;color:#e8e8f0;font-size:.95rem}
.step-timer b{color:#6ba8ff;font-size:1.05rem}
</style>
""", unsafe_allow_html=True)

if "auth" not in st.session_state:
    st.session_state.auth = False

if not st.session_state.auth:
    st.markdown("<div class='main-title'>🔐 Private App</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        pwd = st.text_input("Password", type="password", label_visibility="collapsed", placeholder="Password")
        if st.button("Login", use_container_width=True):
            if pwd == PASSWORD:
                st.session_state.auth = True
                st.rerun()
            else:
                st.error("Password မှား")
    st.stop()


# ==================== Utility Functions ====================

def vid_info(p):
    pr = ffmpeg.probe(p)
    v = next(s for s in pr['streams'] if s['codec_type'] == 'video')
    return int(v['width']), int(v['height']), float(pr['format']['duration'])


def t2s(s):
    ms = int(round((s - int(s)) * 1000)); tot = int(s)
    if ms >= 1000: tot += 1; ms = 0
    h, r = divmod(tot, 3600); m, sec = divmod(r, 60)
    return f"{h:02d}:{m:02d}:{sec:02d},{ms:03d}"


def s2t(ts):
    ts = ts.strip()
    m = re.match(r'^(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})$', ts)
    if m:
        h, mi, se, ms = m.groups()
        return int(h)*3600 + int(mi)*60 + int(se) + int(ms.ljust(3,'0'))/1000
    return None


def render_png(text, out, fp, W, H, fs=30, pos_y=100, bh=100, ba=100,
                box_width_ratio=BOX_WIDTH_RATIO,
                padding_y=PADDING_Y, corner_radius=CORNER_RADIUS):
    img = Image.new("RGBA", (W, H), (0,0,0,0))
    d = ImageDraw.Draw(img)
    try: f = ImageFont.truetype(fp, fs)
    except: f = ImageFont.load_default()

    mc = max(15, int(W/(fs*0.9)))
    lines, cur = [], ""
    for w in text.split():
        if len(cur)+len(w)+1 <= mc: cur = cur+" "+w if cur else w
        else:
            if cur: lines.append(cur)
            cur = w
    if cur: lines.append(cur)
    if not lines: return out

    lh = int(fs * 1.3)
    text_h = len(lines) * lh
    box_w = int(W * box_width_ratio)
    box_h = text_h + padding_y * 2
    box_x = (W - box_w) // 2
    max_y = H - box_h
    box_y = int((pos_y / 100) * max_y)
    box_y = max(0, min(box_y, max_y))

    d.rounded_rectangle(
        [box_x, box_y, box_x + box_w, box_y + box_h],
        radius=corner_radius, fill=(0, 0, 0, ba)
    )

    ty = box_y + padding_y
    for ln in lines:
        bb = d.textbbox((0, 0), ln, font=f)
        lw = bb[2] - bb[0]
        lx = box_x + (box_w - lw) // 2
        for dx in [-2,-1,0,1,2]:
            for dy in [-2,-1,0,1,2]:
                d.text((lx+dx, ty+dy), ln, font=f, fill=(0,0,0,255))
        d.text((lx, ty), ln, font=f, fill=(255,255,255,255))
        ty += lh

    img.save(out, "PNG")
    return out


def scr_to_srt(scr, dur, path, mc=30):
    sents = [s.strip()+"။" for s in scr.replace("။","။|").split("|") if s.strip()]
    if not sents: return None
    parts = []
    for s in sents:
        s = s.replace("။။","။")
        if len(s) <= mc: parts.append(s)
        else:
            cur = ""
            for w in s.split():
                if len(cur)+len(w)+1 <= mc: cur = cur+" "+w if cur else w
                else:
                    if cur: parts.append(cur.strip())
                    cur = w
            if cur: parts.append(cur.strip())
    if not parts: return None
    tot = sum(len(p) for p in parts); cur = 0.0
    with open(path, "w", encoding="utf-8") as f:
        for i, p in enumerate(parts, 1):
            d = (len(p)/tot)*dur
            f.write(f"{i}\n{t2s(cur)} --> {t2s(cur+d)}\n{p}\n\n"); cur += d
    return path


def parse_srt(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read().replace("\r\n","\n").replace("\r","\n")
    segs = []
    for ck in re.split(r"\n\s*\n", raw.strip()):
        ls = [l for l in ck.split("\n") if l.strip()]
        if len(ls) < 3: continue
        ts = next((l for l in ls if "-->" in l), None)
        if not ts: continue
        p = re.split(r"\s*-->\s*", ts)
        if len(p) != 2: continue
        a, b = s2t(p[0]), s2t(p[1])
        if a is None or b is None: continue
        idx = ls.index(ts); txt = " ".join(ls[idx+1:]).strip()
        if txt: segs.append({"start": a, "end": b, "text": txt})
    return segs


def overlay(vp, sp, op, fp, fs=30, pos_y=100, bh=100, ba=100,
            box_width_ratio=BOX_WIDTH_RATIO):
    W, H, _ = vid_info(vp); segs = parse_srt(sp)
    if not segs: raise Exception("SRT empty")
    os.makedirs("subtitle_pngs", exist_ok=True)

    def render_one(args):
        i, s = args
        p = f"subtitle_pngs/s_{i:04d}.png"
        render_png(s["text"], p, fp, W, H, fs, pos_y, bh, ba,
                   box_width_ratio=box_width_ratio)
        return i, {"p": p, "a": s["start"], "b": s["end"]}

    pngs = [None] * len(segs)
    with concurrent.futures.ThreadPoolExecutor(max_workers=PNG_WORKERS) as ex:
        for idx, item in ex.map(render_one, enumerate(segs)):
            pngs[idx] = item

    cmd = ["ffmpeg","-y","-i",vp] + sum([["-i",x["p"]] for x in pngs], [])
    flt, cur = [], "[0:v]"
    for i, x in enumerate(pngs):
        lbl = f"[v{i}]"
        flt.append(f"{cur}[{i+1}:v]overlay=0:0:enable='between(t,{x['a']:.3f},{x['b']:.3f})'{lbl}")
        cur = lbl
    cmd += ["-filter_complex",";".join(flt),"-map",cur,"-map","0:a?",
            "-c:v","libx264","-crf",str(ENC_CRF),"-preset",ENC_PRESET,
            "-tune","fastdecode","-c:a","copy",op]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0: raise Exception(f"FFmpeg: {(r.stderr or '')[-500:]}")
    for x in pngs:
        try: os.remove(x["p"])
        except: pass
    return op


def normalize_script(t):
    """Script ကို TTS/Subtitle အတွက် သန့်စင် (quote, emoji, စာကြောင်းလွတ်, ။ ထပ်)"""
    t = re.sub(r'[\U0001F000-\U0001FFFF\u2600-\u27BF\uFE0F]', '', t)
    t = re.sub(r'[\u201c\u201d"`*_#<>\[\]{}()\uff08\uff09\u300c\u300d\u300e\u300f\u00ab\u00bb~^|\\/]', ' ', t)
    out = []
    for ln in t.splitlines():
        ln = re.sub(r"\s+", " ", ln).strip()
        if not ln: continue
        if not ln.endswith(("။", "၊", "!", "?")): ln += "။"
        out.append(ln)
    t = " ".join(out)
    t = re.sub(r"။(\s*။)+", "။", t)
    return t.strip()


def has_speech(t):
    """ပြောလို့ရတဲ့ အက္ခရာ/ဂဏန်း ပါမပါ (။ ၊ သင်္ကေတချည်းဆိုရင် False)"""
    return re.search(r"[\u1000-\u1049\u1050-\u109F\w]", t) is not None


def split_scr(t, mc=TTS_CHUNK):
    sents = []
    for p in t.replace("။","။|").split("|"):
        p = p.strip()
        if not p: continue
        sents.append(p if p.endswith("။") else p + "။")
    out, cur = [], ""
    for s in sents:
        if len(cur)+len(s) <= mc: cur += s
        else:
            if cur: out.append(cur)
            if len(s) > mc:
                for i in range(0, len(s), mc): out.append(s[i:i+mc])
                cur = ""
            else: cur = s
    if cur: out.append(cur)
    return out


def video_bypass(input_video, output_video="bypass.mp4",
                 crop_ratio=0.95, mirror=True):
    W, H, dur = vid_info(input_video)
    filters = []

    if crop_ratio != 1.0:
        cw = int(W * crop_ratio); ch = int(H * crop_ratio)
        if cw % 2 != 0: cw -= 1
        if ch % 2 != 0: ch -= 1
        cx = (W - cw) // 2; cy = (H - ch) // 2
        filters.append(f"crop={cw}:{ch}:{cx}:{cy}")

    if mirror:
        filters.append("hflip")

    vf = ",".join(filters) if filters else "null"

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy", output_video
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise Exception(f"FFmpeg: {(r.stderr or '')[-300:]}")
    return output_video


# ==================== TikTok Neon Border (Static + Chase) ====================

def bypass_and_neon(input_video, output_video,
                    crop_ratio=0.95, mirror=True, thickness=30):
    """Bypass + Static Neon Border — ၃ လွှာ (Video မဖုံး)"""
    W, H, dur = vid_info(input_video)
    pad = thickness
    filters = []

    if crop_ratio != 1.0:
        cw = int(W * crop_ratio); ch = int(H * crop_ratio)
        if cw % 2 != 0: cw -= 1
        if ch % 2 != 0: ch -= 1
        cx = (W - cw) // 2; cy = (H - ch) // 2
        filters.append(f"crop={cw}:{ch}:{cx}:{cy}")

    if mirror:
        filters.append("hflip")

    filters.append(f"pad={W}:{H}:{pad}:{pad}:color=black@0")
    filters.append(f"drawbox=x=0:y=0:w={W}:h={H}:color={TIKTOK_BLACK}@1.0:t={pad}:replace=0")
    filters.append(f"drawbox=x=0:y=0:w={W}:h={H}:color={TIKTOK_CYAN}@1.0:t={max(3, pad//3)}:replace=0")
    filters.append(f"drawbox=x={pad}:y={pad}:w={W-2*pad}:h={H-2*pad}:color={TIKTOK_MAGENTA}@1.0:t={max(3, pad//3)}:replace=0")

    vf = ",".join(filters)

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy", output_video
    ]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise Exception(f"FFmpeg: {(r.stderr or '')[-300:]}")
    return output_video


def bypass_and_neon_chase(input_video, output_video,
                          crop_ratio=0.95, mirror=True,
                          thickness=30, speed=1.0, tail=0.35):
    """Bypass + ပတ်ပြေးနေတဲ့ Neon အလင်းတန်း (comet / မြားဦး style)"""
    base = "bypass_base.mp4"
    video_bypass(input_video, base, crop_ratio, mirror)

    W, H, _ = vid_info(base)
    pr = ffmpeg.probe(base)
    vs = next(s for s in pr['streams'] if s['codec_type'] == 'video')
    n, d = vs['r_frame_rate'].split('/')
    fps = float(n) / float(d)

    th = thickness
    P = 2 * (W + H)

    # Border ring pixel တွေရဲ့ ပတ်လမ်းအတိုင်း နေရာ (S) ကို တစ်ခါတည်း တွက်
    yy, xx = np.mgrid[0:H, 0:W]
    ring = (xx < th) | (xx >= W - th) | (yy < th) | (yy >= H - th)
    ys, xs = np.nonzero(ring)
    dt, db, dl, dr = ys, H - 1 - ys, xs, W - 1 - xs
    m = np.minimum.reduce([dt, db, dl, dr])
    S = np.where(m == dt, xs,
        np.where(m == dr, W + ys,
        np.where(m == db, 2 * W + H + (W - xs),
                 2 * W + 2 * H - ys))).astype(np.float32)

    BASE = np.array([15, 15, 15], np.float32)       # အနက်ရောင် အောက်ခံ
    CYAN = np.array([238, 244, 37], np.float32)     # BGR
    MAGENTA = np.array([85, 44, 254], np.float32)   # BGR

    def comet(head):
        dist = (head - S) % P                       # head ရဲ့ နောက်ဘက် ဝေးမှု
        a = np.clip(1.0 - dist / (tail * P), 0, 1)  # tail မှိန်သွား
        return (a ** 1.5)[:, None]

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
        "-i", base,
        "-map", "0:v", "-map", "1:a?",
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest",
        output_video
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)

    cap = cv2.VideoCapture(base)
    i = 0
    while True:
        ok, fr = cap.read()
        if not ok: break
        t = i / fps
        head = (t * speed * P / 4.0) % P            # speed=1 → ၄ စက္ကန့်/အပတ်
        c = BASE + CYAN * comet(head) + MAGENTA * comet(head + P / 2)
        fr[ys, xs] = np.clip(c, 0, 255).astype(np.uint8)
        proc.stdin.write(fr.tobytes())
        i += 1
    cap.release()
    proc.stdin.close()
    proc.wait()
    if proc.returncode != 0:
        raise Exception("Chase Neon: ffmpeg fail")
    return output_video


def keep_count(video_in, segments):
    """Speech အပိုင်းထဲက frame အရေအတွက် (encode မလုပ်ဘဲ တွက်)"""
    pr = ffmpeg.probe(video_in)
    vs = next(x for x in pr['streams'] if x['codec_type'] == 'video')
    n_, d_ = vs['r_frame_rate'].split('/')
    fps = float(n_) / float(d_)
    total = int(round(float(pr['format']['duration']) * fps))
    t = np.arange(total) / fps
    mask = np.zeros(total, dtype=bool)
    for a, b in segments:
        mask |= (t >= a) & (t <= b)
    return int(mask.sum()), fps


def simple_merge(video_in, audio_in, output_video, tempo, segments=None):
    vf = []
    if segments:
        sel = "+".join(f"between(t,{a:.3f},{b:.3f})" for a, b in segments)
        vf = ["-vf", f"select='{sel}',setpts=N/FRAME_RATE/TB"]
    cmd = ["ffmpeg", "-y", "-i", video_in, "-i", audio_in,
           "-af", f"atempo={tempo}", "-map", "0:v", "-map", "1:a"] + vf + [
           "-c:v", "libx264", "-crf", "20", "-preset", "veryfast",
           "-c:a", "aac", "-b:a", AUDIO_BITRATE, "-shortest", output_video]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise Exception(f"FFmpeg: {(r.stderr or '')[-300:]}")
    return output_video


def final_render(video_in, audio_in, output_video, tempo,
                 srt_path=None, fp=FONT_FILE, fs=FS, pos_y=100, bh=BH, ba=BA,
                 use_neon=True, thickness=15, speed=0.4, tail=0.5,
                 crop_ratio=0.95, mirror=True, segments=None):
    """Crop + Mirror + Chase Neon + Subtitle + Audio — encode တစ်ခါတည်း"""
    W0, H0, _ = vid_info(video_in)
    pr = ffmpeg.probe(video_in)
    vs = next(s for s in pr['streams'] if s['codec_type'] == 'video')
    n, d = vs['r_frame_rate'].split('/')
    fps = float(n) / float(d)

    if crop_ratio != 1.0:
        cw = int(W0 * crop_ratio); ch = int(H0 * crop_ratio)
        if cw % 2: cw -= 1
        if ch % 2: ch -= 1
        cx = (W0 - cw) // 2; cy = (H0 - ch) // 2
    else:
        cw, ch, cx, cy = W0, H0, 0, 0
    W, H = cw, ch

    # ---- Subtitle PNG များကို အကြိုပြင် (parallel) ----
    subs = []
    if srt_path:
        segs = parse_srt(srt_path)
        os.makedirs("subtitle_pngs", exist_ok=True)

        def prep(args):
            i, sg = args
            p = f"subtitle_pngs/s_{i:04d}.png"
            render_png(sg["text"], p, fp, W, H, fs, pos_y, bh, ba,
                       box_width_ratio=BOX_WIDTH_RATIO)
            rgba = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            try: os.remove(p)
            except: pass
            if rgba is None: return None
            rows = np.where(rgba[:, :, 3].any(axis=1))[0]
            if len(rows) == 0: return None
            y0, y1 = int(rows[0]), int(rows[-1]) + 1
            crop = rgba[y0:y1]
            alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
            pre = crop[:, :, :3].astype(np.float32) * alpha
            return {"a": sg["start"], "b": sg["end"], "y0": y0, "y1": y1,
                    "inv": 1.0 - alpha, "pre": pre}

        with concurrent.futures.ThreadPoolExecutor(max_workers=PNG_WORKERS) as ex:
            subs = [x for x in ex.map(prep, enumerate(segs)) if x]

    # ---- Neon ring ကြိုတွက် ----
    if use_neon:
        th = thickness
        P = 2 * (W + H)
        yy, xx = np.mgrid[0:H, 0:W]
        ring = (xx < th) | (xx >= W - th) | (yy < th) | (yy >= H - th)
        ys, xs = np.nonzero(ring)
        dt, db, dl, dr = ys, H - 1 - ys, xs, W - 1 - xs
        m = np.minimum.reduce([dt, db, dl, dr])
        S = np.where(m == dt, xs,
            np.where(m == dr, W + ys,
            np.where(m == db, 2 * W + H + (W - xs),
                     2 * W + 2 * H - ys))).astype(np.float32)
        BASE = np.array([12, 8, 10], np.float32)         # အောက်ခံ (အနက်ဖြစ်မနေအောင် အရောင်အနည်းငယ်)
        CYAN = np.array([255, 255, 0], np.float32)       # BGR → #00FFFF လင်းလင်း
        MAGENTA = np.array([110, 30, 255], np.float32)   # BGR → #FF1E6E လင်းလင်း
        IDLE = 0.14                                      # အမြီးမရှိတဲ့နေရာမှာလည်း အရောင်အလင်း အနည်းငယ်

        Si = (S.astype(np.int64)) % P
        P_arr = np.arange(P, dtype=np.float32)

        def layer(h, color):
            dist = (h - P_arr) % P
            a_ = np.clip(1.0 - dist / (tail * P), 0, 1)
            glow = IDLE + (1.0 - IDLE) * (a_ ** 0.55)                    # အမြီး ပိုကြာကြာ လင်း
            core = np.clip(1.0 - dist / (0.06 * P), 0, 1) ** 2           # ဦးခေါင်း အဖြူရောင် တောက်
            return color * glow[:, None] + 255.0 * 0.9 * core[:, None]

        def make_strip(head):
            c_ = BASE + layer(head, CYAN) + layer(head + P / 2, MAGENTA)
            return np.clip(c_, 0, 255).astype(np.uint8)

    cmd = [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{W}x{H}", "-r", str(fps), "-i", "-",
    ]
    if audio_in:
        cmd += ["-i", audio_in, "-af", f"atempo={tempo}", "-map", "0:v", "-map", "1:a"]
    else:
        cmd += ["-an"]
    cmd += ["-c:v", "libx264", "-crf", str(FINAL_CRF), "-preset", FINAL_PRESET,
            "-pix_fmt", "yuv420p", "-threads", "0"]
    if audio_in:
        cmd += ["-c:a", "aac", "-b:a", AUDIO_BITRATE, "-shortest"]
    cmd += [output_video]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    cap = cv2.VideoCapture(video_in)
    i = 0; j = 0; k = 0; p = 0
    try:
        while True:
            if not cap.grab(): break
            t_in = i / fps
            i += 1
            if segments:
                while p < len(segments) and segments[p][1] < t_in: p += 1
                if not (p < len(segments) and segments[p][0] <= t_in): continue
            ok, fr = cap.retrieve()
            if not ok: break
            t = j / fps
            fr = fr[cy:cy + ch, cx:cx + cw]
            if mirror: fr = fr[:, ::-1]
            fr = np.ascontiguousarray(fr)

            if use_neon:
                head = (t * speed * P / 4.0) % P
                fr[ys, xs] = make_strip(head)[Si]

            while k < len(subs) and subs[k]["b"] < t: k += 1
            if k < len(subs) and subs[k]["a"] <= t:
                sb = subs[k]
                reg = fr[sb["y0"]:sb["y1"]].astype(np.float32) * sb["inv"] + sb["pre"]
                fr[sb["y0"]:sb["y1"]] = reg.astype(np.uint8)

            proc.stdin.write(fr.tobytes())
            j += 1
    finally:
        cap.release()
        try: proc.stdin.close()
        except: pass
        proc.wait()
    if proc.returncode != 0:
        raise Exception("Final render: ffmpeg fail")
    return output_video


def mux_audio(video_in, audio_in, output_video, tempo):
    """Video ကို ပြန် encode မလုပ်ဘဲ အသံပေါင်းပေး (copy)"""
    cmd = ["ffmpeg", "-y", "-i", video_in, "-i", audio_in,
           "-af", f"atempo={tempo}", "-map", "0:v", "-map", "1:a",
           "-c:v", "copy", "-c:a", "aac", "-b:a", AUDIO_BITRATE,
           "-shortest", output_video]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0:
        raise Exception(f"Mux: {(r.stderr or '')[-300:]}")
    return output_video


def prepare_video_job(video_in, script_text, use_sub, fw_model,
                      pos_y, use_neon, thickness, speed):
    """Background: Speech ရှာ → Subtitle → အသံမပါ Video render (TTS နဲ့ တပြိုင်တည်း)"""
    segments = whisper_fast(video_in, fw_model)
    if not segments: raise Exception("Speech မတွေ့")
    kept, vfps = keep_count(video_in, segments)
    if kept == 0: raise Exception("Speech မတွေ့")
    vdur = kept / vfps
    render_err = None
    try:
        sp = scr_to_srt(script_text, vdur, "sub.srt") if use_sub else None
        final_render(video_in, None, "video_only.mp4", 1.0,
                     srt_path=sp, fp=FONT_FILE, fs=FS, pos_y=pos_y,
                     bh=BH, ba=BA, use_neon=use_neon,
                     thickness=thickness, speed=speed, segments=segments)
    except Exception as e:
        render_err = str(e)
    return {"segments": segments, "vdur": vdur, "render_err": render_err}


# ==================== Preview ====================

def draw_tiktok_border_preview(img, thickness=30, animated_phase=0.0):
    """Preview frame ပေါ်မှာ TikTok 3-layer border ဆွဲ"""
    d = ImageDraw.Draw(img, "RGBA")
    W, H = img.size
    inner_c = max(3, thickness // 3)

    # 1. အနက်ရောင် border
    d.rectangle([0, 0, W - 1, H - 1],
                outline=(0, 0, 0, 255), width=thickness)

    # 2. Cyan border (Animated ဆိုရင် အရောင် ပြောင်း)
    if animated_phase > 0:
        import math
        phase = (math.sin(animated_phase * math.pi) + 1) / 2
        cyan_r = int(37 * (1 - phase) + 254 * phase)
        cyan_g = int(244 * (1 - phase) + 44 * phase)
        cyan_b = int(238 * (1 - phase) + 85 * phase)
        d.rectangle([0, 0, W - 1, H - 1],
                    outline=(cyan_r, cyan_g, cyan_b, 255), width=inner_c)
    else:
        d.rectangle([0, 0, W - 1, H - 1],
                    outline=(37, 244, 238, 255), width=inner_c)

    # 3. Magenta border
    offset = thickness
    d.rectangle([offset, offset, W - 1 - offset, H - 1 - offset],
                outline=(254, 44, 85, 255), width=inner_c)

    return img


# ==================== TTS Functions ====================

def tts_demo(chunks, ref, space, cb=None):
    cl = Client(space); files = []; rf = handle_file(ref) if ref else None
    for i, c in enumerate(chunks):
        if cb: cb(i, len(chunks), c)
        res = cl.predict(
            text_input=c,
            control_instruction="A warm young woman, calm and expressive",
            reference_wav_path_input=rf,
            use_prompt_text=False, prompt_text_input="",
            cfg_value_input=2.0, do_normalize=True, denoise=False,
            api_name="/generate"
        )
        p = res[0] if isinstance(res, (tuple, list)) else res
        dst = f"chunk_{i}.wav"; shutil.copy(p, dst); files.append(dst)
    return files


def tts_burmese(chunks, ref, space, cb=None):
    cl = Client(space); files = []
    if not ref: raise Exception("Reference Audio needed")
    rf = handle_file(ref)
    for i, c in enumerate(chunks):
        if cb: cb(i, len(chunks), c)
        res = cl.predict(
            target_text=c, ref_audio=rf,
            ref_text="မြန်မာ အသံနမူနာ", cfg_value=2.0,
            inference_timesteps=10, api_name="/tts"
        )
        p = res[0] if isinstance(res, (tuple, list)) else res
        dst = f"chunk_b_{i}.wav"; shutil.copy(p, dst); files.append(dst)
    return files


async def _edge_tts_async(text, out_file, voice):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(out_file)


def edge_tts_run(chunks, out_path, cb=None, workers=TTS_WORKERS):
    voice_id = EDGE_VOICES[EDGE_VOICE_FIXED]

    # စာတိုတိုပြန်ခွဲ + ပြောလို့မရတဲ့ chunk ဖယ်
    small = []
    for c in chunks:
        for x in split_scr(c, EDGE_CHUNK):
            if has_speech(x): small.append(x)
    chunks = small
    if not chunks: raise Exception("TTS လုပ်စရာ စာမတွေ့ပါ")

    def run_async(coro_fn):
        try: asyncio.run(coro_fn())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(coro_fn())
            loop.close()

    def tts_one(args):
        i, c = args
        os.makedirs("tts_cache", exist_ok=True)
        key = hashlib.md5((voice_id + c).encode("utf-8")).hexdigest()
        dst = f"tts_cache/{key}.mp3"
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            return (i, dst)   # ယခင်ထုတ်ပြီးသား — request ပြန်မခေါ်
        for attempt in range(4):
            try:
                if os.path.exists(dst): os.remove(dst)
                run_async(lambda: _edge_tts_async(c, dst, voice_id))
                if os.path.exists(dst) and os.path.getsize(dst) > 0:
                    time.sleep(0.7)   # request တစ်ခုနဲ့တစ်ခုကြား နားချိန်
                    return (i, dst)
            except Exception:
                pass
            time.sleep(3 * (attempt + 1))   # rate limit ရှောင်ဖို့ တဖြည်းဖြည်း စောင့်
        return (i, None)

    results = [None] * len(chunks); done = 0; skipped = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        for idx, dst in ex.map(tts_one, enumerate(chunks)):
            results[idx] = dst; done += 1
            if dst is None: skipped.append(idx)
            if cb: cb(done - 1, len(chunks), chunks[idx])

    good = [a for a in results if a]
    if not good:
        raise Exception("Edge TTS အသံမရပါ — edge-tts ကို update လုပ်ပါ (edge-tts>=7.0.0) / ခဏနေ ပြန်စမ်းပါ")
    if skipped:
        st.warning("⚠️ ကျော်လိုက်တဲ့ အပိုင်း: " + " | ".join(chunks[i][:30] for i in skipped))

    with open("edge_concat.txt", "w", encoding="utf-8") as f:
        for a in good: f.write(f"file '{a}'\n")

    ffmpeg.input("edge_concat.txt", format="concat", safe=0).output(
        out_path, acodec="libmp3lame", audio_bitrate=AUDIO_BITRATE, ar=48000
    ).run(overwrite_output=True)
    return out_path


def tts_free(chunks, out, cb=None):
    """Edge TTS သီဟ"""
    edge_tts_run(chunks, out, cb=cb)
    st.success("✅ Edge TTS — 👨 သီဟ (Thiha)")
    return out


def tts_all(text, out, ref=None, cb=None, use_voxcpm=True):
    chunks = split_scr(text, TTS_CHUNK)

    if not use_voxcpm:
        return tts_free(chunks, out, cb=cb)

    files = None
    for s in SPACES:
        try:
            st.info(f"🎙️ VoxCPM2 — {s['space']} — စမ်းနေသည်...")
            if s["type"] == "demo":
                files = tts_demo(chunks, ref, s["space"], cb)
            else:
                files = tts_burmese(chunks, ref, s["space"], cb)
            st.success("✅ VoxCPM2 — အောင်မြင်")
            break
        except Exception as e:
            st.warning(f"⚠️ VoxCPM2 — Fail: {str(e)[:80]}")
            files = None
            continue

    if files is None:
        st.warning("⚠️ VoxCPM2 — Busy/Fail — Edge TTS သီဟ Auto")
        return tts_free(chunks, out, cb=cb)

    with open("concat.txt", "w", encoding="utf-8") as f:
        for a in files: f.write(f"file '{a}'\n")
    ffmpeg.input("concat.txt", format="concat", safe=0).output(
        out, acodec="libmp3lame", audio_bitrate=AUDIO_BITRATE, ar=48000
    ).run(overwrite_output=True)
    return out


@st.cache_resource(show_spinner=False)
def get_fw_model():
    from faster_whisper import WhisperModel
    return WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8",
                        cpu_threads=os.cpu_count() or 4)


def whisper_fast(video_path, model=None):
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-ar", "16000", "-ac", "1",
        "-c:a", "pcm_s16le", "whisper_audio.wav"
    ], capture_output=True, check=True)

    if USE_FAST_VAD:
        try:
            from faster_whisper.audio import decode_audio
            from faster_whisper.vad import get_speech_timestamps, VadOptions
            audio = decode_audio("whisper_audio.wav", sampling_rate=16000)
            ts = get_speech_timestamps(
                audio, VadOptions(min_silence_duration_ms=700, speech_pad_ms=200))
            segs_vad = [(t["start"] / 16000.0, t["end"] / 16000.0) for t in ts]
            if segs_vad:
                return segs_vad
        except Exception:
            pass   # VAD မရရင် Whisper နဲ့ ဆက်သွား

    speech_segments = []
    try:
        model = model or get_fw_model()
        segments, _ = model.transcribe(
            "whisper_audio.wav", language=WHISPER_LANG,
            vad_filter=False, beam_size=1,
            condition_on_previous_text=False, temperature=0
        )
        for seg in segments:
            speech_segments.append((seg.start, seg.end))
    except Exception:
        import whisper
        model = whisper.load_model(WHISPER_MODEL)
        result = model.transcribe(
            "whisper_audio.wav", language=WHISPER_LANG,
            condition_on_previous_text=False, beam_size=1, temperature=0
        )
        for seg in result["segments"]:
            speech_segments.append((seg["start"], seg["end"]))
    return speech_segments


def silence_cut_v2(input_video, output_video="input_cut.mp4"):
    t0 = time.time()
    speech_segments = whisper_fast(input_video)
    whisper_time = time.time() - t0

    if not speech_segments: raise Exception("Speech မတွေ့")

    select_exprs = [f"between(t,{s:.3f},{e:.3f})" for s, e in speech_segments]
    select_str = "+".join(select_exprs)

    cmd = ["ffmpeg", "-y", "-i", input_video,
           "-vf", f"select='{select_str}',setpts=N/FRAME_RATE/TB",
           "-af", f"aselect='{select_str}',asetpts=N/SR/TB",
           "-c:v", "libx264", "-crf", "23", "-preset", "ultrafast",
           "-c:a", "aac", "-b:a", "128k", output_video]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if r.returncode != 0: raise Exception(f"FFmpeg: {(r.stderr or '')[-300:]}")

    total = sum(e - s for s, e in speech_segments)
    return {"segments": len(speech_segments), "duration": total, "whisper_time": whisper_time}


# ==================== UI ====================

st.markdown("<div class='main-title'>🎬 Myanmar TTS Recap</div>", unsafe_allow_html=True)
st.markdown("<div class='main-sub'>Video → Script → VoxCPM2 / Edge TTS သီဟ → Recap</div>", unsafe_allow_html=True)
st.divider()

if "script" not in st.session_state: st.session_state.script = ""
script = st.text_area("Script", value=st.session_state.script, height=180,
                       label_visibility="collapsed", placeholder="မြန်မာ Script paste...")
st.session_state.script = script

c1, c2 = st.columns([3, 1])
with c1: st.caption(f"📝 {len(script):,}")
with c2:
    if st.button("🗑️ Clear", use_container_width=True):
        st.session_state.script = ""; st.rerun()

def _canvas_body():
    st.link_button("↗️ Canvas ကို Tab အသစ်မှာ ဖွင့်", CANVAS_URL, use_container_width=True)
    components.html(f"""
    <button onclick="window.open('{CANVAS_URL}','canvas_win','width=480,height=820,left=40,top=40')"
      style="width:100%;padding:12px;border-radius:10px;border:1px solid #667eea;
             background:#1a1a35;color:#e8e8f0;font-size:15px;font-weight:600;cursor:pointer">
      🪟 Popup Window နဲ့ ဖွင့် (PC မှာ ဘေးချင်းကပ်ကြည့်လို့ရ)
    </button>""", height=56)
    if st.toggle("👁️ ဒီနေရာထဲမှာ iframe နဲ့ စမ်းကြည့်မယ်", value=False, key="canvas_iframe"):
        components.iframe(CANVAS_URL, height=480, scrolling=True)
        st.caption("⚠️ အလွတ်ပဲ ပြရင် Google က ပိတ်ထားတာ — အပေါ်က ခလုတ်နှစ်ခုကို သုံးပါ")
    pasted = st.text_area("Canvas ကနေ Copy → ဒီမှာ Paste", height=240, key="canvas_paste")
    if st.button("➡️ Script ထဲ ထည့်မယ်", key="canvas_apply", use_container_width=True):
        lines = [l.strip().strip('"\u201c\u201d').strip() for l in pasted.splitlines()]
        st.session_state.script = "\n\n".join(l for l in lines if l)
        st.rerun()


if hasattr(st, "dialog"):
    @st.dialog("📄 Gemini Canvas")
    def canvas_dialog():
        _canvas_body()

    if st.button("📄 Transcript ထုတ်ယူမယ် (Gemini Canvas)", use_container_width=True):
        canvas_dialog()
else:
    with st.expander("📄 Transcript ထုတ်ယူမယ် (Gemini Canvas)"):
        _canvas_body()
st.divider()

st.subheader("📁 Step 2 — Video")
vid = st.file_uploader("📹", type=["mp4","mov","avi","mkv"], label_visibility="collapsed")
if vid: st.success(f"✅ {vid.size/(1024*1024):.1f} MB")
st.divider()

# TikTok Neon Border — UI မပြဘဲ နောက်ကွယ်မှာ Auto (ပုံသေ)
use_neon = True
neon_animated = True
neon_thickness = 15
neon_speed = 0.4

# Step 4 — Subtitle
st.subheader("📝 Step 4 — Subtitle")
use_sub = st.toggle("Burn-in", value=True)
pos_y = 100
if use_sub:
    pos_y = st.slider("📍 Position", 0, 100, 100, 1)
st.divider()

# Step 5 — Preview
if vid:
    st.subheader("🖼️ Step 5 — Preview")
    with st.spinner("Preview..."):
        pkey = (vid.name, vid.size)
        if st.session_state.get("pkey") != pkey or not os.path.exists("preview.mp4"):
            vid.seek(0)
            with open("preview.mp4", "wb") as f: f.write(vid.read())
            st.session_state.pkey = pkey
        W, H, _ = vid_info("preview.mp4")

        render_png("စာတန်းထိုး Preview", "prev.png", FONT_FILE, W, H, FS, pos_y, BH, BA,
                   box_width_ratio=BOX_WIDTH_RATIO)

        cap = cv2.VideoCapture("preview.mp4")
        ok, fr = cap.read()
        cap.release()

        if ok:
            bg = Image.fromarray(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)).convert("RGBA")
            fg = Image.open("prev.png").convert("RGBA")
            comp = Image.alpha_composite(bg, fg)

            if use_neon:
                phase = 0.5 if neon_animated else 0.0
                comp = draw_tiktok_border_preview(comp, thickness=neon_thickness,
                                                   animated_phase=phase)

            pw = 720
            comp.resize((pw, int(H * (pw / W))), Image.LANCZOS).convert("RGB").save("prev_out.png")
            st.image("prev_out.png", use_container_width=True)
            if use_neon and neon_animated:
                st.caption("🎬 Animated — Output Video မှာ အလင်းတန်း ပတ်ပြေးနေမည်")
st.divider()

# Step 6 — Generate
st.subheader("🚀 Step 6 — Generate")

use_voxcpm = st.toggle("🎙️ VoxCPM2 သုံးမလား?", value=True)

if use_voxcpm:
    st.info("✅ VoxCPM2 သုံးမယ် — Fail/Busy ရင် — Edge TTS သီဟ Auto")
    ref = st.file_uploader("🎤 Ref Audio (VoxCPM2) — Optional",
                            type=["wav","mp3","m4a"], key="ref_up")
    if ref:
        with open("ref.wav", "wb") as f: f.write(ref.read())
        st.session_state.ref = "ref.wav"
        st.success("✅ Ref Audio")
    else:
        st.session_state.ref = None
else:
    st.info("⚡ Edge TTS သီဟ (Thiha) — ပဲ သုံးမယ်")
    st.session_state.ref = None

if st.button("✨ Generate Recap Video", type="primary", use_container_width=True):
    if not script.strip(): st.error("Script paste"); st.stop()
    if vid is None: st.error("Video Upload"); st.stop()

    total_start = time.time(); step_times = {}
    script_n = normalize_script(script)
    if not script_n.strip(): st.error("Script မှာ စာမပါပါ"); st.stop()
    vid.seek(0)
    with open("input.mp4", "wb") as f: f.write(vid.read())
    _, _, vdur = vid_info("input.mp4")

    # ၁။ Cut + Render ကို Background မှာ — TTS နဲ့ တပြိုင်တည်း (အသံမပါ Video အရင်ထုတ်)
    fw_model = None
    if not USE_FAST_VAD:
        try: fw_model = get_fw_model()
        except Exception: pass
    job_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    job = job_pool.submit(prepare_video_job, "input.mp4", script_n, use_sub, fw_model,
                          pos_y, use_neon, neon_thickness, neon_speed)

    # ၂။ TTS
    t0 = time.time()
    pb = st.progress(0); txt = st.empty()
    def cb(i, tot, c): pb.progress((i+1)/tot); txt.caption(f"[{i+1}/{tot}]")
    try:
        tts_all(script_n, "voice.mp3", ref=st.session_state.get("ref"),
                cb=cb, use_voxcpm=use_voxcpm)
    except Exception as e:
        st.error(f"TTS: {e}"); st.stop()
    step_times["🎙️ TTS"] = time.time() - t0

    # ၃။ Background job စောင့် + အသံပေါင်း
    t0 = time.time()
    try:
        info = job.result()
    except Exception as e:
        st.error(f"❌ Cut: {e}"); st.stop()
    job_pool.shutdown(wait=False)
    segments, vdur = info["segments"], info["vdur"]

    adur = float(ffmpeg.probe("voice.mp3")['format']['duration'])
    tempo = max(0.5, min(2.0, adur/vdur))

    with st.spinner("🎬 Finalizing..."):
        try:
            if info["render_err"]: raise Exception(info["render_err"])
            mux_audio("video_only.mp4", "voice.mp3", "final.mp4", tempo)
        except Exception as e:
            st.warning(f"⚠️ Fail: {e}")
            simple_merge("input.mp4", "voice.mp3", "final.mp4", tempo, segments)
    step_times["🎬 Render (စောင့်ချိန်)"] = time.time() - t0

    total_elapsed = time.time() - total_start

    st.markdown(f"""
    <div class="timer-box">
        <div class="timer-title">⏱️ TOTAL TIME</div>
        <div class="timer-value">{total_elapsed:.1f}<span class="timer-unit">sec</span></div>
    </div>
    """, unsafe_allow_html=True)

    for name, t in step_times.items():
        st.markdown(f"<div class='step-timer'>{name} — <b>{t:.1f}s</b></div>", unsafe_allow_html=True)

    st.success(f"✅ Done — ⏱️ {total_elapsed:.1f}s")
    st.video("final.mp4")
    with open("final.mp4", "rb") as f:
        st.download_button("📥 Download", f, file_name="recap.mp4")
