import os
import re
import json
import time
import asyncio
import shutil
import subprocess
import tempfile

import pandas as pd
import streamlit as st

st.set_page_config(page_title="Recap → Myanmar", page_icon="🎙️", layout="centered")

DEFAULT_MODELS = (
    "gemini-3.5-flash, gemini-3.5-flash-lite, gemini-3.6-flash, "
    "gemini-3.7-flash, gemini-3.1-flash-lite"
)
TARGET, MAXD, BATCH = 4.2, 5.6, 40

TRIM = (
    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,"
    "silenceremove=start_periods=1:start_threshold=-50dB,areverse,apad=pad_dur=0.2"
)


# ------------------------------------------------------------------ helpers
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


def tts_one(text, voice, rate, mp3, proxy=None):
    import edge_tts

    async def run():
        kw = {"proxy": proxy} if proxy else {}
        await edge_tts.Communicate(text, voice, rate=rate, **kw).save(mp3)

    asyncio.run(run())


def reset_job():
    wd = st.session_state.get("wd")
    if wd and os.path.isdir(wd):
        shutil.rmtree(wd, ignore_errors=True)
    for k in ("wd", "units", "vdur", "result"):
        st.session_state.pop(k, None)


# ------------------------------------------------------------------ UI
st.title("🎙️ Recap → Myanmar")
st.caption("Whisper → Gemini → Edge TTS → Smart Sync · Video တိုတိုနဲ့ စမ်းပါ (CPU သာ)")

with st.expander("⚙️ Settings", expanded=not secret("GEMINI_API_KEY")):
    api_key = st.text_input("Gemini API key", type="password",
                            value=secret("GEMINI_API_KEY"))
    c1, c2 = st.columns(2)
    voice = c1.selectbox("Voice", ["my-MM-ThihaNeural", "my-MM-NilarNeural"],
                         format_func=lambda v: "Thiha (ယောက်ျား)" if "Thiha" in v else "Nilar (မိန်းမ)")
    rate_n = c2.slider("Speed %", -50, 100, 0, 5)
    c3, c4 = st.columns(2)
    src_lang = c3.selectbox("မူရင်းဘာသာ", ["auto", "en", "zh", "ko", "ja", "hi", "th", "vi", "id"])
    wsize = c4.selectbox("Whisper model", ["tiny", "base", "small"], index=1,
                         help="small = တိကျ/နှေး၊ tiny = မြန်/အမှားများ")
    c5, c6 = st.columns(2)
    height = c5.select_slider("Output height", [360, 480, 720, 1080], value=480)
    max_speed = c6.slider("ရုပ် အမြန်ဆုံး ဆ", 1.0, 2.0, 1.25, 0.05)
    models_txt = st.text_input("Gemini models (comma ခြား)", DEFAULT_MODELS)
    max_min = st.slider("Video အရှည် ကန့်သတ် (မိနစ်)", 1, 15, 6)

rate = f"{rate_n:+d}%"
models = [m.strip() for m in models_txt.split(",") if m.strip()]
proxy = secret("EDGE_TTS_PROXY") or None

up = st.file_uploader("Video ရွေးပါ", type=["mp4", "mov", "mkv", "webm", "m4v"])

# ------------------------------------------------------------------ Stage 1
if st.button("1️⃣ စာသားထုတ် + မြန်မာပြန်", type="primary",
             disabled=not (up and api_key), use_container_width=True):
    reset_job()
    wd = tempfile.mkdtemp(prefix="recap_")
    st.session_state["wd"] = wd
    src = os.path.join(wd, "src.mp4")
    with open(src, "wb") as f:
        f.write(up.getbuffer())

    try:
        vdur = dur(src)
        if vdur > max_min * 60:
            st.error(f"Video က {vdur/60:.1f} မိနစ်ရှိတယ်။ {max_min} မိနစ်အောက်ပဲ လက်ခံတယ် "
                     "(Settings မှာ ပြောင်းလို့ရ)")
            st.stop()
        st.session_state["vdur"] = vdur

        # ---- Whisper
        with st.status("🎧 Whisper စာသားပြောင်းနေတယ်...", expanded=True) as stt:
            wav = os.path.join(wd, "audio16k.wav")
            sh(["ffmpeg", "-y", "-i", src, "-vn", "-ac", "1", "-ar", "16000", wav])
            wm = load_whisper(wsize)
            segs, info = wm.transcribe(
                wav, language=None if src_lang == "auto" else src_lang,
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

        # ---- Gemini
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
            {"start": units[i][0], "end": units[i][1], "src": units[i][2], "my": my[i]}
            for i in range(len(units))
        ]
    except Exception as ex:  # noqa: BLE001
        st.error(f"Error: {ex}")

# ------------------------------------------------------------------ Edit + Stage 2
if st.session_state.get("units"):
    units = st.session_state["units"]
    st.subheader("✏️ မြန်မာစာသား စစ်/ပြင်")
    df = pd.DataFrame({"#": range(len(units)),
                       "မူရင်း": [u["src"] for u in units],
                       "မြန်မာ": [u["my"] for u in units]})
    edited = st.data_editor(df, disabled=["#", "မူရင်း"], hide_index=True,
                            use_container_width=True, key="editor")

    if st.button("2️⃣ အသံထုတ် + Video ပေါင်း", type="primary", use_container_width=True):
        wd, vdur = st.session_state["wd"], st.session_state["vdur"]
        src = os.path.join(wd, "src.mp4")
        my_list = [clean_my(str(t)) or units[i]["my"] for i, t in enumerate(edited["မြန်မာ"])]
        n = len(units)
        try:
            # ---- TTS
            with st.status("🔊 မြန်မာအသံ ထုတ်နေတယ်...", expanded=True) as tst:
                bar = st.progress(0.0)
                for i in range(n):
                    mp3, wav = f"{wd}/a{i:04d}.mp3", f"{wd}/a{i:04d}.wav"
                    for attempt in range(4):
                        try:
                            tts_one(my_list[i], voice, rate, mp3, proxy)
                            break
                        except Exception as ex:  # noqa: BLE001
                            if attempt == 3:
                                raise RuntimeError(
                                    f"Edge TTS မအောင်မြင်ပါ (စာကြောင်း #{i}): {ex}\n\n{my_list[i]}")
                            time.sleep(2)
                    sh(["ffmpeg", "-y", "-i", mp3, "-af", TRIM, "-ar", "44100", wav])
                    bar.progress((i + 1) / n)
                tst.update(label="🔊 အသံ ပြီးပြီ", state="complete")

            # ---- Smart Sync
            with st.status("🎬 Smart Sync render...", expanded=True) as rst:
                bounds = [0.0] + [(units[i][1] + units[i + 1][0]) / 2 for i in range(n - 1)] + [vdur]
                clips, total_audio = [], 0.0
                bar = st.progress(0.0)
                for i in range(n):
                    s, e = bounds[i], bounds[i + 1]
                    w = max(e - s, 0.2)
                    wav = f"{wd}/a{i:04d}.wav"
                    d = dur(wav)
                    total_audio += d
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
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "24",
                        "-pix_fmt", "yuv420p", "-c:a", "aac", "-ar", "44100", "-ac", "2", out])
                    clips.append(out)
                    bar.progress((i + 1) / n)

                lst = f"{wd}/list.txt"
                with open(lst, "w") as f:
                    for c in clips:
                        f.write(f"file '{c}'\n")
                final = f"{wd}/final.mp4"
                sh(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", final])
                with open(final, "rb") as f:
                    st.session_state["result"] = f.read()
                rst.update(label=f"🎬 ပြီးပါပြီ ({dur(final):.1f}s / မူရင်း {vdur:.1f}s)",
                           state="complete")
        except Exception as ex:  # noqa: BLE001
            st.error(f"Error: {ex}")

# ------------------------------------------------------------------ Result
if st.session_state.get("result"):
    st.subheader("✅ ရလဒ်")
    st.video(st.session_state["result"])
    st.download_button("⬇️ Download MP4", st.session_state["result"],
                       file_name="final_myanmar.mp4", mime="video/mp4",
                       use_container_width=True)
    if st.button("🗑️ အသစ်ပြန်စ", use_container_width=True):
        reset_job()
        st.rerun()
