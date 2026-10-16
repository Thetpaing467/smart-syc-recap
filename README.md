# Recap → Myanmar (Streamlit)

English / Chinese / တခြားဘာသာ recap video ကို မြန်မာအသံနဲ့ ပြန်ထုတ်ပေးတဲ့ web app။

Whisper (စာသားထုတ်) → Gemini (မြန်မာပြန်) → Edge TTS (မြန်မာအသံ) → Smart Sync (ရုပ်ကို အသံအရှည်နဲ့ ညှိ)

## Deploy (Streamlit Community Cloud)
1. ဒီ file တွေကို GitHub repo အသစ်ထဲ တင်ပါ (app.py, requirements.txt, packages.txt, .streamlit/config.toml)
2. share.streamlit.io → Create app → repo / branch main / Main file: `app.py`
3. Advanced settings → Python 3.12 → Secrets ထဲ `GEMINI_API_KEY = "..."`
4. Deploy

## သတိ
- CPU သာ ရှိတယ် (GPU မရှိ) → video တိုတို (၅ မိနစ်အောက်) နဲ့ စမ်းပါ
- Edge TTS က cloud IP တချို့ကို 403 ပေးနိုင်တယ်
- App အိပ်သွားရင် (inactive) Wake up နှိပ်ပါ၊ တင်ထားတဲ့ file တွေ ပျောက်ပါမယ်
# smart-syc-recap