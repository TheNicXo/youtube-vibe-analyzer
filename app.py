import streamlit as st
import librosa
import numpy as np
import whisper
from textblob import TextBlob
import plotly.express as px
import pandas as pd
import google.genai as genai
from google.genai import types
import json
from youtube_extractor import download_audio_from_youtube


st.set_page_config(page_title="YouTube Vibe Analyzer", page_icon="🎵", layout="wide")

st.markdown("""
    <style>
    .stApp {
        background-color: #121212;
        color: #FFFFFF;
    }
    h1 {
        color: #1DB954 !important;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    .stButton>button {
        background-color: #1DB954 !important;
        color: white !important;
        border-radius: 20px !important;
        border: none !important;
        font-weight: bold !important;
    }
    .stButton>button:hover {
        background-color: #1ed760 !important;
        transform: scale(1.02);
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 YouTube Vibe & Audio Intelligence Analyzer")
st.write("Analysez la vibe sonore et textuelle de vos fichiers audio avec l'Intelligence Artificielle.")

if "yt_audio_ready" not in st.session_state:
    st.session_state.yt_audio_ready = False
    st.session_state.yt_video_title = None
    st.session_state.yt_video_id = None


def analyze_audio_vibe(file_path):
    y, sr = librosa.load(file_path, sr=None)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    if isinstance(tempo, np.ndarray):
        tempo = float(tempo)
    mean_energy = float(np.mean(librosa.feature.rms(y=y)))
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    danceability = float(np.std(onset_env))
    mean_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    profile = "Électronique / Moderne" if mean_centroid > 2500 else "Acoustique / Organique"
    return {"tempo": tempo, "energy": mean_energy, "danceability": danceability, "profile": profile}


def analyze_nlp(file_path):
    model = whisper.load_model("tiny")
    result = model.transcribe(file_path)
    extracted_text = result['text'].strip()
    blob = TextBlob(extracted_text)
    sentiment_score = blob.sentiment.polarity
    segments = [
        {"start": s["start"], "end": s["end"], "text": s["text"].strip()}
        for s in result.get("segments", [])
    ]
    return {
        "text": extracted_text,
        "valence": sentiment_score,
        "language": result.get('language', 'Inconnue'),
        "segments": segments,
    }


def format_timestamp(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    return f"{m:02d}:{s:02d}"


def generate_summary(client, text):
    response = client.models.generate_content(
        model='gemini-flash-latest',
        contents=f"Fais un résumé structuré, clair et punchy en français du texte suivant extrait d'un enregistrement audio : {text}",
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2048,
        )
    )

    if not response.candidates:
        return None, "⚠️ Aucune réponse générée (candidats vides — probablement filtré)."

    candidate = response.candidates[0]
    finish_reason = getattr(candidate, "finish_reason", None)
    warning = None
    if finish_reason is not None and str(finish_reason) not in ("STOP", "FinishReason.STOP"):
        warning = f"⚠️ Génération incomplète ou filtrée (finish_reason: {finish_reason})."

    return getattr(response, "text", None), warning


def generate_chapters(client, segments):
    segments_text = "\n".join(
        f"[{s['start']:.1f}s -> {s['end']:.1f}s] {s['text']}" for s in segments
    )
    prompt = f"""Voici une transcription audio segmentée avec timestamps :
{segments_text}

Génère des chapitres/vibes horodatés en identifiant les changements de ton, d'intensité ou de sujet.
Réponds UNIQUEMENT avec un JSON valide, sans texte autour, au format :
[{{"timestamp": "MM:SS", "label": "titre court de la vibe/section"}}]"""

    response = client.models.generate_content(
        model='gemini-flash-latest',
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.4,
            max_output_tokens=2048,
        )
    )

    if not response.candidates or not getattr(response, "text", None):
        return None

    raw = response.text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


# with st.expander("🔍 Vérifier les modèles Gemini disponibles"):
#     if st.button("Lister mes modèles"):
#         try:
#             client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
#             models = [m.name for m in client.models.list()]
#             st.write(models)
#         except Exception as e:
#             st.error(f"Erreur : {e}")

source_mode = st.radio("Source audio", ["Fichier local", "URL YouTube"], horizontal=True)

audio_ready = False

if source_mode == "Fichier local":
    st.session_state.yt_audio_ready = False  # reset l'état YouTube si on change de mode
    uploaded_file = st.file_uploader("Choisissez un fichier audio (.wav)", type=["wav"])
    if uploaded_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(uploaded_file, format="audio/wav")
        audio_ready = True

else:
    youtube_url = st.text_input("Colle l'URL YouTube ici")
    cookiefile = st.text_input(
        "Chemin vers ton fichier cookies.txt (optionnel, réduit les blocages YouTube)",
        placeholder="ex: /Users/toi/cookies.txt"
    )
    if youtube_url and st.button("📥 Télécharger l'audio depuis YouTube"):
        with st.spinner("Téléchargement et extraction audio en cours..."):
            try:
                result = download_audio_from_youtube(
                    youtube_url,
                    output_path="temp_audio",
                    cookiefile=cookiefile if cookiefile else None
                )
                st.session_state.yt_audio_ready = True
                st.session_state.yt_video_title = result["title"]
                st.session_state.yt_video_id = result["video_id"]
            except Exception as e:
                st.session_state.yt_audio_ready = False
                st.error(f"❌ Échec du téléchargement : {e}")
                st.info("Si l'erreur mentionne un blocage ou une authentification, fournis un fichier cookies.txt exporté depuis ton navigateur.")

    if st.session_state.yt_audio_ready:
        st.success(f"Audio extrait : {st.session_state.yt_video_title}")
        st.audio("temp_audio.wav", format="audio/wav")
        audio_ready = True

if audio_ready:
    if st.button("🔥 Lancer l'Analyse d'Intelligence Multimodale"):
        with st.spinner("Analyse en cours... Veuillez patienter (Traitement Signal + NLP)..."):

            audio_metrics = analyze_audio_vibe("temp_audio.wav")
            nlp_metrics = analyze_nlp("temp_audio.wav")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Métriques Audio & Style")
                st.metric("Tempo (BPM)", f"{audio_metrics['tempo']:.2f}")
                st.metric("Énergie Globale", f"{audio_metrics['energy']:.4f}")
                st.metric("Texture Spectrale", audio_metrics['profile'])

                norm_tempo = min(audio_metrics['tempo'] / 160.0, 1.0)
                norm_energy = min(audio_metrics['energy'] * 2.0, 1.0)
                norm_dance = min(audio_metrics['danceability'] / 2.0, 1.0)
                norm_valence = (nlp_metrics['valence'] + 1) / 2

                df_radar = pd.DataFrame(dict(
                    r=[norm_tempo, norm_energy, norm_dance, norm_valence],
                    theta=['Tempo / Vitesse', 'Énergie / Puissance', 'Dansabilité / Rythme', 'Positivité / Valence']
                ))
                fig = px.line_polar(df_radar, r='r', theta='theta', line_close=True, range_r=[0, 1])
                fig.update_traces(fill='toself', line_color='#1DB954')
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.subheader("📝 Analyse Textuelle & Résumé IA")
                st.write(f"**Langue détectée :** {nlp_metrics['language'].upper()}")
                st.write(f"**Score de Positivité (Valence) :** {nlp_metrics['valence']:.2f}")

                st.markdown("### 🤖 Résumé Automatique de l'IA")
                try:
                    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                    summary_text, warning = generate_summary(client, nlp_metrics['text'])

                    if warning:
                        st.warning(warning)

                    if summary_text:
                        st.success(summary_text)
                    else:
                        st.warning("⚠️ L'IA a répondu mais aucun texte n'a pu être extrait.")

                except Exception as e:
                    st.error(f"❌ Erreur d'appel à l'API Gemini : {e}")

                st.markdown("### 🕒 Chapitres par Vibe")
                try:
                    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
                    chapters = generate_chapters(client, nlp_metrics['segments'])

                    if chapters:
                        df_chapters = pd.DataFrame(chapters)
                        st.dataframe(df_chapters, use_container_width=True, hide_index=True)
                    else:
                        st.warning("⚠️ Impossible de générer les chapitres (réponse vide ou JSON invalide).")

                except Exception as e:
                    st.error(f"❌ Erreur de génération des chapitres : {e}")

                st.text_area("Texte intégral transcrit par l'IA", nlp_metrics['text'], height=200)