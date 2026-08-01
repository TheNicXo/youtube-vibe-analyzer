import streamlit as st
import librosa
import numpy as np
import whisper
from textblob import TextBlob
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import google.genai as genai
from google.genai import types
import json
from youtube_extractor import download_audio_from_youtube, get_video_comments
from report_generator import generate_pdf_report


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
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        background-color: #1a1a1a;
        padding: 10px 18px;
        border-radius: 16px;
        border: 1px solid #262626;
        display: inline-flex;
        gap: 28px;
    }
    div[data-testid="stRadio"] label {
        font-size: 1rem;
    }
    div[data-testid="stRadio"] label:hover {
        color: #1DB954;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🚀 YouTube Vibe & Audio Intelligence Analyzer")
st.write("Analysez la vibe sonore et textuelle de vos fichiers audio avec l'Intelligence Artificielle.")

if "yt_audio_ready" not in st.session_state:
    st.session_state.yt_audio_ready = False
    st.session_state.yt_video_title = None
    st.session_state.yt_video_id = None
    st.session_state.yt_url_confirmed = None
    st.session_state.yt_cookiefile_confirmed = None
    st.session_state.yt_from_cache = False
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None


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


def analyze_comments_sentiment(comments):
    if not comments:
        return None
    scores = [TextBlob(c["text"]).sentiment.polarity for c in comments]
    positive = sum(1 for s in scores if s > 0.1)
    negative = sum(1 for s in scores if s < -0.1)
    neutral = len(scores) - positive - negative
    return {
        "average_valence": sum(scores) / len(scores),
        "count": len(scores),
        "positive_count": positive,
        "neutral_count": neutral,
        "negative_count": negative,
    }


def describe_comments_sentiment(comments_result, video_valence):
    avg = comments_result["average_valence"]
    count = comments_result["count"]
    pos_pct = comments_result["positive_count"] / count * 100
    neu_pct = comments_result["neutral_count"] / count * 100
    neg_pct = comments_result["negative_count"] / count * 100

    if avg > 0.15:
        qualifier = "globalement positif"
    elif avg < -0.15:
        qualifier = "globalement négatif"
    else:
        qualifier = "globalement neutre"

    if avg > video_valence + 0.05:
        comparison = "plus positif que"
    elif avg < video_valence - 0.05:
        comparison = "moins positif que"
    else:
        comparison = "similaire à"

    para1 = (
        f"Sur {count} commentaires analysés, le ressenti du public est {qualifier} "
        f"(valence moyenne de {avg:.2f} sur une échelle de -1 à +1)."
    )
    para2 = f"Répartition : {pos_pct:.0f}% positifs, {neu_pct:.0f}% neutres, {neg_pct:.0f}% négatifs."
    para3 = f"Le public apparaît {comparison} le contenu de la vidéo lui-même (valence {video_valence:.2f})."

    return [para1, para2, para3]


RADAR_CAPTION = (
    "L'axe Positivité/Valence est normalisé de -1..1 vers 0..1 : 0.5 = neutre, 0 = très négatif, "
    "1 = très positif. Seul cet axe est comparable entre vidéo et public — les autres axes n'ont "
    "pas d'équivalent dans les commentaires texte. Le pointillé indique la portée maximale de l'axe "
    "pour situer l'écart."
)

COOKIE_TOOLTIP = (
    "Optionnel — laisser vide fonctionne dans la majorité des cas et évite les erreurs 403 "
    "parfois causées par des cookies mal exportés ou expirés. Ne renseigne ce champ que si le "
    "téléchargement échoue avec un message mentionnant une restriction d'âge ou une vidéo "
    "privée/non répertoriée."
)

source_mode = st.radio("Source audio", ["Fichier local", "URL YouTube"], horizontal=True)

audio_ready = False

if source_mode == "Fichier local":
    st.session_state.yt_audio_ready = False
    st.session_state.yt_video_id = None
    st.session_state.yt_url_confirmed = None
    uploaded_file = st.file_uploader("Choisissez un fichier audio (.wav)", type=["wav"])
    if uploaded_file is not None:
        with open("temp_audio.wav", "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.audio(uploaded_file, format="audio/wav")
        audio_ready = True

else:
    with st.form("youtube_input_form"):
        youtube_url_input = st.text_input("Colle l'URL YouTube ici")

        spacer_col, cookie_col = st.columns([1, 10])
        with cookie_col:
            st.markdown(
                f"Chemin vers ton fichier cookies.txt "
                f"<span title=\"{COOKIE_TOOLTIP}\" style='cursor: help; display: inline-flex; align-items: center; "
                f"justify-content: center; width: 16px; height: 16px; border-radius: 50%; border: 1.5px solid #1DB954; "
                f"color: #1DB954; font-size: 11px; font-weight: bold;'>?</span>",
                unsafe_allow_html=True
            )
            cookiefile_input = st.text_input(
                "cookies_path_input",
                placeholder="ex: /Users/toi/cookies.txt",
                label_visibility="collapsed"
            )

        url_submitted = st.form_submit_button("✅ Valider l'URL (ou appuie sur Entrée)")

    if url_submitted:
        st.session_state.yt_url_confirmed = youtube_url_input
        st.session_state.yt_cookiefile_confirmed = cookiefile_input
        st.session_state.yt_audio_ready = False

    if st.session_state.yt_url_confirmed:
        if st.button("📥 Télécharger l'audio depuis YouTube"):
            with st.spinner("Téléchargement et extraction audio en cours..."):
                try:
                    result = download_audio_from_youtube(
                        st.session_state.yt_url_confirmed,
                        output_path="temp_audio",
                        cookiefile=st.session_state.yt_cookiefile_confirmed if st.session_state.yt_cookiefile_confirmed else None
                    )
                    st.session_state.yt_audio_ready = True
                    st.session_state.yt_video_title = result["title"]
                    st.session_state.yt_video_id = result["video_id"]
                    st.session_state.yt_from_cache = result.get("from_cache", False)
                    st.session_state.analysis_results = None
                except Exception as e:
                    st.session_state.yt_audio_ready = False
                    st.error(f"❌ Échec du téléchargement : {e}")
                    st.info("Si l'erreur mentionne un blocage ou une authentification, fournis un fichier cookies.txt exporté depuis ton navigateur.")

        if st.session_state.yt_audio_ready:
            cache_note = " (depuis le cache local — aucune donnée consommée)" if st.session_state.get("yt_from_cache") else ""
            st.success(f"Audio extrait : {st.session_state.yt_video_title}{cache_note}")
            st.audio("temp_audio.wav", format="audio/wav")
            audio_ready = True

if audio_ready:
    if st.button("🔥 Lancer l'Analyse d'Intelligence Multimodale"):
        with st.spinner("Analyse en cours... Veuillez patienter (Traitement Signal + NLP)..."):

            audio_metrics = analyze_audio_vibe("temp_audio.wav")
            nlp_metrics = analyze_nlp("temp_audio.wav")

            norm_tempo = min(audio_metrics['tempo'] / 160.0, 1.0)
            norm_energy = min(audio_metrics['energy'] * 2.0, 1.0)
            norm_dance = min(audio_metrics['danceability'] / 2.0, 1.0)
            norm_valence = (nlp_metrics['valence'] + 1) / 2

            categories = ['Tempo / Vitesse', 'Énergie / Puissance', 'Dansabilité / Rythme', 'Positivité / Valence']
            raw_values = [audio_metrics['tempo'], audio_metrics['energy'], audio_metrics['danceability'], nlp_metrics['valence']]
            video_values = [norm_tempo, norm_energy, norm_dance, norm_valence]

            video_id = st.session_state.get("yt_video_id")
            comments_result = None
            comments_error = None
            comments_paragraphs = None

            if video_id:
                try:
                    youtube_api_key = st.secrets["YOUTUBE_API_KEY"]
                    comments = get_video_comments(video_id, youtube_api_key, max_results=100)
                    comments_result = analyze_comments_sentiment(comments)
                    if comments_result:
                        comments_paragraphs = describe_comments_sentiment(comments_result, nlp_metrics['valence'])
                except PermissionError as e:
                    comments_error = str(e)
                except Exception as e:
                    comments_error = f"Erreur API YouTube : {e}"

            has_pie = comments_result is not None

            fig = make_subplots(
                rows=1, cols=2,
                specs=[[{'type': 'polar'}, {'type': 'domain'}]],
                column_widths=[0.62, 0.38],
                subplot_titles=('Vibe Vidéo vs Public', 'Répartition des commentaires' if has_pie else '')
            )

            fig.add_trace(go.Scatterpolar(
                r=video_values + [video_values[0]],
                theta=categories + [categories[0]],
                fill='toself',
                name='Vidéo',
                line=dict(color='#1DB954', width=3),
                fillcolor='rgba(29, 185, 84, 0.35)',
                marker=dict(size=6),
                customdata=raw_values + [raw_values[0]],
                hovertemplate='%{theta}<br>Normalisé: %{r:.2f}<br>Brut: %{customdata:.2f}<extra></extra>',
            ), row=1, col=1)

            if comments_result:
                norm_public_valence = (comments_result["average_valence"] + 1) / 2

                fig.add_trace(go.Scatterpolar(
                    r=[0, norm_public_valence],
                    theta=['Positivité / Valence', 'Positivité / Valence'],
                    mode='lines+markers',
                    name='Public (commentaires)',
                    line=dict(color='#FF4B4B', width=6),
                    marker=dict(size=10, color='#FF4B4B'),
                    hovertemplate=f'Public<br>Normalisé: %{{r:.2f}}<br>Brut: {comments_result["average_valence"]:.2f}<extra></extra>',
                ), row=1, col=1)

                fig.add_trace(go.Scatterpolar(
                    r=[0, 1],
                    theta=['Positivité / Valence', 'Positivité / Valence'],
                    mode='lines',
                    line=dict(color='#FF4B4B', width=1, dash='dot'),
                    showlegend=False,
                    hoverinfo='skip',
                ), row=1, col=1)

                fig.add_trace(go.Pie(
                    labels=['Positifs', 'Neutres', 'Négatifs'],
                    values=[
                        comments_result['positive_count'],
                        comments_result['neutral_count'],
                        comments_result['negative_count'],
                    ],
                    marker=dict(colors=['#1DB954', '#9E9E9E', '#FF4B4B']),
                    hole=0.4,
                    textinfo='percent',
                    showlegend=False,
                ), row=1, col=2)

            fig.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1]),
                    bgcolor='white',
                ),
                showlegend=True,
                legend=dict(orientation='h', yanchor='bottom', y=1.12),
                margin=dict(t=80),
            )

            try:
                radar_chart_bytes = fig.to_image(format="png", engine="kaleido", width=1100, height=550)
            except Exception:
                radar_chart_bytes = None

            client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
            summary_text, summary_warning = generate_summary(client, nlp_metrics['text'])
            chapters = generate_chapters(client, nlp_metrics['segments'])

            pdf_bytes = None
            pdf_error = None
            try:
                pdf_bytes = generate_pdf_report(
                    video_title=st.session_state.get("yt_video_title"),
                    audio_metrics=audio_metrics,
                    nlp_metrics=nlp_metrics,
                    summary_text=summary_text,
                    chapters=chapters,
                    radar_chart_bytes=radar_chart_bytes,
                    radar_caption=RADAR_CAPTION,
                    transcript_text=nlp_metrics['text'],
                    comments_paragraphs=comments_paragraphs,
                )
            except Exception as e:
                pdf_error = str(e)

            st.session_state.analysis_results = {
                "audio_metrics": audio_metrics,
                "nlp_metrics": nlp_metrics,
                "summary_text": summary_text,
                "summary_warning": summary_warning,
                "chapters": chapters,
                "comments_result": comments_result,
                "comments_error": comments_error,
                "comments_paragraphs": comments_paragraphs,
                "fig": fig,
                "pdf_bytes": pdf_bytes,
                "pdf_error": pdf_error,
            }

if st.session_state.analysis_results:
    results = st.session_state.analysis_results
    audio_metrics = results["audio_metrics"]
    nlp_metrics = results["nlp_metrics"]

    action_spacer, action_col = st.columns([4, 1])
    with action_col:
        if results["pdf_bytes"]:
            st.download_button(
                label="⬇️ Export PDF",
                data=results["pdf_bytes"],
                file_name="vibe_analysis_report.pdf",
                mime="application/pdf",
            )
        elif results["pdf_error"]:
            st.caption(f"PDF indisponible : {results['pdf_error']}")

    st.write("")
    st.write("")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📊 Métriques Audio & Style")
        st.metric("Tempo (BPM)", f"{audio_metrics['tempo']:.2f}")
        st.metric("Énergie Globale", f"{audio_metrics['energy']:.4f}")
        st.metric("Texture Spectrale", audio_metrics['profile'])

        st.plotly_chart(results["fig"], use_container_width=True)
        st.caption(RADAR_CAPTION)

        if results["comments_paragraphs"]:
            st.markdown("### 💬 Analyse des Commentaires du Public")
            for para in results["comments_paragraphs"]:
                st.write(para)
        elif results["comments_error"]:
            st.info(f"ℹ️ Analyse des commentaires indisponible : {results['comments_error']}")

        st.markdown("### 🕒 Chapitres par Vibe")
        if results["chapters"]:
            df_chapters = pd.DataFrame(results["chapters"])
            st.dataframe(df_chapters, use_container_width=True, hide_index=True)
        else:
            st.warning("⚠️ Impossible de générer les chapitres (réponse vide ou JSON invalide).")

    with col2:
        st.subheader("📝 Analyse Textuelle & Résumé IA")
        st.write(f"**Langue détectée :** {nlp_metrics['language'].upper()}")
        st.write(f"**Score de Positivité (Valence) :** {nlp_metrics['valence']:.2f} (brut, -1 à +1)")

        st.markdown("### 🤖 Résumé Automatique de l'IA")
        if results["summary_warning"]:
            st.warning(results["summary_warning"])
        if results["summary_text"]:
            st.success(results["summary_text"])
        else:
            st.warning("⚠️ L'IA a répondu mais aucun texte n'a pu être extrait.")

        st.text_area("Texte intégral transcrit par l'IA", nlp_metrics['text'], height=200)