import librosa
import numpy as np
import whisper
from textblob import TextBlob
import google.genai as genai
from google.genai import types
import json


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


def generate_summary(api_key, text):
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model='gemini-flash-latest',
        contents=f"Fais un résumé structuré, clair et punchy en français du texte suivant extrait d'un enregistrement audio : {text}",
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=2048)
    )
    if not response.candidates:
        return None
    return getattr(response, "text", None)


def generate_chapters(api_key, segments):
    client = genai.Client(api_key=api_key)
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
        config=types.GenerateContentConfig(temperature=0.4, max_output_tokens=2048)
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