# 🎵 YouTube Vibe & Audio Intelligence Analyzer

![Subject](https://img.shields.io/badge/Subject-Audio_DSP_/_NLP_/_Generative_AI-blue)
![Type](https://img.shields.io/badge/Type-Multimodal_Pipeline-orange)
![Technology](https://img.shields.io/badge/AI-Gemini_2.5_%2B_Whisper-red)
![Architecture](https://img.shields.io/badge/Architecture-FastAPI_%2B_Celery_%2B_Redis-teal)
![Status](https://img.shields.io/badge/Status-Portfolio_Ready-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)

Application web multimodale qui analyse la « vibe » sonore et textuelle de n'importe quel contenu YouTube (musique, podcast, vlog), en recalculant localement des métriques audio inspirées de Spotify, en transcrivant et analysant le sentiment du contenu parlé/chanté, et en confrontant la vibe de la vidéo à celle de son public via les commentaires.

---

## 📌 Vue d'ensemble

Plutôt que de s'appuyer sur une API tierce type Spotify pour obtenir des métriques audio (tempo, énergie, dansabilité), ce projet **recalcule ces signaux directement depuis le signal brut** via traitement du signal (`librosa`), puis les enrichit avec :
1. **Transcription & NLP** : Speech-to-Text local (Whisper) + analyse de sentiment (TextBlob).
2. **IA Générative** : résumé structuré et découpage en chapitres horodatés par vibe via Gemini.
3. **Analyse sociale** : sentiment agrégé des commentaires YouTube, comparé à la vibe intrinsèque de la vidéo.
4. **Export & Architecture** : rapport PDF téléchargeable, et démonstration d'une architecture asynchrone de production (FastAPI + Celery + Redis) en parallèle de l'interface Streamlit.

---

## 🛠 Stack Technique

* **Language :** Python 3.9+
* **Frontend & Visualisation :** Streamlit, Plotly (thème sombre Spotify custom)
* **Audio Signal Processing :** `librosa`, `numpy` (tempo, énergie RMS, dansabilité, centroïde spectral)
* **NLP & Speech-to-Text :** `openai-whisper` (modèle `tiny`, local), `textblob` (sentiment)
* **IA Générative :** Google Gemini API (`gemini-flash-latest`) — résumés, chapitres, via `google-genai`
* **Ingestion YouTube :** `yt-dlp` (extraction audio only), YouTube Data API v3 (commentaires)
* **Export :** `reportlab` + `kaleido` (rapport PDF avec graphiques intégrés)
* **Architecture Asynchrone :** FastAPI, Celery, Redis (traitement en tâche de fond, suivi de progression)
* **Environnement :** `venv`, secrets gérés via `.streamlit/secrets.toml`

---

## 🛠️ Pipeline

1. **Ingestion (`youtube_extractor.py`) :**
   * Téléchargement audio-only depuis une URL YouTube ou upload direct d'un `.wav`.
   * Cache local par `video_id` pour éviter les re-téléchargements redondants.
   * Gestion optionnelle de cookies de session pour les vidéos restreintes.
2. **Traitement du Signal (Data Science) :**
   * Extraction du tempo (BPM), de l'énergie RMS moyenne, de la dansabilité (variance de l'onset strength) et de la texture spectrale (centroïde).
3. **NLP & Transcription :**
   * Speech-to-Text local via Whisper avec détection automatique de la langue.
   * Score de valence (sentiment) via TextBlob, sur le texte transcrit et sur les commentaires publics.
4. **IA Générative (Gemini) :**
   * Résumé structuré et synthétique du contenu.
   * Génération de chapitres horodatés par changement de vibe/ton.
5. **Analyse Sociale :**
   * Récupération et scoring de 100 commentaires via YouTube Data API v3.
   * Comparaison visuelle Vidéo vs Public sur un radar chart + répartition en camembert.
6. **Restitution :**
   * Dashboard interactif Streamlit avec cache de session (pas de recalcul redondant).
   * Export PDF complet (métriques, radar, résumé, chapitres, transcription, analyse commentaires).
7. **Architecture de Production (démonstration) :**
   * Endpoints FastAPI (`/analyze/youtube`, `/analyze/file`, `/status/{task_id}`) déclenchant des tâches Celery asynchrones sur Redis, avec suivi de progression en temps réel — logique métier découplée dans `analysis_core.py`.

---

## 📁 Structure du Projet

```text
.
├── app.py                    # Interface Streamlit (point d'entrée principal)
├── youtube_extractor.py      # Téléchargement audio YouTube + cache + commentaires
├── report_generator.py       # Génération du rapport PDF
├── analysis_core.py          # Logique métier réutilisable (signal, NLP, IA générative)
├── celery_app.py             # Configuration Celery (broker/backend Redis)
├── tasks.py                  # Tâche Celery orchestrant le pipeline complet
├── main.py                   # API FastAPI (endpoints asynchrones)
├── requirements.txt          # Dépendances Python
├── .streamlit/secrets.toml   # Clés API (non versionné)
└── .gitignore                # Exclusion des secrets, venv, cache audio, cookies
```

---

## 🚀 Installation & Lancement

### Prérequis communs

```bash
brew install ffmpeg          # requis par Whisper
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m textblob.download_corpora
```

Configurer `.streamlit/secrets.toml` :

```toml
GEMINI_API_KEY = "ta_clé_gemini"
YOUTUBE_API_KEY = "ta_clé_youtube_data_v3"
```

### Mode 1 — Interface Streamlit (démo simple, déployable telle quelle)

```bash
streamlit run app.py
```

### Mode 2 — Architecture asynchrone complète (FastAPI + Celery + Redis)

Nécessite trois process en parallèle :

```bash
# Terminal 1 — Broker Redis
brew install redis
redis-server

# Terminal 2 — Worker Celery (pool solo recommandé sur macOS/Apple Silicon)
source venv/bin/activate
celery -A celery_app worker --loglevel=info --pool=solo

# Terminal 3 — API FastAPI
source venv/bin/activate
uvicorn main:app --reload
```

Test rapide :

```bash
curl -X POST http://localhost:8000/analyze/youtube \
  -F "url=https://www.youtube.com/watch?v=VIDEO_ID" \
  -F "gemini_api_key=TA_CLE" \
  -F "youtube_api_key=TA_CLE_YOUTUBE"

curl http://localhost:8000/status/{task_id}
```

---

## ⚠️ Notes techniques

* **Extraction audio uniquement** (pas de vidéo) : préserve la bande passante, un point d'attention particulier en usage mobile/partage de connexion.
* **`yt-dlp` non figé en version** (`requirements.txt`) : YouTube fait évoluer ses défenses en continu, une version datée casse tôt ou tard.
* **Cookies de session** : à utiliser uniquement en cas de blocage (vidéo restreinte) — leur usage systématique peut au contraire provoquer des blocages 403 avec certains exports mal formés.
* **Deux architectures cohabitent volontairement** : Streamlit autonome pour la simplicité de déploiement, FastAPI/Celery/Redis comme brique de démonstration d'une architecture de production asynchrone.