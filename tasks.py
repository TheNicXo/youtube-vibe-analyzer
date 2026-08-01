from celery_app import celery_app
from analysis_core import analyze_audio_vibe, analyze_nlp, generate_summary, generate_chapters, analyze_comments_sentiment
from youtube_extractor import download_audio_from_youtube, get_video_comments
import os


@celery_app.task(bind=True)
def run_full_analysis(self, source_type, source_value, gemini_api_key, youtube_api_key=None, cookiefile=None):
    work_dir = f"/tmp/vibe_analyzer_{self.request.id}"
    os.makedirs(work_dir, exist_ok=True)
    audio_path = os.path.join(work_dir, "audio")

    video_title = None
    video_id = None

    if source_type == "youtube_url":
        self.update_state(state="PROGRESS", meta={"step": "Téléchargement YouTube..."})
        result = download_audio_from_youtube(source_value, output_path=audio_path, cookiefile=cookiefile)
        audio_file = result["path"]
        video_title = result["title"]
        video_id = result["video_id"]
    else:
        audio_file = source_value

    self.update_state(state="PROGRESS", meta={"step": "Analyse du signal audio..."})
    audio_metrics = analyze_audio_vibe(audio_file)

    self.update_state(state="PROGRESS", meta={"step": "Transcription (Whisper)..."})
    nlp_metrics = analyze_nlp(audio_file)

    self.update_state(state="PROGRESS", meta={"step": "Génération du résumé IA..."})
    summary_text = generate_summary(gemini_api_key, nlp_metrics["text"])

    self.update_state(state="PROGRESS", meta={"step": "Génération des chapitres..."})
    chapters = generate_chapters(gemini_api_key, nlp_metrics["segments"])

    comments_result = None
    if video_id and youtube_api_key:
        self.update_state(state="PROGRESS", meta={"step": "Analyse des commentaires..."})
        try:
            comments = get_video_comments(video_id, youtube_api_key, max_results=100)
            comments_result = analyze_comments_sentiment(comments)
        except Exception:
            comments_result = None

    return {
        "video_title": video_title,
        "audio_metrics": audio_metrics,
        "nlp_metrics": nlp_metrics,
        "summary_text": summary_text,
        "chapters": chapters,
        "comments_result": comments_result,
    }