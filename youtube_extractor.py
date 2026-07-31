import yt_dlp
import os


def download_audio_from_youtube(url, output_path="temp_audio", cookiefile=None):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_path}.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
        'quiet': True,
        'no_warnings': True,
        'extractor_args': {
            'youtube': {'player_client': ['android', 'ios', 'web']}
        },
    }

    if cookiefile:
        ydl_opts['cookiefile'] = cookiefile

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        video_id = info.get('id')
        title = info.get('title')

    final_path = f"{output_path}.wav"
    if not os.path.exists(final_path):
        raise FileNotFoundError("Le téléchargement a échoué : fichier audio introuvable après extraction.")

    return {"path": final_path, "video_id": video_id, "title": title}