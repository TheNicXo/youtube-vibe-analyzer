import yt_dlp
import os
import shutil
import requests

CACHE_DIR = "audio_cache"


def download_audio_from_youtube(url, output_path="temp_audio", cookiefile=None, use_cache=True):
    os.makedirs(CACHE_DIR, exist_ok=True)

    video_id = None
    title = None

    if use_cache:
        try:
            probe_opts = {
                'quiet': True,
                'no_warnings': True,
                'skip_download': True,
                'extractor_args': {'youtube': {'player_client': ['ios', 'android', 'web']}},
            }
            if cookiefile:
                probe_opts['cookiefile'] = cookiefile
            with yt_dlp.YoutubeDL(probe_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                video_id = info.get('id')
                title = info.get('title')
        except Exception:
            video_id = None

        if video_id:
            cached_file = os.path.join(CACHE_DIR, f"{video_id}.wav")
            if os.path.exists(cached_file):
                final_path = f"{output_path}.wav"
                shutil.copyfile(cached_file, final_path)
                return {"path": final_path, "video_id": video_id, "title": title, "from_cache": True}

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
            'youtube': {'player_client': ['ios', 'android', 'web']}
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

    cached_file = os.path.join(CACHE_DIR, f"{video_id}.wav")
    try:
        shutil.copyfile(final_path, cached_file)
    except Exception:
        pass

    return {"path": final_path, "video_id": video_id, "title": title, "from_cache": False}


def get_video_comments(video_id, api_key, max_results=100):
    url = "https://www.googleapis.com/youtube/v3/commentThreads"
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": min(max_results, 100),
        "order": "relevance",
        "textFormat": "plainText",
        "key": api_key,
    }

    response = requests.get(url, params=params, timeout=10)

    if response.status_code == 403:
        raise PermissionError("Commentaires désactivés ou inaccessibles pour cette vidéo.")
    response.raise_for_status()

    data = response.json()
    comments = []
    for item in data.get("items", []):
        snippet = item["snippet"]["topLevelComment"]["snippet"]
        comments.append({
            "text": snippet.get("textDisplay", ""),
            "likes": snippet.get("likeCount", 0),
        })

    return comments