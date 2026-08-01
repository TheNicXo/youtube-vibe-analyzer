from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from tasks import run_full_analysis
from celery_app import celery_app
import shutil
import uuid
import os

app = FastAPI(title="YouTube Vibe Analyzer API")


@app.post("/analyze/youtube")
def analyze_youtube(url: str = Form(...), gemini_api_key: str = Form(...),
                     youtube_api_key: str = Form(None), cookiefile: str = Form(None)):
    task = run_full_analysis.delay("youtube_url", url, gemini_api_key, youtube_api_key, cookiefile)
    return {"task_id": task.id}


@app.post("/analyze/file")
def analyze_file(file: UploadFile = File(...), gemini_api_key: str = Form(...)):
    temp_path = f"/tmp/upload_{uuid.uuid4()}.wav"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    task = run_full_analysis.delay("local_file", temp_path, gemini_api_key, None, None)
    return {"task_id": task.id}


@app.get("/status/{task_id}")
def get_status(task_id: str):
    result = celery_app.AsyncResult(task_id)

    if result.state == "PENDING":
        return {"state": "PENDING"}
    elif result.state == "PROGRESS":
        return {"state": "PROGRESS", "step": result.info.get("step")}
    elif result.state == "SUCCESS":
        return {"state": "SUCCESS", "result": result.result}
    elif result.state == "FAILURE":
        return {"state": "FAILURE", "error": str(result.info)}
    else:
        return {"state": result.state}