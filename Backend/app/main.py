import asyncio
import json
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from . import config
from .llm import analyze_jd, clean_jd_analysis, evaluate_resume
from .parsers import ParseError, extract_text
app = FastAPI(title="Resume Shortlisting API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model": config.LLM_MODEL,
        "api_key_configured": config.api_key_configured(),
        "max_resumes": config.MAX_RESUMES,
    }
async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > config.MAX_FILE_BYTES:
        raise ParseError(f"File is larger than {config.MAX_FILE_BYTES // (1024 * 1024)} MB.")
    return data
async def _read_jd(jd_text: str, jd_file: UploadFile | None) -> str:
    if jd_file is not None and jd_file.filename:
        try:
            jd = extract_text(jd_file.filename, await _read_upload(jd_file))
        except ParseError as exc:
            raise HTTPException(400, f"Job description: {exc}")
    else:
        jd = jd_text.strip()
    if len(jd) < 50:
        raise HTTPException(400, "Please provide a job description (upload a file or paste at least a few lines).")
    return jd
@app.post("/api/analyze-jd")
async def analyze_job_description(jd_text: str = Form(""), jd_file: UploadFile | None = File(None)):
    # Step 1: HR reviews and edits this analysis (must-haves, minimum years) before screening
    if not config.api_key_configured():
        raise HTTPException(500, "LLM_API_KEY is not set. Add your key to Backend/.env and restart the server.")
    jd = await _read_jd(jd_text, jd_file)
    try:
        return await analyze_jd(jd)
    except Exception as exc:
        raise HTTPException(502, f"Could not analyse the job description: {exc}")
@app.post("/api/screen")
async def screen(
    resumes: list[UploadFile] = File(...),
    jd_text: str = Form(""),
    jd_file: UploadFile | None = File(None),
    jd_analysis: str = Form(""),
    ids: str = Form(""),
):
    # The UI sends the JD analysis HR reviewed; retries also send the original result ids so rows are replaced in place
    try:
        previous_analysis = clean_jd_analysis(json.loads(jd_analysis)) if jd_analysis else None
        resume_ids = json.loads(ids) if ids else list(range(len(resumes)))
    except (json.JSONDecodeError, ValueError):
        raise HTTPException(400, "Invalid job description analysis or retry data.")
    if len(resume_ids) != len(resumes):
        raise HTTPException(400, "Invalid retry data.")

    if not config.api_key_configured():
        raise HTTPException(500, "LLM_API_KEY is not set. Add your key to Backend/.env and restart the server.")
    if not resumes:
        raise HTTPException(400, "Please upload at least one resume.")
    if len(resumes) > config.MAX_RESUMES:
        raise HTTPException(400, f"You can upload at most {config.MAX_RESUMES} resumes at a time.")

    jd = await _read_jd(jd_text, jd_file)

    items = []
    for resume_id, file in zip(resume_ids, resumes):
        item = {"id": resume_id,"file_name": file.filename, "text": None, "error": None}
        try:
            item["text"] = extract_text(file.filename or "", await _read_upload(file))
        except ParseError as exc:
            item["error"] = str(exc)
        items.append(item)

    semaphore = asyncio.Semaphore(config.LLM_CONCURRENCY)
    async def run(item: dict, analysis: dict) -> dict:
        base = {"id": item["id"], "file_name": item["file_name"]}
        if item["error"]:
            return {**base, "status": "error", "error": item["error"]}
        try:
            async with semaphore:
                evaluation = await evaluate_resume(jd, analysis, item["text"])
            return {**base, "status": "ok", **evaluation}
        except Exception as exc:
            return {**base, "status": "error", "error": f"AI evaluation failed: {exc}"}

    async def stream():
        # Newline-delimited JSON so the UI can show each result as soon as it is ready
        yield _event("start", total=len(items))
        if previous_analysis:
            analysis = previous_analysis
        else:
            try:
                analysis = await analyze_jd(jd)
            except Exception as exc:
                yield _event("error", message=f"Could not analyse the job description: {exc}")
                return
        yield _event("jd", data=analysis)

        tasks = [asyncio.create_task(run(item, analysis)) for item in items]
        for next_done in asyncio.as_completed(tasks):
            yield _event("result", data=await next_done)
        yield _event("done")

    return StreamingResponse(stream(), media_type="application/x-ndjson")
def _event(event_type: str, **fields) -> str:
    return json.dumps({"type": event_type, **fields}) + "\n"
