# Resume Shortlister

HR uploads one Job Description (PDF, DOCX or pasted text) and up to 20 resumes (PDF or DOCX).
It works for any department (technology, finance, sales, HR, operations and so on).

**How it works**

1. **The AI reads the JD once.** It decides how much each criterion matters for this role (weights that add up to
   100%), finds the **minimum years of experience**, and lists the other **must-haves**: only mandatory items a
   CV can show, such as degrees, licenses, certifications, specific tools and required languages. HR sees this
   on screen.
2. **Every CV is checked against that same analysis**, so all candidates are judged by the same rules.
   The AI rates Skills, Experience, Education & certifications and Role fit from 0 to 10, checks each must-have
   with evidence from the CV, and the app calculates the weighted score.
3. **Experience is calculated by the app, not the AI.** The AI lists each job with its dates and marks whether it
   is relevant to the role. The app counts the months, treating "Present" as today's date and counting overlapping
   jobs once, and checks the minimum years.
4. **Generic soft skills** (problem solving, communication, teamwork and similar) are never must-haves, even when
   the JD says "required", because almost every CV claims them. They only earn credit when the CV backs them with
   real evidence, such as an achievement.
5. Each candidate gets a category:

| Category        | Rule                                  |
|-----------------|---------------------------------------|
| Highly Suitable | Score 80–100                          |
| Suitable        | Score 60–79                           |
| Moderate        | Score 40–59                           |
| Not Suitable    | Score under 40, **or below the minimum experience, or missing any must-have** |

For each candidate HR sees the must-have check, matched and missing skills, strengths, concerns and a short
summary. The full shortlist can be exported to Excel (CSV). About 20 CVs take under a minute.

```
Frontend/   React + Vite UI           (http://localhost:5173)
Backend/    Python FastAPI + LLM      (http://localhost:8000)
```

## 1. Get an API key

The backend works with any **OpenAI-compatible** API. By default it uses NVIDIA's hosted open-source models:

1. Go to https://build.nvidia.com and sign in.
2. Open any model and click **Get API Key**.
3. Put the key in `Backend/.env`:

```
LLM_API_KEY=nvapi-...
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=z-ai/glm-5.3-flash
LLM_FALLBACK_MODEL=openai/gpt-oss-20b
LLM_REASONING_EFFORT=low
LLM_CONCURRENCY=3
```

NVIDIA retires free models from time to time. If the main model stops working, the fallback model is used
automatically. To replace a retired model, pick a new one from https://build.nvidia.com and change `LLM_MODEL`.
Other providers, such as Groq, OpenRouter, Together or a local Ollama (`http://localhost:11434/v1`), work too.
Just change `LLM_BASE_URL` and `LLM_MODEL`.

## 2. Run the backend

```powershell
cd Backend
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env      # then edit .env and add your key
uvicorn app.main:app --reload --port 8000
```

Check it at http://localhost:8000/api/health. The API docs are at http://localhost:8000/docs.

## 3. Run the frontend

```powershell
cd Frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Notes

- **Scanned PDFs** (images with no text layer) can't be read. They show up as "Could not process".
- Old **.doc** files aren't supported. Save them as .docx first.
- Results aren't stored anywhere. Export the CSV if you need to keep them.
- `LLM_CONCURRENCY` (default 3) controls how many resumes are evaluated at the same time. NVIDIA's free tier
  rejects bigger bursts; the app automatically waits and retries when that happens.
- If the JD lists skills under a heading like "Required:", the AI treats them as must-haves, and candidates
  missing any of them become Not Suitable. Put optional skills under "Nice to have" or "Preferred".

## Production

- Build the frontend with `npm run build` and serve `Frontend/dist` with any web server (IIS, nginx).
- Set `VITE_API_URL` in `Frontend/.env` to the backend URL before building, and add the frontend URL to
  `CORS_ORIGINS` in `Backend/.env`.
- Run the backend with `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2`.
