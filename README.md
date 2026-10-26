# Resume Shortlister

HR uploads one Job Description (PDF, DOCX or pasted text) and up to 20 resumes (PDF or DOCX).
It works for any department (technology, finance, sales, HR, operations and so on).

**How it works**

1. **The AI reads the JD once.** It decides how much each criterion matters for this role (weights that add up to
   100%), finds the **minimum years of experience**, and lists the other **must-haves**: only mandatory items a
   CV can show, such as degrees, licenses, certifications, specific tools and required languages.
2. **HR reviews the must-haves before screening.** HR fixes the wording, removes what does not matter, adds what is
   missing and sets the minimum years of experience. A missing must-have is shown as a warning and does not change
   the category. A candidate missing a **Deal-breaker** is Not Suitable whatever the score. The AI pre-ticks only
   formal qualifications the JD strictly requires (a degree, a professional license, a legally required
   certification), never skills or tools, and not when the JD accepts "or equivalent experience". HR can untick
   them or tick other true requirements.
   For a degree, the AI copies its end date and the app checks it against today: a degree still in progress
   ("Expected 2029", or an end date after today) does not meet the requirement, and one that already ended does.
   After the results come in, **Edit must-haves** goes back to this step.
3. **Every CV is checked against that reviewed analysis**, so all candidates are judged by the same rules.
   The AI rates Skills, Experience, Education & certifications and Role fit from 0 to 10, checks each must-have
   with evidence from the CV, and the app calculates the weighted score.
4. **Experience is calculated by the app, not the AI.** The AI lists each job with its dates and marks whether it
   is relevant to the role. The app counts the months, treating "Present" as today's date and counting overlapping
   jobs once, and checks the minimum years. Being below the minimum is shown as a warning.
5. **Generic soft skills** (problem solving, communication, teamwork and similar) are never must-haves, even when
   the JD says "required", because almost every CV claims them. They only earn credit when the CV backs them with
   real evidence, such as an achievement.
6. Each candidate gets a category from the score:

| Category        | Rule                                  |
|-----------------|---------------------------------------|
| Highly Suitable | Score 78–100                          |
| Suitable        | Score 51–77                           |
| Moderate        | Score 41–50                           |
| Not Suitable    | Score 40 or under, **or missing a must-have HR marked as a deal-breaker** |

For each candidate HR sees the must-have check (missing must-haves are listed as warnings, missing deal-breakers
in red), matched and missing skills, strengths, concerns and a short summary. The full shortlist can be exported to Excel (CSV). About 20 CVs take under a minute.

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
- The AI is asked to keep must-haves broad: a required degree is always one, long lists of required tools are
  grouped by area (e.g. "Node.js/Express or PHP/Laravel backend"), and practices CVs rarely name (security standards,
  code reviews, load testing) are left to the score. Smaller models do not always follow this, which is why HR
  reviews the list before screening.
- A must-have only passes when the AI quotes the resume and the app finds that quote in the resume text. Anything
  the AI can only infer, or whose quote is not on the resume, shows as "?" (unverified): it does not reject the
  candidate even when it is a deal-breaker, but it is not counted as met either. An unverified deal-breaker is
  shown as "? Check manually" on the candidate's row so HR checks the resume. The same happens when the AI skips a
  requirement, says a degree is unfinished although its end date has passed, or gives a degree date that is not on
  the resume or falls in the current year without a month.

## Production

- Build the frontend with `npm run build` and serve `Frontend/dist` with any web server (IIS, nginx).
- Set `VITE_API_URL` in `Frontend/.env` to the backend URL before building, and add the frontend URL to
  `CORS_ORIGINS` in `Backend/.env`.
- Run the backend with `uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2`.
