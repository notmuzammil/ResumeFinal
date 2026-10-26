"""JD analysis and resume evaluation through an OpenAI-compatible LLM endpoint (NVIDIA NIM by default).

Flow: the JD is analysed once (weights, minimum experience, must-haves), then every resume is rated against that
same analysis so all candidates are judged by identical rules. Experience years are calculated in code from the
dates the model extracts, never by the model itself.
"""
import asyncio
import datetime
import json
import logging
import random
import re
import unicodedata

from openai import APIError, AsyncOpenAI, AuthenticationError, RateLimitError

from . import config

logger = logging.getLogger(__name__)

MAX_JD_CHARS = 8000
MAX_RESUME_CHARS = 12000

CRITERIA = ["skills", "experience", "education", "fit"]
DEFAULT_WEIGHTS = {"skills": 40, "experience": 30, "education": 15, "fit": 15}
MAX_MUST_HAVES = 6
MAX_EDITED_MUST_HAVES = 10  # HR may add a few more after reviewing the AI's list
PRESENT_WORDS = {"present", "current", "now", "till date", "to date", "ongoing"}
YEARS_PATTERN = re.compile(r"\d+(\.\d+)?\s*\+?\s*(years?|yrs?)\b", re.IGNORECASE)
# Free API tiers reject bursts with 429; wait 3s, 6s, 12s, 24s before giving up
RATE_LIMIT_RETRIES = 4

SYSTEM_PROMPT = (
    "You are a senior HR recruiter with experience hiring across all departments. "
    "Be objective, evidence-based, and practical: credit genuine qualifications while understanding "
    "real-world nuances (e.g., treat equivalent terms, reasonable experience overlap like 3.8 vs 4 years, "
    "or close skill synonyms as matching). "
    "Your goal is to screen candidates fairly, avoiding both over-strict literal disqualifications "
    "and false claims. Judge job-relevant qualifications only; ignore name, gender, age, religion, "
    "nationality, marital status, photos, and personal attributes. "
    "Respond with a single valid JSON object and nothing else."
)

JD_PROMPT = """Analyse this job description so that every candidate can be screened against it in a consistent yet practical manner.

1. Decide how important each criterion is for THIS role. Weights are whole numbers that add up to 100:
  - skills: job-specific skills, tools, knowledge and competencies
  - experience: relevant work experience, years and seniority
  - education: degrees, certifications and licenses
  - fit: fit with the role's responsibilities, domain and industry
   Base the weights on what the JD emphasises. For example, an entry-level role weighs education more,
   a senior role weighs experience more, and a licensed role (doctor, CA, pilot) weighs education/certifications more.

2. Minimum experience: if the JD states a minimum number of years of experience, put the exact number in
   "min_experience_years" and describe it in "experience_requirement" (e.g. "4+ years of accounting experience").
   Otherwise use null for both. Do not include experience years in "must_haves" (experience will be evaluated separately).

3. List MUST-HAVE requirements. A missing must-have rejects the candidate outright, so keep them few, broad and
   checkable on a CV. A must-have is something a candidate would list on a CV by name: a degree, license,
   certification, language, or a core tool, technology or skill area the job cannot be done without.
   - If the JD names a required degree or qualification (often under "Qualification" or "Education"), ALWAYS include
     it, listed first, with the alternatives the JD allows (e.g. "Bachelor's or Master's in Finance or related field").
   - When the JD lists many required tools or skills, group them into a few broad items, one per area, joined with
     "or", so a candidate needs a solid core rather than every item (e.g. "SAP or Oracle ERP", "Python or R for data
     analysis", "Java or C# backend development"). NEVER write one item that demands every tool in a list.
   - Name the skill only, without depth qualifiers such as "advanced", "expert", "design and optimization" or
     "secure" (write "Financial modelling", not "Advanced financial modelling and forecast optimization"). Depth is
     judged in the skills rating.
   - NEVER include practices, standards, methods or duties: things a person does on the job but rarely lists by name
     on a CV (e.g. following security or quality standards, best practices, testing, reviews, documentation,
     optimization, reporting routines). These are judged in the skills rating instead.
   - NEVER include soft skills or leadership (e.g. problem solving, communication, teamwork, team management);
     seniority and leadership are judged in the experience and fit ratings.
   - NEVER include preferred, "nice to have", "plus", or "good to have" skills; put them in "nice_to_haves".
   - When in doubt, leave it out.
   Maximum {max_must_haves}. An empty list is fine.

Return JSON with exactly these keys:
{{
  "job_title": "string",
  "department": "string",
  "seniority": "string, e.g. Entry, Mid, Senior, Manager",
  "weights": {{"skills": int, "experience": int, "education": int, "fit": int}},
  "weight_reason": "one sentence explaining the weights",
  "min_experience_years": number or null,
  "experience_requirement": "string or null",
  "must_haves": ["string"],
  "nice_to_haves": ["string"]
}}

=== JOB DESCRIPTION ===
{jd}
"""

RESUME_PROMPT = """Today's date is {today}. Evaluate the candidate's resume against the job description with a balanced HR mindset.

First, list every job, internship, or work position in "positions". Copy the dates as YYYY-MM
(use month 01 when only the year is given) and write "Present" for a current job. Set "relevant" to true
when the work is relevant to the role in this JD. Leave out education. Do not calculate total years yourself:
the app calculates total years from these dates.

Rate the candidate from 0 to 10 on each criterion:
- skills: job-specific skills, tools, knowledge, and competencies asked in JD
- experience: relevant work experience, years, and seniority ("Present" means {today})
- education: degrees, certifications, and licenses asked in JD
- fit: fit with role responsibilities, domain, and industry
Rating guide: 9-10 fully meets or exceeds; 7-8 meets most; 5-6 partly meets; 2-4 meets little; 0-1 nothing relevant.
Judge experience and fit against the seniority the role needs: for a senior, lead or manager role, a fresh graduate,
intern or junior with no comparable responsibility rates 0-3, however many skills they list.
Soft skills get credit ONLY when backed by concrete achievements or responsibilities on the resume.

Evaluating Requirements & Rules:
1. EXPERIENCE TOLERANCE: Be realistic. If a candidate has ~3.7 - 3.9 years of relevant experience for a 4-year requirement, treat it as substantially meeting the expectation. Do NOT harshly penalize minor experience gaps if work quality and skills align.
2. MUST-HAVES EVALUATION:
   - "met" is `true` when the resume names the requirement or a direct, standard equivalent (e.g. MySQL for "SQL databases", ACCA for "professional accounting qualification"). For a requirement with alternatives ("A or B", "A, B or C"), one alternative is enough. Only check that the skill itself appears; how deep or advanced it is belongs in the ratings, not here.
   - "met" is `false` when the resume does not mention it, or contradicts it. A degree still in progress ("expected 2027") does not meet a degree requirement.
   - "met" is `null` when you can only infer it (the candidate "likely", "probably" or "implicitly" has it). Never mark `true` on inference.
   - "evidence": when met is true, copy the exact words from the resume that show it (a short quote, no paraphrasing). Otherwise write "Not mentioned in resume" or explain briefly.

Must-have requirements:
{must_haves}

Return JSON with exactly these keys:
{{
  "candidate_name": "string",
  "email": "string or null",
  "phone": "string or null",
  "current_title": "string or null",
  "positions": [{{"title": "string", "company": "string or null", "start": "YYYY-MM", "end": "YYYY-MM or Present", "relevant": true or false}}],
  "ratings": {{"skills": int, "experience": int, "education": int, "fit": int}},
  "must_haves": [{{"requirement": "copied exactly from the list above", "met": true, false or null, "evidence": "string"}}],
  "matched_skills": ["JD requirements the resume clearly shows, by name or a direct equivalent"],
  "missing_skills": ["important JD requirements not found in the resume"],
  "strengths": ["up to 4 short points"],
  "concerns": ["up to 4 short points"],
  "summary": "2-3 sentences explaining the candidate's fit, noting any acceptable minor gaps or strong trade-offs for HR"
}}

=== JOB DESCRIPTION ===
{jd}

=== RESUME ===
{resume}
"""

_client: AsyncOpenAI | None = None
_client_loop: asyncio.AbstractEventLoop | None = None


def _get_client() -> AsyncOpenAI:
    global _client, _client_loop
    # The HTTP client is tied to the event loop it was created on, so recreate it if the loop changes
    loop = asyncio.get_running_loop()
    if _client is None or _client_loop is not loop:
        _client_loop = loop
        _client = AsyncOpenAI(
            api_key=config.LLM_API_KEY,
            base_url=config.LLM_BASE_URL,
            timeout=config.LLM_TIMEOUT,
            max_retries=0,  # rate limits are retried in _create; other failures go to the fallback model
        )
    return _client


def categorize(score: int) -> str:
    # Keep in sync with CATEGORIES in Frontend/src/categories.js
    if score > 77:
        return "Highly Suitable"
    if score > 50:
        return "Suitable"
    if score > 40:
        return "Moderate"
    return "Not Suitable"


async def analyze_jd(jd_text: str) -> dict:
    prompt = JD_PROMPT.format(jd=jd_text[:MAX_JD_CHARS], max_must_haves=MAX_MUST_HAVES)
    data = await _complete_json(prompt, max_tokens=4000)
    min_years = _number(data.get("min_experience_years"))
    if not min_years or min_years <= 0:
        min_years = None
    must_haves = _str_list(data.get("must_haves"))
    if min_years:
        # The model sometimes repeats the years requirement here; it is checked in code instead
        must_haves = [m for m in must_haves if not YEARS_PATTERN.search(m)]
    return clean_jd_analysis({**data, "min_experience_years": min_years, "must_haves": must_haves[:MAX_MUST_HAVES]})

def clean_jd_analysis(data) -> dict:
    # Also used on the analysis HR edits in the browser, so every field is checked again
    if not isinstance(data, dict):
        raise ValueError("Invalid job description analysis.")
    min_years = _number(data.get("min_experience_years"))
    if not min_years or min_years <= 0:
        min_years = None
    return {
        "job_title": _str(data.get("job_title")) or "Untitled role",
        "department": _str(data.get("department")),
        "seniority": _str(data.get("seniority")),
        "weights": _normalize_weights(data.get("weights")),
        "weight_reason": _str(data.get("weight_reason")) or "",
        "min_experience_years": min_years,
        "experience_requirement": (
            _str(data.get("experience_requirement")) or f"Minimum {min_years:g} years of relevant experience"
        ) if min_years else None,
        "must_haves": list(dict.fromkeys(_str_list(data.get("must_haves"))))[:MAX_EDITED_MUST_HAVES],
        "nice_to_haves": _str_list(data.get("nice_to_haves"))[:10],
    }

async def evaluate_resume(jd_text: str, jd: dict, resume_text: str) -> dict:
    today = datetime.date.today()
    must_haves = "\n".join(f"- {m}" for m in jd["must_haves"]) or "(none - return an empty list)"
    resume = resume_text[:MAX_RESUME_CHARS]
    prompt = RESUME_PROMPT.format(
        today=today.strftime("%B %Y"),
        must_haves=must_haves,
        jd=jd_text[:MAX_JD_CHARS],
        resume=resume,
    )
    # Thinking models spend part of max_tokens on reasoning, so leave plenty of room for the JSON
    return _normalize_resume(await _complete_json(prompt, max_tokens=6000), jd, today, resume)

async def _complete_json(prompt: str, max_tokens: int) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    models = [m for m in (config.LLM_MODEL, config.LLM_FALLBACK_MODEL) if m]
    for i, model in enumerate(models):
        try:
            return await _complete_with(model, messages, max_tokens)
        except AuthenticationError:
            raise  # a bad key fails the same way on every model
        except (APIError, ValueError) as exc:
            if i == len(models) - 1:
                raise
            logger.warning("Model %s failed (%s), falling back to %s", model, exc, models[i + 1])

async def _complete_with(model: str, messages: list[dict], max_tokens: int) -> dict:
    last_error: Exception | None = None
    for _ in range(2):  # one retry if the model returns malformed JSON
        response = await _create(model, messages, max_tokens)
        content = response.choices[0].message.content or ""
        try:
            return _extract_json(content)
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc
    raise ValueError(f"Model returned an unreadable response: {last_error}")

async def _create(model: str, messages: list[dict], max_tokens: int):
    extra = {"reasoning_effort": config.LLM_REASONING_EFFORT} if config.LLM_REASONING_EFFORT else {}
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        try:
            return await _get_client().chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=max_tokens,
                **extra,
            )
        except RateLimitError:
            if attempt == RATE_LIMIT_RETRIES:
                raise
            logger.warning("Rate limited by %s, retry %d of %d", model, attempt + 1, RATE_LIMIT_RETRIES)
            await asyncio.sleep(3 * 2**attempt + random.random())

def _extract_json(content: str) -> dict:
    # Reasoning models (e.g. DeepSeek R1) wrap their thinking in <think> tags
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.S).strip()
    match = re.search(r"\{.*\}", content, flags=re.S)
    if not match:
        raise ValueError("no JSON object found")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    return data

def _normalize_weights(raw) -> dict:
    if not isinstance(raw, dict):
        return dict(DEFAULT_WEIGHTS)
    values = {key: max(0.0, _number(raw.get(key)) or 0.0) for key in CRITERIA}
    total = sum(values.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS)
    weights = {key: round(value * 100 / total) for key, value in values.items()}
    # Rounding can leave the total at 99 or 101; correct it on the largest weight
    weights[max(weights, key=weights.get)] += 100 - sum(weights.values())
    return weights

def _normalize_resume(data: dict, jd: dict, today: datetime.date, resume_text: str) -> dict:
    raw = data.get("ratings") if isinstance(data.get("ratings"), dict) else {}
    ratings = {key: _clamp(raw.get(key), 10) for key in CRITERIA}
    weights = jd["weights"]
    # The score is always computed here from the JD weights so every candidate is scored by the same rules
    score = round(sum(weights[key] * ratings[key] for key in CRITERIA) / 10)

    # Years are calculated here from the dates, not by the model, so "Present" always means today
    positions = _work_history(data.get("positions"), today)
    relevant = [p for p in positions if p["relevant"]]
    relevant_years = _years(relevant)

    must_haves = _match_must_haves(jd["must_haves"], data.get("must_haves"), resume_text)
    if jd["min_experience_years"]:
        must_haves.insert(0, _experience_check(jd, relevant, relevant_years))
    failed = [m["requirement"] for m in must_haves if m["met"] is False]

    return {
        "candidate_name": _str(data.get("candidate_name")) or "Unknown",
        "email": _str(data.get("email")),
        "phone": _str(data.get("phone")),
        "current_title": _str(data.get("current_title")),
        "total_experience_years": _years(positions) if positions else None,
        "relevant_experience_years": relevant_years if positions else None,
        "positions": [{key: p[key] for key in ("title", "company", "period", "years", "relevant")} for p in positions],
        "score": score,
        "category": "Not Suitable" if failed else categorize(score),
        "failed_must_haves": failed,
        "must_haves": must_haves,
        "breakdown": {key: {"rating": ratings[key], "weight": weights[key]} for key in CRITERIA},
        "matched_skills": _str_list(data.get("matched_skills")),
        "missing_skills": _str_list(data.get("missing_skills")),
        "strengths": _str_list(data.get("strengths"))[:4],
        "concerns": _str_list(data.get("concerns"))[:4],
        "summary": _str(data.get("summary")) or "",
    }


def _work_history(raw, today: datetime.date) -> list[dict]:
    """Dated positions with start/end as month numbers (year * 12 + month - 1); undated ones are skipped."""
    now = today.year * 12 + today.month - 1
    positions = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        start, end = _month(item.get("start"), now), _month(item.get("end"), now)
        if start is None or end is None:
            continue
        end = min(end, now)  # future end dates (contracts, "expected") only count up to today
        if start > end:
            continue
        positions.append({
            "title": _str(item.get("title")) or "Untitled position",
            "company": _str(item.get("company")),
            "start": start,
            "end": end,
            "period": f"{_month_label(start)} – {'Present' if end == now else _month_label(end)}",
            "years": round((end - start + 1) / 12, 1),
            "relevant": _bool(item.get("relevant")) is True,
        })
    return positions

def _month(value, now: int) -> int | None:
    text = (_str(value) or "").lower()
    if text in PRESENT_WORDS:
        return now
    match = re.match(r"(\d{4})(?:-(\d{1,2}))?", text)
    if not match:
        return None
    month = int(match.group(2) or 1)
    return int(match.group(1)) * 12 + month - 1 if 1 <= month <= 12 else None

def _month_label(month: int) -> str:
    return datetime.date(month // 12, month % 12 + 1, 1).strftime("%b %Y")


def _years(positions: list[dict]) -> float:
    # Count distinct months so overlapping jobs are counted once; end months are inclusive
    months = set()
    for p in positions:
        months.update(range(p["start"], p["end"] + 1))
    return round(len(months) / 12, 1)

def _experience_check(jd: dict, relevant: list[dict], relevant_years: float) -> dict:
    jobs = "; ".join(f"{p['title']} ({p['period']})" for p in relevant)
    return {
        "requirement": jd["experience_requirement"],
        "met": relevant_years >= jd["min_experience_years"],
        "evidence": f"{relevant_years:g} years of relevant experience" + (f": {jobs}" if jobs else " found in resume"),
    }

def _match_must_haves(expected: list[str], returned, resume_text: str) -> list[dict]:
    items = [r for r in returned if isinstance(r, dict)] if isinstance(returned, list) else []
    by_text = {(_str(r.get("requirement")) or "").lower(): r for r in items}
    resume = _compact(resume_text)
    result = []
    for i, requirement in enumerate(expected):
        # Prefer an exact text match; fall back to position when the model kept the order but reworded
        item = by_text.get(requirement.lower()) or (items[i] if len(items) == len(expected) else None)
        # None means the model did not or could not check it; that never disqualifies a candidate
        met = _bool(item.get("met")) if item else None
        evidence = _str(item.get("evidence")) if item else None
        if met is True and not _quoted_in_resume(evidence, resume):
            # The model claimed a pass but its quote is not on the resume, so the pass is not trusted
            met = None
            evidence = f"Could not verify on the resume. The AI said: {evidence or 'no evidence given'}"
        result.append({"requirement": requirement, "met": met, "evidence": evidence})
    return result

def _compact(text: str) -> str:
    # Lowercase letters and digits only, so PDF artifacts ("F ull-Stack", one word per line, ligatures) still match
    return re.sub(r"[^a-z0-9]+", "", unicodedata.normalize("NFKC", text).lower())

def _quoted_in_resume(evidence: str | None, resume: str) -> bool:
    if not evidence:
        return False
    # The evidence may hold several quotes; one found on the resume is enough
    for part in re.split(r'[";|…]|\.\.\.', evidence):
        if len(_compact(part)) >= 6 and _compact(part) in resume:
            return True
    # Allow small rewording: nearly every word of the evidence must appear on the resume
    words = [w for w in re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", evidence).lower()) if len(w) > 2]
    return bool(words) and sum(w in resume for w in words) / len(words) >= 0.9

def _bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return {"true": True, "yes": True, "false": False, "no": False}.get(value.strip().lower())
    return None

def _clamp(value, limit: int) -> int:
    try:
        return max(0, min(limit, round(float(value))))
    except (TypeError, ValueError):
        return 0

def _number(value):
    try:
        return round(float(value), 1)
    except (TypeError, ValueError):
        return None

def _str(value):
    if value is None:
        return None
    text = str(value).strip()
    return None if text.lower() in {"", "null", "none", "n/a"} else text
def _str_list(value) -> list[str]:
    if not isinstance(value, list):
        return []
    return [s for s in (_str(v) for v in value) if s]
