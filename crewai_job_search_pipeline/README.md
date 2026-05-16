# CrewAI Job Search Pipeline

A production-ready multi-agent job search pipeline built on **CrewAI Flows + Crews**.  
Given a job query and your resume details, it finds live postings, scores fit, tailors
your resume per role, and surfaces hiring manager contacts.

---

## Architecture recap

```
User Input (query + resume)
        │
        ▼  Flow @start — deterministic
  Parse & Enrich Input → CandidateProfile
        │
        ▼  Flow @listen — agent intelligence
  SearchCrew (sequential)
    ├─ Job Researcher       → live job listings
    ├─ Company Analyst      → company profiles
    └─ JD Extractor         → structured requirements
        │
        ▼  Flow @router — deterministic
  Quality Gate  ──(retry up to 2×)──▶ retry_search
        │ pass
        ▼  Flow @listen — agent intelligence
  MatchCrew (sequential)
    ├─ Skills Matcher       → overlap scores
    ├─ Fit Scorer           → holistic score 0–100
    ├─ Tailoring Agent      → per-job resume + cover letter
    └─ Hiring Manager Finder → recruiter contacts
        │
        ▼  Flow @listen — deterministic
  Validate & Guardrails   (clamp scores, de-dupe, strip fake URLs)
        │
        ▼  Flow @listen — deterministic
  Generate Market Insights
        │
        ▼
  JobSearchReport  ✅
```

---

## 1. Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.11 + |
| pip | 23 + |

---

## 2. Install dependencies

```bash
# From the repo root (one level above crewai_job_search/)
pip install crewai crewai-tools pydantic python-dotenv
```

Full pinned install (recommended for production):

```bash
pip install \
  "crewai>=0.80.0" \
  "crewai-tools>=0.30.0" \
  "pydantic>=2.7" \
  "python-dotenv>=1.0"
```

---

## 3. Set environment variables

Create a `.env` file next to `crewai_job_search/`:

```dotenv
# .env

# LLM — OpenAI (default) or swap to any litellm-compatible provider
OPENAI_API_KEY=sk-...

# Web search — get a free key at https://serper.dev
SERPER_API_KEY=...

# Optional: enhanced research
# EXA_API_KEY=...
```

Load it before running:

```bash
export $(cat .env | xargs)
# or just let python-dotenv pick it up automatically
```

---

## 4. Project structure

```
.
├── .env
├── crewai_job_search/
│   ├── __init__.py
│   ├── models.py       # All Pydantic schemas
│   ├── tools.py        # Custom CrewAI tools
│   ├── crews.py        # SearchCrew + MatchCrew definitions
│   ├── flow.py         # Deterministic Flow backbone
│   └── main.py         # Entrypoint + example resume
└── README.md
```

---

## 5. Run the built-in example

`main.py` ships with a pre-built sample resume (Alex Rivera, Senior Python Engineer).

```bash
# From the directory containing crewai_job_search/
python -m crewai_job_search.main
```

You'll see agent reasoning streamed to the terminal, then a formatted report printed
and `job_search_report.json` written to disk.

---

## 6. Use your own resume programmatically

```python
from crewai_job_search.models import (
    JobSearchInput, ResumeProfile,
    WorkExperience, Education,
    JobType, SeniorityLevel,
)
from crewai_job_search.main import run_job_search, print_report

# --- Build your resume ---
my_resume = ResumeProfile(
    name="Jane Smith",
    summary="Full-stack engineer with 4 years in React + Node.js...",
    skills=["TypeScript", "React", "Node.js", "PostgreSQL", "Docker", "AWS"],
    experience=[
        WorkExperience(
            title="Software Engineer",
            company="Acme Inc.",
            years=4.0,
            description="Built customer-facing dashboards and REST APIs.",
            skills_used=["React", "Node.js", "PostgreSQL"],
        )
    ],
    education=[
        Education(
            degree="B.Sc. Computer Science",
            institution="MIT",
            year=2020,
        )
    ],
    certifications=[],
    preferred_salary=140_000,
    remote_preference="remote",
)

# --- Define your search ---
job_input = JobSearchInput(
    query="Full Stack Engineer React Node",
    location="remote",
    resume=my_resume,
    max_results=8,
    job_type=JobType.FULL_TIME,
    seniority=SeniorityLevel.MID,
)

# --- Run ---
report = run_job_search(job_input)
print_report(report, top_n=5)

# Access typed data
for match in report.top_matches(3):
    print(match.listing.title, match.fit_score)
    if match.materials:
        print(match.materials.cover_letter)
```

---

## 7. Save & export results

```python
import json
from pathlib import Path

# Full JSON
Path("report.json").write_text(report.model_dump_json(indent=2))

# Just top matches as plain dicts
top = [m.model_dump() for m in report.top_matches(5)]
print(json.dumps(top, indent=2))

# Cover letters only
for match in report.matches:
    if match.materials:
        fname = f"cover_{match.listing.company.replace(' ', '_')}.md"
        Path(fname).write_text(match.materials.cover_letter)
```

---

## 8. Swap the LLM provider

CrewAI uses [LiteLLM](https://docs.litellm.ai) under the hood — swap any agent to any model:

```python
from crewai import LLM

# Anthropic Claude
claude = LLM(model="anthropic/claude-sonnet-4-5", api_key="sk-ant-...")

# Inside build_search_crew(), pass llm= to any Agent:
job_researcher = Agent(
    role="Senior Job Researcher",
    llm=claude,
    ...
)
```

---

## 9. Run verbosity & debug

```bash
# Quiet (just final report)
CREWAI_VERBOSE=false python -m crewai_job_search.main

# Debug all agent steps
CREWAI_VERBOSE=true python -m crewai_job_search.main 2>&1 | tee run.log
```

---

## 10. Common errors

| Error | Fix |
|---|---|
| `SERPER_API_KEY not set` | Add key to `.env` and re-export |
| `openai.AuthenticationError` | Check `OPENAI_API_KEY` |
| `pydantic.ValidationError` on input | Check required fields in `JobSearchInput` |
| 0 listings returned | Try a broader `query`, or check SERPER quota |
| `ModuleNotFoundError: crewai_tools` | `pip install crewai-tools` |

---

## 11. Production tips

- **Retry budget**: `quality_gate` retries up to 2× automatically. Increase `retry_count < N` in `flow.py` for flaky networks.  
- **Memory persistence**: CrewAI's long-term memory writes to SQLite by default. Point it at a shared DB for multi-user deployments.  
- **Rate limits**: Add `max_rpm=10` to each `Agent()` to avoid hitting OpenAI / Serper rate limits.  
- **Async**: Call `flow.kickoff_async()` inside an `asyncio` event loop to run searches concurrently.  
- **Observability**: Set `CREWAI_TELEMETRY=false` to opt out, or pipe logs to your APM tool.
