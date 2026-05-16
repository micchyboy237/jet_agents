"""
Custom CrewAI tools for the Job Search Pipeline.

Built on top of crewai.tools.BaseTool so agents can discover
and invoke them like any built-in CrewAI tool.
"""

from __future__ import annotations
import json
import re
import urllib.request
from typing import Optional, Type
from pydantic import BaseModel, Field
from crewai.tools import BaseTool


# ─────────────────────────────────────────────
# Skill Vector / Semantic Match Tool
# ─────────────────────────────────────────────

class SkillMatchInput(BaseModel):
    resume_skills:  list[str] = Field(description="List of skills from the candidate's resume")
    jd_skills:      list[str] = Field(description="List of required skills from the job description")


class SkillMatchTool(BaseTool):
    """
    Computes a semantic overlap score between resume skills and JD requirements.

    Returns a JSON object with:
      - matched_skills: skills present in both (normalised)
      - missing_skills: JD skills absent from resume
      - overlap_score: 0–100 float
    """
    name:        str = "Skill Match Tool"
    description: str = (
        "Compares candidate resume skills against job description requirements. "
        "Returns matched skills, missing skills, and an overlap score 0-100."
    )
    args_schema: Type[BaseModel] = SkillMatchInput

    def _run(self, resume_skills: list[str], jd_skills: list[str]) -> str:  # type: ignore[override]
        def _norm(s: str) -> str:
            return s.lower().strip().replace("-", " ").replace("_", " ")

        normed_resume = {_norm(s) for s in resume_skills}
        normed_jd     = [_norm(s) for s in jd_skills]

        matched  = [s for s in normed_jd if s in normed_resume]
        missing  = [s for s in normed_jd if s not in normed_resume]
        score    = round(len(matched) / max(len(normed_jd), 1) * 100, 1)

        result = {
            "matched_skills": matched,
            "missing_skills": missing,
            "overlap_score":  score,
        }
        return json.dumps(result)


# ─────────────────────────────────────────────
# Fit Scoring Tool
# ─────────────────────────────────────────────

class FitScoringInput(BaseModel):
    overlap_score:      float  = Field(ge=0, le=100, description="Skills overlap score from SkillMatchTool")
    candidate_seniority: str   = Field(description="Candidate seniority level, e.g. 'senior'")
    job_seniority:      str    = Field(description="Job required seniority level, e.g. 'senior'")
    remote_pref:        str    = Field(description="Candidate remote preference: remote/hybrid/onsite")
    job_location:       str    = Field(description="Job location or 'remote'")
    years_exp:          float  = Field(description="Candidate total years of experience")
    required_years:     Optional[float] = Field(default=None, description="Min years required by JD")


SENIORITY_RANK = {
    "intern": 0, "junior": 1, "mid": 2, "senior": 3,
    "staff": 4, "principal": 4, "manager": 3, "director": 5,
}


class FitScoringTool(BaseTool):
    """
    Computes a holistic fit score (0–100) combining skills overlap,
    seniority match, location/remote alignment, and experience match.
    """
    name:        str = "Fit Scoring Tool"
    description: str = (
        "Produces a holistic fit score 0-100 for a candidate-job pair "
        "based on skills overlap, seniority alignment, location, and experience."
    )
    args_schema: Type[BaseModel] = FitScoringInput

    def _run(  # type: ignore[override]
        self,
        overlap_score: float,
        candidate_seniority: str,
        job_seniority: str,
        remote_pref: str,
        job_location: str,
        years_exp: float,
        required_years: Optional[float] = None,
    ) -> str:
        # Skills weight: 50%
        skills_component = overlap_score * 0.50

        # Seniority weight: 25%
        c_rank = SENIORITY_RANK.get(candidate_seniority.lower(), 2)
        j_rank = SENIORITY_RANK.get(job_seniority.lower(), 2)
        seniority_diff = abs(c_rank - j_rank)
        seniority_score = max(0, 100 - seniority_diff * 25)
        seniority_component = seniority_score * 0.25

        # Location weight: 15%
        remote_match = (
            ("remote" in job_location.lower() and remote_pref == "remote") or
            ("remote" not in job_location.lower() and remote_pref == "onsite") or
            remote_pref == "hybrid"
        )
        location_component = 15.0 if remote_match else 7.5

        # Experience weight: 10%
        exp_ok = (required_years is None) or (years_exp >= required_years * 0.8)
        exp_component = 10.0 if exp_ok else 5.0

        total = round(
            skills_component + seniority_component + location_component + exp_component, 1
        )
        return json.dumps({
            "fit_score":            total,
            "skills_component":     round(skills_component, 1),
            "seniority_component":  round(seniority_component, 1),
            "location_component":   location_component,
            "experience_component": exp_component,
        })


# ─────────────────────────────────────────────
# URL Alive Check Tool (validator step)
# ─────────────────────────────────────────────

class URLCheckInput(BaseModel):
    urls: list[str] = Field(description="List of URLs to verify are reachable")


class URLAliveCheckTool(BaseTool):
    """
    Verifies that job posting URLs are reachable (HTTP 200/301/302).
    Returns a dict mapping URL → alive (bool).
    """
    name:        str = "URL Alive Check Tool"
    description: str = (
        "Checks whether a list of job posting URLs are alive and return HTTP 2xx/3xx. "
        "Use before including URLs in final output."
    )
    args_schema: Type[BaseModel] = URLCheckInput

    def _run(self, urls: list[str]) -> str:  # type: ignore[override]
        results: dict[str, bool] = {}
        for url in urls:
            try:
                req = urllib.request.Request(url, method="HEAD")
                req.add_header("User-Agent", "Mozilla/5.0")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    results[url] = resp.status < 400
            except Exception:
                results[url] = False
        return json.dumps(results)


# ─────────────────────────────────────────────
# Resume Template Tool
# ─────────────────────────────────────────────

class TemplateInput(BaseModel):
    candidate_name:    str
    tailored_summary:  str
    tailored_bullets:  list[str]
    skills:            list[str]
    job_title:         str
    company:           str


class ResumeTemplateTool(BaseTool):
    """
    Renders a clean Markdown resume from tailored content.
    Returns a formatted Markdown string ready for PDF conversion.
    """
    name:        str = "Resume Template Tool"
    description: str = (
        "Renders a polished Markdown resume from tailored summary, bullets, and skills. "
        "Returns clean Markdown text."
    )
    args_schema: Type[BaseModel] = TemplateInput

    def _run(  # type: ignore[override]
        self,
        candidate_name: str,
        tailored_summary: str,
        tailored_bullets: list[str],
        skills: list[str],
        job_title: str,
        company: str,
    ) -> str:
        bullets_md = "\n".join(f"- {b}" for b in tailored_bullets)
        skills_md  = " · ".join(skills)
        return f"""# {candidate_name}

## Professional Summary
{tailored_summary}

## Experience
{bullets_md}

## Key Skills
{skills_md}

---
*Tailored for: {job_title} at {company}*
"""
