"""
Pydantic models for the CrewAI Job Search Pipeline.
All inputs, intermediate state, and outputs are strictly typed.
"""

from __future__ import annotations
from pydantic import BaseModel, Field, HttpUrl, field_validator
from typing import Optional
from enum import Enum


# ─────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────

class SeniorityLevel(str, Enum):
    INTERN    = "intern"
    JUNIOR    = "junior"
    MID       = "mid"
    SENIOR    = "senior"
    STAFF     = "staff"
    PRINCIPAL = "principal"
    MANAGER   = "manager"
    DIRECTOR  = "director"


class JobType(str, Enum):
    FULL_TIME  = "full_time"
    PART_TIME  = "part_time"
    CONTRACT   = "contract"
    FREELANCE  = "freelance"
    INTERNSHIP = "internship"


# ─────────────────────────────────────────────
# INPUT MODELS
# ─────────────────────────────────────────────

class WorkExperience(BaseModel):
    title:       str
    company:     str
    years:       float = Field(ge=0)
    description: str
    skills_used: list[str] = Field(default_factory=list)


class Education(BaseModel):
    degree:      str
    institution: str
    year:        Optional[int] = None
    field:       Optional[str] = None


class ResumeProfile(BaseModel):
    """Structured resume information provided by the user."""
    name:             str
    summary:          str
    skills:           list[str]
    experience:       list[WorkExperience]
    education:        list[Education]
    certifications:   list[str]          = Field(default_factory=list)
    preferred_salary: Optional[int]      = None   # USD per year
    remote_preference: str               = "hybrid"  # remote / hybrid / onsite


class JobSearchInput(BaseModel):
    """Top-level user input validated before entering the Flow."""
    query:        str   = Field(description="Job role / query, e.g. 'Senior Python Engineer'")
    location:     str   = Field(description="City, country, or 'remote'")
    resume:       ResumeProfile
    max_results:  int   = Field(default=10, ge=1, le=50)
    job_type:     JobType         = JobType.FULL_TIME
    seniority:    Optional[SeniorityLevel] = None   # auto-inferred if None
    extra_filters: dict[str, str] = Field(default_factory=dict)

    @field_validator("query")
    @classmethod
    def query_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()


# ─────────────────────────────────────────────
# INTERMEDIATE / AGENT OUTPUT MODELS
# ─────────────────────────────────────────────

class CandidateProfile(BaseModel):
    """Parsed & enriched resume, produced by the Flow's parse step."""
    name:               str
    top_skills:         list[str]           # Top 10 extracted skills
    years_total_exp:    float
    inferred_seniority: SeniorityLevel
    industries:         list[str]           # Preferred / experienced industries
    keywords:           list[str]           # For search query augmentation
    resume_summary:     str                 # Condensed for agent context


class RawJobListing(BaseModel):
    """Output of Job Researcher agent per posting."""
    title:          str
    company:        str
    location:       str
    url:            str
    posted_date:    Optional[str]   = None
    job_type:       Optional[str]   = None
    salary_range:   Optional[str]   = None
    raw_description: str


class CompanyProfile(BaseModel):
    """Output of Company Analyst agent."""
    company:          str
    website:          Optional[str]  = None
    industry:         Optional[str]  = None
    size:             Optional[str]  = None
    funding_stage:    Optional[str]  = None
    glassdoor_rating: Optional[float] = None
    tech_stack:       list[str]      = Field(default_factory=list)
    culture_notes:    str            = ""


class StructuredJD(BaseModel):
    """Output of JD Extractor agent."""
    required_skills:    list[str]
    nice_to_have_skills: list[str]
    responsibilities:   list[str]
    qualifications:     list[str]
    seniority_level:    Optional[str] = None
    keywords:           list[str]


class Contact(BaseModel):
    name:        str
    title:       str
    company:     str
    linkedin_url: Optional[str] = None
    email:       Optional[str]  = None


class ApplicationMaterials(BaseModel):
    """Tailored resume bullets + cover letter produced by Tailoring Agent."""
    job_url:            str
    tailored_summary:   str
    tailored_bullets:   list[str]   # Rewritten experience bullets
    cover_letter:       str
    key_talking_points: list[str]


class JobMatch(BaseModel):
    """Full enriched record for one job. Core output unit."""
    listing:        RawJobListing
    company:        CompanyProfile
    structured_jd:  StructuredJD
    fit_score:      float = Field(ge=0, le=100)
    skills_matched: list[str]
    skills_missing: list[str]
    salary_estimate: Optional[str] = None
    materials:      Optional[ApplicationMaterials] = None
    contacts:       list[Contact] = Field(default_factory=list)


class MarketInsights(BaseModel):
    trending_skills:    list[str]
    avg_salary_range:   str
    top_hiring_companies: list[str]
    summary:            str


# ─────────────────────────────────────────────
# FLOW STATE
# ─────────────────────────────────────────────

class JobSearchState(BaseModel):
    """Mutable state object threaded through the entire Flow."""
    # Inputs
    search_input:   Optional[JobSearchInput]   = None
    # Parsed
    candidate:      Optional[CandidateProfile] = None
    # Search results
    raw_listings:   list[RawJobListing]        = Field(default_factory=list)
    # Matched & scored
    job_matches:    list[JobMatch]             = Field(default_factory=list)
    insights:       Optional[MarketInsights]   = None
    # Final output — stored here so kickoff() callers can retrieve it reliably
    report:         Optional["JobSearchReport"] = None
    # Control
    retry_count:    int                        = 0
    errors:         list[str]                  = Field(default_factory=list)


# ─────────────────────────────────────────────
# FINAL OUTPUT
# ─────────────────────────────────────────────

class JobSearchReport(BaseModel):
    """Final output returned to the caller."""
    query:         str
    location:      str
    total_found:   int
    matches:       list[JobMatch]
    insights:      MarketInsights
    run_metadata:  dict[str, str] = Field(default_factory=dict)

    def top_matches(self, n: int = 5) -> list[JobMatch]:
        return sorted(self.matches, key=lambda m: m.fit_score, reverse=True)[:n]
