"""
CrewAI Crews for the Job Search Pipeline.

Two crews:
  1. SearchCrew  — finds and extracts job listings
  2. MatchCrew   — scores, tailors, and finds contacts for top listings
"""

from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileReadTool
from pydantic import BaseModel

from .models import (
    CandidateProfile,
    RawJobListing,
    CompanyProfile,
    StructuredJD,
    JobMatch,
    ApplicationMaterials,
    Contact,
    MarketInsights,
)
from .tools import SkillMatchTool, FitScoringTool, ResumeTemplateTool

# ── Shared tool instances ────────────────────
search_tool  = SerperDevTool()
scrape_tool  = ScrapeWebsiteTool()
skill_tool   = SkillMatchTool()
scoring_tool = FitScoringTool()
template_tool = ResumeTemplateTool()


# ═══════════════════════════════════════════════════════════════
# CREW 1 · SearchCrew
# Agents: Job Researcher → Company Analyst → JD Extractor
# ═══════════════════════════════════════════════════════════════

def build_search_crew(
    query: str,
    location: str,
    candidate: CandidateProfile,
    max_results: int,
    job_type: str,
) -> Crew:
    """
    Constructs and returns the SearchCrew.

    Process: Sequential (Researcher → Analyst → Extractor).
    Each task has a Pydantic output schema, ensuring typed
    data flows into the next agent as context.
    """

    # ── Agent 1: Job Researcher ──────────────────────────────
    job_researcher = Agent(
        role="Senior Job Researcher",
        goal=(
            f"Find {max_results} real, live {job_type} job postings for "
            f"'{query}' in {location}. Focus on roles matching these skills: "
            f"{', '.join(candidate.top_skills[:8])}."
        ),
        backstory=(
            "You are an expert talent scout with years of experience finding "
            "high-quality job opportunities. You know exactly where to search, "
            "how to verify postings are live, and how to extract complete details."
        ),
        tools=[search_tool, scrape_tool],
        verbose=True,
        max_iter=8,
        reasoning=True,
    )

    research_task = Task(
        description=(
            f"Search for {max_results} {job_type} job postings matching: '{query}' "
            f"in {location}.\n\n"
            f"Candidate keywords: {', '.join(candidate.keywords[:10])}\n\n"
            "For each posting collect: title, company, location, URL, posted date, "
            "salary range (if shown), job type, and the full raw description text.\n\n"
            "Only include postings from the last 30 days. Verify URLs are real."
        ),
        expected_output=(
            "A JSON list of job objects with fields: title, company, location, url, "
            "posted_date, job_type, salary_range, raw_description."
        ),
        output_pydantic=list[RawJobListing],
        agent=job_researcher,
    )

    # ── Agent 2: Company Analyst ────────────────────────────
    company_analyst = Agent(
        role="Company Intelligence Analyst",
        goal=(
            "For each company in the job list, research its culture, tech stack, "
            "size, funding stage, and Glassdoor rating."
        ),
        backstory=(
            "You specialise in company intelligence. You know how to quickly surface "
            "the most relevant information about employers: culture, tech, and growth "
            "trajectory — the signals that separate great employers from mediocre ones."
        ),
        tools=[search_tool, scrape_tool],
        verbose=True,
        max_iter=6,
    )

    company_task = Task(
        description=(
            "For each company in the job list (from the previous task), research:\n"
            "  - Company website & industry\n"
            "  - Headcount / size tier (startup / SMB / enterprise)\n"
            "  - Funding stage (if applicable)\n"
            "  - Glassdoor or Levels.fyi rating\n"
            "  - Tech stack (from job ads, engineering blogs)\n"
            "  - Brief culture notes (1-2 sentences)\n\n"
            "Return a profile per company."
        ),
        expected_output=(
            "A JSON list of company profiles with fields: company, website, industry, "
            "size, funding_stage, glassdoor_rating, tech_stack, culture_notes."
        ),
        output_pydantic=list[CompanyProfile],
        agent=company_analyst,
        context=[research_task],
    )

    # ── Agent 3: JD Extractor ───────────────────────────────
    jd_extractor = Agent(
        role="Job Description Extraction Specialist",
        goal=(
            "Parse each job description into structured requirements: required skills, "
            "nice-to-have skills, responsibilities, qualifications, seniority, and keywords."
        ),
        backstory=(
            "You are an NLP specialist who can rapidly parse job descriptions into "
            "clean, structured data. You distinguish hard requirements from preferences "
            "and extract the exact keywords that ATS systems use."
        ),
        tools=[scrape_tool],
        verbose=True,
        max_iter=5,
    )

    jd_task = Task(
        description=(
            "For each job posting from the research task, parse the raw_description into:\n"
            "  - required_skills: must-have technical & soft skills\n"
            "  - nice_to_have_skills: preferred but not required\n"
            "  - responsibilities: key job duties (bullet list)\n"
            "  - qualifications: degree / experience requirements\n"
            "  - seniority_level: inferred (junior/mid/senior/etc.)\n"
            "  - keywords: ATS-relevant keywords for resume tailoring\n\n"
            "Keep lists concise (max 10 items each)."
        ),
        expected_output=(
            "A JSON list of structured job descriptions, one per posting, with fields: "
            "required_skills, nice_to_have_skills, responsibilities, qualifications, "
            "seniority_level, keywords."
        ),
        output_pydantic=list[StructuredJD],
        agent=jd_extractor,
        context=[research_task],
    )

    return Crew(
        agents=[job_researcher, company_analyst, jd_extractor],
        tasks=[research_task, company_task, jd_task],
        process=Process.sequential,
        verbose=True,
        memory=True,
        embedder={"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    )


# ═══════════════════════════════════════════════════════════════
# CREW 2 · MatchCrew
# Agents: Skills Matcher → Fit Scorer → Tailoring Agent → HM Finder
# ═══════════════════════════════════════════════════════════════

def build_match_crew(
    candidate: CandidateProfile,
    raw_listings: list[RawJobListing],
    structured_jds: list[StructuredJD],
    company_profiles: list[CompanyProfile],
) -> Crew:
    """
    Constructs the MatchCrew.

    Process: Sequential. Each agent enriches the job records produced
    by SearchCrew and hands off to the next specialist.
    """

    listings_summary = "\n".join(
        f"- [{i}] {l.title} @ {l.company} ({l.url})"
        for i, l in enumerate(raw_listings)
    )

    candidate_ctx = (
        f"Candidate: {candidate.name}\n"
        f"Seniority: {candidate.inferred_seniority}\n"
        f"Total exp: {candidate.years_total_exp} years\n"
        f"Top skills: {', '.join(candidate.top_skills)}\n"
        f"Remote pref: see resume\n"
    )

    # ── Agent 1: Skills Matcher ──────────────────────────────
    skills_matcher = Agent(
        role="Skills Alignment Specialist",
        goal=(
            "For each job, compute the semantic overlap between the candidate's "
            "skills and the job's required skills. Produce matched and missing skill lists."
        ),
        backstory=(
            "You are a skills taxonomy expert. You understand synonyms, related "
            "technologies, and transferable skills. You never dismiss a candidate "
            "just because they used a slightly different tool name."
        ),
        tools=[skill_tool],
        verbose=True,
        max_iter=5,
    )

    skills_task = Task(
        description=(
            f"{candidate_ctx}\n"
            "For each job in the list, use the Skill Match Tool to compare the "
            "candidate's resume skills against each job's required_skills.\n\n"
            f"Jobs:\n{listings_summary}\n\n"
            "Return per-job: matched_skills, missing_skills, overlap_score."
        ),
        expected_output=(
            "A JSON list, one entry per job, with: job_url, matched_skills, "
            "missing_skills, overlap_score (0-100)."
        ),
        agent=skills_matcher,
    )

    # ── Agent 2: Fit Scorer ──────────────────────────────────
    fit_scorer = Agent(
        role="Candidate-Job Fit Scorer",
        goal=(
            "Produce a holistic fit score (0–100) per job, combining skills overlap, "
            "seniority match, location preference, and experience requirements. "
            "Rank the top matches."
        ),
        backstory=(
            "You are a recruiting strategist who looks at the full picture. You know "
            "that a 100% skills match at the wrong seniority is worse than an 80% "
            "match at the right level. You produce defensible, data-driven scores."
        ),
        tools=[scoring_tool, search_tool],
        verbose=True,
        max_iter=5,
    )

    scoring_task = Task(
        description=(
            f"{candidate_ctx}\n"
            "Using the skills overlap data from the previous task and the structured JDs, "
            "compute a holistic fit score for each job using the Fit Scoring Tool.\n\n"
            "Also use web search to estimate salary ranges for roles you don't have data for.\n\n"
            "Return: job_url, fit_score, salary_estimate. Sort descending by fit_score."
        ),
        expected_output=(
            "A JSON list of jobs with: job_url, fit_score (0-100), salary_estimate. "
            "Sorted best-match first."
        ),
        agent=fit_scorer,
        context=[skills_task],
    )

    # ── Agent 3: Tailoring Agent ─────────────────────────────
    tailoring_agent = Agent(
        role="Resume & Cover Letter Specialist",
        goal=(
            "For the top 5 scoring jobs, rewrite the candidate's resume bullets "
            "to align with the JD keywords, and craft a personalised cover letter."
        ),
        backstory=(
            "You are an expert resume writer and career coach. You know how to "
            "highlight transferable skills, quantify achievements, and mirror the "
            "language of job descriptions without being dishonest. Every application "
            "you prepare increases callback rates."
        ),
        tools=[template_tool],
        verbose=True,
        max_iter=8,
        reasoning=True,
    )

    tailoring_task = Task(
        description=(
            f"{candidate_ctx}\n"
            "For the TOP 5 jobs by fit_score (from previous task):\n"
            "  1. Rewrite 4-6 resume experience bullets to mirror the JD keywords\n"
            "  2. Write a tailored_summary (3 sentences) for that role\n"
            "  3. Draft a cover letter (3 paragraphs: hook, fit, close)\n"
            "  4. List 3-5 key talking points for the interview\n\n"
            "Use the Resume Template Tool to render the final resume Markdown.\n"
            "Be specific — reference the company name and role in the cover letter."
        ),
        expected_output=(
            "A JSON list (one per top-5 job) with: job_url, tailored_summary, "
            "tailored_bullets, cover_letter, key_talking_points."
        ),
        output_pydantic=list[ApplicationMaterials],
        agent=tailoring_agent,
        context=[scoring_task],
    )

    # ── Agent 4: Hiring Manager Finder ───────────────────────
    hm_finder = Agent(
        role="Hiring Manager Research Specialist",
        goal=(
            "For the top 5 companies, find the likely hiring manager or recruiter "
            "handling this role. Provide name, title, LinkedIn URL if available."
        ),
        backstory=(
            "You are a talent intelligence researcher. You know how to find the "
            "right people at companies using LinkedIn, company websites, and "
            "professional networks. You always verify contacts before listing them."
        ),
        tools=[search_tool, scrape_tool],
        verbose=True,
        max_iter=6,
    )

    hm_task = Task(
        description=(
            "For each of the top 5 companies (from the fit scoring task), find:\n"
            "  - The hiring manager or engineering manager for this type of role\n"
            "  - Any recruiter or talent acquisition specialist at the company\n"
            "  - Their LinkedIn profile URL (if findable)\n"
            "  - Their title\n\n"
            "Search: '[Company] [Role] hiring manager LinkedIn' and '[Company] recruiter LinkedIn'.\n"
            "Only include contacts you can verify exist — do not hallucinate."
        ),
        expected_output=(
            "A JSON list of contacts with: name, title, company, linkedin_url (or null)."
        ),
        output_pydantic=list[Contact],
        agent=hm_finder,
        context=[scoring_task],
    )

    return Crew(
        agents=[skills_matcher, fit_scorer, tailoring_agent, hm_finder],
        tasks=[skills_task, scoring_task, tailoring_task, hm_task],
        process=Process.sequential,
        verbose=True,
        memory=True,
        embedder={"provider": "openai", "config": {"model": "text-embedding-3-small"}},
    )
