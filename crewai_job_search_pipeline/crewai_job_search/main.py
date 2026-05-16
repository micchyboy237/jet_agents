"""
main.py — Production entrypoint for the CrewAI Job Search Pipeline.

Usage:
    python main.py

Or import and call run_job_search() programmatically.

Environment variables required:
    OPENAI_API_KEY   — LLM provider (or set a different LLM on each agent)
    SERPER_API_KEY   — SerperDevTool (web search)

Optional:
    EXA_API_KEY      — EXASearchTool (enhanced research)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# ── Setup logging ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("job_search.main")

# ── Validate env vars ─────────────────────────────────────────
_REQUIRED_ENV = ["OPENAI_API_KEY", "SERPER_API_KEY"]
for _var in _REQUIRED_ENV:
    if not os.getenv(_var):
        logger.warning("Environment variable %s is not set.", _var)


from .models import (
    Education,
    JobSearchInput,
    JobSearchReport,
    JobSearchState,
    JobType,
    ResumeProfile,
    SeniorityLevel,
    WorkExperience,
)
from .flow import JobSearchFlow


def run_job_search(search_input: JobSearchInput) -> JobSearchReport:
    """
    Main pipeline entry point.

    Args:
        search_input: Validated JobSearchInput with query + resume.

    Returns:
        JobSearchReport with ranked matches, materials, contacts, and insights.
    """
    logger.info(
        "Starting Job Search Pipeline: query='%s' location='%s' max_results=%d",
        search_input.query,
        search_input.location,
        search_input.max_results,
    )

    # Initialise Flow with state
    initial_state = JobSearchState(search_input=search_input)
    flow = JobSearchFlow(initial_state=initial_state)

    # Kick off the Flow — returns the JobSearchReport from the final step
    report: JobSearchReport = flow.kickoff()

    logger.info(
        "Pipeline complete. Found %d matches. Top score: %.1f",
        report.total_found,
        report.matches[0].fit_score if report.matches else 0.0,
    )
    return report


def print_report(report: JobSearchReport, top_n: int = 5) -> None:
    """Pretty-print the top matches to stdout."""
    print("\n" + "═" * 60)
    print(f"  JOB SEARCH REPORT")
    print(f"  Query: {report.query} | Location: {report.location}")
    print(f"  Total matches: {report.total_found}")
    print("═" * 60)

    print(f"\n📊 MARKET INSIGHTS")
    ins = report.insights
    print(f"  Salary range:    {ins.avg_salary_range}")
    print(f"  Top companies:   {', '.join(ins.top_hiring_companies[:3])}")
    print(f"  Trending skills: {', '.join(ins.trending_skills[:6])}")
    print(f"  Summary:         {ins.summary}")

    print(f"\n🏆 TOP {top_n} MATCHES")
    print("─" * 60)
    for i, match in enumerate(report.top_matches(top_n), 1):
        l  = match.listing
        co = match.company
        print(f"\n  [{i}] {l.title}  ·  {l.company}")
        print(f"      📍 {l.location}  |  💯 Fit score: {match.fit_score:.1f}/100")
        if l.salary_range or match.salary_estimate:
            print(f"      💰 {l.salary_range or match.salary_estimate}")
        print(f"      🔗 {l.url}")
        if match.skills_matched:
            print(f"      ✅ Matched: {', '.join(match.skills_matched[:5])}")
        if match.skills_missing:
            print(f"      ⚠️  Missing: {', '.join(match.skills_missing[:3])}")
        if co.glassdoor_rating:
            print(f"      ⭐ Glassdoor: {co.glassdoor_rating}")
        if match.contacts:
            c = match.contacts[0]
            print(f"      👤 Contact: {c.name} ({c.title})" +
                  (f"  {c.linkedin_url}" if c.linkedin_url else ""))
        if match.materials:
            print(f"      📝 Application materials: ready")

    print("\n" + "═" * 60 + "\n")


# ─────────────────────────────────────────────────────────────
# Example / demo run
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # ── Build a sample resume ─────────────────────────────────
    sample_resume = ResumeProfile(
        name="Alex Rivera",
        summary=(
            "Backend engineer with 6 years building scalable Python APIs and "
            "data pipelines. Strong in distributed systems, cloud-native architecture, "
            "and AI/ML integration. Looking for a senior or staff role at a "
            "growth-stage company."
        ),
        skills=[
            "Python", "FastAPI", "Django", "PostgreSQL", "Redis",
            "Kafka", "Docker", "Kubernetes", "AWS", "GCP",
            "Terraform", "CI/CD", "REST APIs", "gRPC", "GraphQL",
            "Pytest", "SQLAlchemy", "Celery", "OpenAI API", "LangChain",
        ],
        experience=[
            WorkExperience(
                title="Senior Backend Engineer",
                company="Stripe",
                years=3.0,
                description=(
                    "Built and maintained high-throughput payment processing APIs "
                    "handling 50K req/sec. Led migration to event-driven architecture "
                    "using Kafka. Mentored 3 junior engineers."
                ),
                skills_used=["Python", "Kafka", "PostgreSQL", "AWS", "gRPC"],
            ),
            WorkExperience(
                title="Backend Engineer",
                company="Datadog",
                years=2.5,
                description=(
                    "Developed internal data pipeline tooling in Python/Celery. "
                    "Reduced pipeline latency by 40% through query optimisation. "
                    "Contributed to open-source observability tooling."
                ),
                skills_used=["Python", "Celery", "Redis", "Docker", "GCP"],
            ),
            WorkExperience(
                title="Junior Python Developer",
                company="Acme Corp",
                years=0.5,
                description="Built internal REST APIs with Django. Onboarding project.",
                skills_used=["Python", "Django", "PostgreSQL"],
            ),
        ],
        education=[
            Education(
                degree="B.Sc. Computer Science",
                institution="University of California, Berkeley",
                year=2018,
                field="Computer Science",
            )
        ],
        certifications=["AWS Certified Solutions Architect", "CKA (Kubernetes)"],
        preferred_salary=180_000,
        remote_preference="hybrid",
    )

    # ── Build search input ────────────────────────────────────
    job_input = JobSearchInput(
        query="Senior Backend Engineer Python",
        location="San Francisco, CA",
        resume=sample_resume,
        max_results=10,
        job_type=JobType.FULL_TIME,
        seniority=SeniorityLevel.SENIOR,
    )

    # ── Run the pipeline ──────────────────────────────────────
    try:
        report = run_job_search(job_input)
        print_report(report, top_n=5)

        # Save full JSON report
        output_path = Path("job_search_report.json")
        output_path.write_text(report.model_dump_json(indent=2))
        logger.info("Full report saved to %s", output_path)

    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user.")
        sys.exit(0)
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        sys.exit(1)
