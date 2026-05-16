"""
CrewAI Flow — Deterministic backbone for the Job Search Pipeline.

Architecture (from CrewAI best practices):
  Flow owns structure, state, routing, and guardrails.
  Crews provide intelligence at specific steps.

Steps:
  @start  parse_and_enrich_input
    ↓ @listen → run_search_crew
    ↓ @router → quality_gate  (retry | continue)
    ↓ @listen → run_match_crew
    ↓ @listen → validate_and_compile
    ↓ @listen → generate_insights
    → returns JobSearchReport
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Optional

from crewai.flow.flow import Flow, listen, router, start

from .models import (
    CandidateProfile,
    CompanyProfile,
    JobMatch,
    JobSearchInput,
    JobSearchReport,
    JobSearchState,
    MarketInsights,
    RawJobListing,
    SeniorityLevel,
    StructuredJD,
)
from .crews import build_search_crew, build_match_crew

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helper: infer seniority from experience years
# ─────────────────────────────────────────────────────────────

def _infer_seniority(years: float) -> SeniorityLevel:
    if years < 1:   return SeniorityLevel.INTERN
    if years < 2:   return SeniorityLevel.JUNIOR
    if years < 5:   return SeniorityLevel.MID
    if years < 8:   return SeniorityLevel.SENIOR
    if years < 12:  return SeniorityLevel.STAFF
    return SeniorityLevel.PRINCIPAL


def _extract_keywords(resume: "ResumeProfile") -> list[str]:  # type: ignore[name-defined]
    """Pull searchable keywords from resume for query augmentation."""
    kws: set[str] = set(resume.skills)
    for exp in resume.experience:
        kws.update(exp.skills_used)
    for cert in resume.certifications:
        kws.add(cert)
    return list(kws)[:20]


# ─────────────────────────────────────────────────────────────
# The Flow
# ─────────────────────────────────────────────────────────────

class JobSearchFlow(Flow[JobSearchState]):
    """
    Deterministic Flow that orchestrates the two CrewAI crews.
    State is immutably threaded through all steps via JobSearchState.
    """

    # ── STEP 1: Parse & Enrich ───────────────────────────────
    @start()
    def parse_and_enrich_input(self) -> str:
        """
        Pure deterministic step — no LLM call.
        Validates input, derives CandidateProfile, augments keywords.
        Returns 'ready' to trigger the search crew.
        """
        inp: JobSearchInput = self.state.search_input  # type: ignore[assignment]
        if inp is None:
            raise ValueError("JobSearchInput not set on state before kickoff.")

        resume = inp.resume

        # Total experience
        years_total = sum(e.years for e in resume.experience)

        # Infer seniority
        seniority = inp.seniority or _infer_seniority(years_total)

        # Industry detection (simple heuristic on job titles)
        industries = list({
            exp.company for exp in resume.experience
        })[:5]

        keywords = _extract_keywords(resume)

        candidate = CandidateProfile(
            name=resume.name,
            top_skills=resume.skills[:10],
            years_total_exp=years_total,
            inferred_seniority=seniority,
            industries=industries,
            keywords=keywords,
            resume_summary=resume.summary[:500],
        )

        self.state.candidate = candidate
        logger.info(
            "CandidateProfile built: %s | %s | %.1f yrs | %d skills",
            candidate.name,
            candidate.inferred_seniority,
            candidate.years_total_exp,
            len(candidate.top_skills),
        )
        return "ready"

    # ── STEP 2: Search Crew ──────────────────────────────────
    @listen(parse_and_enrich_input)
    def run_search_crew(self) -> str:
        """
        Kicks off SearchCrew (Job Researcher → Company Analyst → JD Extractor).
        Parses crew output into typed lists on the state.
        """
        inp = self.state.search_input
        candidate = self.state.candidate

        logger.info("Launching SearchCrew for query='%s' location='%s'", inp.query, inp.location)

        crew = build_search_crew(
            query=inp.query,
            location=inp.location,
            candidate=candidate,
            max_results=inp.max_results,
            job_type=inp.job_type.value,
        )

        result = crew.kickoff()

        # The crew's final task output is a list of StructuredJDs.
        # Tasks are sequential so we can pull by index.
        try:
            raw_output = result.tasks_output

            # Task 0 → RawJobListings
            raw_listings: list[RawJobListing] = _safe_parse_list(
                raw_output[0].raw if raw_output else "[]",
                RawJobListing,
            )
            # Task 1 → CompanyProfiles (stored as JSON in raw for later use)
            company_profiles: list[CompanyProfile] = _safe_parse_list(
                raw_output[1].raw if len(raw_output) > 1 else "[]",
                CompanyProfile,
            )
            # Task 2 → StructuredJDs
            structured_jds: list[StructuredJD] = _safe_parse_list(
                raw_output[2].raw if len(raw_output) > 2 else "[]",
                StructuredJD,
            )
        except Exception as exc:
            logger.warning("SearchCrew output parse error: %s", exc)
            raw_listings, company_profiles, structured_jds = [], [], []

        self.state.raw_listings = raw_listings
        # Stash company / JD data as JSON in extra state storage
        self._company_profiles = company_profiles
        self._structured_jds   = structured_jds

        logger.info("SearchCrew found %d listings", len(raw_listings))
        return "search_done"

    # ── STEP 3: Quality Gate Router ──────────────────────────
    @router(run_search_crew)
    def quality_gate(self) -> str:
        """
        Deterministic routing: if we found enough results, continue to match.
        Otherwise retry search up to 2 times.
        """
        if len(self.state.raw_listings) >= max(3, self.state.search_input.max_results // 3):
            logger.info("Quality gate PASSED — %d listings", len(self.state.raw_listings))
            return "match"
        if self.state.retry_count < 2:
            self.state.retry_count += 1
            logger.warning(
                "Quality gate FAILED (only %d listings). Retry %d/2",
                len(self.state.raw_listings),
                self.state.retry_count,
            )
            return "retry_search"
        logger.error("Quality gate failed after max retries. Proceeding with %d results.",
                     len(self.state.raw_listings))
        return "match"  # Best-effort

    @listen("retry_search")
    def retry_search(self) -> str:
        """Widen query and re-run search crew."""
        # Broaden query by appending a top skill
        inp = self.state.search_input
        top_skill = (self.state.candidate.top_skills or [""])[0]
        broadened = f"{inp.query} {top_skill}".strip()
        logger.info("Retrying search with broadened query: '%s'", broadened)
        # Temporarily mutate for retry (Flow state is mutable during execution)
        original_query = inp.query
        object.__setattr__(inp, "query", broadened)
        result = self.run_search_crew()
        object.__setattr__(inp, "query", original_query)
        return result

    # ── STEP 4: Match Crew ───────────────────────────────────
    @listen("match")
    def run_match_crew(self) -> str:
        """
        Kicks off MatchCrew (Skills Matcher → Fit Scorer → Tailoring → HM Finder).
        """
        candidate = self.state.candidate
        listings  = self.state.raw_listings

        if not listings:
            logger.error("No listings to match. Skipping MatchCrew.")
            self.state.job_matches = []
            return "validate"

        logger.info("Launching MatchCrew for %d listings", len(listings))

        crew = build_match_crew(
            candidate=candidate,
            raw_listings=listings,
            structured_jds=getattr(self, "_structured_jds", []),
            company_profiles=getattr(self, "_company_profiles", []),
        )

        result = crew.kickoff()

        # Assemble JobMatch records from crew task outputs
        try:
            raw_output = result.tasks_output
            # task[1] → scoring (job_url + fit_score)
            scoring_data: list[dict] = _parse_json_list(
                raw_output[1].raw if len(raw_output) > 1 else "[]"
            )
            # task[2] → application materials
            materials_list: list[dict] = _parse_json_list(
                raw_output[2].raw if len(raw_output) > 2 else "[]"
            )
            # task[3] → contacts
            contacts_list: list[dict] = _parse_json_list(
                raw_output[3].raw if len(raw_output) > 3 else "[]"
            )
        except Exception as exc:
            logger.warning("MatchCrew parse error: %s", exc)
            scoring_data, materials_list, contacts_list = [], [], []

        # Build JobMatch objects
        score_map:    dict[str, dict] = {d.get("job_url", ""): d for d in scoring_data}
        material_map: dict[str, dict] = {d.get("job_url", ""): d for d in materials_list}
        company_map:  dict[str, CompanyProfile] = {
            cp.company: cp
            for cp in getattr(self, "_company_profiles", [])
        }
        jd_map: dict[str, StructuredJD] = {}
        for i, listing in enumerate(listings):
            jds = getattr(self, "_structured_jds", [])
            if i < len(jds):
                jd_map[listing.url] = jds[i]

        job_matches: list[JobMatch] = []
        for listing in listings:
            score_info = score_map.get(listing.url, {})
            mat_info   = material_map.get(listing.url, {})

            jm = JobMatch(
                listing=listing,
                company=company_map.get(listing.company, CompanyProfile(company=listing.company)),
                structured_jd=jd_map.get(listing.url, StructuredJD(
                    required_skills=[], nice_to_have_skills=[],
                    responsibilities=[], qualifications=[], keywords=[],
                )),
                fit_score=score_info.get("fit_score", 0.0),
                skills_matched=score_info.get("matched_skills", []),
                skills_missing=score_info.get("missing_skills", []),
                salary_estimate=score_info.get("salary_estimate"),
                materials=ApplicationMaterials(**mat_info) if mat_info else None,
                contacts=[],
            )
            job_matches.append(jm)

        # Attach contacts to top-5
        top_companies = {m.listing.company for m in
                         sorted(job_matches, key=lambda x: x.fit_score, reverse=True)[:5]}
        for jm in job_matches:
            if jm.listing.company in top_companies:
                jm.contacts = [
                    c for c in contacts_list
                    if isinstance(c, dict) and c.get("company") == jm.listing.company
                ]

        self.state.job_matches = sorted(job_matches, key=lambda x: x.fit_score, reverse=True)
        logger.info("MatchCrew produced %d scored matches", len(self.state.job_matches))
        return "validate"

    # ── STEP 5: Validate & Guardrails ────────────────────────
    @listen("validate")
    def validate_and_compile(self) -> str:
        """
        Deterministic guardrails:
          - Clamp scores to [0, 100]
          - Strip null / placeholder URLs
          - De-duplicate by URL
          - Flag any missing required fields
        """
        seen_urls: set[str] = set()
        clean: list[JobMatch] = []

        for match in self.state.job_matches:
            url = match.listing.url or ""

            # Skip empty/duplicate URLs
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)

            # Skip placeholder / obviously fake URLs
            if re.search(r"(example\.com|placeholder|test\.com)", url, re.IGNORECASE):
                continue

            # Clamp score
            clamped = max(0.0, min(100.0, match.fit_score))
            object.__setattr__(match, "fit_score", clamped)

            clean.append(match)

        self.state.job_matches = clean
        logger.info("Validation: %d valid matches after guardrails", len(clean))
        return "insights"

    # ── STEP 6: Market Insights ──────────────────────────────
    @listen("insights")
    def generate_insights(self) -> str:
        """
        Deterministic aggregation of market intelligence from match data.
        No LLM call — pure computation.
        """
        matches = self.state.job_matches

        if not matches:
            self.state.insights = MarketInsights(
                trending_skills=[],
                avg_salary_range="N/A",
                top_hiring_companies=[],
                summary="No matches found. Try broadening your query.",
            )
            return "done"

        # Trending skills = skills most frequently required across all JDs
        from collections import Counter
        skill_counts: Counter = Counter()
        for m in matches:
            skill_counts.update(m.structured_jd.required_skills)

        trending = [s for s, _ in skill_counts.most_common(10)]

        # Salary aggregation
        salary_strings = [
            m.salary_estimate for m in matches
            if m.salary_estimate and "$" in (m.salary_estimate or "")
        ]
        avg_salary = salary_strings[0] if salary_strings else "Not disclosed"

        # Top hiring companies (by frequency in results)
        company_counts: Counter = Counter(m.listing.company for m in matches)
        top_companies = [c for c, _ in company_counts.most_common(5)]

        # Summary
        top3 = sorted(matches, key=lambda x: x.fit_score, reverse=True)[:3]
        top3_names = ", ".join(f"{m.listing.title} @ {m.listing.company}" for m in top3)

        self.state.insights = MarketInsights(
            trending_skills=trending,
            avg_salary_range=avg_salary,
            top_hiring_companies=top_companies,
            summary=(
                f"Found {len(matches)} matching roles. "
                f"Top matches: {top3_names}. "
                f"Most in-demand skills: {', '.join(trending[:5])}."
            ),
        )
        logger.info("Insights generated: %d trending skills", len(trending))
        return "done"

    # ── STEP 7: Build Report ─────────────────────────────────
    @listen("done")
    def build_report(self) -> None:
        """
        Final step: assemble the JobSearchReport and store it on state.
        We store on state (not return) because flow.kickoff() returns the
        raw return value of the last method — storing on state lets main.py
        reliably retrieve the typed report after kickoff() completes.
        """
        inp = self.state.search_input
        report = JobSearchReport(
            query=inp.query,
            location=inp.location,
            total_found=len(self.state.job_matches),
            matches=self.state.job_matches,
            insights=self.state.insights,
            run_metadata={
                "completed_at": datetime.utcnow().isoformat() + "Z",
                "candidate":    self.state.candidate.name,
                "retry_count":  str(self.state.retry_count),
                "crew_version": "2.0",
            },
        )
        self.state.report = report
        logger.info("JobSearchReport ready: %d matches", report.total_found)


# ─────────────────────────────────────────────────────────────
# Parse helpers
# ─────────────────────────────────────────────────────────────

def _parse_json_list(raw: str) -> list[dict]:
    """Safely parse a JSON list string, stripping markdown fences."""
    try:
        cleaned = re.sub(r"```json|```", "", raw).strip()
        data = json.loads(cleaned)
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _safe_parse_list(raw: str, model) -> list:
    """Parse JSON list and construct Pydantic model instances, skipping bad items."""
    items = _parse_json_list(raw)
    result = []
    for item in items:
        try:
            result.append(model(**item))
        except Exception:
            pass
    return result
