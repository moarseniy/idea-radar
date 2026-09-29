from __future__ import annotations

FEATURE_SCHEMA_VERSION = "weak-signals-63-v2"

FEATURE_NAMES = [
    "stage_research", "stage_poc", "stage_pilot", "stage_early_adoption",
    "stage_scaling_or_mature", "first_evidence_age_years",
    "first_commercial_event_age_years", "verified_pilots_log_24m",
    "production_deployments_log", "commercial_vendor_count_log",
    "named_customer_count_log", "final_standard_flag", "regulatory_precursor_count",
    "procurement_presence", "mass_market_flag",
    "papers_log_3y", "paper_growth_2y", "paper_acceleration",
    "paper_volume_percentile_domain", "citation_velocity_median", "preprint_share_2y",
    "institution_count_log_3y", "industry_affiliation_share_3y",
    "science_country_count_log_3y",
    "patent_families_log_3y", "patent_growth_2y", "patent_acceleration",
    "patent_first_priority_age_years", "patent_new_assignee_share_3y",
    "patent_assignee_count_log_3y", "patent_assignee_hhi_3y", "patent_active_share",
    "media_volume_normalized_12m", "media_growth_6m", "media_burstiness_12m",
    "independent_media_domains_log_12m", "press_release_share_12m",
    "source_type_diversity", "independent_high_trust_source_count_log",
    "unverifiable_source_share", "source_organization_hhi",
    "funding_log_24m", "funding_growth_24m", "funding_round_count_24m",
    "early_stage_funding_share", "strategic_investor_count", "funding_company_hhi",
    "public_grants_log_36m",
    "marketing_claim_density", "technical_term_density", "claim_specificity",
    "is_technology_flag", "rebranding_similarity_to_mature",
    "patent_to_paper_ratio", "patent_minus_paper_growth",
    "media_to_technical_evidence_ratio", "cross_channel_confirmation_count",
    "first_time_convergence_flag",
    "missing_science", "missing_patents", "missing_media", "missing_funding",
    "missing_adoption",
]

assert len(FEATURE_NAMES) == 63
assert len(set(FEATURE_NAMES)) == 63
