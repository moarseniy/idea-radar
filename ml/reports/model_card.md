# Карточка модели logreg_v1

Эта карточка описывает JSON-артефакт логистической регрессии, который использует сервис.
Сводные метрики и протокол валидации приведены в [docs/12-final-metrics.md](../../docs/12-final-metrics.md).

Логистическая регрессия, 46 входных показателей, C=0.03, веса классов сбалансированы, калибровка Платта на out-of-fold, порог 0.42.
Обучение: `ml/datasets/model_compare/gpt-5.6-luna/training.csv`, 1327 строк (390 позитивов). C, калибровка и порог выбраны только на обучении (StratifiedGroupKFold по технологии).

| | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier |
|---|---:|---:|---:|---:|---:|---:|
| CV на обучении | 0.672 | 0.938 | 0.783 | 0.922 | 0.799 | 0.106 |
| Валидация: золото + негативы | 0.786 | 0.939 | 0.856 | 0.918 | 0.911 | 0.12 |

Валидация: 98 позитивов золота, 89 негативов (`validation_negatives.csv`). Доля позитивов золота с уверенностью > 0,75: 0.857.

## Исключённые признаки

- `rebranding_similarity_to_mature` — направление противоречит смыслу: у зрелых технологий LLM ставит 0
- `paper_acceleration` — в паре с ростом публикаций получает вес обратного знака; CV AUC без него −0,0015
- `patent_acceleration` — то же, что paper_acceleration (патентный канал сейчас не измеряется)
- константные на обучении (не измеряются): `institution_count_log_3y`, `industry_affiliation_share_3y`, `science_country_count_log_3y`, `patent_families_log_3y`, `patent_growth_2y`, `patent_first_priority_age_years`, `patent_new_assignee_share_3y`, `patent_assignee_count_log_3y`, `patent_assignee_hhi_3y`, `patent_active_share`, `patent_to_paper_ratio`, `patent_minus_paper_growth`, `first_time_convergence_flag`, `missing_patents`

## Веса (на стандартизованных признаках, до калибровки)

| Признак | Вес |
|---|---:|
| `is_technology_flag` | +1.409 |
| `mass_market_flag` | -0.639 |
| `stage_scaling_or_mature` | -0.595 |
| `first_evidence_age_years` | -0.403 |
| `commercial_vendor_count_log` | +0.302 |
| `stage_research` | +0.299 |
| `stage_poc` | +0.212 |
| `strategic_investor_count` | +0.211 |
| `technical_term_density` | +0.210 |
| `paper_growth_2y` | +0.197 |
| `first_commercial_event_age_years` | -0.193 |
| `production_deployments_log` | -0.164 |
| `named_customer_count_log` | +0.158 |
| `stage_early_adoption` | +0.144 |
| `missing_funding` | -0.137 |
| `stage_pilot` | +0.116 |
| `missing_media` | +0.104 |
| `missing_science` | +0.103 |
| `paper_volume_percentile_domain` | -0.099 |
| `verified_pilots_log_24m` | +0.095 |
| `preprint_share_2y` | +0.094 |
| `media_growth_6m` | +0.089 |
| `unverifiable_source_share` | +0.083 |
| `funding_round_count_24m` | +0.072 |
| `funding_growth_24m` | -0.065 |
| `missing_adoption` | -0.061 |
| `marketing_claim_density` | +0.056 |
| `press_release_share_12m` | +0.053 |
| `media_volume_normalized_12m` | -0.046 |
| `early_stage_funding_share` | +0.046 |
| `public_grants_log_36m` | -0.042 |
| `papers_log_3y` | -0.038 |
| `funding_log_24m` | -0.036 |
| `media_burstiness_12m` | -0.027 |
| `independent_media_domains_log_12m` | -0.025 |
| `claim_specificity` | -0.025 |
| `source_organization_hhi` | -0.023 |
| `citation_velocity_median` | +0.023 |
| `cross_channel_confirmation_count` | -0.022 |
| `procurement_presence` | +0.017 |
| `final_standard_flag` | -0.007 |
| `regulatory_precursor_count` | -0.005 |
| `funding_company_hhi` | -0.005 |
| `source_type_diversity` | -0.004 |
| `media_to_technical_evidence_ratio` | -0.003 |
| `independent_high_trust_source_count_log` | -0.001 |
