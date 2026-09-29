from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION

SNAPSHOT_DATE = date(2026, 9, 22)
DEFAULT_OUTPUT = ROOT / "data" / "training" / "positive_weak_signals_global.csv"
CACHE_DIR = ROOT / "storage" / "openalex_dataset_cache"
CROSSREF_CACHE_DIR = ROOT / "storage" / "crossref_dataset_cache"
OPENALEX = "https://api.openalex.org/works"
CROSSREF = "https://api.crossref.org/works"

DOMAIN_SPECS = [
    ("AI security", "Защита ИИ", [
        ("AI agent identity and delegated authorization", "идентичность ИИ-агентов и делегированная авторизация"),
        ("memory poisoning defense for AI agents", "защита памяти ИИ-агентов от отравления"),
        ("machine unlearning", "машинное разобучение"),
        ("AI model provenance and software bills of materials", "провенанс моделей ИИ и ведомости компонентов"),
        ("continuous automated AI red teaming", "непрерывный автоматизированный red teaming ИИ"),
    ], [
        ("financial service agents", "агентов финансовых сервисов"),
        ("healthcare copilots", "медицинских копилотов"),
        ("industrial control agents", "агентов промышленного управления"),
        ("government service agents", "агентов государственных сервисов"),
        ("edge AI systems", "периферийных систем ИИ"),
        ("multi-agent enterprise workflows", "многоагентных корпоративных процессов"),
        ("autonomous software engineering", "автономной разработки программного обеспечения"),
        ("critical infrastructure operations", "эксплуатации критической инфраструктуры"),
        ("scientific research agents", "агентов научных исследований"),
        ("autonomous supply chains", "автономных цепочек поставок"),
        ("autonomous healthcare decisions", "автономных решений в здравоохранении"),
        ("defense systems", "оборонных систем"),
    ]),
    ("AI infrastructure", "Инфраструктура ИИ", [
        ("photonic neural accelerators", "фотонные нейросетевые ускорители"),
        ("analog in-memory computing", "аналоговые вычисления в памяти"),
        ("neuromorphic processors", "нейроморфные процессоры"),
        ("optical circuit switching", "оптическая коммутация каналов"),
        ("disaggregated KV-cache storage", "дезагрегированное хранение KV-кэша"),
    ], [
        ("long-context language model inference", "инференса языковых моделей с длинным контекстом"),
        ("energy-efficient edge inference", "энергоэффективного edge-инференса"),
        ("onboard robot computing", "бортовых вычислений роботов"),
        ("sovereign AI clouds", "суверенных облаков ИИ"),
        ("low-energy AI data centers", "энергоэффективных центров обработки данных ИИ"),
        ("scientific foundation models", "научных фундаментальных моделей"),
        ("real-time video analytics", "аналитики видео в реальном времени"),
        ("distributed model serving", "распределённого обслуживания моделей"),
        ("privacy-preserving inference", "конфиденциального инференса"),
        ("large-scale recommendation systems", "крупномасштабных рекомендательных систем"),
        ("generative AI serving", "обслуживания генеративного ИИ"),
        ("high-performance scientific computing", "высокопроизводительных научных вычислений"),
    ]),
    ("Edge AI", "Edge", [
        ("event-based vision sensors", "событийные сенсоры машинного зрения"),
        ("in-sensor computing", "вычисления внутри сенсора"),
        ("tinyML neural processing units", "нейропроцессоры tinyML"),
        ("on-device model fine-tuning", "дообучение моделей на устройстве"),
        ("adaptive edge-to-cloud inference routing", "адаптивная маршрутизация инференса между устройством и облаком"),
    ], [
        ("battery-powered IoT", "IoT-устройств с батарейным питанием"),
        ("wearable health monitoring", "носимого мониторинга здоровья"),
        ("industrial predictive maintenance", "предиктивного обслуживания оборудования"),
        ("autonomous vehicles", "автономного транспорта"),
        ("precision agriculture", "точного земледелия"),
        ("point-of-care diagnostics", "диагностики непосредственно у пациента"),
        ("wildlife monitoring", "мониторинга дикой природы"),
        ("smart building controls", "управления интеллектуальными зданиями"),
        ("personalized mobile assistants", "персонализированных мобильных ассистентов"),
        ("remote infrastructure monitoring", "мониторинга удалённой инфраструктуры"),
        ("drone navigation", "навигации беспилотников"),
        ("factory machine monitoring", "мониторинга промышленного оборудования"),
    ]),
    ("Robotics", "Роботы", [
        ("tactile vision-language-action models", "тактильные vision-language-action модели"),
        ("electrohydraulic soft actuators", "электрогидравлические мягкие приводы"),
        ("zero-shot simulation-to-reality transfer", "zero-shot перенос из симуляции в реальность"),
        ("decentralized robot swarms", "децентрализованные рои роботов"),
        ("teleoperation data engines for robot learning", "платформы телеоперационных данных для обучения роботов"),
    ], [
        ("flexible manufacturing", "гибкого производства"),
        ("crop harvesting", "сбора урожая"),
        ("subsea inspection", "подводной инспекции"),
        ("physical rehabilitation", "физической реабилитации"),
        ("hazardous facility inspection", "инспекции опасных объектов"),
        ("warehouse fulfillment", "складской комплектации"),
        ("construction automation", "автоматизации строительства"),
        ("elderly home assistance", "домашней помощи пожилым людям"),
        ("last-mile delivery", "доставки последней мили"),
        ("laboratory automation", "автоматизации лабораторий"),
        ("surgical assistance", "хирургической помощи"),
        ("nuclear decommissioning", "вывода ядерных объектов из эксплуатации"),
    ]),
    ("Industrial AI", "Индустриальный ИИ", [
        ("foundation models for industrial time series", "фундаментальные модели промышленных временных рядов"),
        ("reinforcement learning on industrial controllers", "обучение с подкреплением на промышленных контроллерах"),
        ("compiler-verified language model generated PLC code", "генерация кода ПЛК языковыми моделями с проверкой компилятором"),
        ("machine-readable P&ID generation", "генерация машиночитаемых P&ID-схем"),
        ("adaptive vision-guided robotic welding", "адаптивная роботизированная сварка с машинным зрением"),
    ], [
        ("chemical process plants", "химических производств"),
        ("semiconductor fabrication", "полупроводниковых фабрик"),
        ("mineral processing", "обогащения полезных ископаемых"),
        ("industrial microgrids", "промышленных микросетей"),
        ("aerospace assembly", "аэрокосмической сборки"),
        ("food manufacturing", "пищевого производства"),
        ("pharmaceutical manufacturing", "фармацевтического производства"),
        ("water treatment plants", "станций водоочистки"),
        ("steel production", "производства стали"),
        ("oil and gas operations", "нефтегазовых операций"),
        ("pulp and paper production", "целлюлозно-бумажного производства"),
        ("cement production", "производства цемента"),
    ]),
    ("Fintech", "Финтех", [
        ("AI agent payment authorization protocols", "протоколы авторизации платежей ИИ-агентов"),
        ("tokenized commercial bank deposits", "токенизированные депозиты коммерческих банков"),
        ("zero-knowledge reusable identity credentials", "переиспользуемые идентификационные данные с доказательствами с нулевым разглашением"),
        ("federated learning for financial crime detection", "федеративное обучение для выявления финансовых преступлений"),
        ("purpose-bound programmable money", "программируемые деньги с целевыми ограничениями"),
    ], [
        ("cross-border settlement", "трансграничных расчётов"),
        ("SME lending", "кредитования малого и среднего бизнеса"),
        ("institutional collateral mobility", "мобильности институционального обеспечения"),
        ("machine-to-machine commerce", "машинной коммерции"),
        ("emerging market financial inclusion", "финансовой доступности на развивающихся рынках"),
        ("offline retail payments", "офлайн-розничных платежей"),
        ("trade finance", "торгового финансирования"),
        ("insurance underwriting", "страхового андеррайтинга"),
        ("central bank digital currency", "цифровых валют центральных банков"),
        ("digital asset custody", "хранения цифровых активов"),
        ("anti-money laundering", "противодействия отмыванию денег"),
        ("programmable settlement", "программируемых расчётов"),
    ]),
    ("Biotechnology", "Биотехнологии и медицина", [
        ("organoid intelligence", "органоидный интеллект"),
        ("biohybrid living actuators", "биогибридные приводы из живой ткани"),
        ("magnetically guided medical microrobots", "магнитоуправляемые медицинские микророботы"),
        ("programmable epigenome editing", "программируемое редактирование эпигенома"),
        ("federated learning across clinical institutions", "федеративное обучение между клиническими учреждениями"),
    ], [
        ("drug screening", "скрининга лекарств"),
        ("precision diagnostics", "точной диагностики"),
        ("targeted drug delivery", "адресной доставки лекарств"),
        ("neurorehabilitation", "нейрореабилитации"),
        ("rare disease treatment", "лечения редких заболеваний"),
        ("antimicrobial therapy", "антимикробной терапии"),
        ("personalized cancer therapy", "персонализированной терапии рака"),
        ("tissue regeneration", "регенерации тканей"),
        ("brain-computer interfaces", "интерфейсов мозг–компьютер"),
        ("organ transplantation", "трансплантации органов"),
        ("gene therapy delivery", "доставки генотерапевтических препаратов"),
        ("digital pathology", "цифровой патологии"),
    ]),
    ("Advanced materials", "Передовые материалы", [
        ("self-healing electronic skin", "самовосстанавливающаяся электронная кожа"),
        ("two-dimensional semiconductor devices", "устройства на двумерных полупроводниках"),
        ("programmable mechanical metamaterials", "программируемые механические метаматериалы"),
        ("solid-state battery electrolytes", "твердотельные электролиты для аккумуляторов"),
        ("carbon-negative cementitious materials", "цементные материалы с отрицательным углеродным следом"),
    ], [
        ("soft robotic systems", "мягких робототехнических систем"),
        ("low-power sensing", "малопотребляющих сенсоров"),
        ("aerospace structures", "аэрокосмических конструкций"),
        ("grid-scale energy storage", "сетевых накопителей энергии"),
        ("high-flux thermal management", "управления высокими тепловыми потоками"),
        ("circular manufacturing", "циклического производства"),
        ("biodegradable electronics", "биоразлагаемой электроники"),
        ("water purification", "очистки воды"),
        ("flexible medical devices", "гибких медицинских устройств"),
        ("hydrogen storage", "хранения водорода"),
        ("wearable electronics", "носимой электроники"),
        ("corrosion protection", "защиты от коррозии"),
    ]),
    ("Clean energy", "Энергетика и климат", [
        ("perovskite-silicon tandem photovoltaics", "перовскит-кремниевые тандемные солнечные элементы"),
        ("sodium-ion batteries", "натрий-ионные аккумуляторы"),
        ("thermochemical heat batteries", "термохимические тепловые аккумуляторы"),
        ("electrochemical ammonia synthesis", "электрохимический синтез аммиака"),
        ("electrochemical direct air carbon capture", "электрохимический прямой захват углерода из воздуха"),
    ], [
        ("AI data centers", "центров обработки данных ИИ"),
        ("decarbonizing heavy industry", "декарбонизации тяжёлой промышленности"),
        ("zero-carbon shipping", "безуглеродного судоходства"),
        ("building heat management", "управления теплом в зданиях"),
        ("remote microgrids", "удалённых микросетей"),
        ("seasonal energy storage", "сезонного хранения энергии"),
        ("green steel production", "производства зелёной стали"),
        ("industrial waste heat recovery", "утилизации промышленного сбросного тепла"),
        ("aviation fuel production", "производства авиационного топлива"),
        ("long-duration grid storage", "длительного сетевого хранения энергии"),
        ("green hydrogen production", "производства зелёного водорода"),
        ("carbon-neutral chemicals", "углеродно-нейтрального производства химикатов"),
    ]),
    ("Quantum technology", "Квантовые технологии", [
        ("neutral-atom quantum processors", "квантовые процессоры на нейтральных атомах"),
        ("quantum repeater networks", "сети квантовых повторителей"),
        ("diamond quantum sensors", "алмазные квантовые сенсоры"),
        ("integrated photonic quantum processors", "интегральные фотонные квантовые процессоры"),
        ("quantum-inspired tensor optimization", "квантово-инспирированная тензорная оптимизация"),
    ], [
        ("navigation without satellites", "навигации без спутников"),
        ("molecular drug discovery", "молекулярного поиска лекарств"),
        ("electric grid optimization", "оптимизации электрических сетей"),
        ("financial risk simulation", "моделирования финансовых рисков"),
        ("materials discovery", "поиска новых материалов"),
        ("secure telecommunications", "защищённой телекоммуникации"),
        ("medical imaging", "медицинской визуализации"),
        ("underground infrastructure sensing", "мониторинга подземной инфраструктуры"),
        ("high-precision timing", "высокоточной синхронизации времени"),
        ("battery material simulation", "моделирования материалов аккумуляторов"),
        ("geophysical exploration", "геофизической разведки"),
        ("quantum chemistry", "квантовой химии"),
    ]),
    ("Space and ocean", "Космос и океан", [
        ("orbital edge computing", "орбитальные периферийные вычисления"),
        ("space-based solar power", "космическая солнечная энергетика"),
        ("autonomous underwater sensor networks", "автономные подводные сенсорные сети"),
        ("onboard hyperspectral satellite AI", "бортовой ИИ для гиперспектральных спутников"),
        ("in-space robotic manufacturing", "роботизированное производство в космосе"),
    ], [
        ("disaster response", "реагирования на стихийные бедствия"),
        ("climate monitoring", "мониторинга климата"),
        ("mineral exploration", "разведки полезных ископаемых"),
        ("precision agriculture", "точного земледелия"),
        ("maritime security", "морской безопасности"),
        ("resilient communications", "устойчивой связи"),
        ("wildfire detection", "выявления лесных пожаров"),
        ("ocean carbon monitoring", "мониторинга углерода в океане"),
        ("polar research", "полярных исследований"),
        ("offshore energy inspection", "инспекции морской энергетической инфраструктуры"),
        ("fisheries monitoring", "мониторинга рыболовства"),
        ("lunar construction", "строительства на Луне"),
    ]),
    ("Food and agriculture", "Пищевые и агротехнологии", [
        ("precision fermentation", "прецизионная ферментация"),
        ("electrochemical fertilizer production", "электрохимическое производство удобрений"),
        ("vision-guided robotic weeding", "роботизированная прополка с машинным зрением"),
        ("plant bioelectronic interfaces", "биоэлектронные интерфейсы растений"),
        ("engineered microbial biostimulants", "инженерные микробные биостимуляторы"),
    ], [
        ("low-carbon protein production", "низкоуглеродного производства белка"),
        ("crop drought resilience", "устойчивости культур к засухе"),
        ("soil health monitoring", "мониторинга здоровья почв"),
        ("controlled-environment agriculture", "земледелия в контролируемой среде"),
        ("smallholder farming", "малых фермерских хозяйств"),
        ("early crop disease detection", "раннего выявления болезней растений"),
        ("livestock health monitoring", "мониторинга здоровья сельскохозяйственных животных"),
        ("post-harvest food preservation", "послеуборочного хранения продуктов"),
        ("alternative dairy production", "альтернативного производства молочных продуктов"),
        ("agricultural carbon measurement", "измерения углерода в сельском хозяйстве"),
        ("vertical farming", "вертикального земледелия"),
        ("sustainable aquaculture", "устойчивой аквакультуры"),
    ]),
]

STOP = {"for", "and", "the", "of", "to", "in", "on", "with", "across", "a", "an",
        "de", "del", "la", "las", "los", "para", "por", "com", "da", "do", "dos", "das",
        "des", "du", "les", "une", "und", "der", "die", "für", "на", "для", "по", "и", "в", "о"}
MARKETING = {"revolutionary", "breakthrough", "game-changing", "unprecedented", "transformative", "disruptive"}
TECHNICAL = {"model", "sensor", "processor", "computing", "learning", "electrochemical", "photonic",
             "quantum", "robot", "network", "protocol", "algorithm", "controller", "semiconductor",
             "battery", "electrolyte", "actuator", "metamaterial", "microfluidic", "federated"}

# Native-language search seeds. Every candidate still needs a dated Crossref source;
# entries without a suitable source are rejected by the same rules as English seeds.
MULTILINGUAL_CANDIDATES = [
    ("es", "Энергетика и климат", "baterías de sodio-ion", "натрий-ионные аккумуляторы", "almacenamiento de energía", "хранения энергии"),
    ("es", "Роботы", "robótica blanda", "мягкая робототехника", "rehabilitación médica", "медицинской реабилитации"),
    ("es", "Пищевые и агротехнологии", "fermentación de precisión", "прецизионная ферментация", "proteínas alternativas", "альтернативных белков"),
    ("pt", "Энергетика и климат", "baterias de íons de sódio", "натрий-ионные аккумуляторы", "armazenamento de energia", "хранения энергии"),
    ("pt", "Пищевые и агротехнологии", "fermentação de precisão", "прецизионная ферментация", "produção de proteínas", "производства белка"),
    ("fr", "Энергетика и климат", "batteries sodium-ion", "натрий-ионные аккумуляторы", "stockage d'énergie", "хранения энергии"),
    ("fr", "Роботы", "robotique souple", "мягкая робототехника", "rééducation", "реабилитации"),
    ("de", "Энергетика и климат", "Natrium-Ionen-Batterien", "натрий-ионные аккумуляторы", "Energiespeicherung", "хранения энергии"),
    ("de", "Квантовые технологии", "Quantensensoren", "квантовые сенсоры", "Navigation", "навигации"),
    ("zh", "Энергетика и климат", "钠离子电池", "натрий-ионные аккумуляторы", "储能", "хранения энергии"),
    ("zh", "Передовые материалы", "柔性电子皮肤", "гибкая электронная кожа", "机器人", "роботов"),
    ("ja", "Энергетика и климат", "ペロブスカイト太陽電池", "перовскитные солнечные элементы", "建築", "строительства"),
    ("ru", "Энергетика и климат", "натрий-ионные аккумуляторы", "натрий-ионные аккумуляторы", "накопления энергии", "накопления энергии"),
    ("ru", "Роботы", "биогибридные роботы", "биогибридные роботы", "медицинского применения", "медицинского применения"),
]

NATIVE_TECH_ANCHORS = {
    "baterías de sodio-ion": "sodio",
    "baterias de íons de sódio": "sódio",
    "batteries sodium-ion": "sodium",
    "Natrium-Ionen-Batterien": "natrium",
    "натрий-ионные аккумуляторы": "натрий",
    "биогибридные роботы": "биогибрид",
    "fermentación de precisión": "fermentación",
    "fermentação de precisão": "fermentação",
}


def candidates():
    for domain_en, domain_ru, technologies, applications in DOMAIN_SPECS:
        for tech_en, tech_ru in technologies:
            for app_en, app_ru in applications:
                yield {
                    "domain_original": domain_en,
                    "domain_ru": domain_ru,
                    "technology_original": tech_en,
                    "technology_ru": tech_ru,
                    "application_original": app_en,
                    "application_ru": app_ru,
                    "name_original": f"{tech_en} for {app_en}",
                    "name_ru": f"{tech_ru} для {app_ru}",
                    "original_language": "en",
                    "search_query": f"{tech_en} {app_en}",
                }
    connectors = {"es": "para", "pt": "para", "fr": "pour", "de": "für", "zh": "用于", "ja": "向け", "ru": "для"}
    for language, domain_ru, tech, tech_ru, app, app_ru in MULTILINGUAL_CANDIDATES:
        connector = connectors[language]
        name_original = f"{tech}{connector}{app}" if language in {"zh", "ja"} else f"{tech} {connector} {app}"
        yield {
            "domain_original": domain_ru,
            "domain_ru": domain_ru,
            "technology_original": tech,
            "technology_ru": tech_ru,
            "application_original": app,
            "application_ru": app_ru,
            "name_original": name_original,
            "name_ru": f"{tech_ru} для {app_ru}",
            "original_language": language,
            "search_query": f"{tech} {app}",
        }


def load_candidates_file(path: Path) -> list[dict[str, str]]:
    """Read arbitrary areas/languages; the bundled 12-area seed is only a fallback."""
    required = {
        "domain_original", "domain_ru", "technology_original", "technology_ru",
        "application_original", "application_ru", "name_original", "name_ru",
        "original_language", "search_query",
    }
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Candidate file is missing columns: {', '.join(sorted(missing))}")
        result = []
        seen = set()
        for line, raw in enumerate(reader, start=2):
            item = {key: (raw.get(key) or "").strip() for key in required}
            if any(not value for value in item.values()):
                raise ValueError(f"Candidate file has blank required fields on line {line}")
            key = item["name_original"].casefold()
            if key in seen:
                raise ValueError(f"Duplicate original candidate name on line {line}")
            seen.add(key)
            result.append(item)
    if not result:
        raise ValueError("Candidate file contains no rows")
    return result


def cache_key(params):
    return hashlib.sha256(json.dumps(params, sort_keys=True).encode()).hexdigest()


class OpenAlexClient:
    def __init__(self, api_key: str = "", delay: float = 0.08):
        self.api_key = api_key
        self.delay = delay
        self.client = httpx.Client(timeout=30, headers={"User-Agent": "IDEA-weak-signals/1.0"})
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get(self, params):
        params = dict(params)
        if self.api_key:
            params["api_key"] = self.api_key
        key = cache_key(params)
        path = CACHE_DIR / f"{key}.json"
        if path.exists():
            return json.loads(path.read_text())
        error = None
        for attempt in range(5):
            try:
                response = self.client.get(OPENALEX, params=params)
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                data = response.json()
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(data, ensure_ascii=False))
                temp.replace(path)
                time.sleep(self.delay)
                return data
            except Exception as exc:  # noqa: BLE001
                error = exc
                time.sleep(1 + attempt)
        raise RuntimeError(f"OpenAlex request failed: {type(error).__name__}")

    def close(self):
        self.client.close()


class CrossrefClient:
    """Polite cached Crossref client used when OpenAlex budget is unavailable."""

    def __init__(self, delay: float = 0.12):
        self.delay = delay
        self.client = httpx.Client(timeout=45, headers={"User-Agent": "IDEA-weak-signals/1.0"})
        CROSSREF_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def get(self, technology: str, snapshot: date):
        params = {
            "query.bibliographic": technology,
            "filter": f"from-pub-date:2018-01-01,until-pub-date:{snapshot.isoformat()}",
            "rows": 200,
            "select": "DOI,title,URL,published,is-referenced-by-count,type,author,abstract,publisher",
        }
        path = CROSSREF_CACHE_DIR / f"{cache_key(params)}.json"
        if path.exists():
            return json.loads(path.read_text())
        last_error = None
        for attempt in range(5):
            try:
                response = self.client.get(CROSSREF, params=params)
                if response.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                data = response.json()["message"]
                temp = path.with_suffix(".tmp")
                temp.write_text(json.dumps(data, ensure_ascii=False))
                temp.replace(path)
                time.sleep(self.delay)
                return data
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                time.sleep(1 + attempt)
        raise RuntimeError(f"Crossref request failed: {type(last_error).__name__}")

    def close(self):
        self.client.close()


def abstract_text(work):
    if work.get("abstract_text"):
        return work["abstract_text"]
    index = work.get("abstract_inverted_index") or {}
    positions = [(pos, word) for word, values in index.items() for pos in values]
    return " ".join(word for _, word in sorted(positions))


def tokens(text):
    return {x for x in re.findall(r"[^\W_][\w-]*", text.casefold()) if len(x) > 1 and x not in STOP}


def relevance(candidate, work):
    text = f"{work.get('title') or ''} {abstract_text(work)}".lower()
    anchor = NATIVE_TECH_ANCHORS.get(candidate["technology_original"])
    if anchor and anchor.casefold() not in text.casefold():
        return 0.0
    tech = tokens(candidate["technology_original"])
    app = tokens(candidate["application_original"])
    tech_score = len(tech & tokens(text)) / max(1, len(tech))
    app_score = len(app & tokens(text)) / max(1, len(app))
    # A source must support both the base technology and the stated application.
    # Otherwise one broad paper could be reused to justify unrelated applications.
    if tech_score < 0.5 or app_score < 0.5:
        return 0.0
    return 0.65 * tech_score + 0.35 * app_score


def source_excerpt(candidate, work):
    """Return a short, auditable passage; metadata matching is not a proof of deployment."""
    title = re.sub(r"\s+", " ", work.get("title") or "").strip()
    abstract = re.sub(r"\s+", " ", abstract_text(work)).strip()
    tech = tokens(candidate["technology_original"])
    app = tokens(candidate["application_original"])
    if abstract:
        passages = re.split(r"(?<=[.!?])\s+", abstract)
        scored = []
        for passage in passages:
            present = tokens(passage)
            tech_overlap = len(tech & present) / max(1, len(tech))
            app_overlap = len(app & present) / max(1, len(app))
            scored.append((min(tech_overlap, app_overlap), tech_overlap + app_overlap, passage))
        if scored:
            best = max(scored)
            if best[0] > 0:
                return best[2][:600], "same_abstract_sentence"
        return abstract[:600], "title_or_abstract_terms"
    return title[:600], "title_terms_only"


def query_openalex(client, candidate, snapshot):
    common_filter = f"from_publication_date:2018-01-01,to_publication_date:{snapshot.isoformat()}"
    # One cached scientific corpus per technology is reused across all applications.
    # This reduces 960 external calls for 480 candidates to 120 calls for 60 technologies.
    search_term = candidate["technology_original"]
    works = client.get({
        "search": search_term, "filter": common_filter, "per-page": 100,
        "select": "id,title,display_name,publication_date,publication_year,doi,cited_by_count,type,language,authorships,abstract_inverted_index,primary_location",
    }).get("results", [])
    groups = client.get({
        "search": search_term, "filter": common_filter,
        "group_by": "publication_year", "per-page": 200,
    }).get("group_by", [])
    year_counts = {int(x["key"]): int(x["count"]) for x in groups if str(x.get("key", "")).isdigit()}
    ranked = sorted(((relevance(candidate, w), w) for w in works), key=lambda x: (x[0], x[1].get("publication_date") or ""), reverse=True)
    ranked = [(score, work) for score, work in ranked if score >= 0.28]
    return ranked, year_counts


def crossref_date(item):
    parts = ((item.get("published") or {}).get("date-parts") or [[]])[0]
    if not parts:
        return None
    year = int(parts[0])
    month = int(parts[1]) if len(parts) > 1 else 1
    day = int(parts[2]) if len(parts) > 2 else 1
    try:
        return date(year, month, day).isoformat()
    except ValueError:
        return f"{year:04d}-01-01"


def crossref_date_precision(item):
    parts = ((item.get("published") or {}).get("date-parts") or [[]])[0]
    return {1: "year", 2: "month"}.get(len(parts), "day")


def normalize_crossref_work(item):
    published = crossref_date(item)
    raw_abstract = re.sub(r"<[^>]+>", " ", item.get("abstract") or "")
    raw_abstract = re.sub(r"&lt;[^&]+?&gt;", " ", raw_abstract)
    raw_abstract = re.sub(r"\s+", " ", raw_abstract).strip()
    work_type = "preprint" if item.get("type") == "posted-content" else item.get("type")
    title = re.sub(r"\s+", " ", (item.get("title") or [""])[0]).strip()
    return {
        "id": f"https://api.crossref.org/works/{item.get('DOI')}" if item.get("DOI") else item.get("URL"),
        "title": title, "publication_date": published,
        "publication_date_precision": crossref_date_precision(item),
        "publication_year": int(published[:4]) if published else None,
        "doi": f"https://doi.org/{item['DOI']}" if item.get("DOI") else None,
        "cited_by_count": item.get("is-referenced-by-count") or 0, "type": work_type,
        "language": "unknown", "authorships": [], "abstract_text": raw_abstract,
        "publisher": item.get("publisher"),
    }


def query_crossref(client, candidate, snapshot):
    data = client.get(candidate["technology_original"], snapshot)
    works = [normalize_crossref_work(item) for item in data.get("items", [])]
    ranked = sorted(((relevance(candidate, w), w) for w in works),
                    key=lambda x: (x[0], x[1].get("publication_date") or ""), reverse=True)
    ranked = [(score, work) for score, work in ranked if score >= 0.28]
    year_counts = Counter(w["publication_year"] for score, w in ranked if w.get("publication_year"))
    return ranked, dict(year_counts)


def institution_stats(works):
    institutions, countries, company_works = set(), set(), 0
    hhi_counts = Counter()
    for work in works:
        has_company = False
        work_institutions = set()
        for authorship in work.get("authorships") or []:
            for institution in authorship.get("institutions") or []:
                iid = institution.get("id")
                if iid:
                    institutions.add(iid)
                    work_institutions.add(iid)
                if institution.get("country_code"):
                    countries.add(institution["country_code"])
                if institution.get("type") == "company":
                    has_company = True
        company_works += int(has_company)
        for iid in work_institutions:
            hhi_counts[iid] += 1
    total = sum(hhi_counts.values())
    hhi = sum((count / total) ** 2 for count in hhi_counts.values()) if total else ""
    return institutions, countries, company_works / max(1, len(works)), hhi


def stage_features(text):
    low = text.lower()
    poc = bool(re.search(r"prototype|proof[- ]of[- ]concept|experimental demonstrat", low))
    pilot = bool(re.search(r"\bpilot\b|field trial|clinical trial", low))
    adoption = bool(re.search(r"real[- ]world|field demonstrat|deployed|deployment|first-in-human", low))
    mature = bool(re.search(r"mass production|commercially available|large[- ]scale deployment|industry standard", low))
    # The schema describes one-hot stage columns, not independent term flags.
    if mature:
        return [0, 0, 0, 0, 1]
    if adoption:
        return [0, 0, 0, 1, 0]
    if pilot:
        return [0, 0, 1, 0, 0]
    if poc:
        return [0, 1, 0, 0, 0]
    return [1, 0, 0, 0, 0]


def log_growth(current, previous):
    return math.log1p(current) - math.log1p(previous)


def row_from_candidate(candidate, ranked, years, snapshot, *, allow_mature=False, require_recent=True):
    score = ranked[0][0]
    relevant = [work for rel, work in ranked if rel >= max(0.28, score - 0.25)]
    recent_cutoff = snapshot - timedelta(days=365 * 2)
    science_cutoff = snapshot - timedelta(days=365 * 3)
    def published_date(work):
        try:
            return date.fromisoformat(work["publication_date"])
        except (TypeError, ValueError):
            return None
    recent_works = [w for w in relevant if (d := published_date(w)) and recent_cutoff <= d <= snapshot]
    if require_recent and not recent_works:
        return None
    recent_ranked = [(rel, work) for rel, work in ranked if work in recent_works]
    recent_ranked.sort(key=lambda item: (item[0], item[1].get("publication_date") or ""), reverse=True)
    score, primary = recent_ranked[0] if recent_ranked else ranked[0]
    previous_cutoff = recent_cutoff - timedelta(days=365 * 2)
    older_cutoff = previous_cutoff - timedelta(days=365 * 2)
    recent = len(recent_works)
    previous = sum(previous_cutoff <= d < recent_cutoff for w in relevant if (d := published_date(w)))
    older = sum(older_cutoff <= d < previous_cutoff for w in relevant if (d := published_date(w)))
    growth = log_growth(recent, previous)
    acceleration = growth - log_growth(previous, older)
    dated = [d for w in relevant if (d := published_date(w)) and d <= snapshot]
    first_date = min(dated) if dated else snapshot
    citations = []
    for work in recent_works:
        try:
            published = date.fromisoformat(work["publication_date"])
            months = max(1, (snapshot.year - published.year) * 12 + snapshot.month - published.month)
            citations.append((work.get("cited_by_count") or 0) / months)
        except (TypeError, ValueError):
            pass
    science_works = [w for w in relevant if (d := published_date(w)) and science_cutoff <= d <= snapshot]
    institutions, countries, company_share, _ = institution_stats(science_works)
    combined_text = " ".join(f"{w.get('title') or ''} {abstract_text(w)}" for w in relevant)
    stages = stage_features(combined_text)
    if stages[4] and not allow_mature:
        return None
    words = re.findall(r"[a-z][a-z0-9-]+", combined_text.lower())
    marketing_density = 1000 * sum(w in MARKETING for w in words) / max(1, len(words))
    technical_density = sum(w in TECHNICAL for w in words) / max(1, len(words))
    specificity = min(1.0, (len(re.findall(r"\b\d+(?:\.\d+)?\s*(?:%|x|nm|µm|mm|w|kw|mw|kwh|wh|ms|ghz|mhz)\b", combined_text.lower())) +
                            len(re.findall(r"compared with|outperform|reduced|increased|efficiency|accuracy", combined_text.lower()))) / max(1, len(relevant)))
    preprints = sum((w.get("type") or "") == "preprint" for w in recent_works)
    primary_url = primary.get("doi") or primary.get("id")
    excerpt, match_method = source_excerpt(candidate, primary)
    feature = {name: "" for name in FEATURE_NAMES}
    feature.update({
        "stage_research": stages[0], "stage_poc": stages[1], "stage_pilot": stages[2],
        "stage_early_adoption": stages[3], "stage_scaling_or_mature": stages[4],
        "first_evidence_age_years": round((snapshot - first_date).days / 365.25, 3),
        # A word such as "pilot" in an abstract is not a verified customer pilot.
        # Adoption features remain missing until deployment registries are checked.
        "papers_log_3y": round(math.log1p(len(science_works)), 6),
        "paper_growth_2y": round(growth, 6), "paper_acceleration": round(acceleration, 6),
        "citation_velocity_median": round(statistics.median(citations), 6) if citations else "",
        "preprint_share_2y": round(preprints / max(1, len(recent_works)), 6),
        # Crossref author affiliation coverage is sparse. An empty list means
        # unknown, not zero independent institutions or zero industry authors.
        "institution_count_log_3y": round(math.log1p(len(institutions)), 6) if institutions else "",
        "industry_affiliation_share_3y": round(company_share, 6) if institutions else "",
        "science_country_count_log_3y": round(math.log1p(len(countries)), 6) if countries else "",
        "source_type_diversity": 1,
        # DOI existence does not establish high trust, independence or URL reachability.
        "marketing_claim_density": round(marketing_density, 6),
        "technical_term_density": round(technical_density, 6),
        "claim_specificity": round(specificity, 6),
        "cross_channel_confirmation_count": int(growth > 0 and recent >= 2),
        "missing_science": 0, "missing_patents": 1, "missing_media": 1,
        "missing_funding": 1, "missing_adoption": 1,
    })
    stable = hashlib.sha256(f"{candidate['name_original']}|{snapshot}".encode()).hexdigest()[:16]
    technology_group_id = hashlib.sha256(candidate["technology_original"].casefold().encode()).hexdigest()[:12]
    return {
        "id": f"pos_{stable}", "label": 1, "data_tier": "bronze",
        "label_status": "provisional_positive", "review_status": "needs_human_review",
        "snapshot_date": snapshot.isoformat(), "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "technology_group_id": f"tech_{technology_group_id}",
        **candidate,
        "primary_source_title_original": primary.get("title") or primary.get("display_name"),
        "primary_source_url": primary_url,
        "primary_source_crossref_id": primary.get("id"), "primary_source_doi": primary.get("doi"),
        "primary_source_date": primary.get("publication_date"),
        "primary_source_date_precision": primary.get("publication_date_precision") or "unknown",
        "primary_source_language": primary.get("language") or "unknown",
        "primary_source_publisher": primary.get("publisher") or "unknown",
        "source_evidence_excerpt_original": excerpt,
        "source_match_method": match_method,
        "source_countries": "|".join(sorted(countries)),
        "source_relevance_score": round(score, 6),
        "science_sample_size": len(relevant),
        "science_count_scope": "relevant_works_in_top_200_crossref_results",
        "selection_rule": "curated_application+crossref_title_abstract_term_overlap+recent_science",
        "evidence_note": "Crossref publication supports search-term overlap only; weak-signal status and exact application require expert review.",
        **feature,
    }


def balanced_selection(rows, target):
    groups = {}
    for row in rows:
        groups.setdefault(row["domain_ru"], []).append(row)
    for group in groups.values():
        group.sort(key=lambda r: (float(r["source_relevance_score"]), r["primary_source_date"]), reverse=True)
    selected = []
    domains = sorted(groups)
    while len(selected) < target and domains:
        remaining = []
        for domain in domains:
            if groups[domain] and len(selected) < target:
                selected.append(groups[domain].pop(0))
            if groups[domain]:
                remaining.append(domain)
        domains = remaining
    return selected


def build(output: Path, target: int, snapshot: date, candidate_file: Path | None = None,
          enrich_external: bool = False, news_only: bool = False):
    client = CrossrefClient()
    rows, failures = [], []
    source_candidates = load_candidates_file(candidate_file) if candidate_file else list(candidates())
    try:
        for index, candidate in enumerate(source_candidates, 1):
            try:
                ranked, years = query_crossref(client, candidate, snapshot)
                row = row_from_candidate(candidate, ranked, years, snapshot) if ranked else None
                if row:
                    rows.append(row)
                else:
                    failures.append({"name": candidate["name_original"], "reason": "no relevant recent non-mature source"})
            except Exception as exc:  # noqa: BLE001
                failures.append({"name": candidate["name_original"], "reason": type(exc).__name__})
            if index % 20 == 0:
                print(f"processed={index} accepted={len(rows)} failures={len(failures)}", flush=True)
    finally:
        client.close()
    if len(rows) < 300:
        raise RuntimeError(f"Only {len(rows)} candidates passed; at least 300 required")
    # A percentile of curated positives would leak the construction of this set.
    # This feature needs an independent, domain-wide background corpus.
    rows = balanced_selection(rows, target)
    if enrich_external:
        from scripts.enrich_public_sources import enrich_rows, enrich_technology_context
        rows = enrich_rows(rows, snapshot, channels=("news",) if news_only else ("patent", "news"))
        rows = enrich_technology_context(rows, snapshot)
    from scripts.enrich_public_sources import EXTRA_COLUMNS
    for row in rows:
        for column in EXTRA_COLUMNS:
            row.setdefault(column, "")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = [
        "id", "label", "data_tier", "label_status", "review_status", "snapshot_date",
        "feature_schema_version", "technology_group_id", "domain_original", "domain_ru", "technology_original",
        "technology_ru", "application_original", "application_ru", "name_original", "name_ru",
        "original_language", "search_query", "primary_source_title_original", "primary_source_url",
        "primary_source_crossref_id", "primary_source_doi", "primary_source_date", "primary_source_date_precision",
        "primary_source_language", "primary_source_publisher", "source_evidence_excerpt_original", "source_match_method",
        "source_countries", "source_relevance_score", "science_sample_size", "science_count_scope", "selection_rule",
        "evidence_note",
        *EXTRA_COLUMNS,
    ]
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=metadata + FEATURE_NAMES, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    report_dir = ROOT / "storage" / "dataset_reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    report = report_dir / f"{output.stem}.report.json"
    report.write_text(json.dumps({
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "snapshot_date": snapshot.isoformat(), "rows": len(rows), "features": len(FEATURE_NAMES),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "candidate_source": str(candidate_file) if candidate_file else "bundled_seed_examples",
        "candidate_pool_size": len(source_candidates),
        "candidate_pool_domains": dict(Counter(row["domain_ru"] for row in source_candidates)),
        "domains": Counter(r["domain_ru"] for r in rows), "failed_candidates": failures,
        "source_match_methods": Counter(r["source_match_method"] for r in rows),
        "source_date_precision": Counter(r["primary_source_date_precision"] for r in rows),
        "feature_nonmissing_count": {name: sum(r[name] != "" for r in rows) for name in FEATURE_NAMES},
        "feature_variable_count": sum(len({r[name] for r in rows}) > 1 for name in FEATURE_NAMES),
        "unique_primary_sources": len({r["primary_source_url"] for r in rows}),
        "corroborating_sources": {
            channel: sum(bool(r.get(f"{channel}_source_url")) for r in rows)
            for channel in ("patent", "media", "official", "funding", "deployment")
        },
        "technology_context_sources": {
            channel: sum(bool(r.get(f"technology_{channel}_source_url")) for r in rows)
            for channel in ("media", "official", "funding")
        },
        "limitations": [
            "Bronze positive candidates: human review is required before supervised training.",
            "Crossref term overlap between the curated technology and application is evidence of topical relevance, not proof that the application exists or is a weak signal.",
            "Science counts are from the relevant subset of the top 200 Crossref search results per technology, not exhaustive global publication counts.",
            "Crossref publication dates may have only month or year precision; the CSV preserves this uncertainty.",
            "Patent and media search results are corroborating records, not exhaustive counts. Funding and deployment snippets are leads for human checking, not measured totals.",
            "Patent, media, funding and most adoption numeric features remain missing until complete channel-specific series are verified; missing indicators are present.",
            "Candidate names are curated English phrases and prepared Russian renderings; source titles retain their original language.",
        ],
    }, ensure_ascii=False, indent=2, default=dict))
    print(f"saved {len(rows)} rows to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--target", type=int, default=350, choices=range(300, 401), metavar="300..400")
    parser.add_argument("--snapshot-date", type=date.fromisoformat, default=SNAPSHOT_DATE)
    parser.add_argument("--candidates-file", type=Path, help="CSV with arbitrary domains and languages; overrides bundled examples")
    parser.add_argument("--enrich-external", action="store_true", help="Add keyless public patent and news corroboration")
    parser.add_argument("--news-only", action="store_true", help="Skip patents when public search is unavailable")
    args = parser.parse_args()
    build(args.output, args.target, args.snapshot_date, args.candidates_file,
          args.enrich_external, args.news_only)
