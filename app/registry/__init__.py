"""Реестр источников: тип, уровень доверия и роль каждого домена.

Единственное место, где задано доверие к источникам. Им пользуются:
  - радар (app/radar/sources.py) — тип и доверие каждого прочитанного документа;
  - признаки модели (app/ml/features/extract.py) — доли пресс-релизов, изданий высокого
    доверия и непроверяемых источников среди упоминаний;
  - правило подтверждения (supports_signal) — соцсети, блоги и агрегаторы по ТЗ служат
    только первичным индикатором, пресс-релиз не может быть единственным подтверждением.

Данные — app/registry/sources.json. Сначала применяются шаблоны URL для пресс-релизов,
затем точный домен или его поддомен (самое длинное совпадение), затем шаблоны доменов
(.gov, .edu и международные варианты), иначе «веб-источник» с неустановленным доверием.

Изменение реестра меняет признаки модели: после правки нужно пересобрать датасет из кэша
и переобучить модель (docs/08-dataset.md, раздел «Реестр источников»).
"""
from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

REGISTRY_PATH = Path(__file__).resolve().parent / "sources.json"
ROOT = Path(__file__).resolve().parents[2]
OVERRIDES_PATH = ROOT / "storage" / "source_registry_overrides.json"
TRUST_ORDER = {"high": 3, "medium": 2, "low": 1, "unverified": 0}
TRUST_LABELS = {"high": "Высокая", "medium": "Средняя", "low": "Пониженная", "unverified": "Не установлена"}
_WRITE_LOCK = threading.RLock()


@dataclass(frozen=True)
class SourceClass:
    host: str
    type: str
    type_label: str
    trust: str
    primary_only: bool
    reason: str
    name: str | None = None
    matched: str | None = None  # домен или шаблон реестра, по которому определено

    def as_dict(self) -> dict:
        return asdict(self)


def _modified(path: Path) -> int:
    try:
        return path.stat().st_mtime_ns
    except FileNotFoundError:
        return 0


@lru_cache(maxsize=8)
def _load_registry(registry_mtime: int, overrides_mtime: int) -> dict:
    data = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    overrides = json.loads(OVERRIDES_PATH.read_text(encoding="utf-8")) if overrides_mtime else {}
    deleted = set(overrides.get("deleted", []))
    custom = overrides.get("upsert", {})
    domains = {d["domain"]: dict(d) for d in data["domains"] if d["domain"] not in deleted}
    for domain, entry in custom.items():
        domains[domain] = {"domain": domain, "name": entry["name"], "type": entry["type"],
                           "lang": entry["lang"], "customized": True}
    for domain, entry in domains.items():
        entry.setdefault("customized", domain in custom)
    data["domains"] = sorted(domains.values(), key=lambda d: d["domain"])
    data["_custom_domains"] = set(custom)
    data["_overrides_updated"] = overrides.get("updated")
    unknown = {d["type"] for d in data["domains"]} - set(data["types"])
    if unknown:
        raise ValueError(f"реестр источников: неизвестные типы {sorted(unknown)}")
    data["_by_domain"] = {d["domain"]: d for d in data["domains"]}
    return data


def load() -> dict:
    """Load bundled defaults plus persistent user edits."""
    return _load_registry(_modified(REGISTRY_PATH), _modified(OVERRIDES_PATH))


def normalize_domain(value: str) -> str:
    domain = (value or "").strip().lower().rstrip(".")
    if not domain or any(char in domain for char in "/:@?#") or ".." in domain:
        raise ValueError("Введите домен без протокола, пути или порта, например example.org.")
    domain = domain.removeprefix("www.")
    try:
        ascii_domain = domain.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError("Не удалось распознать домен.") from exc
    labels = ascii_domain.split(".")
    if (len(ascii_domain) > 253 or len(labels) < 2
            or any(not label or len(label) > 63
                   or not re.fullmatch(r"[a-z0-9](?:[a-z0-9-]*[a-z0-9])?", label)
                   for label in labels)):
        raise ValueError("Введите корректный домен, например example.org.")
    return ascii_domain


def _write_overrides(data: dict) -> None:
    OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    temp = OVERRIDES_PATH.with_suffix(".tmp")
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(OVERRIDES_PATH)
    _load_registry.cache_clear()


def save_domain(domain: str, name: str, type_: str, lang: str) -> dict:
    domain = normalize_domain(domain)
    name = (name or "").strip()
    lang = (lang or "").strip().lower()
    if not name or len(name) > 120:
        raise ValueError("Укажите название источника (не более 120 символов).")
    if not re.fullmatch(r"[a-z]{2,8}", lang):
        raise ValueError("Язык укажите коротким кодом, например ru или en.")
    if type_ not in load()["types"]:
        raise ValueError("Выберите тип источника из списка.")
    with _WRITE_LOCK:
        overrides = (json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
                     if OVERRIDES_PATH.exists() else {"upsert": {}, "deleted": []})
        overrides.setdefault("upsert", {})[domain] = {"name": name, "type": type_, "lang": lang}
        overrides["deleted"] = [d for d in overrides.setdefault("deleted", []) if d != domain]
        overrides["updated"] = datetime.now(timezone.utc).date().isoformat()
        _write_overrides(overrides)
    return registry_table()


def delete_domain(value: str) -> bool:
    domain = normalize_domain(value)
    data = load()
    if domain not in data["_by_domain"]:
        return False
    bundled = {d["domain"] for d in json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))["domains"]}
    with _WRITE_LOCK:
        overrides = (json.loads(OVERRIDES_PATH.read_text(encoding="utf-8"))
                     if OVERRIDES_PATH.exists() else {"upsert": {}, "deleted": []})
        overrides.setdefault("upsert", {}).pop(domain, None)
        deleted = set(overrides.setdefault("deleted", []))
        if domain in bundled:
            deleted.add(domain)
        else:
            deleted.discard(domain)
        overrides["deleted"] = sorted(deleted)
        overrides["updated"] = datetime.now(timezone.utc).date().isoformat()
        _write_overrides(overrides)
    return True


def host_of(url_or_host: str) -> str:
    value = (url_or_host or "").strip().lower()
    host = urlsplit(value).hostname if "://" in value else value.split("/")[0]
    host = (host or "").removeprefix("www.")
    try:
        return host.encode("idna").decode("ascii")
    except UnicodeError:
        return host


def _type_class(host: str, type_: str, matched: str | None, name: str | None = None) -> SourceClass:
    spec = load()["types"][type_]
    return SourceClass(host=host, type=type_, type_label=spec["label"], trust=spec["trust"],
                       primary_only=spec["primary_only"], reason=spec["reason"], name=name, matched=matched)


def classify(url_or_host: str) -> SourceClass:
    data = load()
    host = host_of(url_or_host)
    parts = host.split(".")
    # Content-level classifications (for example a press-release URL) take
    # precedence over publisher-domain trust. An official or reputable domain
    # can still publish a first-party announcement that cannot independently
    # confirm its own claims.
    for pattern in data["patterns"]:
        if "url_regex" in pattern and re.search(pattern["url_regex"], url_or_host or "", re.IGNORECASE):
            return _type_class(host, pattern["type"], pattern["url_regex"])
    for i in range(len(parts) - 1):  # от полного хоста к родительским доменам
        entry = data["_by_domain"].get(".".join(parts[i:]))
        if entry:
            return _type_class(host, entry["type"], entry["domain"], entry.get("name"))
    for pattern in data["patterns"]:
        if "suffix" in pattern and (host.endswith(pattern["suffix"]) or host == pattern["suffix"].lstrip(".")):
            return _type_class(host, pattern["type"], "*" + pattern["suffix"])
    return _type_class(host, "website", None)


def is_press_release(url_or_host: str) -> bool:
    return classify(url_or_host).type == "press_release"


def is_high_trust(url_or_host: str) -> bool:
    return classify(url_or_host).trust == "high"


def is_unverifiable(url_or_host: str) -> bool:
    """Соцсети, блоги, агрегаторы — по ТЗ только первичный индикатор."""
    return not host_of(url_or_host) or classify(url_or_host).primary_only


def supports_signal(sources: list[dict]) -> tuple[bool, list[str]]:
    """Достаточно ли подтверждающих источников для слабого сигнала.

    sources — источники, на которые опираются подтверждающие цитаты (поля url, trust
    и primary_only, как в радаре). Возвращает (достаточно, ограничения для карточки).
    """
    if not sources:
        return False, ["Нет подтверждающих цитат из прочитанных источников."]
    classes = [classify(s["url"]) for s in sources]
    independent = [c for c in classes if not c.primary_only and c.type != "press_release"]
    notes = []
    if not independent:
        kinds = ", ".join(sorted({c.type_label.lower() for c in classes}))
        return False, [(f"Подтверждено только первичными индикаторами ({kinds}); "
                        "по ТЗ нужен независимый источник.")]
    if len({c.host for c in independent}) == 1:
        notes.append(f"Подтверждение из одного независимого источника ({independent[0].host}).")
    if not any(TRUST_ORDER[c.trust] >= TRUST_ORDER["medium"] for c in independent):
        notes.append("Среди подтверждающих источников нет изданий среднего или высокого доверия.")
    return True, notes


def role(type_: str) -> str:
    """Роль типа в правиле подтверждения supports_signal() — для интерфейса."""
    spec = load()["types"][type_]
    if spec["primary_only"]:
        return "Только первичный индикатор"
    if type_ == "press_release":
        return "Не может быть единственным основанием"
    if spec["trust"] == "unverified":
        return "Подтверждающий, доверие не установлено"
    return "Подтверждающий"


def registry_table() -> dict:
    """Реестр для интерфейса: типы с правилами и домены."""
    data = load()
    return {"version": data["version"], "updated": data["_overrides_updated"] or data["updated"],
            "types": [{"type": t, **spec, "trust_label": TRUST_LABELS[spec["trust"]], "role": role(t),
                       "count": sum(d["type"] == t for d in data["domains"])} for t, spec in data["types"].items()],
            "patterns": data["patterns"],
            "domains": [{**d, "trust": data["types"][d["type"]]["trust"],
                         "trust_label": TRUST_LABELS[data["types"][d["type"]]["trust"]],
                         "type_label": data["types"][d["type"]]["label"],
                         "primary_only": data["types"][d["type"]]["primary_only"]} for d in data["domains"]]}
