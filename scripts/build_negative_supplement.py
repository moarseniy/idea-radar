"""Build a supplemental, source-backed negative pool without changing the frozen split.

The pool deliberately mixes named software products (N3b) and officially
challenged claims (N2). Both are provisional labels and need human review.
"""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from scripts.build_negative_dataset import ROOT, SNAPSHOT

SOURCE = ROOT / "data/training/negative_weak_signals_global.csv"
OUTPUT = ROOT / "data/training/negative_weak_signals_additional_300.csv"

TOPICS = {
    "agriculture": ("Сельское хозяйство", "сельское хозяйство"),
    "analytics": ("Аналитика", "аналитика"),
    "chatbot": ("Чат-боты", "чат-боты"),
    "crm": ("CRM", "CRM"),
    "ecommerce": ("Электронная коммерция", "электронная коммерция"),
    "education": ("Образование", "образование"),
    "education-tech": ("Образовательные технологии", "образовательные технологии"),
    "fintech": ("Финансовые технологии", "финансовые технологии"),
    "healthcare": ("Здравоохранение", "здравоохранение"),
    "health-tech": ("Медицинские технологии", "медицинские технологии"),
    "home_automation": ("Автоматизация дома", "автоматизация дома"),
    "manufacturing": ("Производство", "производство"),
    "personal_finance": ("Личные финансы", "личные финансы"),
    "photo_management": ("Управление фотографиями", "управление фотографиями"),
    "project_management": ("Управление проектами", "управление проектами"),
    "robotics": ("Робототехника", "робототехника"),
    "self_hosted": ("Самостоятельно размещаемое ПО", "самостоятельно размещаемое ПО"),
    "sustainability": ("Устойчивое развитие", "устойчивое развитие"),
}

# title, date, domain, technology, application, evidence, URL, source publisher
# Entries describe the challenged claim/event, never assert that the underlying
# technology itself is invalid. Sources are regulator primary pages.
CLAIMS = [
    ("MelApp melanoma-risk diagnosis claim", "2015-04-20", "Digital health", "smartphone image analysis", "melanoma risk assessment", "FTC final order bars unsupported claims that MelApp could assess melanoma risk from a mole photo.", "https://www.ftc.gov/news-events/news/press-releases/2015/04/ftc-approves-final-order-barring-misleading-claims-about-apps-ability-diagnose-or-assess-risk", "FTC"),
    ("Mole Detective melanoma detection claim", "2015-08-19", "Digital health", "smartphone image analysis", "melanoma detection", "FTC action barred deceptive health claims made for the Mole Detective app.", "https://www.ftc.gov/news-events/news/press-releases/2015/08/melanoma-detection-app-sellers-barred-making-deceptive-health-claims", "FTC"),
    ("Lumosity brain-training efficacy claim", "2016-01-05", "Education technology", "cognitive training software", "memory and cognitive performance", "FTC charged Lumos Labs with deceptive claims about improved performance and protection from cognitive decline.", "https://www.ftc.gov/news-events/news/press-releases/2016/01/lumosity-pay-2-million-settle-ftc-deceptive-advertising-charges-its-brain-training-program", "FTC"),
    ("LearningRx treatment and school-performance claims", "2016-05-09", "Education technology", "cognitive training", "ADHD and academic performance", "FTC settlement required the marketers to stop unsupported claims about treating serious conditions and improving real-world outcomes.", "https://www.ftc.gov/news-events/news/press-releases/2016/05/marketers-one-one-brain-training-programs-settle-ftc-charges-claims-about-ability-treat-severe", "FTC"),
    ("Jungle Rangers brain-training claims", "2015-04-13", "Education technology", "brain-training game", "children's attention and school performance", "FTC final order barred unsupported claims that the game permanently improved children's cognition, school performance, or ADHD symptoms.", "https://www.ftc.gov/news-events/news/press-releases/2015/04/ftc-approves-final-order-barring-company-making-unsubstantiated-claims-related-products-brain", "FTC"),
    ("BrainStrong memory improvement claims", "2014-08-08", "Consumer health", "DHA supplement", "adult memory improvement", "FTC final order required clinical evidence for claims that BrainStrong Adult improved memory.", "https://www.ftc.gov/news-events/news/press-releases/2014/08/ftc-approves-final-order-settling-charges-brainstrong-adult-supplement-marketers-made-deceptive", "FTC"),
    ("Geniux cognitive-enhancement claims", "2019-04-10", "Consumer health", "cognitive supplement", "memory, focus, and IQ improvement", "FTC described unsupported cognitive claims and fabricated or misleading scientific support in Geniux marketing.", "https://www.ftc.gov/news-events/news/press-releases/2019/04/geniux-dietary-supplement-sellers-barred-unsupported-cognitive-improvement-claims", "FTC"),
    ("Quell wearable pain-relief claims", "2020-03-05", "Digital health", "wearable transcutaneous electrical nerve stimulation", "chronic pain relief", "FTC challenged claims that Quell wearable devices relieved chronic pain without adequate substantiation.", "https://www.ftc.gov/business-guidance/blog/2020/03/ftc-challenges-claims-quell-pain-relief-device", "FTC"),
    ("Office Depot PC Health Check diagnostic claims", "2019-03-27", "Consumer software", "PC diagnostic software", "malware and computer-performance diagnosis", "FTC alleged consumers were misled by scans that overstated malware or performance problems and were used to sell repair services.", "https://www.ftc.gov/news-events/news/press-releases/2019/03/office-depot-tech-support-firm-will-pay-35-million-settle-ftc-allegations-they-tricked-consumers", "FTC"),
    ("Mobile Money Code income scheme", "2018-06-12", "Fintech", "mobile payment app", "guaranteed online income", "FTC case alleged consumers were deceived by a get-rich-quick scheme marketed through mobile-money claims.", "https://www.ftc.gov/news-events/news/press-releases/2018/06/defendants-settle-allegations-they-deceived-consumers-through-get-rich-quick-scheme", "FTC"),
    ("Vitagene DNA privacy and deletion claims", "2023-06-16", "Digital health", "direct-to-consumer genetic testing", "DNA data privacy and deletion", "FTC alleged the company failed to protect genetic data and changed privacy terms retroactively.", "https://www.ftc.gov/news-events/news/press-releases/2023/06/ftc-says-genetic-testing-company-1health-failed-protect-privacy-security-dna-data-unfairly-changed", "FTC"),
    ("CRI Genetics DNA-report accuracy claims", "2023-11-20", "Digital health", "genetic testing and matching algorithm", "ancestry and health reports", "FTC alleged numerous misrepresentations about test accuracy, matching, and artificial-intelligence use.", "https://www.ftc.gov/news-events/news/press-releases/2023/11/ftc-california-obtain-order-against-dna-testing-firm-over-charges-it-made-myriad-misrepresentations", "FTC"),
    ("Decision Diagnostics rapid COVID blood-test claim", "2020-12-18", "Diagnostics", "finger-prick blood test", "rapid COVID-19 diagnosis", "SEC alleged the company claimed a working rapid test existed when the idea had not materialized into a product.", "https://www.sec.gov/newsroom/press-releases/2020-327", "SEC"),
    ("Nate AI shopping automation claim", "2025-04-11", "E-commerce", "artificial intelligence", "automated mobile purchases", "SEC charged the founder with misleading investors about AI completing purchases without human involvement.", "https://www.sec.gov/enforcement-litigation/litigation-releases/lr-26282", "SEC"),
    ("Joonko AI hiring and customer claims", "2024-06-11", "Human resources", "AI-assisted recruitment", "diverse candidate matching", "SEC charged the founder over allegedly false statements about customers, candidates, revenue, and the AI recruitment service.", "https://www.sec.gov/newsroom/press-releases/2024-70", "SEC"),
    ("GameOn AI chat commercial-success claims", "2025-01-24", "Conversational software", "AI chat technology", "customer contracts and revenue", "SEC charged the former CEO over alleged fabricated customer contracts, revenue, and financial records.", "https://www.sec.gov/enforcement-litigation/litigation-releases/lr-26232", "SEC"),
    ("Presto Voice automation disclosure", "2025-01-14", "Restaurant technology", "AI-assisted speech recognition", "drive-through order taking", "SEC said the company made misleading statements about the technology powering its deployed voice product, including third-party operation.", "https://www.sec.gov/enforcement-litigation/administrative-proceedings/33-11352-s", "SEC"),
    ("Zymergen product-market and sales statements", "2024-09-13", "Synthetic biology", "engineered bio-materials", "commercial product sales", "SEC alleged misleading statements about product market potential, customer demand, and expected sales.", "https://www.sec.gov/newsroom/press-releases/2024-129", "SEC"),
    ("Kubient KAI fraud-detection performance claims", "2024-09-16", "Advertising technology", "machine-learning fraud detection", "real-time ad fraud prevention", "SEC charged the former chairman and CEO with fraud and lying to auditors concerning KAI-related revenue.", "https://www.sec.gov/newsroom/press-releases/2024-131", "SEC"),
    ("Rimar AI automated trading claims", "2024-10-10", "Fintech", "artificial intelligence", "automated securities trading", "SEC charged Rimar entities and their owner over false statements about purported AI-driven trading.", "https://www.sec.gov/newsroom/press-releases/2024-167", "SEC"),
    ("3001 AD virtual-reality product and IPO hype", "2009-09-29", "Consumer electronics", "virtual reality headset", "immersive video games", "SEC alleged the company hyped VR products, business relationships, and an imminent IPO while making false investor claims.", "https://www.sec.gov/newsroom/press-releases/2009-210-sec-charges-virtual-reality-product-maker-individuals-boiler-room-fraud", "SEC"),
    ("Melanoma thermography as standalone screening", "2019-02-25", "Medical imaging", "infrared thermography", "breast-cancer screening", "FDA warned that thermography was not cleared as a standalone replacement for mammography and acted against an unapproved package.", "https://www.fda.gov/news-events/press-announcements/fda-issues-warning-letter-clinic-illegally-marketing-unapproved-thermography-device-warns-consumers", "FDA"),
    ("Inova pharmacogenetic medication-response claims", "2019-04-04", "Digital health", "pharmacogenetic testing", "predicting response to prescription medications", "FDA issued a warning letter because the tests had not been reviewed and the claimed medication-response utility was not established.", "https://www.fda.gov/news-events/press-announcements/fda-issues-warning-letter-genomics-lab-illegally-marketing-genetic-test-claims-predict-patients", "FDA"),
    ("Everalbum facial-recognition consent claims", "2021-05-07", "Consumer software", "facial recognition", "photo organization and biometric identification", "FTC finalized an order related to misuse of facial-recognition technology and representations about user consent.", "https://www.ftc.gov/news-events/news/press-releases/2021/05/ftc-finalizes-settlement-photo-app-developer-related-misuse-facial-recognition-technology", "FTC"),
    ("Flo fertility-app data-sharing promises", "2021-06-22", "Digital health", "fertility tracking app", "menstrual and reproductive health tracking", "FTC finalized an order after alleging that sensitive health data was shared despite privacy promises.", "https://www.ftc.gov/news-events/news/press-releases/2021/06/ftc-finalizes-order-flo-health-fertility-tracking-app-shared-sensitive-health-data-facebook-google", "FTC"),
    ("GoodRx health-data privacy promises", "2023-02-01", "Digital health", "telehealth and prescription platform", "prescription discounts and telehealth", "FTC alleged sensitive health data was shared for advertising contrary to privacy promises.", "https://www.ftc.gov/news-events/news/press-releases/2023/02/ftc-enforcement-action-bar-goodrx-sharing-consumers-sensitive-health-info-advertising", "FTC"),
    ("Premom fertility-app health-data claims", "2023-05-17", "Digital health", "fertility tracking app", "fertility and ovulation tracking", "FTC alleged the app disclosed sensitive health information through third-party tracking tools contrary to promises.", "https://www.ftc.gov/news-events/news/press-releases/2023/05/ovulation-tracking-app-premom-will-be-barred-sharing-health-data-advertising-under-proposed-ftc", "FTC"),
    ("Zoom security and encryption representations", "2020-11-09", "Communications software", "video conferencing encryption", "secure online meetings", "FTC order required stronger security practices after allegations that Zoom misrepresented aspects of its security and encryption.", "https://www.ftc.gov/news-events/news/press-releases/2020/11/ftc-requires-zoom-enhance-its-security-practices-part-settlement", "FTC"),
    ("Ring home-camera privacy and security", "2023-05-31", "Smart home", "connected cameras and computer vision", "home surveillance", "FTC alleged employees accessed customer videos and the company failed to prevent account takeovers.", "https://www.ftc.gov/news-events/news/press-releases/2023/05/ftc-says-ring-employees-illegally-surveilled-customers-failed-stop-hackers-taking-control-users", "FTC"),
    ("D-Link router and camera security representations", "2017-01-05", "Cybersecurity", "network routers and IP cameras", "consumer network security", "FTC alleged inadequate security put users of connected routers and cameras at risk.", "https://www.ftc.gov/news-events/news/press-releases/2017/01/ftc-charges-d-link-put-consumers-privacy-risk-due-inadequate-security-its-computer-routers-cameras", "FTC"),
    ("Avast antivirus privacy representations", "2025-02-24", "Cybersecurity", "browser tracking and antivirus software", "online privacy protection", "FTC announced a refund process for customers affected by deceptive privacy claims about Avast software.", "https://www.ftc.gov/news-events/news/press-releases/2025/02/ftc-announces-refund-claims-process-avast-customers-impacted-deceptive-privacy-claims", "FTC"),
    ("SpyFone stalkerware privacy claims", "2021-09-07", "Cybersecurity", "mobile surveillance software", "phone monitoring and safety", "FTC banned the provider from the surveillance business after allegations about covert monitoring and security practices.", "https://www.ftc.gov/news-events/news/press-releases/2021/09/ftc-bans-stalkerware-app-provider-surveillance-business", "FTC"),
    ("Microsoft 365 Copilot subscription pricing representations", "2025-10-27", "Productivity software", "generative AI assistant", "office productivity subscription", "ACCC alleged consumers were misled about subscription options and the cost implications of Copilot plans.", "https://www.accc.gov.au/media-release/microsoft-in-court-for-allegedly-misleading-millions-of-australians-over-microsoft-365-subscriptions", "ACCC"),
    ("JustAnswer expert-service pricing and affiliation claims", "2026-07-08", "Online services", "online expert Q&A platform", "paid access to expert answers", "ACCC said the service made misleading pricing and affiliation representations; this is a claim about marketing, not the underlying Q&A technology.", "https://www.accc.gov.au/media-release/justanswer-to-pay-10m-in-penalties-for-misleading-pricing-representations-and-misleading-affiliation-claims", "ACCC"),
    ("Gravity Defyer footwear pain-relief claims", "2025-02-12", "Consumer health", "shock-absorbing footwear", "pain relief", "FTC court order barred unsupported pain-relief claims for the footwear product.", "https://www.ftc.gov/news-events/news/press-releases/2025/02/ftc-secures-court-order-barring-gravity-defyer-its-owner-making-unsupported-pain-relief-claims", "FTC"),
    ("BetterHelp online counselling confidentiality claims", "2023-03-02", "Digital health", "online counseling platform", "remote mental-health care", "FTC alleged sensitive health data was disclosed for advertising despite privacy promises.", "https://www.ftc.gov/news-events/news/press-releases/2023/03/ftc-finalizes-order-requiring-betterhelp-pay-7-million-consumers-sharing-sensitive-health-info", "FTC"),
    ("Rite Aid facial-recognition deployment safeguards", "2023-12-19", "Retail technology", "AI facial recognition", "retail loss prevention", "FTC alleged the company deployed facial recognition without reasonable safeguards, leading to erroneous matches.", "https://www.ftc.gov/news-events/news/press-releases/2023/12/ftc-says-retailer-deployed-facial-recognition-without-reasonable-safeguards", "FTC"),
    ("Decision Diagnostics COVID test product-existence claim", "2020-12-18", "Diagnostics", "rapid blood testing", "COVID-19 screening", "SEC alleged promotional releases described a working product and regulatory progress that did not exist as represented.", "https://www.sec.gov/newsroom/press-releases/2020-327", "SEC"),
    ("Stimwave wireless neurostimulation product claims", "2023-12-19", "Medical devices", "wireless neurostimulation implant", "chronic pain treatment", "SEC announced fraud charges involving alleged misrepresentations about the medical-device product and its costs.", "https://www.sec.gov/newsroom/press-releases/2023-255", "SEC"),
    ("SKAEL AI startup commercial and fundraising claims", "2024-09-24", "Enterprise software", "AI automation", "enterprise workflow automation", "SEC charged the former CEO with fraud concerning investor fundraising and company representations.", "https://www.sec.gov/newsroom/press-releases/2024-146", "SEC"),
    ("Bitwise Industries technology-company fundraising claims", "2023-11-09", "Enterprise software", "software services", "digital transformation projects", "SEC charged former co-CEOs over falsified documents used while raising investor funds.", "https://www.sec.gov/newsroom/press-releases/2023-233", "SEC"),
    ("Slync.io logistics-platform investor claims", "2023-02-14", "Logistics technology", "workflow automation software", "freight and logistics operations", "SEC charged the former CEO over an offering fraud involving a logistics-software company.", "https://www.sec.gov/newsroom/press-releases/2023-26", "SEC"),
    ("CytoDyn biotech development claims", "2022-12-20", "Biotechnology", "monoclonal antibody therapy", "infectious disease and cancer treatment", "SEC charged the former CEO with fraud and insider trading related to company representations.", "https://www.sec.gov/newsroom/press-releases/2022-232", "SEC"),
    ("Virtual reality 3001 AD product and partnership hype", "2009-09-29", "Consumer electronics", "virtual reality headset", "360-degree video game interaction", "SEC described false investor claims and exaggerated business relationships around a VR product offering.", "https://www.sec.gov/newsroom/press-releases/2009-210-sec-charges-virtual-reality-product-maker-individuals-boiler-room-fraud", "SEC"),
]

DOMAIN_RU = {
    "Digital health": "Цифровое здравоохранение", "Education technology": "Образовательные технологии",
    "Consumer health": "Потребительское здоровье", "Fintech": "Финансовые технологии",
    "Consumer software": "Потребительское ПО", "Diagnostics": "Диагностика",
    "E-commerce": "Электронная коммерция", "Conversational software": "Разговорное ПО",
    "Restaurant technology": "Ресторанные технологии", "Synthetic biology": "Синтетическая биология",
    "Advertising technology": "Рекламные технологии", "Medical imaging": "Медицинская визуализация",
    "Consumer electronics": "Потребительская электроника", "Cybersecurity": "Кибербезопасность",
    "Productivity software": "Программное обеспечение для продуктивности",
    "Online services": "Онлайн-сервисы", "Medical devices": "Медицинские устройства",
    "Human resources": "Управление персоналом", "Retail technology": "Розничные технологии",
    "Communications software": "Программное обеспечение для коммуникаций",
    "Enterprise software": "Корпоративное ПО", "Logistics technology": "Логистические технологии",
    "Biotechnology": "Биотехнологии",
}
TECH_RU = {
    "smartphone image analysis": "анализ изображений со смартфона",
    "cognitive training software": "ПО для когнитивных тренировок",
    "brain-training game": "игра для тренировки мозга", "DHA supplement": "добавка с ДГК",
    "cognitive supplement": "добавка для когнитивных функций",
    "wearable transcutaneous electrical nerve stimulation": "носимая чрескожная электростимуляция нервов",
    "PC diagnostic software": "диагностическое ПО для компьютеров",
    "mobile payment app": "мобильное платёжное приложение",
    "direct-to-consumer genetic testing": "генетическое тестирование напрямую для потребителя",
    "genetic testing and matching algorithm": "генетическое тестирование и алгоритм сопоставления",
    "finger-prick blood test": "анализ крови из капли",
    "artificial intelligence": "искусственный интеллект", "AI-assisted speech recognition": "распознавание речи с ИИ",
    "engineered bio-materials": "биоматериалы, созданные методами биоинженерии",
    "machine-learning fraud detection": "обнаружение мошенничества методами машинного обучения",
    "virtual reality headset": "шлем виртуальной реальности",
    "infrared thermography": "инфракрасная термография",
    "pharmacogenetic testing": "фармакогенетическое тестирование",
    "facial recognition": "распознавание лиц", "fertility tracking app": "приложение для отслеживания фертильности",
    "telehealth and prescription platform": "платформа телемедицины и рецептурных препаратов",
    "video conferencing encryption": "шифрование видеоконференций",
    "connected cameras and computer vision": "подключённые камеры и компьютерное зрение",
    "network routers and IP cameras": "сетевые маршрутизаторы и IP-камеры",
    "browser tracking and antivirus software": "отслеживание браузерной активности и антивирусное ПО",
    "mobile surveillance software": "ПО для слежения через мобильные устройства",
    "generative AI assistant": "генеративный ИИ-ассистент",
    "online expert Q&A platform": "онлайн-платформа вопросов и ответов с экспертами",
    "shock-absorbing footwear": "обувь с амортизирующей подошвой",
    "online counseling platform": "платформа онлайн-консультирования",
    "AI facial recognition": "распознавание лиц с ИИ", "rapid blood testing": "экспресс-анализ крови",
    "wireless neurostimulation implant": "беспроводной нейростимуляционный имплант",
    "AI automation": "автоматизация с помощью ИИ", "software services": "программные услуги",
    "workflow automation software": "ПО для автоматизации рабочих процессов",
    "monoclonal antibody therapy": "терапия моноклональными антителами",
}
APP_RU = {
    "melanoma risk assessment": "оценка риска меланомы", "melanoma detection": "выявление меланомы",
    "memory and cognitive performance": "память и когнитивные способности",
    "ADHD and academic performance": "СДВГ и успеваемость", "children's attention and school performance": "внимание и успеваемость детей",
    "adult memory improvement": "улучшение памяти взрослых", "memory, focus, and IQ improvement": "улучшение памяти, концентрации и IQ",
    "chronic pain relief": "облегчение хронической боли", "malware and computer-performance diagnosis": "поиск вредоносного ПО и проблем производительности компьютера",
    "guaranteed online income": "гарантированный доход в интернете", "DNA data privacy and deletion": "конфиденциальность и удаление генетических данных",
    "ancestry and health reports": "отчёты о происхождении и здоровье", "rapid COVID-19 diagnosis": "экспресс-диагностика COVID-19",
    "automated mobile purchases": "автоматизация покупок в мобильном приложении", "diverse candidate matching": "подбор разнообразных кандидатов",
    "customer contracts and revenue": "клиентские контракты и выручка", "drive-through order taking": "приём заказов в ресторанах с обслуживанием в автомобиле",
    "commercial product sales": "коммерческие продажи продукта", "real-time ad fraud prevention": "предотвращение рекламного мошенничества в реальном времени",
    "automated securities trading": "автоматическая торговля ценными бумагами", "immersive video games": "иммерсивные видеоигры",
    "breast-cancer screening": "скрининг рака молочной железы", "predicting response to prescription medications": "прогноз реакции на рецептурные лекарства",
    "photo organization and biometric identification": "организация фотографий и биометрическая идентификация",
    "menstrual and reproductive health tracking": "отслеживание менструального цикла и репродуктивного здоровья",
    "prescription discounts and telehealth": "скидки на лекарства и телемедицина", "fertility and ovulation tracking": "отслеживание фертильности и овуляции",
    "secure online meetings": "безопасные онлайн-встречи", "home surveillance": "видеонаблюдение дома",
    "consumer network security": "безопасность домашних сетей", "online privacy protection": "защита конфиденциальности в интернете",
    "phone monitoring and safety": "мониторинг телефонов и безопасность", "office productivity subscription": "подписка на офисные инструменты для продуктивности",
    "paid access to expert answers": "платный доступ к ответам экспертов", "pain relief": "облегчение боли",
    "remote mental-health care": "дистанционная психологическая помощь", "retail loss prevention": "предотвращение потерь в розничной торговле",
    "COVID-19 screening": "скрининг COVID-19", "chronic pain treatment": "лечение хронической боли",
    "enterprise workflow automation": "автоматизация корпоративных рабочих процессов", "digital transformation projects": "проекты цифровой трансформации",
    "freight and logistics operations": "грузовые и логистические операции", "infectious disease and cancer treatment": "лечение инфекционных заболеваний и рака",
    "360-degree video game interaction": "управление видеоигрой с обзором 360 градусов",
}

EXCLUDE = re.compile(r"awesome|tutorial|documentation|\bdocs\b|benchmark|paper|dataset|course|boilerplate|starter|template|\bsdk\b|\blibrary\b|\bframework\b|\bprotocol\b|\balgorithm\b|\bdriver\b", re.I)
PRODUCT = re.compile(r"\b(app|application|assistant|platform|client|workspace|product|studio|workbench|server|system|dashboard|manager|software|service|self.hosted|open.source|tool)\b", re.I)


def feature_defaults():
    result = {name: "" for name in FEATURE_NAMES}
    result.update({"source_type_diversity": 1, "missing_science": 1, "missing_patents": 1,
                   "missing_media": 1, "missing_funding": 1, "missing_adoption": 1})
    return result


def claim_row(item, index):
    name, day, domain, tech, app, evidence, url, publisher = item
    base = f"{name}|{SNAPSHOT}"
    group = hashlib.sha256(tech.casefold().encode()).hexdigest()[:12]
    features = feature_defaults()
    row = {
        "id": f"neg_{hashlib.sha256(base.encode()).hexdigest()[:16]}", "label": 0,
        "data_tier": "bronze", "label_status": "provisional_negative",
        "review_status": "needs_human_review", "negative_type": "N2", "stage": "unknown",
        "trend": "unknown", "snapshot_date": SNAPSHOT.isoformat(),
        "feature_schema_version": FEATURE_SCHEMA_VERSION, "technology_group_id": f"claim_{group}_{index:02d}",
        "domain_original": domain, "domain_ru": DOMAIN_RU.get(domain, domain), "technology_original": tech,
        "technology_ru": TECH_RU.get(tech, tech), "application_original": app,
        "application_ru": APP_RU.get(app, app), "name_original": name,
        "name_ru": f"Заявление: {TECH_RU.get(tech, tech)} для {APP_RU.get(app, app)}", "original_language": "en",
        "search_query": f"{tech} {app}", "primary_source_title_original": name,
        "primary_source_url": url, "primary_source_date": day,
        "primary_source_date_precision": "day", "primary_source_language": "en",
        "primary_source_publisher": publisher, "source_evidence_excerpt_original": evidence,
        "source_match_method": "official_regulatory_claim_or_finding",
        "source_match_strength": "regulatory_source", "source_relevance_score": 1,
        "science_count_scope": "not_searched", "selection_rule": "official_claim_noise_v1",
        "evidence_note": "The source challenges a specific claim or conduct; it does not establish that the underlying technology is ineffective. Check allegation/order status in the linked source.",
        "official_source_title_original": name, "official_source_url": url,
        "official_source_date": day, "patent_search_status": "not_searched",
        "media_search_status": "not_searched", **features,
    }
    return row


def products(existing_rows, target):
    used_urls = {r["primary_source_url"].casefold() for r in existing_rows}
    used_names = {r["technology_original"].casefold() for r in existing_rows}
    candidates = []
    cache_dir = ROOT / "storage/github_negative_cache"
    for topic in TOPICS:
        path = cache_dir / f"{topic}.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = sorted(payload.get("items", []), key=lambda x: x.get("stargazers_count") or 0, reverse=True)
        for repo in items:
            name, desc, url = repo.get("name", ""), repo.get("description") or "", repo.get("html_url", "")
            if not name or not desc or not url or url.casefold() in used_urls or name.casefold() in used_names:
                continue
            if repo.get("fork") or repo.get("archived") or repo.get("created_at", "9999")[:10] > SNAPSHOT.isoformat():
                continue
            if EXCLUDE.search(f"{name} {desc}") or not (PRODUCT.search(desc) or repo.get("homepage")):
                continue
            if len(desc.strip()) < 18:
                continue
            candidates.append((topic, repo))
    # Round-robin topic buckets so one GitHub search term cannot dominate.
    buckets = {topic: [] for topic in TOPICS}
    for topic, repo in candidates:
        buckets[topic].append(repo)
    selected = []
    while len(selected) < target and any(buckets.values()):
        for topic, bucket in buckets.items():
            while bucket and (bucket[0]["html_url"].casefold() in used_urls
                              or bucket[0]["name"].casefold() in used_names):
                bucket.pop(0)
            if bucket and len(selected) < target:
                repo = bucket.pop(0)
                selected.append((topic, repo))
                used_urls.add(repo["html_url"].casefold())
                used_names.add(repo["name"].casefold())
    rows = []
    for topic, repo in selected:
        domain_ru, _ = TOPICS[topic]
        name = repo["name"]
        full_name = repo["full_name"]
        stable = hashlib.sha256(f"supplement|{full_name}|{SNAPSHOT}".encode()).hexdigest()[:16]
        group = hashlib.sha256(full_name.casefold().encode()).hexdigest()[:12]
        row = {
            "id": f"neg_{stable}", "label": 0, "data_tier": "bronze",
            "label_status": "provisional_negative", "review_status": "needs_human_review",
            "negative_type": "N3b", "stage": "unknown", "trend": "unknown",
            "snapshot_date": SNAPSHOT.isoformat(), "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "technology_group_id": f"product_{group}", "domain_original": topic,
            "domain_ru": domain_ru, "technology_original": name, "technology_ru": name,
            "application_original": "named software product", "application_ru": "готовый программный продукт",
            "name_original": f"{name} (software product)", "name_ru": f"{name} (программный продукт)",
            "original_language": "en", "search_query": f"{name} {topic} software product",
            "primary_source_title_original": full_name, "primary_source_url": repo["html_url"],
            "primary_source_date": repo["created_at"][:10], "primary_source_date_precision": "day",
            "primary_source_language": "unknown", "primary_source_publisher": "GitHub repository",
            "source_evidence_excerpt_original": repo["description"][:600],
            "source_match_method": "named_repository_product_identity", "source_match_strength": "repository_entity",
            "source_relevance_score": 1, "science_count_scope": "not_searched",
            "selection_rule": f"github_product_topic_supplement_v1:{topic}",
            "evidence_note": "Public repository identifies a named software product. N3b is an entity-vs-technology negative subtype; repo existence alone does not establish broad adoption or product success.",
            "patent_search_status": "not_searched", "media_search_status": "not_searched",
            **feature_defaults(),
        }
        rows.append(row)
    if len(rows) != target:
        raise RuntimeError(f"Only {len(rows)} unique products available; requested {target}")
    return rows


def main():
    with SOURCE.open(newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fields, existing = reader.fieldnames, list(reader)
    existing_urls = {r["primary_source_url"].casefold() for r in existing}
    existing_names = {r["name_original"].casefold() for r in existing}
    claims = []
    seen = set()
    seen_urls = set()
    for item in CLAIMS:
        if item[6].casefold() in existing_urls or item[0].casefold() in existing_names:
            continue
        key = (item[0].casefold(), item[3].casefold(), item[4].casefold())
        if key not in seen and item[6].casefold() not in seen_urls:
            claims.append(item)
            seen.add(key)
            seen_urls.add(item[6].casefold())
    # Keep claim cases varied; use all independently curated, unique records.
    if len(claims) > 50:
        claims = claims[:50]
    prod_rows = products(existing, 300 - len(claims))
    claim_rows = [claim_row(item, i + 1) for i, item in enumerate(claims)]
    rows = prod_rows + claim_rows
    if len(rows) != 300:
        raise RuntimeError(f"Expected 300 rows, got {len(rows)}")
    all_urls = [r["primary_source_url"].casefold() for r in rows]
    all_ids = [r["id"] for r in rows]
    if len(set(all_urls)) != 300 or len(set(all_ids)) != 300:
        raise RuntimeError("Duplicate primary source URLs or ids in supplement")
    if any(not all(r.get(col) is not None for col in FEATURE_NAMES) for r in rows):
        raise RuntimeError("A row is missing one or more 63 feature columns")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"output": str(OUTPUT), "rows": len(rows),
                      "types": dict(Counter(r["negative_type"] for r in rows)),
                      "product_topics": dict(Counter(r["domain_original"] for r in prod_rows)),
                      "feature_count": len(FEATURE_NAMES),
                      "distinct_urls": len(set(all_urls))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
