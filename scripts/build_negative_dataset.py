"""Collect auditable, provisional hard negatives from pre-2019 Crossref works.

An old paper establishes that a technology/application pair is not new; it does
not, by itself, prove mass adoption. Every row therefore requires human review.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import statistics
import time
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

import httpx

from app.ml.features import FEATURE_NAMES, FEATURE_SCHEMA_VERSION
from scripts.build_positive_dataset import (ROOT, CrossrefClient, normalize_crossref_work,
                                            query_crossref, row_from_candidate,
                                            source_excerpt, tokens)
from scripts.enrich_public_sources import EXTRA_COLUMNS, enrich_rows

SNAPSHOT = date(2026, 9, 22)
OUTPUT = ROOT / "data/training/negative_weak_signals_global.csv"

# Each pair is queried independently and retained only if a pre-2019 DOI title
# mentions the technology and application. These are hypotheses, not gold labels.
# Format: domain EN, domain RU, technology EN, technology RU, applications EN|RU.
SPECS = [
 ("Retail", "Ритейл", "barcode", "штрихкод", "retail checkout|кассовая продажа,product identification|идентификация товара,inventory management|управление запасами,warehouse logistics|складская логистика,supply chain tracking|отслеживание поставок,food traceability|прослеживаемость продуктов,pharmaceutical tracking|отслеживание лекарств,library management|управление библиотекой,parcel sorting|сортировка посылок,manufacturing traceability|прослеживаемость производства"),
 ("Retail", "Ритейл", "RFID", "радиочастотная идентификация", "inventory management|управление запасами,warehouse logistics|складская логистика,supply chain tracking|отслеживание поставок,asset tracking|учёт активов,retail stores|розничные магазины,library management|управление библиотекой,patient identification|идентификация пациентов,livestock tracking|отслеживание скота,vehicle identification|идентификация транспорта,manufacturing traceability|прослеживаемость производства"),
 ("Telecommunications", "Телекоммуникации", "Wi-Fi", "Wi-Fi", "home networking|домашние сети,public hotspots|общественные точки доступа,university campus|университетский кампус,industrial networks|промышленные сети,hospital networks|больничные сети,wireless sensing|беспроводные датчики,smart buildings|умные здания,retail stores|розничные магазины,warehouse connectivity|складская связь,vehicle connectivity|связь транспорта"),
 ("Telecommunications", "Телекоммуникации", "Bluetooth", "Bluetooth", "wireless headphones|беспроводные наушники,wearable devices|носимые устройства,medical devices|медицинские устройства,indoor positioning|позиционирование в помещениях,smart homes|умные дома,automotive connectivity|автомобильная связь,fitness tracking|отслеживание фитнеса,industrial sensors|промышленные датчики,wireless keyboards|беспроводные клавиатуры,asset tracking|учёт активов"),
 ("Telecommunications", "Телекоммуникации", "4G LTE", "мобильная связь 4G LTE", "mobile broadband|мобильный интернет,video streaming|видеотрансляция,vehicle connectivity|связь транспорта,telemedicine|телемедицина,smart city networks|сети умного города,industrial IoT|промышленный интернет вещей,emergency communications|экстренная связь,remote education|дистанционное образование,mobile payments|мобильные платежи,public transport|общественный транспорт"),
 ("Telecommunications", "Телекоммуникации", "fiber optic", "волоконно-оптическая связь", "broadband access|широкополосный доступ,data centers|центры обработки данных,submarine cables|подводные кабели,industrial networks|промышленные сети,medical imaging|медицинская визуализация,temperature sensing|измерение температуры,oil pipelines|нефтепроводы,telecommunications networks|телекоммуникационные сети,railway monitoring|мониторинг железных дорог,power grids|электросети"),
 ("Payments", "Платежи", "NFC", "бесконтактная связь NFC", "contactless payments|бесконтактные платежи,transit ticketing|оплата транспорта,access control|контроль доступа,mobile wallets|мобильные кошельки,retail checkout|кассовая продажа,hotel access|доступ в гостиницу,smart posters|умные плакаты,device pairing|сопряжение устройств,patient identification|идентификация пациентов,loyalty programs|программы лояльности"),
 ("Payments", "Платежи", "QR code", "QR-код", "mobile payments|мобильные платежи,restaurant menus|меню ресторанов,event tickets|билеты на мероприятия,product authentication|аутентификация товара,marketing campaigns|маркетинговые кампании,contact tracing|отслеживание контактов,public transport|общественный транспорт,library management|управление библиотекой,retail checkout|кассовая продажа,patient identification|идентификация пациентов"),
 ("Manufacturing", "Производство", "CNC machining", "обработка на станках с ЧПУ", "aerospace parts|детали авиации,automotive parts|детали автомобилей,medical implants|медицинские имплантаты,precision manufacturing|точное производство,mold making|изготовление пресс-форм,woodworking|обработка древесины,metal cutting|резка металла,dental prosthetics|зубные протезы,shipbuilding|судостроение,tool manufacturing|изготовление инструментов"),
 ("Manufacturing", "Производство", "industrial robot", "промышленный робот", "automotive welding|сварка автомобилей,assembly lines|сборочные линии,material handling|перемещение материалов,painting operations|покрасочные работы,packaging automation|автоматизация упаковки,semiconductor manufacturing|производство полупроводников,food processing|пищевая промышленность,electronics assembly|сборка электроники,warehouse palletizing|укладка паллет,machine tending|обслуживание станков"),
 ("Manufacturing", "Производство", "laser cutting", "лазерная резка", "sheet metal fabrication|изготовление листового металла,automotive parts|детали автомобилей,textile manufacturing|производство текстиля,medical devices|медицинские устройства,aerospace parts|детали авиации,woodworking|обработка древесины,electronics manufacturing|производство электроники,precision manufacturing|точное производство,packaging production|производство упаковки,shipbuilding|судостроение"),
 ("Logistics", "Логистика", "GPS tracking", "спутниковое отслеживание GPS", "fleet management|управление автопарком,vehicle navigation|навигация транспорта,asset tracking|учёт активов,maritime navigation|морская навигация,precision agriculture|точное земледелие,emergency response|экстренное реагирование,wildlife monitoring|мониторинг животных,parcel delivery|доставка посылок,construction equipment|строительная техника,public transport|общественный транспорт"),
 ("Logistics", "Логистика", "route optimization", "оптимизация маршрутов", "parcel delivery|доставка посылок,vehicle fleets|автопарки,public transport|общественный транспорт,waste collection|сбор отходов,food delivery|доставка еды,ambulance dispatch|диспетчеризация скорой помощи,airline scheduling|расписание авиарейсов,warehouse picking|складской отбор,maritime shipping|морские перевозки,school buses|школьные автобусы"),
 ("Healthcare", "Здравоохранение", "PCR", "полимеразная цепная реакция", "infectious disease diagnosis|диагностика инфекций,genetic testing|генетическое тестирование,cancer diagnostics|диагностика рака,food safety|безопасность пищевых продуктов,forensic analysis|криминалистический анализ,environmental monitoring|экологический мониторинг,pathogen detection|обнаружение патогенов,blood screening|скрининг крови,veterinary diagnostics|ветеринарная диагностика,water quality|качество воды"),
 ("Healthcare", "Здравоохранение", "MRI", "магнитно-резонансная томография", "brain imaging|визуализация мозга,cardiac imaging|визуализация сердца,cancer diagnosis|диагностика рака,spinal imaging|визуализация позвоночника,joint imaging|визуализация суставов,liver imaging|визуализация печени,pediatric imaging|детская визуализация,breast imaging|визуализация молочной желез,stroke diagnosis|диагностика инсульта,musculoskeletal imaging|визуализация опорно-двигательной системы"),
 ("Healthcare", "Здравоохранение", "ultrasound", "ультразвуковая диагностика", "prenatal imaging|пренатальная визуализация,cardiac imaging|визуализация сердца,vascular imaging|визуализация сосудов,abdominal imaging|визуализация брюшной полости,emergency medicine|экстренная медицина,breast imaging|визуализация молочной желез,liver imaging|визуализация печени,thyroid imaging|визуализация щитовидной желез,musculoskeletal imaging|визуализация опорно-двигательной системы,prostate imaging|визуализация простаты"),
 ("Healthcare", "Здравоохранение", "computed tomography", "компьютерная томография", "lung imaging|визуализация лёгких,stroke diagnosis|диагностика инсульта,trauma imaging|визуализация травм,cancer screening|скрининг рака,cardiac imaging|визуализация сердца,abdominal imaging|визуализация брюшной полости,dental imaging|стоматологическая визуализация,bone imaging|визуализация костей,vascular imaging|визуализация сосудов,brain imaging|визуализация мозга"),
 ("Healthcare", "Здравоохранение", "electrocardiography", "электрокардиография", "arrhythmia detection|обнаружение аритмии,cardiac monitoring|мониторинг сердца,emergency medicine|экстренная медицина,wearable monitoring|носимый мониторинг,heart failure diagnosis|диагностика сердечной недостаточности,athlete monitoring|мониторинг спортсменов,remote patient monitoring|дистанционный мониторинг пациентов,ischemia detection|обнаружение ишемии,preoperative assessment|предоперационная оценка,ambulance care|помощь в скорой"),
 ("Healthcare", "Здравоохранение", "laparoscopy", "лапароскопия", "gallbladder surgery|операция на желчном пузыре,colon surgery|операция на кишечнике,gynecological surgery|гинекологическая хирургия,urological surgery|урологическая хирургия,hernia repair|лечение грыжи,bariatric surgery|бариатрическая хирургия,appendectomy|аппендэктомия,cancer surgery|онкологическая хирургия,pediatric surgery|детская хирургия,abdominal surgery|абдоминальная хирургия"),
 ("Energy", "Энергетика", "solar photovoltaic", "солнечная фотоэлектрика", "rooftop electricity|электричество на крышах,utility scale generation|крупные электростанции,off grid systems|автономные системы,water pumping|перекачка воды,telecom base stations|телекоммуникационные базовые станции,agricultural irrigation|сельскохозяйственное орошение,electric vehicle charging|зарядка электромобилей,building electricity|электроснабжение зданий,remote villages|отдалённые посёлки,street lighting|уличное освещение"),
 ("Energy", "Энергетика", "lithium ion battery", "литий-ионный аккумулятор", "electric vehicles|электромобили,mobile phones|мобильные телефоны,grid storage|сетевое хранение энергии,laptop computers|ноутбуки,power tools|электроинструменты,medical devices|медицинские устройства,electric buses|электробусы,home storage|домашние накопители,drones|беспилотники,portable electronics|портативная электроника"),
 ("Energy", "Энергетика", "wind turbine", "ветровая турбина", "electricity generation|производство электроэнергии,offshore wind farms|морские ветропарки,remote microgrids|удалённые микросети,agricultural pumping|сельскохозяйственные насосы,hybrid power systems|гибридные энергосистемы,grid integration|интеграция в сеть,island electricity|электроснабжение островов,industrial electricity|промышленное электроснабжение,distributed generation|распределённая генерация,rural electrification|электрификация сельской местности"),
 ("Energy", "Энергетика", "heat pump", "тепловой насос", "residential heating|отопление жилья,commercial buildings|коммерческие здания,water heating|нагрев воды,district heating|централизованное теплоснабжение,industrial process heat|промышленное тепло,space cooling|охлаждение помещений,agricultural greenhouses|сельскохозяйственные теплицы,hotel heating|отопление гостиниц,food drying|сушка пищевых продуктов,swimming pools|бассейны"),
 ("Energy", "Энергетика", "LED lighting", "светодиодное освещение", "street lighting|уличное освещение,home lighting|домашнее освещение,commercial buildings|коммерческие здания,automotive headlights|автомобильные фары,greenhouse lighting|освещение теплиц,industrial facilities|промышленные объекты,retail displays|витрины магазинов,hospital lighting|освещение больниц,aircraft lighting|освещение самолётов,architectural lighting|архитектурное освещение"),
 ("Environment", "Экология", "reverse osmosis", "обратный осмос", "seawater desalination|опреснение морской воды,drinking water treatment|очистка питьевой воды,industrial wastewater|промышленные сточные воды,food processing|пищевая промышленность,pharmaceutical water|вода для фармацевтики,agricultural irrigation|сельскохозяйственное орошение,brackish water treatment|очистка солоноватой воды,municipal wastewater|городские сточные воды,boiler feedwater|котловая вода,dairy processing|переработка молока"),
 ("Environment", "Экология", "activated carbon", "активированный уголь", "drinking water treatment|очистка питьевой воды,air filtration|фильтрация воздуха,wastewater treatment|очистка сточных вод,industrial gas treatment|очистка промышленных газов,soil remediation|очистка почвы,food processing|пищевая промышленность,odor control|удаление запахов,pharmaceutical purification|очистка фармацевтических веществ,mercury removal|удаление ртути,chemical separation|химическое разделение"),
 ("Agriculture", "Сельское хозяйство", "drip irrigation", "капельное орошение", "vineyard irrigation|орошение виноградников,greenhouse cultivation|тепличное выращивание,orchard irrigation|орошение садов,vegetable production|производство овощей,water conservation|экономия воды,arid agriculture|земледелие в засушливых районах,fertilizer delivery|внесение удобрений,olive cultivation|выращивание оливок,cotton production|производство хлопка,urban farming|городское сельское хозяйство"),
 ("Agriculture", "Сельское хозяйство", "precision agriculture", "точное земледелие", "variable rate fertilization|дифференцированное внесение удобрений,crop monitoring|мониторинг посевов,soil mapping|картирование почвы,irrigation management|управление орошением,pest detection|обнаружение вредителей,yield mapping|картирование урожайности,tractor guidance|навигация тракторов,weed control|борьба с сорняками,grain production|производство зерна,vineyard management|управление виноградниками"),
 ("Construction", "Строительство", "building information modeling", "информационное моделирование зданий", "building design|проектирование зданий,construction planning|планирование строительства,facility management|эксплуатация объектов,cost estimation|оценка стоимости,structural engineering|строительное проектирование,bridge construction|строительство мостов,energy analysis|энергетический анализ,heritage conservation|сохранение наследия,clash detection|поиск коллизий,infrastructure projects|инфраструктурные проекты"),
 ("Construction", "Строительство", "concrete reinforcement", "армирование бетона", "bridge construction|строительство мостов,high rise buildings|высотные здания,tunnel construction|строительство тоннелей,marine structures|морские сооружения,earthquake resistance|сейсмостойкость,pavement construction|строительство дорог,industrial floors|промышленные полы,precast elements|сборные элементы,water tanks|резервуары воды,foundation construction|строительство фундаментов"),
 ("Cybersecurity", "Кибербезопасность", "two factor authentication", "двухфакторная аутентификация", "online banking|онлайн-банкинг,enterprise login|корпоративный вход,cloud services|облачные сервисы,health records|медицинские записи,e commerce|электронная коммерция,government portals|государственные порталы,remote work|удалённая работа,mobile payments|мобильные платежи,university accounts|университетские учётные записи,social networks|социальные сети"),
 ("Cybersecurity", "Кибербезопасность", "firewall", "межсетевой экран", "enterprise networks|корпоративные сети,data centers|центры обработки данных,industrial control|промышленное управление,cloud services|облачные сервисы,university networks|университетские сети,hospital networks|больничные сети,home networking|домашние сети,mobile networks|мобильные сети,government networks|государственные сети,small businesses|малый бизнес"),
 ("Software", "Программное обеспечение", "container orchestration", "оркестрация контейнеров", "cloud applications|облачные приложения,microservices|микросервисы,data processing|обработка данных,web services|веб-сервисы,edge computing|периферийные вычисления,enterprise software|корпоративное ПО,machine learning deployment|развёртывание машинного обучения,telecom networks|телекоммуникационные сети,big data analytics|аналитика больших данных,continuous delivery|непрерывная доставка"),
 ("Software", "Программное обеспечение", "relational database", "реляционная база данных", "banking transactions|банковские транзакции,hospital records|больничные записи,retail inventory|товарные запасы,airline reservations|бронирование авиабилетов,government records|государственные реестры,e commerce|электронная коммерция,university administration|управление университетом,customer relationship management|управление клиентами,hotel bookings|бронирование гостиниц,supply chain management|управление цепями поставок"),
 ("Media", "Медиа", "video streaming", "потоковое видео", "online education|онлайн-образование,live sports|прямые трансляции спорта,telemedicine|телемедицина,entertainment platforms|развлекательные платформы,corporate training|корпоративное обучение,video conferencing|видеоконференции,news broadcasting|новостное вещание,social media|социальные сети,remote monitoring|дистанционный мониторинг,public events|публичные мероприятия"),
]

# Public regulatory findings, collected separately from historical DOI candidates.
# The negative is the *specific overclaim*, not the underlying technology.
HYPE_CASES = [
 ("AI lawyer equivalent to human legal advice", "ИИ-юрист как полноценная замена консультации человека", "Legal technology", "Юридические технологии", "AI lawyer", "ИИ-юрист", "legal advice", "юридические консультации", "FTC final order: DoNotPay robot lawyer claims", "https://www.ftc.gov/news-events/news/press-releases/2025/02/ftc-finalizes-order-donotpay-prohibits-deceptive-ai-lawyer-claims-imposes-monetary-relief-requires", "2025-02-11", "FTC final order prohibits deceptive claims that the service performs like a human lawyer."),
 ("AI content detector with claimed 98 percent accuracy", "ИИ-детектор контента с заявленной точностью 98%", "AI detection", "Детекция ИИ", "AI content detector", "ИИ-детектор контента", "general purpose text", "тексты общего назначения", "FTC final order: Workado AI detector accuracy", "https://www.ftc.gov/news-events/news/press-releases/2025/08/ftc-approves-final-order-against-workado-llc-which-misrepresented-accuracy-its-artificial", "2025-08-28", "FTC says the 98 percent accuracy claim was misleading; independent testing found 53 percent on general-purpose content."),
 ("AI website plugin guaranteeing WCAG compliance", "ИИ-плагин с гарантией соответствия WCAG", "Web accessibility", "Веб-доступность", "AI accessibility plugin", "ИИ-плагин доступности", "WCAG compliance", "соответствие WCAG", "FTC final order: accessiBe compliance claims", "https://www.ftc.gov/news-events/news/press-releases/2025/04/ftc-approves-final-order-requiring-accessibe-pay-1-million", "2025-04-22", "FTC says the claim that accessWidget can make any website WCAG-compliant was false, misleading or unsubstantiated."),
 ("AI weapons scanner detecting all weapons", "ИИ-сканер с заявлением об обнаружении любого оружия", "Security screening", "Досмотр безопасности", "AI weapons scanner", "ИИ-сканер оружия", "school security", "безопасность школ", "FTC action: Evolv screening claims", "https://www.ftc.gov/news-events/news/press-releases/2024/11/ftc-takes-action-against-evolv-technologies-deceiving-users-about-its-ai-powered-security-screening", "2024-11-26", "FTC alleged deceptive advertising that the scanners detect all weapons and outperform traditional metal detectors."),
 ("Facial recognition claimed free of demographic bias", "Распознавание лиц с заявленным отсутствием демографической ошибки", "Computer vision", "Компьютерное зрение", "facial recognition", "распознавание лиц", "demographic fairness", "демографическая справедливость", "FTC final order: IntelliVision bias claims", "https://www.ftc.gov/news-events/news/press-releases/2025/01/ftc-finalizes-order-prohibiting-intellivision-making-deceptive-claims-about-its-facial-recognition", "2025-01-14", "FTC says zero gender or racial bias claims lacked evidence."),
 ("AI moderation claimed to prevent cyberbullying", "ИИ-модерация с заявленным предотвращением кибербуллинга", "Social media", "Социальные медиа", "AI content moderation", "ИИ-модерация контента", "anonymous messaging", "анонимные сообщения", "FTC order: NGL AI moderation claims", "https://www.ftc.gov/news-events/news/press-releases/2024/07/ftc-order-will-ban-ngl-labs-its-founders-offering-anonymous-messaging-apps-kids-under-18-halt", "2024-07-09", "FTC and Los Angeles DA alleged false claims that AI moderation filtered cyberbullying and harmful messages."),
 ("AI investment adviser using client data", "ИИ-советник с заявленным использованием данных клиентов", "Fintech", "Финтех", "AI investment advice", "ИИ-инвестиционные советы", "personalized investing", "персональные инвестиции", "SEC settled charges: Delphia and Global Predictions AI claims", "https://www.sec.gov/newsroom/press-releases/2024-36", "2024-03-18", "SEC charged two advisers with false or misleading claims about purported AI use in their investment processes."),
 ("Finger-prick analyzer promising comprehensive blood tests", "Анализатор по капле крови с обещанием комплексных тестов", "Diagnostics", "Диагностика", "finger-prick blood analyzer", "анализатор крови по капле", "comprehensive blood tests", "комплексные анализы крови", "SEC charges: Theranos technology claims", "https://www.sec.gov/newsroom/press-releases/2018-41", "2018-03-14", "SEC charged Theranos with exaggerated or false statements about portable blood analyzer capabilities."),
 ("AI storefront promising passive income", "ИИ-магазин с обещанием пассивного дохода", "E-commerce", "Электронная коммерция", "AI storefront automation", "ИИ-автоматизация магазина", "passive income", "пассивный доход", "FTC order: Ascend Ecom AI income claims", "https://www.ftc.gov/news-events/news/press-releases/2025/06/ftc-case-leads-order-banning-ascend-ecom-its-owners-business-opportunity-marketing", "2025-06-17", "FTC says AI-powered storefront income claims were false and the scheme defrauded consumers."),
 ("Conversational AI promising guaranteed business returns", "Разговорный ИИ с обещанием гарантированной прибыли", "E-commerce", "Электронная коммерция", "conversational AI", "разговорный ИИ", "business income", "доход бизнеса", "FTC complaint: Air AI earnings claims", "https://www.ftc.gov/news-events/news/press-releases/2025/08/ftc-sues-stop-air-ai-using-deceptive-claims-about-business-growth-earnings-potential-refund", "2025-08-25", "FTC alleges deceptive promises that buyers would rapidly earn substantial profits using conversational AI services."),
]

NON_ENGLISH_PAIRS = [
 ("Pagos", "Платежи", "código QR", "QR-код", "pagos móviles", "мобильные платежи", "es"),
 ("Salud", "Здравоохранение", "reacción en cadena de la polimerasa", "полимеразная цепная реакция", "diagnóstico de enfermedades", "диагностика заболеваний", "es"),
 ("Energía", "Энергетика", "energía solar fotovoltaica", "солнечная фотоэлектрика", "generación eléctrica", "производство электроэнергии", "es"),
 ("Agricultura", "Сельское хозяйство", "riego por goteo", "капельное орошение", "cultivo de tomate", "выращивание томатов", "es"),
 ("Salud", "Здравоохранение", "resonancia magnética", "магнитно-резонансная томография", "diagnóstico cerebral", "диагностика мозга", "es"),
 ("Energía", "Энергетика", "bomba de calor", "тепловой насос", "calefacción residencial", "отопление жилья", "es"),
 ("Saúde", "Здравоохранение", "ultrassonografia", "ультразвуковая диагностика", "diagnóstico cardíaco", "диагностика сердца", "pt"),
 ("Energia", "Энергетика", "energia solar fotovoltaica", "солнечная фотоэлектрика", "geração de eletricidade", "производство электроэнергии", "pt"),
 ("Agricultura", "Сельское хозяйство", "irrigação por gotejamento", "капельное орошение", "produção agrícola", "сельскохозяйственное производство", "pt"),
 ("Saúde", "Здравоохранение", "tomografia computadorizada", "компьютерная томография", "diagnóstico de câncer", "диагностика рака", "pt"),
 ("Energie", "Энергетика", "Wärmepumpe", "тепловой насос", "Gebäudeheizung", "отопление зданий", "de"),
 ("Industrie", "Промышленность", "Industrieroboter", "промышленный робот", "Automobilproduktion", "производство автомобилей", "de"),
 ("Energie", "Энергетика", "Photovoltaik", "солнечная фотоэлектрика", "Stromerzeugung", "производство электроэнергии", "de"),
 ("Santé", "Здравоохранение", "imagerie par résonance magnétique", "магнитно-резонансная томография", "diagnostic du cancer", "диагностика рака", "fr"),
 ("Énergie", "Энергетика", "pompe à chaleur", "тепловой насос", "chauffage des bâtiments", "отопление зданий", "fr"),
 ("Agriculture", "Сельское хозяйство", "irrigation goutte à goutte", "капельное орошение", "culture maraîchère", "овощеводство", "fr"),
]

TECHNOLOGY_CONTEXT = {
    "barcode": "https://www.gs1.org/standards/barcodes/ean-upc",
    "RFID": "https://www.gs1.org/standards/rfid",
    "Bluetooth": "https://www.bluetooth.com/specifications/specs/",
    "NFC": "https://nfc-forum.org/build/specifications",
    "solar photovoltaic": "https://www.iea.org/energy-system/renewables/solar-pv",
    "lithium ion battery": "https://afdc.energy.gov/vehicles/electric-batteries",
    "heat pump": "https://www.iea.org/reports/heat-pump-monitor-2026/key-findings",
    "building information modeling": "https://technical.buildingsmart.org/standards/ifc/",
}

ENTITY_QUERIES = {"ai_assistant": "topic:ai-assistant", "chatbot": "topic:chatbot"}
ENTITY_EXCLUDE = re.compile(
    r"awesome|guide|tutorial|skills|library|sdk|framework|template|mcp|plugin|nvim|docs|examples|workflow builder|tools list|curated",
    re.IGNORECASE,
)
ENTITY_INCLUDE = re.compile(
    r"\b(app|assistant|platform|client|workspace|agent|product|studio|workbench|server|system|chatbot|bot)\b",
    re.IGNORECASE,
)


def candidates():
    for domain, domain_ru, tech, tech_ru, applications in SPECS:
        for item in applications.split(","):
            app, app_ru = item.split("|", 1)
            yield {
                "domain_original": domain, "domain_ru": domain_ru,
                "technology_original": tech, "technology_ru": tech_ru,
                "application_original": app, "application_ru": app_ru,
                "name_original": f"{tech} for {app}",
                "name_ru": f"{tech_ru} для {app_ru}",
                "original_language": "en", "search_query": f"{tech} {app}",
            }
    for domain, domain_ru, tech, tech_ru, app, app_ru, language in NON_ENGLISH_PAIRS:
        yield {"domain_original": domain, "domain_ru": domain_ru,
               "technology_original": tech, "technology_ru": tech_ru,
               "application_original": app, "application_ru": app_ru,
               "name_original": f"{tech} para {app}" if language in {"es", "pt"} else f"{tech} — {app}",
               "name_ru": f"{tech_ru} для {app_ru}",
               "original_language": language, "search_query": f"{tech} {app}"}


def fetch(candidate):
    """Crossref exact-pair search; require old work and explicit title overlap."""
    params = {
        "query.bibliographic": candidate["search_query"],
        "filter": "from-pub-date:1990-01-01,until-pub-date:2018-12-31",
        "rows": 20,
        "select": "DOI,title,URL,published,is-referenced-by-count,type,abstract,publisher",
    }
    cache = ROOT / "storage/crossref_negative_cache"
    cache.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(candidate["search_query"].encode()).hexdigest()
    path = cache / f"{digest}.json"
    if path.exists():
        import json
        items = json.loads(path.read_text())
    else:
        import json
        last_error = None
        for attempt in range(5):
            try:
                with httpx.Client(timeout=30, headers={"User-Agent": "IDEA-weak-signals/1.0"}) as client:
                    response = client.get("https://api.crossref.org/works", params=params)
                if response.status_code == 429:
                    time.sleep(min(20, 2 ** attempt + 1))
                    continue
                response.raise_for_status()
                items = response.json()["message"]["items"]
                break
            except httpx.HTTPError as exc:
                last_error = exc
                time.sleep(min(20, 2 ** attempt + 1))
        else:
            raise RuntimeError(f"Crossref request failed after retries: {last_error}")
        path.write_text(json.dumps(items, ensure_ascii=False))
        time.sleep(1.1)
    tech_terms = tokens(candidate["technology_original"])
    app_terms = tokens(candidate["application_original"])
    ranked = []
    for item in items:
        work = normalize_crossref_work(item)
        title_terms = tokens(work["title"])
        tech_match = len(tech_terms & title_terms) / max(1, len(tech_terms))
        app_match = len(app_terms & title_terms) / max(1, len(app_terms))
        if tech_match >= 0.5 and app_match >= 0.5 and work.get("doi"):
            ranked.append((0.65 * tech_match + 0.35 * app_match, work))
    ranked.sort(key=lambda pair: (pair[0], pair[1].get("publication_date") or ""), reverse=True)
    return candidate, ranked


def make_row(candidate, ranked):
    score, source = ranked[0]
    excerpt, match = source_excerpt(candidate, source)
    stable = hashlib.sha256(f"{candidate['name_original']}|{SNAPSHOT}".encode()).hexdigest()[:16]
    tech_group = hashlib.sha256(candidate["technology_original"].casefold().encode()).hexdigest()[:12]
    dates = [w.get("publication_date") for _, w in ranked if w.get("publication_date")]
    years = Counter(int(d[:4]) for d in dates)
    # Counts cover only the first 20 search results, not the literature. Do not
    # populate papers_log_3y, paper growth, citations, or missing_science=0.
    features = {name: "" for name in FEATURE_NAMES}
    features.update({"source_type_diversity": 1, "missing_science": 1,
                     "missing_patents": 1, "missing_media": 1,
                     "missing_funding": 1, "missing_adoption": 1})
    row = {
        "id": f"neg_{stable}", "label": 0, "data_tier": "bronze",
        "label_status": "provisional_negative", "review_status": "needs_human_review",
        "negative_type": "N1", "stage": "unknown", "trend": "unknown",
        "snapshot_date": SNAPSHOT.isoformat(), "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "technology_group_id": f"tech_{tech_group}", **candidate,
        "primary_source_title_original": source["title"],
        "primary_source_url": source["doi"],
        "primary_source_crossref_id": source["id"],
        "primary_source_doi": source["doi"],
        "primary_source_date": source["publication_date"],
        "primary_source_date_precision": source["publication_date_precision"],
        "primary_source_language": source.get("language") or "unknown",
        "primary_source_publisher": source.get("publisher") or "unknown",
        "source_evidence_excerpt_original": excerpt,
        "source_match_method": f"pre_2019_{match}",
        "source_match_strength": "full_title_term_overlap" if score >= 0.999 else "partial_title_term_overlap",
        "source_countries": "", "source_relevance_score": round(score, 6),
        "science_sample_size": len(ranked),
        "science_count_scope": "matched_titles_in_top_20_crossref_pair_results_pre_2019",
        "selection_rule": "curated_established_pair+pre_2019_doi_title_overlap",
        "evidence_note": "Pre-2019 DOI proves prior work on this pair, not mass adoption. Verify N1 for this exact application before training.",
        "historical_evidence_years": "|".join(map(str, sorted(years))),
        **{col: "" for col in EXTRA_COLUMNS}, **features,
    }
    row["patent_search_status"] = "not_searched"
    row["media_search_status"] = "not_searched"
    return row


def hype_rows():
    result = []
    for name, name_ru, domain, domain_ru, tech, tech_ru, app, app_ru, title, url, day, evidence in HYPE_CASES:
        candidate = {"domain_original": domain, "domain_ru": domain_ru,
                     "technology_original": tech, "technology_ru": tech_ru,
                     "application_original": app, "application_ru": app_ru,
                     "name_original": name, "name_ru": name_ru,
                     "original_language": "en", "search_query": f"{tech} {app}"}
        stable = hashlib.sha256(f"{name}|{SNAPSHOT}".encode()).hexdigest()[:16]
        tech_group = hashlib.sha256(tech.casefold().encode()).hexdigest()[:12]
        features = {key: "" for key in FEATURE_NAMES}
        features.update({"source_type_diversity": 1, "missing_science": 1,
                         "missing_patents": 1, "missing_media": 1,
                         "missing_funding": 1, "missing_adoption": 1})
        row = {"id": f"neg_{stable}", "label": 0, "data_tier": "bronze",
               "label_status": "provisional_negative", "review_status": "needs_human_review",
               "negative_type": "N2", "stage": "unknown", "trend": "unknown",
               "snapshot_date": SNAPSHOT.isoformat(), "feature_schema_version": FEATURE_SCHEMA_VERSION,
               "technology_group_id": f"tech_{tech_group}", **candidate,
               "primary_source_title_original": title, "primary_source_url": url,
               "primary_source_date": day, "primary_source_date_precision": "day",
               "primary_source_language": "en",
               "primary_source_publisher": "FTC" if "ftc.gov" in url else "SEC",
               "source_evidence_excerpt_original": evidence,
               "source_match_method": "regulatory_finding_on_specific_claim",
               "source_match_strength": "regulatory_source",
               "source_relevance_score": 1, "science_sample_size": "",
               "science_count_scope": "not_searched",
               "selection_rule": "manual_regulatory_claim",
               "evidence_note": "The cited regulator disputes this specific marketing claim; the underlying technology is not classified as invalid.",
               "historical_evidence_years": "",
               **{col: "" for col in EXTRA_COLUMNS}, **features}
        row["official_source_title_original"] = title
        row["official_source_url"] = url
        row["official_source_date"] = day
        row["patent_search_status"] = "not_searched"
        row["media_search_status"] = "not_searched"
        result.append(row)
    return result


def entity_rows(limit=60):
    """Names of concrete software products, not names of technical methods."""
    cache_dir = ROOT / "storage/github_negative_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    repositories = {}
    for cache_name, query in ENTITY_QUERIES.items():
        path = cache_dir / f"{cache_name}.json"
        if path.exists():
            payload = json.loads(path.read_text())
        else:
            with httpx.Client(timeout=30, headers={"Accept": "application/vnd.github+json",
                                                  "User-Agent": "IDEA-weak-signals/1.0"}) as client:
                response = client.get("https://api.github.com/search/repositories",
                                      params={"q": query, "sort": "stars", "per_page": 100})
                response.raise_for_status()
                payload = response.json()
            path.write_text(json.dumps(payload, ensure_ascii=False))
        for repo in payload.get("items", []):
            repositories[repo["full_name"].casefold()] = repo
    selected = []
    for repo in sorted(repositories.values(), key=lambda x: x.get("stargazers_count") or 0, reverse=True):
        description = repo.get("description") or ""
        name = repo.get("name") or ""
        if not repo.get("homepage") or repo.get("fork") or repo.get("archived"):
            continue
        if repo.get("created_at", "9999")[:10] > SNAPSHOT.isoformat():
            continue
        if ENTITY_EXCLUDE.search(f"{name} {description}") or not ENTITY_INCLUDE.search(description):
            continue
        selected.append(repo)
        if len(selected) >= limit:
            break
    rows = []
    for repo in selected:
        name = repo["name"]
        description = repo.get("description") or ""
        full_name = repo["full_name"]
        stable = hashlib.sha256(f"{full_name}|{SNAPSHOT}".encode()).hexdigest()[:16]
        group = hashlib.sha256(full_name.casefold().encode()).hexdigest()[:12]
        features = {key: "" for key in FEATURE_NAMES}
        features.update({"source_type_diversity": 1,
                         "missing_science": 1, "missing_patents": 1,
                         "missing_media": 1, "missing_funding": 1,
                         "missing_adoption": 1})
        row = {"id": f"neg_{stable}", "label": 0, "data_tier": "bronze",
               "label_status": "provisional_negative", "review_status": "needs_human_review",
               "negative_type": "N3b", "stage": "unknown", "trend": "unknown",
               "snapshot_date": SNAPSHOT.isoformat(), "feature_schema_version": FEATURE_SCHEMA_VERSION,
               "technology_group_id": f"product_{group}",
               "domain_original": "Software products", "domain_ru": "Программные продукты",
               "technology_original": name, "technology_ru": name,
               "application_original": "software product", "application_ru": "программный продукт",
               "name_original": f"{name} (software product)",
               "name_ru": f"{name} (программный продукт)",
               "original_language": "en", "search_query": f"{name} software product",
               "primary_source_title_original": full_name,
               "primary_source_url": repo["html_url"],
               "primary_source_date": repo["created_at"][:10],
               "primary_source_date_precision": "day",
               "primary_source_language": "unknown",
               "primary_source_publisher": "GitHub repository",
               "source_evidence_excerpt_original": description[:600],
               "source_match_method": "named_repository_product_identity",
               "source_match_strength": "repository_entity",
               "source_relevance_score": 1,
               "science_count_scope": "not_searched",
               "selection_rule": "github_topic+named_product+homepage+description_filter",
               "evidence_note": "Repository identifies a named software product, not a technology/application method. Verify N3b identity before training.",
               "historical_evidence_years": "",
               **{col: "" for col in EXTRA_COLUMNS}, **features}
        row["patent_search_status"] = "not_searched"
        row["media_search_status"] = "not_searched"
        rows.append(row)
    return rows


def enrich_recent_science(rows):
    """Use the same recent-science feature code as positives, where available."""
    client = CrossrefClient(delay=1.1)
    try:
        for row in rows:
            if row["negative_type"] == "N3b":
                continue
            try:
                ranked, years = query_crossref(client, row, SNAPSHOT)
                if not ranked:
                    continue
                proxy = row_from_candidate(row, ranked, years, SNAPSHOT,
                                           allow_mature=True, require_recent=False)
                if proxy:
                    row.update({name: proxy[name] for name in FEATURE_NAMES})
                    # Regulatory source or historical DOI remains primary.
                    row["science_count_scope"] = "relevant_works_in_top_200_crossref_results_2018_to_snapshot"
                    row["science_sample_size"] = proxy["science_sample_size"]
            except Exception as exc:
                row["evidence_note"] += f" Recent science feature search failed: {type(exc).__name__}."
    finally:
        client.close()


def add_technology_context(rows):
    for row in rows:
        url = TECHNOLOGY_CONTEXT.get(row["technology_original"])
        if url:
            row["technology_official_source_url"] = url
            row["technology_context_scope"] = "technology_only_not_application"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=350)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--enrich-science", action="store_true")
    parser.add_argument("--enrich-news", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    found, failures = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch, candidate): candidate for candidate in candidates()}
        for future in as_completed(futures):
            try:
                candidate, ranked = future.result()
                if ranked:
                    found.append(make_row(candidate, ranked))
            except Exception as exc:
                failures.append((futures[future]["name_original"], str(exc)))
    found.sort(key=lambda r: (-float(r["source_relevance_score"]), r["name_original"]))
    entities = entity_rows()
    rows = hype_rows() + entities + found[:max(0, args.target - len(HYPE_CASES) - len(entities))]
    if args.enrich_science:
        enrich_recent_science(rows)
    if args.enrich_news:
        searchable = [row for row in rows if row["negative_type"] != "N3b"]
        enriched = enrich_rows(searchable, SNAPSHOT, workers=3, channels=("news",))
        by_id = {row["id"]: row for row in enriched}
        rows = [by_id.get(row["id"], row) for row in rows]
    add_technology_context(rows)
    if not rows:
        raise RuntimeError(f"No sourced examples; first failures: {failures[:3]}")
    with (ROOT / "data/training/positive_weak_signals_global.csv").open(newline="", encoding="utf-8") as f:
        positive_fields = csv.DictReader(f).fieldnames
    fields = positive_fields[:5] + ["negative_type", "stage", "trend"] + positive_fields[5:-len(FEATURE_NAMES)] + ["source_match_strength", "historical_evidence_years"] + list(FEATURE_NAMES)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"candidates={len(list(candidates()))} sourced={len(found)} output={len(rows)} failures={len(failures)} file={args.output}")
    print("domains", dict(Counter(r["domain_original"] for r in rows)))
    print("negative_types", dict(Counter(r["negative_type"] for r in rows)))
    if failures:
        print("sample_failures", failures[:3])


if __name__ == "__main__":
    main()
