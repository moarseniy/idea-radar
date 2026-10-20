# Внешние датасеты

## jrc_weak_signals.csv: каталог слабых сигналов JRC

Каталог опубликованных «слабых сигналов» в науке и технологиях из двух отчётов Совместного исследовательского центра Еврокомиссии (JRC). Нужен машине времени: на дату `published_date` эксперты уже знали о сигнале.

Файл: `jrc_weak_signals.csv`, 314 строк (93 из отчёта 2021 и 221 из отчёта 2024), UTF-8, колонки:
`signal_id, report, report_year, published_date, title_en, cluster, short_description, page, source_url`.

### Источники и лицензия

| Отчёт | Записей | PDF | Запись репозитория (DOI) | published_date |
|---|---|---|---|---|
| Weak signals in Science and Technologies in 2021 (Eulaerts et al., EUR 31171 EN) | 93 | https://publications.jrc.ec.europa.eu/repository/bitstream/JRC129501/kjna31171enn_1.pdf | https://publications.jrc.ec.europa.eu/repository/handle/JRC129501 (10.2760/700257) | 2022-08-22 |
| Weak signals in Science and Technologies - 2024 (Eulaerts, Grabowska, Bergamini, EUR 40213 EN) | 221 | https://publications.jrc.ec.europa.eu/repository/bitstream/JRC140959/JRC140959_01.pdf | https://publications.jrc.ec.europa.eu/repository/handle/JRC140959 (10.2760/6571994) | 2025-02-17 |

- Лицензия: CC BY 4.0 (в PDF сказано: «Creative Commons Attribution 4.0 International (CC BY 4.0)»). Атрибуция: European Commission, Joint Research Centre; Eulaerts O. и соавт., «Weak signals in Science and Technologies», Publications Office of the EU, Luxembourg.
- Источник данных: только официальные отчёты JRC. Репозиторий remark13/horizon и его `catalog.v6.json` не использовались (у него нет открытой лицензии).
- SHA-256 скачанных PDF (на момент сборки): 2021 — `7ed59081d3849e205d443af281db98c0f5311441946d6fc1463b6ef27e1b3a76`, 2024 — `1e4572af8d004764237bb0732b0846f8bc294ac845dd5df9249ecac0f9b2e0da`.

### Как собран

PDF лежат вне репозитория. Сборка воспроизводима (повторный запуск даёт тот же файл, MD5 `14470a3d52a461c8bcb9477c06bcccb9`):

```
.venv/bin/python ml/scripts/build_jrc_catalog.py \
    --pdf-2021 <путь>/JRC129501_kjna31171enn_1.pdf \
    --pdf-2024 <путь>/JRC140959_01.pdf
```

Скрипт `ml/scripts/build_jrc_catalog.py` извлекает текст через pymupdf (или pypdf) и не ходит в сеть. Он падает, если число сигналов не совпадает с заявленным в отчёте (93 и 221).

- **2021.** Сигналы — подписи «Figure N: Visualisations for …» (номера с 6; обзорные графики 1, 2, 4 и радары не считаются). `cluster` — раздел отчёта (2.1–2.7) по печатным страницам оглавления. `short_description` — абзац под заголовком сигнала в тексте отчёта.
- **2024.** Список сигналов — «Figure 1. Weak signals organized by clusters» (12 кластеров). `cluster` — кластер из этого списка. `short_description` — пункт «a)» из приложения с описаниями сигналов (имя сопоставляется с записью приложения).
- `page` — печатный номер страницы отчёта. Для 2021 это страница подписи к сигналу; для 2024 — страница записи приложения (если записи нет, то страница списка «Figure 1»: 8 или 9).
- `published_date` — дата в записи репозитория JRC (`dc.date.available`), не год в названии отчёта. Для машины времени использовать именно её: отчёт «2021» опубликован 22.08.2022, а «2024» — 17.02.2025.
- `title_en` — название как в отчёте (написание сохранено, включая строчные буквы и опечатки в тексте отчёта).

### Что проверено

- Число записей совпадает с заявленным: 93 (2021) и 221 (2024). В 2024 самом отчёте написано «221 weak signal» (аннотация, раздел 3, 12 кластеров).
- Сумма по кластерам: 2021 — 93, 2024 — 221.
- Выборочно проверено 5 случайных записей (по 2021 и 2024) программно, по тексту PDF: название и начало описания находятся на указанной печатной странице (смещение PDF относительно печатной — 3 для обоих отчётов). Ручного чтения страниц не было. Первая проба выявила ошибку номера страницы в приложении 2024 (номер на части страниц не читается), она исправлена, после чего все проверенные записи совпали.
- Повторный запуск скрипта даёт побайтово тот же CSV.

### Известные пропуски

- **Записи без описания (`short_description` пусто), 21 шт.** 2021: aphantasia, LYTACs, PROTACs (в тексте отчёта нет абзаца под заголовком). 2024: 18 сигналов, для которых в приложении нет пункта «a)»: flexible Zinc ion batteries, urea electrosynthesis, space-Air-Ground Integrated network, 6G V2X, cyber twin architecture, digital twin for connected vehicle simulation, human AI interface, practical byzantine fault tolerance blockchain, optoretinography, Line-field confocal optical coherence tomography, MRI radiomics, intermittently scanned continuous glucose monitoring, internet of health things, edge computing in agriculture, food upcycling, smart local energy systems, sustainable aviation fuel, quantum drug discovery. Описания сочинять не стали.
- **Число сигналов 2024.** В задании было «около 216»; в самом отчёте 221, и скрипт сверяется с 221.
- Для 18 сигналов 2024 без записи в приложении `page` указывает на страницу списка «Figure 1», а не на страницу описания.
- Кластеры 2021 взяты по диапазонам страниц оглавления, а не по тексту подписи. Сигнал, стоящий на границе раздела, мог быть отнесён к соседнему кластеру; это не проверялось по каждой записи.
- Колонки `report_year` и `published_date` для 2021 расходятся: при выборе «на дату» использовать `published_date`.
