# LSERBlm

Семь запусков decoder-only моделей. Данные и BPE собственные; чужие веса не используются.
Демонстрация всех моделей с сохранёнными генерациями — [serbian_llm.ipynb](serbian_llm.ipynb).

## Данные и токенизация

Wikipedia `20231101.sr` и FineWeb2 `srp_Cyrl`, ревизии в `configs/data_mix.json`.
Очистка: NFC, длина, сербская кириллица, язык по метаданным, повторы, списки,
exact/MinHash-дедупликация. При расширении дополнительно удалены служебные разделы
Wikipedia и явные меню/таблицы; train проверен против исходных и очищенных holdout.

| Корпус | Train-документы | Train-токены | Объём train UTF-8 |
|---|---:|---:|---:|
| `data/serbian_mix_v2` | 78 484 | 80 159 914 | 489,9 МБ |
| `data/serbian_expanded_v3` | 250 886 | 267 310 738 | 1 669,2 МБ |

Исходный корпус: около 350 МБ FineWeb и 150 МБ Wikipedia после фильтров, до общей
дедупликации; итог всех split — 499,5 МБ. Первичное разбиение по документам —
98/1/1. При расширении validation (828 документов) и test (796) сохранены побайтово.
Новый аудит не нашёл общих абзацев ≥200 символов между train и holdout.
Фильтрация новой стадии не отменяет прежнего пересечения данных у родителя.

Один **ByteLevel BPE, 8192 токена**, обученный только на исходном train.
NFC, сохранение регистра и диакритики; PAD/BOS/EOS/UNK. Документ кодируется как
`BOS + text + EOS`, `.bin` — uint16. Числа токенов выше включают BOS/EOS.
На новом train: 1,91 токена на слово, UNK=0. BPE при продолжении не переобучался.

## Модели

Общее: контекст 512, causal attention, pre-RMSNorm, SwiGLU, GQA, связанные
embedding/выходной слой. RoPE во всех моделях, кроме absolute-абляции.
RMSNorm контролирует масштаб активаций, SwiGLU добавляет гейтинг FFN;
GQA и связанные embeddings экономят параметры, RoPE обходится без таблицы позиций.
Обучение: AdamW, BF16, PyTorch SDPA, WSD, gradient clipping=1;
effective batch=32 последовательности, 16 384 токена на обновление.
Базовые запуски: 20 000 шагов, 327 680 000 предъявлений токенов, LR 6e-4 → 6e-5.

| Имя запуска | Параметры | Слои × ширина; FFN; Q/KV | Чистое время обучения | Что проверяем |
|---|---:|---|---|---|
| `absolute_41m` | 41 109 888 | 24×384; 1024; 6/2 | 2 ч 12 мин | Обучаемые absolute positions — контроль для RoPE |
| `rope_41m` | 40 913 280 | 24×384; 1024; 6/2 | 2 ч 45 мин | Базовая небольшая модель с RoPE без таблицы позиций |
| `serbian_283m_1` | 283 170 240 | 28×960; 2560; 15/5 | 1 ч 50 мин | Больше ширины и параметров для увеличения ёмкости |
| `serbian_283m_2` | 283 183 872 | 44×768; 2048; 12/4 | 2 ч 10 мин | Более глубокая узкая сеть при том же бюджете параметров |
| `serbian_283m_3` | 283 173 824 | 28×960; 2560; 15/5 | 4 ч 45 мин | QK-RMSNorm перед RoPE + повтор каждого блока A→A→B→B |
| `deep_continue_A` | 283 183 872 | Deep №2 без изменений | Новая стадия ещё не завершена | Продолжение на прежних 80M токенов, контроль дополнительного обучения |
| `deep_continue_B` | 283 183 872 | Deep №2 без изменений | Новая стадия ещё не завершена | Продолжение на очищенном расширенном train, основная ветка |

Время первых пяти моделей — сумма train-шагов, без validation, генераций и записи
checkpoint; оборудование A100 80GB. Текущие время/шаги A/B читаются из файлов в ноутбуке.
В №3 28 уникальных блоков и 56 применений, веса разделяются; это комбинированный
эксперимент, отдельный эффект QK-нормализации здесь не измерен.
Веса FP32 занимают около 164 МБ у 41M и 1,13 ГБ у 283M; training checkpoint с AdamW —
около 0,49 и 3,40 ГБ соответственно.

Полная одинаковая validation (828 документов, stride=256): PPL RoPE-41M **18,613**,
Wide №1 **16,608**, Deep №2 **16,521**, №3 **16,664**. Поэтому родитель A/B — Deep №2.
Для A/B: ещё 30 518 шагов / 500 006 912 токенов, LR 1e-4 → 1e-5, warmup=200;
оптимизатор и расписание сброшены, происхождение записано в `lineage.json`.
Полная оценка A/B ещё не выполнена. PPL из packed-логов и документную PPL не смешиваем.

## Запуск

Из каталога с этим README:

```bash
source /home/aivanova/llm-medicine/.venv/bin/activate
python -m pip install -e . --no-deps
mkdir -p runs/logs
```

Исходные данные, собственный BPE и кодирование:

```bash
python scripts/prepare_data.py --output data/serbian_mix_v2 --fineweb-mb 350 --wiki-mb 150
python -m serbian_llm tokenizer --corpus data/serbian_mix_v2/corpus --output data/serbian_mix_v2/bpe8k --vocab-size 8192
python -m serbian_llm encode --corpus data/serbian_mix_v2/corpus --tokenizer data/serbian_mix_v2/bpe8k
```

Расширение и кодирование прежним BPE:

```bash
python scripts/prepare_data.py --config configs/data_expanded.json --output data/new_candidates_v3 --fineweb-mb 1500 --wiki-mb 200 --candidates-only
python -m serbian_llm expand-corpus --base-corpus data/serbian_mix_v2/corpus --input-jsonl data/new_candidates_v3/raw/wikipedia.jsonl data/new_candidates_v3/raw/fineweb.jsonl --output data/serbian_expanded_v3/corpus
python -m serbian_llm encode --corpus data/serbian_expanded_v3/corpus --tokenizer data/serbian_mix_v2/bpe8k --reuse-tokenizer --output data/serbian_expanded_v3/bpe8k
```

Подготовленные файлы повторно создавать не нужно. Для нового корпуса выбирай новый
выходной каталог: проверка идентичности защищает готовые данные от перезаписи.

Повтор любого из семи запусков (имя из таблицы, GPU, новый каталог результата):

```bash
bash scripts/train_model.sh serbian_283m_2 2 runs/serbian_283m_2_new
bash scripts/train_model.sh deep_continue_B 3 runs/deep_continue_B_new
```

Каждая команда запускает одну выбранную модель через nohup. Конфиги: absolute →
`ablation_absolute.json`, RoPE-41M → `main_rope.json`, №1–3 → `main_rope_1.json`–
`main_rope_3.json`, A/B → `continued_deep.json`. Уже работающие A/B перезапускать не нужно.
Для нового одновременного A/B есть `scripts/run_continuation_ab.sh`.
Логи повторных запусков: `runs/logs/<имя_выходного_каталога>.log`.

Возобновление остановленной B с её состоянием оптимизатора:

```bash
CUDA_VISIBLE_DEVICES=1 nohup python -u -m serbian_llm continue-train --config configs/continued_deep.json --resume runs/deep_continue_B/last.pt --tokenizer data/serbian_expanded_v3/bpe8k --output runs/deep_continue_B --device cuda:0 >> runs/logs/deep_continue_B.log 2>&1 &
```

Инференс и полная validation:

```bash
python -m serbian_llm generate --checkpoint runs/deep_continue_B/best.pt --prompt 'Небо је плаво, а сунце' --max-new-tokens 128 --device cuda:2
python -m serbian_llm evaluate --checkpoint runs/deep_continue_B/best.pt --data data/serbian_mix_v2/corpus/validation.jsonl --output runs/deep_continue_B/full_validation.json --limit 0 --stride 256 --device cuda:2
```

## Код

`src/serbian_llm/`: `streaming.py`, `preparation.py`, `data.py`, `expansion.py` —
данные; `tokenization.py` — BPE и `.bin`; `model.py` — все архитектуры;
`training.py`, `batching.py`, `checkpoints.py` — обучение и resume;
`inference.py`, `evaluation.py` — генерация и document-level NLL/PPL/BPB;
`experiments.py`, `reporting.py` — сводки запусков и графики. `configs/` — только
настройки использованных моделей и данных. `runs/` — веса, конфиги, метрики, отчёты.
