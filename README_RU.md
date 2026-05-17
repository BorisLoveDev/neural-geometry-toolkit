# Neural Geometry Lab

> Лёгкий **исследовательский (exploratory)** инструмент в духе Goodfire-style neural-geometry экспериментов: ищет **круги, циклические представления, Fourier-фичи и модульно-арифметическую геометрию** в скрытых активациях LLM.
>
> 🟡 **Честные рамки (v0.1):** это **starter-kit**, а не фреймворк causal-abstraction. Он выявляет *suggestive* геометрические структуры. Для полноценных causal-экспериментов (activation patching, path steering, output manifolds) используй [**Goodfire Causalab**](https://github.com/goodfire-ai/causalab). **Не affiliated с Goodfire, не реимплементация их стека.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="showcase/01_numbers_heatmap_KEY.png" width="900" alt="Fourier-probe heatmap по слоям — модульная структура чисел в 8B instruct LLM"/>
</p>

<p align="center"><em>Линейная декодируемость <code>(cos 2πn/p, sin 2πn/p)</code> из активаций по слоям. Жёлтое = почти идеальный <code>R²</code>. Модель буквально хранит числа как фазы на нескольких окружностях.</em></p>

---

## Что это

`nglab` — компактный research-инструмент для изучения **геометрии скрытых активаций LLM**:

- Лежат ли дни недели **по кругу** в активациях модели?
- Хранятся ли числа на **нескольких модульных окружностях** одновременно (mod 5, 10, 100)?
- Считает ли модель `a + b` **поворотом** представлений на этих окружностях?
- В каком **слое** появляется каждый из этих эффектов?

Инструмент вдохновлён современными работами по геометрии нейронных представлений
(в частности — публикациями **Goodfire** про circular representations в LLM и
работой **Nanda et al.** про модульную арифметику в gokked-трансформерах) и
позволяет запускать эти эксперименты на **любой causal-LM из Hugging Face** —
от GPT-2 до Llama-3 и Qwen.

### Когда использовать этот toolkit, а когда Goodfire Causalab

| Что нужно | Брать |
|---|---|
| Быстро увидеть, есть ли в активациях круговая/Fourier-структура | **этот toolkit** |
| Однокартовое исследование с layer sweep и красивыми графиками | **этот toolkit** |
| Hypothesis generation перед серьёзным interpretability-исследованием | **этот toolkit** |
| Causal abstraction: задать high-level алгоритм и проверить, реализуют ли его внутренние компоненты | [**Goodfire Causalab**](https://github.com/goodfire-ai/causalab) |
| Activation patching, path steering, output manifolds, pullback | [**Goodfire Causalab**](https://github.com/goodfire-ai/causalab) |
| Воспроизведение их `weekdays_8b_pipeline` / `weekdays_geometry.ipynb` | [**Goodfire Causalab**](https://github.com/goodfire-ai/causalab) |

**Правило большого пальца.** Этот инструмент — *найти* кандидата за 10 минут. Causalab — *доказать*, что он реально реализует алгоритм.

### Возможности

- **CLI + Streamlit UI** — `nglab weekdays`, `nglab numbers`, `nglab addition`, либо браузерный explorer.
- **Layer sweeps** одним флагом (`--layers all`) — получаешь heatmap «где какое представление живёт».
- **Fourier probes** — кросс-валидированный `R²` для линейного декодирования `cos/sin(2πn/p)` по слоям и периодам.
- **Метрики PCA-кругов** — насколько активации лежат на настоящем круге (`angle_mae_deg`, `adjacent_similarity_gap`).
- **Лёгкий activation steering** через хуки — для последующих causal-экспериментов.
- **Никакого зоопарка зависимостей** — `torch`, `transformers`, `numpy`, `pandas`, `sklearn`, `matplotlib`. Работает на одной GPU; маленькие модели — на CPU.

---

## Showcase — что получается на выходе

Подробный разбор результатов на 8B instruction-модели лежит в [`showcase/`](showcase/):

| | |
|:---:|:---:|
| ![Numbers mod-10 ring](showcase/02b_numbers_mod10_layer02_BEST.png) | ![Weekday heptagon](showcase/03b_weekdays_layer19_BEST.png) |
| **Числа `0..99` на mod-10 окружности** — 10 чётких кластеров по 10 точек. | **Дни недели как семиугольник** в правильном циклическом порядке. |

См. [`showcase/README_RU.md`](showcase/README_RU.md) — полный разбор каждой
картинки и интерпретация, включая **отрицательный** результат (модель 8B не
делает сложение по модулю чисто-геометрически).

---

## Установка

```bash
git clone https://github.com/BorisLoveDev/neural-geometry-toolkit.git
cd neural-geometry-toolkit
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

Для Streamlit UI: `pip install -e ".[app]"`.

GPU желательно для моделей крупнее ~1B параметров. CPU / Apple-MPS работает для `gpt2` и подобных.

---

## Быстрый старт

### Дни недели по кругу

```bash
nglab weekdays --model gpt2 --layers all --outdir runs/gpt2_weekdays
```

Создаст PCA-картинки по слоям, arc-vs-chord график и CSV метрик. Главная цифра — `angle_mae_deg`: случайный порядок ≈ 50°, чистый цикл < 25°.

### Числа на модульных окружностях

```bash
nglab numbers --model gpt2 --layers all \
  --start 0 --end 99 \
  --periods 2,5,10,20,50,100 \
  --outdir runs/gpt2_numbers
```

Главный артефакт — `fourier_scores_numbers_heatmap.png`: `R²` по `слой × период`.

### Сложение как геометрический калькулятор

```bash
nglab addition --model gpt2 --layers all \
  --max-a 20 --max-b 20 \
  --periods 2,5,10,20,50 \
  --outdir runs/gpt2_addition
```

Проверяет, читается ли `(a + b) mod p` из активации перед токеном ответа.

### Большая модель (любой HF id)

```bash
nglab weekdays --model Qwen/Qwen2.5-0.5B --layers all --batch-size 8 \
  --outdir runs/qwen05_weekdays

nglab numbers --model meta-llama/Llama-3.2-3B --layers all \
  --device-map auto --dtype bf16 --outdir runs/llama32_3b_numbers
```

### Streamlit UI

```bash
streamlit run streamlit_app.py
```

В сайдбаре выбираешь модель, задачу, слой; графики появляются в браузере.

---

## Как это работает (в одном абзаце)

Для каждой задачи мы строим небольшой датасет промптов, в которых целевой
токен (`Monday`, `17`, …) встречается в разных естественных контекстах.
Прогоняем каждый промпт через модель с `output_hidden_states=True`, усредняем
скрытое состояние по позициям целевого токена, получаем одну активацию на пару
(значение, промпт). Дальше:

- **Метрики круга** — PCA по центроидам классов, дисперсия радиусов, ошибка
  углового порядка, gap «соседи vs не-соседи» по cosine-similarity.
- **Fourier probes** — ridge-регрессия из активации в `cos(2πn/p), sin(2πn/p)`
  для каждого периода `p`, кросс-валидированный `R²`.
- **Layer sweep** — повтор для каждого слоя; heatmap `период × слой`.

Извлечение лежит в [`nglab/core.py`](nglab/core.py), пробы — в
[`nglab/geometry.py`](nglab/geometry.py).

---

## Структура проекта

```
neural-geometry-toolkit/
├── nglab/
│   ├── core.py        # загрузка модели + извлечение активаций
│   ├── datasets.py    # промпт-датасеты под каждую задачу
│   ├── geometry.py    # метрики круга + Fourier probes
│   ├── plotting.py    # графики
│   ├── steering.py    # лёгкий activation steering
│   └── cli.py         # CLI на argparse
├── examples/
│   └── neural_geometry_quickstart.ipynb
├── showcase/          # демо-результаты на 8B модели
├── ng_lab.py          # CLI-entrypoint
└── streamlit_app.py   # браузерный UI
```

---

## Честные ограничения (прочти перед выводами)

Это ранний exploratory-инструмент. Это реальные пробелы, а не придирки:

1. **PCA-картинка показывает структуру, не причину.** Чистый круг в активациях не доказывает, что модель его использует для вычислений. Для causal-claims нужен patching / steering / ablation — чище всего через [Goodfire Causalab](https://github.com/goodfire-ai/causalab).
2. **Линейная декодируемость ≠ модель этим пользуется.** Высокий `R²` Fourier-пробы говорит лишь о наличии информации; она может быть «пассажиром». До любого механистического утверждения нужно вмешательство.
3. **В v0.1 нет baseline-контролей.** Сейчас toolkit не запускает: random-label baseline, random-model baseline, train/test split по *значению* (не только по промпту), held-out на невиданных числах / невиданных шаблонах промптов. **Без этого нельзя исключить «проба сама подогналась».** См. roadmap ниже.
4. **На маленьких моделях сигнал часто слабый.** Если на `gpt2` шум — попробуй 1B–8B instruct-модель.
5. **Токенизация важна.** Экстрактор старается брать активации именно по target-span; на необычных токенизаторах сверяйся с `prompts_*.csv`.
6. **Отрицательные результаты — нормально, но здесь underdetermined.** В 8B-демо в `showcase/` геометрический калькулятор сложения не подтвердился. Без контролей из пункта 3 это **повод для расследования**, а не вывод про модель.

---

## Дорожная карта

**Ближайшее (закрыть пробелы v0.1):**

- **Тесты** — токен-span extraction, индексация слоёв, форма Fourier-пробы, эквивалентность shuffled-label baseline, корректность train/test split.
- **Контроли и baseline'ы** — random-label probe (должен схлопнуться к chance), random-model probe (нижняя планка), split-by-value (не только by-prompt), held-out на невиданных числах и шаблонах.
- **Воспроизводимость** — фиксированные seed'ы, детерминированный порядок данных, маленькие референсные CSV для диффа.

**Среднесрок (causal-mode):**

- **Phase-rotation steering** вдоль найденных направлений круга (двигать θ, а не линейную дельту).
- **Centroid patching** между центроидами классов на выбранном слое.
- **Subspace ablation** — проверка, действительно ли геометрическое подпространство *load-bearing*.
- **Counterfactual activation patching** между циклическим и арифметическим промптами.

Если production-grade causal-режим нужен **сегодня** — [**Goodfire Causalab**](https://github.com/goodfire-ai/causalab) уже даёт `activation_manifold`, `output_manifold`, `path_steering`, `pullback` из коробки. Наш toolkit планирует интеграцию с `transformer-lens` и останется намеренно лёгким.

Активации сохраняются в `activations_*.npz`, чтобы достроить любое из вышеперечисленного поверх существующих ранов.

---

## Связанные работы

- [**Goodfire Causalab**](https://github.com/goodfire-ai/causalab) — официальный open-source фреймворк causal-abstraction interpretability; содержит `weekdays_8b_pipeline`, `weekdays_geometry.ipynb`, анализы `activation_manifold`, `output_manifold`, `path_steering`, `pullback`. **Серьёзная версия того, на что наш toolkit намекает.**
- **Goodfire** — лаборатория interpretability; ключевые публикации про circular & modular representations в LLM.
- Nanda et al., *Progress measures for grokking via mechanistic interpretability* — дискретные Fourier-фичи в трансформерах на модульной арифметике.
- Engels et al. — круги для дней недели / месяцев / часов в LM.
- Anthropic — работы по геометрии features и circuits.

**Не affiliated с Goodfire.** Это независимый MIT-лицензированный exploratory-инструмент. Для цитирования см. [`CITATION.cff`](CITATION.cff).

---

## Контрибьюции

Issues и PR приветствуются. Что особенно полезно:

- Новые задачи (месяцы, часы, стороны света, конвертация систем счисления).
- Интеграция с `transformer-lens` для расширенных hook-вмешательств.
- Утилиты causal steering / patching.
- Воспроизведения на других семействах моделей — присылайте PNG и метрики PR-ом.

---

## Лицензия

MIT — см. [LICENSE](LICENSE).

---

## Другие языки

- 🇬🇧 [README in English](README.md)
