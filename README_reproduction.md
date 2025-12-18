# Reproduction report (Lavender LoRA)

This document summarizes the evaluation results of Llama-3.2-11B-Vision-Instruct (base) vs Lavender LoRA using VLMEvalKit, and provides an interpretation of the observed performance patterns.

## Scope

- Framework: `open-compass/VLMEvalKit`
- Base model: `meta-llama/Llama-3.2-11B-Vision-Instruct` (local path `./models/base`)
- LoRA: `lxasqjc/lavender-llama-3.2-11b-lora` (local path `./models/lavender-lora`)
- Custom wrapper: `VLMEvalKit/vlmeval/vlm/lavender.py`
- Decoding used in this run:
  - `max_new_tokens=512`
  - `do_sample=False`
- Benchmarks:
  - `DocVQA_VAL`
  - `ChartQA_TEST`
  - `MMBench_DEV_EN`
  - `MME`

## Commands

```bash
export PYTHONHASHSEED=42

python run.py \
  --data MMBench_DEV_EN --model \
  llama_vision_base llama_vision_lavender \
  --mode all --work-dir ./outputs/tier1 

python run.py \
  --data DocVQA_VAL \
  --model llama_vision_base llama_vision_lavender \
  --mode all --work-dir ./outputs/tier1 

python run.py \
  --data ChartQA_TEST \
  --model llama_vision_base llama_vision_lavender \
  --mode all --work-dir ./outputs/tier1

python run.py \
  --data MME \
  --model llama_vision_base llama_vision_lavender \
  --mode all --work-dir ./outputs/tier1
```

## Results

### Overall comparison

| Benchmark | Base | Lavender | Δ (Lav-Base) |
| --- | --- | --- | --- |
| DocVQA_VAL (acc) | 80.34 | 85.77 | +5.43 |
| ChartQA_TEST (acc) | 9.96 | 32.60 | +22.64 |
| MMBench_DEV_EN (acc, %) | 31.36 | 30.15 | -1.20 |
| MME (perception + reasoning) | 1608.94 | 1841.99 | +233.05 |

### ChartQA breakdown

| Split | Base | Lavender | Δ |
| --- | --- | --- | --- |
| test_human | 11.52 | 31.44 | +19.92 |
| test_augmented | 8.40 | 33.76 | +25.36 |

### MMBench (selected sub-scores, %)

| Category | Base | Lavender | Δ |
| --- | --- | --- | --- |
| ocr | 53.85 | 56.41 | +2.56 |
| image_topic | 55.56 | 55.56 | +0.00 |
| attribute_recognition | 55.41 | 48.65 | -6.76 |
| celebrity_recognition | 27.27 | 48.48 | +21.21 |
| social_relation | 30.23 | 4.65 | -25.58 |
| image_emotion | 36.00 | 8.00 | -28.00 |
| image_scene | 59.62 | 43.27 | -16.35 |
| image_quality | 20.75 | 37.74 | +16.98 |
| structuralized_imagetext_understanding | 10.26 | 19.23 | +8.97 |

### MME

**Top increases**:

| Metric | Base | Lavender | Δ |
| --- | --- | --- | --- |
| perception | 1327.87 | 1486.28 | +158.41 |
| artwork | 51.00 | 126.50 | +75.50 |
| reasoning | 281.07 | 355.71 | +74.64 |
| text_translation | 27.50 | 70.00 | +42.50 |
| celebrity | 118.24 | 157.06 | +38.82 |
| OCR | 140.00 | 170.00 | +30.00 |
| code_reasoning | 60.00 | 87.50 | +27.50 |
| count | 128.33 | 143.33 | +15.00 |

**Decreases**:

| Metric | Base | Lavender | Δ |
| --- | --- | --- | --- |
| existence | 195.00 | 195.00 | +0.00 |
| landmark | 137.00 | 133.75 | -3.25 |
| numerical_calculation | 80.00 | 72.50 | -7.50 |
| posters | 148.30 | 140.14 | -8.16 |

## Analysis

### 1. Lavender improves document/chart-centric performance

- DocVQA_VAL increases by 5.43 points.
- ChartQA_TEST increases by 22.64 points, with gains on both test_human and test_augmented splits.
- MME increases by 233.05 overall (perception+reasoning), with the largest per-metric gains on:
  - **artwork** (+75.50)
  - **text_translation** (+42.50)
  - **OCR** (+30.00)
  - **celebrity_recognition** (+38.82)
  - **code_reasoning** (+27.50)

This pattern is consistent with Lavender’s positioning as a diffusion-instruction-tuning approach designed to strengthen vision-language grounding and instruction-following on visually rich inputs (including charts/documents).

### 2. MMBench

- **MMBench_DEV_EN**: 31.36% (Base), 30.15% (Lavender).
- **Sub-scores show mixed movement**: OCR/image-style related categories rise, while categories such as social_relation and image_emotion drop substantially.

MMBench is multi-choice, and its absolute score can be highly sensitive to the evaluation pipeline’s answer parsing/judging mode and dataset variant. The Lavender paper reports much higher MMBench numbers using VLMEvalKit with an LLM evaluator for scoring; as a result, the MMBench numbers in this run are not directly comparable to the paper’s headline results without matching the same evaluation settings.

### 3. LoRA effectiveness

The consistent improvements across DocVQA, ChartQA, and multiple MME sub-metrics indicate that:
- The LoRA checkpoint is being applied and is affecting model behavior measurably.
- The main gains are concentrated on tasks with strong text-in-image / structured-visual signals (documents, charts, OCR-adjacent categories), which aligns with the intended scope of Lavender.
