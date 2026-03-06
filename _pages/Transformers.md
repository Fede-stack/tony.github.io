---
permalink: /Transformers/
title: "Transformers-based Features"
author_profile: true
redirect_from: 
  - /md/
  - /Transformers.html
---

{% include toc %}

# Extract Neural Features

## HiTOP


The Hierarchical Taxonomy of Psychopathology (**HiTOP**) is a dimensional framework that organizes psychopathology into a hierarchy of empirically derived constructs, from broad spectra (e.g., Internalizing, Thought Disorder, Detachment) down to specific maladaptive traits such as *anxiousness*, *withdrawal*, *depressivity*, or *emotional lability*. Unlike categorical diagnostic systems (e.g., DSM-5), HiTOP treats psychopathological features as continuous dimensions, enabling a finer-grained, transdiagnostic characterization of psychological functioning. Detecting these traits directly from language is therefore a clinically meaningful task, as each trait represents a stable, observable dimension of personality dysfunction that can manifest in naturalistic text.
With **TONY** you can extract HiTOP traits from text with just a few lines of code, leveraging a lightweight fine-tuned LLM that runs efficiently on consumer hardware. If you are working with an Apple Silicon Mac (M1/M2/M3/M4 chip), you can choose to run the model locally using **MLX**, Apple's machine learning framework optimized for the Metal Performance Shaders backend, enabling fast and energy-efficient inference directly on your device — no GPU server or internet connection required.

```python
from TONY.HiTOP import HiTOP_Predictor, HiTOP_Predictor_mlx
text = 'Some days I keep living, even though I feel completely alone in the world'
hitop = HiTOP_Predictor(model_name='FritzStack/HiTOP-Llama-3.2-3B_4bit-merged')
hitop.predict_HiTOP(text)
# Output: Anhedonia, Withdrawal, Depressivity
```

---

## Interpersonal Risk Factors

The **Interpersonal Risk Factors** (**IRF**) module detects the two core interpersonal risk factors defined by the Interpersonal Theory of Suicide (Van Orden et al., 2010): **Thwarted Belongingness** (TBE) — the painful feeling of being disconnected from others — and **Perceived Burdensomeness** (PBU) - the perception of being a liability to those around oneself. The module not only predicts the presence of each factor but also highlights the supporting textual evidence, providing interpretable outputs for both clinical and research use.

```python
from TONY.IRF import IRFPredictor, IRFPredictor_mlx

text = 'Some days I keep living, even though I feel completely alone in the world'
irf = IRFPredictor(model_name='FritzStack/IRF-Qwen3-8B_4bit-merged')
irf.highlight_evidence_IRF(text)

# Output:
# Question 1: Is there evidence of Thwarted Belongingness?
# Answer: Yes
# Text Evidence: feel completely alone

# Question 2: Is there evidence of Perceived Burdensomeness?
# Answer: No
# Text Evidence: nan
```

If you are working with an Apple Silicon Mac (M1/M2/M3/M4 chip), you can run the model locally using **MLX**, enabling fast and energy-efficient inference without requiring a GPU or internet connection.

---


## Emotions as Markers for Specific Mental Health Conditions

Emotion recognition from text provides a fine-grained signal that complements both diagnostic categories and dimensional trait models. Rather than relying on broad sentiment polarity, detecting discrete emotions, enables a more subtle characterization of affective states that have been consistently linked to specific psychopathological conditions.

**TONY**'s emotion module is built on **GoEmotions** (Demszky et al., 2020), a large-scale dataset of 58k Reddit comments annotated across 27 fine-grained emotion categories. The underlying model is fine-tuned to map naturalistic text onto this taxonomy, providing richer emotional resolution than standard positive/negative sentiment classifiers. As with the other neural modules, the model is available in a quantized 4-bit format and can be run locally on Apple Silicon via **MLX** for fast, privacy-preserving inference.



```python
from TONY.Emotions import Emotions_Predictor, Emotions_Predictor_mlx

text = 'Some days I keep living, even though I feel completely alone in the world'
emotioner = Emotions_Predictor(model_name='FritzStack/QWEN4B-GoEmotions_4bit')
emotioner.predict_emotions(text)
# Output: sadness
```

## Implemented Models *(more coming soon)*

| CATEGORY | Models                 | Model (MLX)                     |
| --------- | ---------------------------------- | ------------------------------------ |
| HiTOP     | FritzStack/HiTOP-QWEN4B_4bit           | FritzStack/HiTOP-QWEN4B-mlx-Q4       |
| HiTOP     | FritzStack/HiTOP-Llama-3B_4bit     | FritzStack/HiTOP-Llama-3B-mlx-Q4     |
| HiTOP     | FritzStack/HiTOP-Phi4_4bit         | FritzStack/HiTOP-Phi4-mlx-Q4         |
| IRF       | FritzStack/IRF-Llama-3B_4bit       | FritzStack/IRF-Llama-3B-mlx-Q4       |
| IRF       | FritzStack/IRF-QWEN4B_4bit         | FritzStack/IRF-QWEN4B-mlx-Q4         |
| IRF       | FritzStack/IRF-QWEN8B_4bit         | FritzStack/IRF-QWEN8B-mlx-Q4         |
| EMOTIONS  | FritzStack/Llama3B-GoEmotions_4bit | FritzStack/Llama3B-GoEmotions-mlx-Q4 |
| EMOTIONS  | FritzStack/QWEN4B-GoEmotions_4bit  | FritzStack/QWEN4B-GoEmotions-mlx-Q4  |
| EMOTIONS  | FritzStack/RACLETTE-fp16  |   |
