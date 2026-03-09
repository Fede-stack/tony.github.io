---
permalink: /Lexicon/
title: "Lexicon-Level Features"
author_profile: true
redirect_from: 
  - /md/
  - /Lexicon.html
---

{% include toc %}


# How to extract Linguistic Markers?



The **Lexicon** module provides a comprehensive suite of interpretable lexical features grounded in the psycholinguistic literature. Each marker has been selected based on its established relevance to mental health research: from first-person pronoun usage as a proxy for self-focus and depression, to absolutist language, negation frequency, and affective dimensions such as valence, arousal, and dominance. Rather than relying on ad hoc feature engineering, TONY consolidates a broad set of psychologically validated indicators — spanning lexical, syntactic, morphosyntactic, stylistic, emotional, and cohesion dimensions — into a single, unified extraction pipeline. This positions TONY as a fully open-source and reproducible alternative to LIWC (Linguistic Inquiry and Word Count), the most widely adopted tool in the field, which remains closed-source and commercially licensed, limiting transparency and extensibility in research workflows. 

> [!NOTE]
> TONY uses spaCy under the hood. Before running the cell below, install the English language model:
> ```bash
> python -m spacy download en_core_web_sm
> ```


```python
from TONY.Lexicon import MarkersExtraction, MarkersExtractionColab

app = MarkersExtractionColab() #Use MarkersExtraction if you are running it locally
```

<br><br>
<img src="https://raw.githubusercontent.com/Fede-stack/TONYpy/main/images/gif_extractmarkers.gif" width="500">

Alternatively, you can extract features without using the ui with two lines of code: 

```python
from TONY.Lexicon import LexiconLevelFeatures

text = 'Some days I keep living, even though I feel completely alone in the world'
markers = LexiconLevelFeatures(language="en")
markers.extract_markers(text)
# Output:
# LinguisticMarkers(lexical_diversity=0.875, lexical_sophistication={'mean_frequency': 0.002983, 'std_frequency': 0.004642}, word_prevalence=0.25, sentence_complexity=0.0, subordination_rate=0.0, coordination_rate=0.0, pronoun_usage={'first_person': 0.125, 'second_person': 0.125, 'third_person': 0.0}, verb_tense_distribution={'past': 0.0, 'present': 1.0, 'future': 0.0}, negation_frequency=0.0, emotion_scores={'joy': 0.0, 'sadness': 0.0, 'anger': 0.0, 'fear': 0.0, 'disgust': 0.0, 'surprise': 0.0, 'anticipation': 0.0, 'trust': 0.0}, sentiment_polarity=0.0, sentiment_intensity=0.4717, cohesion_score=1.0, lexical_overlap=0.0, connectives_usage=0.0, affect_scores={'valence': 0.0, 'arousal': 0.0, 'dominance': 1.0}, cognitive_processes={'insight': 0.0, 'causation': 0.0, 'certainty': 0.0, 'tentative': 0.0}, social_processes=0, readability_index=1.0, average_sentence_length=8.0, graph_connectedness=1.0, semantic_coherence=0.5, absolutist_word_frequency=0.0, death_word_frequency=0.0, anxiety_word_frequency=0.0, sadness_word_frequency=0.0, anger_word_frequency=0.0, question_ratio=0.0, exclamation_ratio=0.0, incomplete_sentence_ratio=0.0, mean_dependency_distance=2.1667, past_future_ratio=0.0, repetition_rate=0.125, body_word_frequency=0.0, achievement_word_frequency=0.0, pos_frequencies={'prep': 0.1429, 'auxverb': 0.1429, 'adverb': 0.0, 'conj': 0.0}, extended_pos={'noun': 0.1429, 'verb': 0.2857, 'adjective': 0.0, 'interjection': 0.0}, morphological_features={'indicative_ratio': 0.3333, 'subjunctive_ratio': 0.0, 'singular_ratio': 1.0, 'plural_ratio': 0.0}, dependency_features={'nsubj_rate': 2.0, 'dobj_rate': 1.0}, ner_features={'person_ref_rate': 0.0, 'temporal_ref_rate': 0.0})
```

# Features Extracted by the Lexicon-Level Module

| Category | Feature | Description |
|---|---|---|
| **Lexical** | `lexical_diversity` | Type-Token Ratio (TTR) |
| | `lexical_sophistication` | Mean/std word frequency (wordfreq) |
| | `word_prevalence` | Function word ratio |
| | `repetition_rate` | Inverse TTR, proxy for rumination |
| **Syntactic** | `sentence_complexity` | Weighted sub/coordination score |
| | `subordination_rate` | Subordinate clause frequency |
| | `coordination_rate` | Coordinate clause frequency |
| | `mean_dependency_distance` | Mean dependency tree distance (spaCy) |
| | `incomplete_sentence_ratio` | Proportion of sentences < 3 words |
| | `average_sentence_length` | Mean words per sentence |
| **Stylistic** | `pronoun_usage` | First, second, and third person rates |
| | `verb_tense_distribution` | Past, present, and future tense ratios |
| | `past_future_ratio` | Log-ratio of past vs future tense usage |
| | `negation_frequency` | Negation word rate |
| | `question_ratio` / `exclamation_ratio` | Sentence-ending punctuation ratios |
| **Emotion & Sentiment** | `emotion_scores` | 8 NRC emotions (normalized) |
| | `sentiment_polarity` | NRC positive/negative ratio |
| | `sentiment_intensity` | VADER compound score |
| | `affect_scores` | Valence, arousal, dominance (VAD space) |
| **Cohesion** | `cohesion_score` | Jaccard similarity across adjacent sentences |
| | `lexical_overlap` | Word overlap across adjacent sentences |
| | `connectives_usage` | Discourse connective rate |
| **Cognitive & Social** | `cognitive_processes` | Insight, causation, certainty, tentative rates |
| | `social_processes` | Social reference word count |
| | `readability_index` | Flesch Reading Ease (normalized) |
| | `graph_connectedness` / `semantic_coherence` | Word co-occurrence graph metrics |
| **Domain Lexicons** | `absolutist_word_frequency` | Absolutist language rate |
| | `death_word_frequency` | Death-related word rate |
| | `anxiety_word_frequency` | Anxiety lexicon rate |
| | `sadness_word_frequency` | Sadness lexicon rate |
| | `anger_word_frequency` | Anger lexicon rate |
| | `body_word_frequency` | Somatic/body lexicon rate |
| | `achievement_word_frequency` | Grandiosity lexicon rate |
| **Morphosyntactic & NER** | `pos_frequencies` | Preposition, auxiliary, adverb, conjunction rates |
| | `extended_pos` | Noun, verb, adjective, interjection rates |
| | `morphological_features` | Indicative/subjunctive mood, singular/plural ratios |
| | `dependency_features` | Subject and object rates per sentence |
| | `ner_features` | Person and temporal entity reference rates |
