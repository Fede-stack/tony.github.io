---
permalink: /bdi/
title: "Filling BDI-II"
author_profile: true
redirect_from: 
  - /md/
  - /bdi.html
---

{% include toc %}

## Filling BDI-II Questionnaire

The BDI-II Scorer module automatically completes the Beck Depression Inventory II (BDI-II) from a user's post history. Given a collection of Reddit posts written by a single user, the module produces a 21-dimensional vector of predicted item scores — one for each BDI-II item — without requiring any manual annotation.
How it works
The module implements the adaptive Retrieval-Augmented Generation (aRAG) pipeline introduced by Ravenda et al. The process unfolds in two stages for each of the 21 BDI-II items:
1. Adaptive retrieval: the most semantically relevant posts are retrieved from the user's history using a mental-health-specialized embedding model (FritzStack/mpnet_MH_embedding). Unlike standard RAG, the number of retrieved posts is not fixed — it adapts to the semantic density of the user's history relative to each item, retrieving more evidence when the signal is rich and fewer posts when it is sparse.
2. Generative scoring: the retrieved posts are passed to a generative LLM together with the four response options for the current BDI-II item. The model selects the option that best reflects the user's expressed state, returning a score from 0 to 3.

**Parameters**

| Parameter | Type | Description |
|---|---|---|
| `retriever_model_name` | `str` | HuggingFace model ID for the embedding retriever |
| `llm_model_name` | `str` | HuggingFace model ID or API model name for the generative LLM |
| `use_hf` | `bool` | If `True`, runs the LLM locally via HuggingFace; if `False`, uses an external API client |
| `client` | `object` | API client instance (e.g. OpenAI, Together, Gemini) — required when `use_hf=False` |

**Input format**

scorer.score() expects:
* reddit_posts: a list of lists, where each inner list contains all posts written by a single user
* bdi_items: a list of 21 items, each containing the 4 BDI-II response options (score 0–3)
* items_names: a list of 21 item label strings


```python
from TONY.BDI import BDIScorer

bdi_items = [
    # 1. Sadness
    [
        "I do not feel sad.",
        "I feel sad much of the time.",
        "I am sad all the time.",
        "I am so sad or unhappy that I can't stand it."
    ],
    # 2. Pessimism
    [
        "I am not discouraged about my future.",
        "I feel more discouraged about my future than I used to be.",
        "I do not expect things to work out for me.",
        "I feel my future is hopeless and will only get worse."
    ],
    # 3. Past Failure
    [
        "I do not feel like a failure.",
        "I have failed more than I should have.",
        "As I look back, I see a lot of failures.",
        "I feel I am a total failure as a person."
    ],
    # 4. Loss of Pleasure
    [
        "I get as much pleasure as I ever did from the things I enjoy.",
        "I don't enjoy things as much as I used to.",
        "I get very little pleasure from the things I used to enjoy.",
        "I can't get any pleasure from the things I used to enjoy."
    ],
    # 5. Guilty Feelings
    [
        "I don't feel particularly guilty.",
        "I feel guilty over many things I have done or should have done.",
        "I feel quite guilty most of the time.",
        "I feel guilty all of the time."
    ],
    # 6. Punishment Feelings
    [
        "I don't feel I am being punished.",
        "I feel I may be punished.",
        "I expect to be punished.",
        "I feel I am being punished."
    ],
    # 7. Self-Dislike
    [
        "I feel the same about myself as ever.",
        "I have lost confidence in myself.",
        "I am disappointed in myself.",
        "I dislike myself."
    ],
    # 8. Self-Criticalness
    [
        "I don't criticize or blame myself more than usual.",
        "I am more critical of myself than I used to be.",
        "I criticize myself for all of my faults.",
        "I blame myself for everything bad that happens."
    ],
    # 9. Suicidal Thoughts or Wishes
    [
        "I don't have any thoughts of killing myself.",
        "I have thoughts of killing myself, but I would not carry them out.",
        "I would like to kill myself.",
        "I would kill myself if I had the chance."
    ],
    # 10. Crying
    [
        "I don't cry anymore than I used to.",
        "I cry more than I used to.",
        "I cry over every little thing.",
        "I feel like crying, but I can't."
    ],
    # 11. Agitation
    [
        "I am no more restless or wound up than usual.",
        "I feel more restless or wound up than usual.",
        "I am so restless or agitated that it's hard to stay still.",
        "I am so restless or agitated that I have to keep moving or doing something."
    ],
    # 12. Loss of Interest
    [
        "I have not lost interest in other people or activities.",
        "I am less interested in other people or things than before.",
        "I have lost most of my interest in other people or things.",
        "It's hard to get interested in anything."
    ],
    # 13. Indecisiveness
    [
        "I make decisions about as well as ever.",
        "I find it more difficult to make decisions than usual.",
        "I have much greater difficulty in making decisions than I used to.",
        "I have trouble making any decisions."
    ],
    # 14. Worthlessness
    [
        "I do not feel I am worthless.",
        "I don't consider myself as worthwhile and useful as I used to.",
        "I feel more worthless as compared to other people.",
        "I feel utterly worthless."
    ],
    # 15. Loss of Energy
    [
        "I have as much energy as ever.",
        "I have less energy than I used to have.",
        "I don't have enough energy to do very much.",
        "I don't have enough energy to do anything."
    ],
    # 16. Changes in Sleeping Pattern
    [
        "I have not experienced any change in my sleeping pattern.",
        "I sleep somewhat more than usual OR I sleep somewhat less than usual.",
        "I sleep a lot more than usual OR I sleep a lot less than usual.",
        "I sleep most of the day OR I wake up 1-2 hours early and can't get back to sleep."
    ],
    # 17. Irritability
    [
        "I am no more irritable than usual.",
        "I am more irritable than usual.",
        "I am much more irritable than usual.",
        "I am irritable all the time."
    ],
    # 18. Changes in Appetite
    [
        "I have not experienced any change in my appetite.",
        "My appetite is somewhat less than usual OR My appetite is somewhat greater than usual.",
        "My appetite is much less than before OR My appetite is much greater than usual.",
        "I have no appetite at all OR I crave food all the time."
    ],
    # 19. Concentration Difficulty
    [
        "I can concentrate as well as ever.",
        "I can't concentrate as well as usual.",
        "It's hard to keep my mind on anything for very long.",
        "I find I can't concentrate on anything."
    ],
    # 20. Tiredness or Fatigue
    [
        "I am no more tired or fatigued than usual.",
        "I get more tired or fatigued more easily than usual.",
        "I am too tired or fatigued to do a lot of the things I used to do.",
        "I am too tired or fatigued to do most of the things I used to do."
    ],
    # 21. Loss of Interest in Sex
    [
        "I have not noticed any recent change in my interest in sex.",
        "I am less interested in sex than I used to be.",
        "I am much less interested in sex now.",
        "I have lost interest in sex completely."
    ]
]

items_names = ['Sadness', 'Pessimism', 'Past Failure', 'Loss of Pleasure', 'Guilty Feelings', 'Punishment Feelings', 
              'Self-Dislike', 'Self-Criticalness', 'Suicidal Thoughts or Wishes', 'Crying', 'Agitation', 
              'Loss of Interest', 'Indecisiveness', 'Worthlessness', 'Loss of Energy', 'Changes in Sleeping Pattern', 
              'Irritability', 'Changes in Appetite', 'Concentration Difficulty', 'Tiredness or Fatigue', 
              'Loss of Interest in Sex']


posts = [['I have been feeling empty for weeks', 'I can barely get out of bed', ...]]  # Each inner list contains all Reddit posts written by a single user
scorer = BDIScorer(
    retriever_model_name='FritzStack/mpnet_MH_embedding',
    llm_model_name='google/gemma-3-27b-it',
    use_hf=False,
    client=client,
)
response_llms = scorer.score(reddit_posts, bdi_items, items_names)

# Output: 21-dimensional vector of predicted BDI-II item scores
```


```bibtex
@inproceedings{ravenda2025llms,
  title={Are LLMs effective psychological assessors? Leveraging adaptive RAG for interpretable mental health screening through psychometric practice},
  author={Ravenda, Federico and Bahrainian, Seyed Ali and Raballo, Andrea and Mira, Antonietta and Kando, Noriko},
  booktitle={Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={8975--8991},
  year={2025}
}
```
