---
permalink: /sae/
title: "Interpret Features with Sparse Autoencoders"
author_profile: true
redirect_from: 
  - /md/
  - /sae.html
---

{% include toc %}


## SAE Interpreter - Interpret Features with a Sparse Autoencoder

The **SAE Interpreter** module provides interpretable latent feature analysis using a Sparse Autoencoder (SAE) trained on 710,000 Reddit posts spanning from casual conversation to mental health-focused communities. Given an input text, the model identifies the most strongly activated latent features, each of which is automatically described in natural language, capturing the psychological and semantic content expressed in the text.

```python
from TONY.SAE import SAEInterpreter

text = 'Some days I keep living, even though I feel completely alone in the world'
interpreter = SAEInterpreter()
result = interpreter.interpret(text, top_k=10)
interpreter.plot_interpretation(result)

```
