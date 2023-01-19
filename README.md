# Awesome-LLM-Large-Language-Models-Notes
***

## Known LLM models classified by year
*Small introduction, paper, code etc.*
   
| Year | Name | Paper | Introduction | Implementation |
| :-: | :-: | :-: | :-: | :-: |
| 2017 | Transformer | [Attention is All you Need](https://arxiv.org/abs/1706.03762) | The focus of the original research was on translation tasks. | |
| 2018 | GPT | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | The first pretrained Transformer model, used for fine-tuning on various NLP tasks and obtained state-of-the-art results | | |
| 2018 | BERT | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | Another large pretrained model, this one designed to produce better summaries of sentences | [PyTorch](https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial) | |
| 2019 | GPT-2 | [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | An improved (and bigger) version of GPT that was not immediately publicly released due to ethical concerns | |
| 2019 | DistilBERT - Distilled BERT | [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) | A distilled version of BERT that is 60% faster, 40% lighter in memory, and still retains 97% of BERT’s performance | |
| 2019 | BART | [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) | Large pretrained models using the same architecture as the original Transformer model. | |
| 2019 | T5 | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) | Large pretrained models using the same architecture as the original Transformer model. | |
| 2019 | ALBERT | [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) |  |  |
| 2019 | RoBERTa - A Robustly Optimized BERT Pretraining Approach | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) | | |
| 2019 | CTRL | [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) | | |
| 2019 | Transformer XL | [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) | | |
| 2020 | GPT-3 | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | An even bigger version of GPT-2 that is able to perform well on a variety of tasks without the need for fine-tuning (called zero-shot learning) | | |
| 2020 | ELECTRA | [ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS](https://openreview.net/pdf?id=r1xMH1BtvB) | | |
| 2020 | mBART | [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) | | |
| 2022 | InstructGPT | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | We’ve trained language models that are much better at following user intentions than GPT-3  | :-: |
***

## Known LLM models classified by name
- [ALBERT](https://arxiv.org/abs/1909.11942)
- [BART](https://arxiv.org/abs/1910.13461) | [BERT](https://arxiv.org/abs/1810.04805) | Big Bird | BLOOM |
- Chinchilla | CLIP | [CTRL](https://arxiv.org/abs/1909.05858)
- DALL-E | DALL-E-2 | Decision Transformers | DialoGPT | [DistilBERT](https://arxiv.org/abs/1910.01108) | DQ-BART |
- [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) | ERNIE |
- Flamingo |
- Gato | Gopher |GLaM | GLIDE | GC-ViT | [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [GPT-3](https://arxiv.org/abs/2005.14165) | GPT-Neo | GPTInstruct |
- Imagen | [InstructGPT](https://arxiv.org/abs/2203.02155) 
- Jurassic-1
- LAMDA
- [mBART](https://arxiv.org/abs/2001.08210) | Megatron | Minerva | MT-NLG
- OPT
- Palm | Pegasus
- [RoBERTa](https://arxiv.org/abs/1907.11692)
- SeeKer | Swin Transformer | Switch
- [Transformer](https://arxiv.org/abs/1706.03762) | [T5](https://arxiv.org/abs/1910.10683) | Trajectory Transformers | [Transformer XL](https://arxiv.org/abs/1901.02860) | Turing-NLG
- ViT
- Wu Dao 2.0 |
- XLM-RoBERTa | XLNet
***

## Classification by architecture

| Architecture | Models | Tasks|
| :-: | :-: | :-: |
| Encoder-only, aka also called auto-encoding Transformer models| ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa | Sentence classification, named entity recognition, extractive question answering |
| Decoder-only, aka auto-regressive Transformer models | CTRL, GPT, GPT-2, Transformer XL | Text generation given a prompt|
| Encoder-Decoder, aka  sequence-to-sequence Transformer models| BART, T5, Marian, mBART | Summarization, translation, generative question answering |
***

## What so special about HuggingFace?
- HuggingFace, a popular NLP library, but it also offers an easy way to deploy models via their Inference API. When you build a model using the HuggingFace library, you can then train it and upload it to their Model Hub. Read more about this [here](https://huggingface.co/pricing).
- [List of notebook](https://huggingface.co/docs/transformers/notebooks)
***

## Must-Read Papers on LLM
- [2022 | A SURVEY ON GPT-3](https://arxiv.org/pdf/2212.00857.pdf)
- https://github.com/thunlp/PLMpapers
*** 

## Blogs
- [Building a synth with ChatGPT](https://jlongster.com/building-a-synth-with-chatgpt)
- [PubMed GPT: a Domain-Specific Large Language Model for Biomedical Text](https://www.mosaicml.com/blog/introducing-pubmed-gpt)
- [ChatGPT - Where it lacks](https://cookup.ai/chatgpt/where-it-lacks/)
- [Awesome ChatGPT Prompts](https://github.com/f/awesome-chatgpt-prompts)
- [ChatGPT vs. GPT3: The Ultimate Comparison](https://dzone.com/articles/chatgpt-vs-gpt3-the-ultimate-comparison-features)
- [Prompt Engineering 101: Introduction and resources](https://amatriain.net/blog/PromptEngineering)
- [Transformer models: an introduction and catalog — 2022 Edition](https://amatriain.net/blog/transformer-models-an-introduction-and-catalog-2d1e9039f376/#GATO)
- [Can GPT-3 or BERT Ever Understand Language?⁠—The Limits of Deep Learning Language Models](https://neptune.ai/blog/gpt-3-bert-limits-of-deep-learning-language-models)
- [10 Things You Need to Know About BERT and the Transformer Architecture That Are Reshaping the AI Landscape](https://neptune.ai/blog/bert-and-the-transformer-architecture)
- [Comprehensive Guide to Transformers](https://neptune.ai/blog/comprehensive-guide-to-transformers)
- [Unmasking BERT: The Key to Transformer Model Performance](https://neptune.ai/blog/unmasking-bert-transformer-model-performance)
- [Transformer NLP Models (Meena and LaMDA): Are They “Sentient” and What Does It Mean for Open-Domain Chatbots?](https://neptune.ai/blog/transformer-nlp-models-meena-lamda-chatbots)
- [Hugging Face Pre-trained Models: Find the Best One for Your Task](https://neptune.ai/blog/hugging-face-pre-trained-models-find-the-best)
***

## Start-up funding landscape
- [NLP Startup Funding in 2022](https://towardsdatascience.com/nlp-startup-funding-in-2022-caad77cb0f0)
***

## Available tutorials
- [Building a search engine with a pre-trained BERT model](https://github.com/kyaiooiayk/Awesome-LLM-Large-Language-Models-Notes/blob/main/tutorials/GitHub_MD_rendering/Building%20a%20search%20engine%20with%20a%20pre-trained%20BERT%20model.ipynb)
- [Fine tuning pre-trained BERT model on Text Classification Task](https://github.com/kyaiooiayk/Awesome-LLM-Large-Language-Models-Notes/blob/main/tutorials/GitHub_MD_rendering/Fine%20tuning%20pre-trained%20BERT%20model%20on%20Text%20Classification%20Task.ipynb)
- [Fine tuning pre-trained BERT model on the Amazon product review dataset](https://github.com/kyaiooiayk/Awesome-LLM-Large-Language-Models-Notes/blob/main/tutorials/GitHub_MD_rendering/Fine%20tuning%20pre-trained%20BERT%20model%20on%20the%20Amazon%20product%20review%20dataset.ipynb)
- [Sentiment analysis with Hugging Face transformer](https://github.com/kyaiooiayk/Awesome-LLM-Large-Language-Models-Notes/blob/main/tutorials/GitHub_MD_rendering/Sentiment%20analysis%20with%20Hugging%20Face%20transformer.ipynb)
- [Fine tuning pre-trained BERT model on YELP review Classification Task](https://github.com/kyaiooiayk/Awesome-LLM-Large-Language-Models-Notes/blob/main/tutorials/GitHub_MD_rendering/Fine%20tuning%20pre-trained%20BERT%20model%20on%20YELP%20review%20Classification%20Task.ipynb)
- [HuggingFace API](https://github.com/kyaiooiayk/Awesome-LLM-Large-Language-Models-Notes/blob/main/tutorials/GitHub_MD_rendering/HuggingFace%20API.ipynb)
- [HuggingFace mask filling](https://github.com/kyaiooiayk/Awesome-LLM-Large-Language-Models-Notes/blob/main/tutorials/GitHub_MD_rendering/HuggingFace%20mask%20filling.ipynb)
- [HuggingFace NER name entity recognition](https://github.com/kyaiooiayk/Awesome-LLM-Large-Language-Models-Notes/blob/main/tutorials/GitHub_MD_rendering/HuggingFace%20NER%20name%20entity%20recognition.ipynb)
- [HuggingFace question answering within context](https://github.com/kyaiooiayk/Awesome-LLM-Large-Language-Models-Notes/blob/main/tutorials/GitHub_MD_rendering/HuggingFace%20question%20answering%20within%20context.ipynb)
- [HuggingFace text generation](https://github.com/kyaiooiayk/Awesome-LLM-Large-Language-Models-Notes/blob/main/tutorials/GitHub_MD_rendering/HuggingFace%20text%20generation.ipynb)
- [HuggingFace text summarisation.ipynb](https://github.com/kyaiooiayk/Awesome-LLM-Large-Language-Models-Notes/blob/main/tutorials/GitHub_MD_rendering/HuggingFace%20text%20summarisation.ipynb)
- [HuggingFace zero-shot learning](https://github.com/kyaiooiayk/Awesome-LLM-Large-Language-Models-Notes/blob/main/tutorials/GitHub_MD_rendering/HuggingFace%20zero-shot%20learning.ipynb)
***

## A small note on the notebook rendering
- Two notebooks are available: 
    - One with coloured boxes and outside folder `GitHub_MD_rendering` 
    - One in black-and-white under folder `GitHub_MD_rendering`
***

## How to run the notebook in Google Colab
- The easiest option would be for you to clone this repository.
- Navigate to Google Colab and open the notebook directly from Colab.
- You can then also write it back to GitHub provided permission to Colab. The whole procedure is automated.
***

## Implementations from scratch
- [How to Code BERT Using PyTorch](https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial)
- [miniGPT in PyTorch](https://github.com/karpathy/minGPT)
- [nanoGPT in PyTorch](https://github.com/karpathy/nanoGPT)
***
