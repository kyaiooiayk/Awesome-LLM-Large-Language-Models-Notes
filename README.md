# Awesome-LLM-Large-Language-Models-Notes
***

## Known LLM models classified by year
*Small introduction, paper, code etc.*
   
| Year | Name | Paper | Info | Implementation |
| :-: | :-: | :-: | :-: | :-: |
| 2017 | Transformer | [Attention is All you Need](https://arxiv.org/abs/1706.03762) | The focus of the original research was on translation tasks. | [TensorFlow](https://github.com/edumunozsala/Transformer-NMT) + [article](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634) |
| 2018 | GPT | [Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | The first pretrained Transformer model, used for fine-tuning on various NLP tasks and obtained state-of-the-art results | | |
| 2018 | BERT | [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) | Another large pretrained model, this one designed to produce better summaries of sentences | [PyTorch](https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial) | |
| 2019 | GPT-2 | [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | An improved (and bigger) version of GPT that was not immediately publicly released due to ethical concerns | |
| 2019 | DistilBERT - Distilled BERT | [DistilBERT, a distilled version of BERT: smaller, faster, cheaper and lighter](https://arxiv.org/abs/1910.01108) | A distilled version of BERT that is 60% faster, 40% lighter in memory, and still retains 97% of BERT’s performance | |
| 2019 | BART | [BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) | Large pretrained models using the same architecture as the original Transformer model. | |
| 2019 | T5 | [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) | Large pretrained models using the same architecture as the original Transformer model. | |
| 2019 | ALBERT | [ALBERT: A Lite BERT for Self-supervised Learning of Language Representations](https://arxiv.org/abs/1909.11942) |  |  |
| 2019 | RoBERTa - A Robustly Optimized BERT Pretraining Approach | [RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) | | |
| 2019 | CTRL | [CTRL: A Conditional Transformer Language Model for Controllable Generation](https://arxiv.org/abs/1909.05858) | | |
| 2019 | Transformer XL | [Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context](https://arxiv.org/abs/1901.02860) | Adopts a recurrence methodology over past state coupled with relative positional encoding enabling longer term dependencies | |
| 2019 | Diablo GPT | [DialoGPT: Large-Scale Generative Pre-training for Conversational Response Generation](https://arxiv.org/abs/1911.00536) | Trained on 147M conversation-like exchanges extracted from Reddit comment chains over a period spanning from 2005 through 2017 | [PyTorch](https://github.com/microsoft/DialoGPT)|
| 2019 | ERNIE | [ERNIE: Enhanced Language Representation with Informative Entities](https://arxiv.org/abs/1905.07129) | In this paper, we utilize both large-scale textual corpora and KGs to train an enhanced language representation model (ERNIE), which can take full advantage of lexical, syntactic, and knowledge information simultaneously. | |
| 2020 | GPT-3 | [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) | An even bigger version of GPT-2 that is able to perform well on a variety of tasks without the need for fine-tuning (called zero-shot learning) | | |
| 2020 | ELECTRA | [ELECTRA: PRE-TRAINING TEXT ENCODERS AS DISCRIMINATORS RATHER THAN GENERATORS](https://openreview.net/pdf?id=r1xMH1BtvB) | | |
| 2020 | mBART | [Multilingual Denoising Pre-training for Neural Machine Translation](https://arxiv.org/abs/2001.08210) | | |
| 2021 | CLIP (Contrastive Language-Image Pre-Training) | [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | CLIP is a neural network trained on a variety of (image, text) pairs. It can be instructed in natural language to predict the most relevant text snippet, given an image, without directly optimizing for the task, similarly to the zero-shot capabilities of GPT-2 and 3. | [PyTorch](https://github.com/openai/CLIP) |
| 2021 | DALL-E | [Zero-Shot Text-to-Image Generation](https://arxiv.org/pdf/2102.12092v2.pdf) |  | [PyTorch](https://github.com/openai/DALL-E) |
| 2021 | Gopher | [Scaling Language Models: Methods, Analysis & Insights from Training Gopher](https://arxiv.org/abs/2112.11446) | | |
| 2021 | Decision Transformer  | [Decision Transformer: Reinforcement Learning via Sequence Modeling](https://arxiv.org/abs/2106.01345) | An architecture that casts the problem of RL as conditional sequence modeling. | [PyTorch](https://github.com/kzl/decision-transformer) |
| 2021 | GLam (Generalist Language Model) | [GLaM: Efficient Scaling of Language Models with Mixture-of-Experts](https://arxiv.org/abs/2112.06905) |  In this paper, we propose and develop a family of language models named GLaM (Generalist Language Model), which uses a sparsely activated mixture-of-experts architecture to scale the model capacity while also incurring substantially less training cost compared to dense variants. | |
| 2022 | chatGPT/InstructGPT | [Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) | This trained language model is much better at following user intentions than GPT-3. The model is optimised (fine tuned) using Reinforcement Learning with Human Feedback (RLHF) to achieve conversational dialogue. The model was trained using a variety of data which were written by people to achieve responses that sounded human-like.  | :-: |
| 2022 | Chinchilla | [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | Uses the same compute budget as Gopher but with 70B parameters and 4x more more data.| :-:|
| 2022 | LaMDA - Language Models for Dialog Applications | [LaMDA](https://arxiv.org/abs/2201.08239) | It is a family of Transformer-based neural language models specialized for dialog | |
| 2022 | DQ-BART | [DQ-BART: Efficient Sequence-to-Sequence Model via Joint Distillation and Quantization](https://arxiv.org/abs/2203.11239) | Propose to jointly distill and quantize the model, where knowledge is transferred from the full-precision teacher model to the quantized and distilled low-precision student model. | | |
| 2022 | Flamingo | [Flamingo: a Visual Language Model for Few-Shot Learning](https://arxiv.org/abs/2204.14198) | Building models that can be rapidly adapted to novel tasks using only a handful of annotated examples is an open challenge for multimodal machine learning research. We introduce Flamingo, a family of Visual Language Models (VLM) with this ability. | |
| 2022 | Gato | [A Generalist Agent](https://openreview.net/forum?id=1ikK0kHjvj) |  Inspired by progress in large-scale language modeling, we apply a similar approach towards building a single generalist agent beyond the realm of text outputs. The agent, which we refer to as Gato, works as a multi-modal, multi-task, multi-embodiment generalist policy. | |
| 2022 | GODEL: Large-Scale Pre-Training for Goal-Directed Dialog | [GODEL: Large-Scale Pre-Training for Goal-Directed Dialog](https://www.microsoft.com/en-us/research/uploads/prod/2022/05/2206.11309.pdf) | In contrast with earlier models such as DialoGPT, GODEL leverages a new phase of grounded pre-training designed to better support adapting GODEL to a wide range of downstream dialog tasks that require information external to the current conversation (e.g., a database or document) to produce good responses. | [PyTorch](https://github.com/microsoft/GODEL) |
| 2023 | GPT-4 | [GPT-4 Technical Report](https://cdn.openai.com/papers/gpt-4.pdf) | The model now accepts multimodal inputs: images and text | :-: |
| 2023 | BloombergGPT | [BloombergGPT: A Large Language Model for Finance](https://arxiv.org/abs/2303.17564) | LLM  specialised in financial domain trained on Bloomberg's extensive data sources | |
| 2023 | BLOOM | [BLOOM: A 176B-Parameter Open-Access Multilingual Language Model](https://arxiv.org/abs/2211.05100) | BLOOM (BigScience Large Open-science Open-access Multilingual Language Model) is a decoder-only Transformer language model that was trained on the ROOTS corpus, a dataset comprising hundreds of sources in 46 natural and 13 programming languages (59 in total) | |
| 2023 | Llama 2| [Llama 2: Open Foundation and Fine-Tuned Chat Models](https://scontent.flhr3-4.fna.fbcdn.net/v/t39.2365-6/10000000_663429262362723_1696968207443577320_n.pdf?_nc_cat=101&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=5ol-jUSglG4AX90dFHP&_nc_ht=scontent.flhr3-4.fna&oh=00_AfC_5ntBQWcl206BQGQDHKRjHsyQfhyuWXuj2zsdxJvr_Q&oe=64BDB0D1) | | [PyTorch #1](https://github.com/facebookresearch/llama) [PyTorch #2](https://blog.briankitano.com/llama-from-scratch/?utm_source=substack&utm_medium=email) |
| 2023 | Claude | [Claude](https://www.anthropic.com/product) | Claude can analyze 75k words (100k tokens). GPT4 can do just 32.7k tokens. | |
| 2023 | SelfCheckGPT | [SelfCheckGPT: Zero-Resource Black-Box Hallucination Detection for Generative Large Language Models](https://arxiv.org/abs/2303.08896) | A simple sampling-based approach that can be used to fact-check black-box models in a zero-resource fashion, i.e. without an external database. | |
***

## LLM models classified by size
|Name | Size (# Parameters) | Training Tokens | Training data|
| :-: | :-: | :-: | :-: |
| [GLaM](https://arxiv.org/abs/2112.06905) | 1.2T | | |
| [Gopher](https://arxiv.org/abs/2112.11446) | 280B | 300B | |
| [BLOOM](https://arxiv.org/abs/2211.05100) | 176B | | ROOTS corpus |
| [GPT-3](https://arxiv.org/abs/2005.14165) | 175B | |
| [LaMDA](https://arxiv.org/abs/2201.08239) | 137B | 168B | 1.56T words of public dialog data and web text |
| [Chinchilla](https://arxiv.org/abs/2203.15556) | 70B | 1.4T | |
| [Llama 2](https://scontent.flhr3-4.fna.fbcdn.net/v/t39.2365-6/10000000_663429262362723_1696968207443577320_n.pdf?_nc_cat=101&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=5ol-jUSglG4AX90dFHP&_nc_ht=scontent.flhr3-4.fna&oh=00_AfC_5ntBQWcl206BQGQDHKRjHsyQfhyuWXuj2zsdxJvr_Q&oe=64BDB0D1) | 7B, 13B and 70B | |
| [BloombergGPT](https://arxiv.org/abs/2303.17564) | 50B | 363B+345B |
| [Falcon40B](https://falconllm.tii.ae/) | 40B | 1T | 1,000B tokens of RefinedWeb |

- M=Million | B=billion | T=Trillion
***

## LLM models classified by name
- [ALBERT](https://arxiv.org/abs/1909.11942) | [Alpaca](https://github.com/tatsu-lab/stanford_alpaca)
- [BART](https://arxiv.org/abs/1910.13461) | [BERT](https://arxiv.org/abs/1810.04805) | [Big Bird](https://arxiv.org/abs/2007.14062) | [BLOOM](https://arxiv.org/abs/2211.05100) |
- [Chinchilla](https://arxiv.org/abs/2203.15556) | [CLIP](https://arxiv.org/abs/2103.00020) | [CTRL](https://arxiv.org/abs/1909.05858) | [chatGPT](https://arxiv.org/abs/2203.02155) | [Claude](https://www.anthropic.com/product) 
- [DALL-E](https://arxiv.org/pdf/2102.12092v2.pdf) | [DALL-E-2](https://arxiv.org/pdf/2204.06125.pdf) | [Decision Transformers](https://arxiv.org/abs/2106.01345) | [DialoGPT](https://arxiv.org/abs/1911.00536) | [DistilBERT](https://arxiv.org/abs/1910.01108) | [DQ-BART](https://arxiv.org/abs/2203.11239) |
- [ELECTRA](https://openreview.net/pdf?id=r1xMH1BtvB) | [ERNIE](https://arxiv.org/abs/1905.07129)
- [Flamingo](https://arxiv.org/abs/2204.14198) | [Falcon40B](https://falconllm.tii.ae/)
- [Gato](https://openreview.net/forum?id=1ikK0kHjvj) | [Gopher](https://arxiv.org/abs/2112.11446) | [GLaM](https://arxiv.org/abs/2112.06905) | [GLIDE](https://arxiv.org/pdf/2112.10741.pdf) | [GPT](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) | [GPT-3](https://arxiv.org/abs/2005.14165) | [GPT-4](https://cdn.openai.com/papers/gpt-4.pdf) | [GPT-Neo](https://huggingface.co/togethercomputer/GPT-NeoXT-Chat-Base-20B) | [Godel](https://www.microsoft.com/en-us/research/uploads/prod/2022/05/2206.11309.pdf) | [GPT-J](https://github.com/kingoflolz/mesh-transformer-jax)
- Imagen | [InstructGPT](https://arxiv.org/abs/2203.02155) 
- Jurassic-1
- [LaMDA](https://arxiv.org/abs/2201.08239) | [Llama 2](https://scontent.flhr3-4.fna.fbcdn.net/v/t39.2365-6/10000000_663429262362723_1696968207443577320_n.pdf?_nc_cat=101&ccb=1-7&_nc_sid=3c67a6&_nc_ohc=5ol-jUSglG4AX90dFHP&_nc_ht=scontent.flhr3-4.fna&oh=00_AfC_5ntBQWcl206BQGQDHKRjHsyQfhyuWXuj2zsdxJvr_Q&oe=64BDB0D1)
- [mBART](https://arxiv.org/abs/2001.08210) | Megatron | Minerva | MT-NLG
- OPT
- Palm | Pegasus
- [RoBERTa](https://arxiv.org/abs/1907.11692)
- SeeKer | Swin Transformer | Switch | [SelfCheckGPT](https://arxiv.org/abs/2303.08896)
- [Transformer](https://arxiv.org/abs/1706.03762) | [T5](https://arxiv.org/abs/1910.10683) | Trajectory Transformers | [Transformer XL](https://arxiv.org/abs/1901.02860) | Turing-NLG
- ViT
- Wu Dao 2.0 |
- XLM-RoBERTa | XLNet
***

## Classification by architecture

| Architecture | Models | Tasks|
| :-: | :-: | :-: |
| Encoder-only, aka also called auto-encoding Transformer models| ALBERT, BERT, DistilBERT, ELECTRA, RoBERTa | Sentence classification, named entity recognition, extractive question answering |
| Decoder-only, aka auto-regressive (or causal) Transformer models | CTRL, GPT, GPT-2, Transformer XL | Text generation given a prompt|
| Encoder-Decoder, aka  sequence-to-sequence Transformer models| BART, T5, Marian, mBART | Summarisation, translation, generative question answering |
***

## What's so special about HuggingFace?
- HuggingFace, a popular NLP library, but it also offers an easy way to deploy models via their Inference API. When you build a model using the HuggingFace library, you can then train it and upload it to their Model Hub. Read more about this [here](https://huggingface.co/pricing).
- [List of notebook](https://huggingface.co/docs/transformers/notebooks)
***

## Must-Read Papers on LLM
- [2014 | Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)
- [2022 | A SURVEY ON GPT-3](https://arxiv.org/pdf/2212.00857.pdf)
- [2022 | Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102)
- [Must-Read Papers on Pre-trained Language Models (PLMs)](https://github.com/thunlp/PLMpapers)
*** 

##  🚀Recap | Bring me up to speed!
- [Catching up on the weird world of LLMs](https://simonwillison.net/2023/Aug/3/weird-world-of-llms/?utm_source=substack&utm_medium=email)
***

## Blog articles
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
- [Large Transformer Model Inference Optimization](https://lilianweng.github.io/posts/2023-01-10-inference-optimization/)
- 4-part tutorial on how transformers work: 
[Part 1](https://towardsdatascience.com/transformers-explained-visually-part-1-overview-of-functionality-95a6dd460452) |
[Part 2](https://towardsdatascience.com/transformers-explained-visually-part-2-how-it-works-step-by-step-b49fa4a64f34) |
[Part 3](https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853) |
[Part 4](https://towardsdatascience.com/transformers-explained-visually-not-just-how-but-why-they-work-so-well-d840bd61a9d3)
- [What Makes a Dialog Agent Useful?](https://huggingface.co/blog/dialog-agents?utm_source=substack&utm_medium=email)
- [Understanding Large Language Models -- A Transformative Reading List](https://sebastianraschka.com/blog/2023/llm-reading-list.html?utm_source=substack&utm_medium=email)
- [Prompt Engineering](https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/?utm_source=substack&utm_medium=email)
- [Building LLM applications for production](https://huyenchip.com/2023/04/11/llm-engineering.html?utm_source=substack&utm_medium=email)
- [Developer's Guide To LLMOps: Prompt Engineering, LLM Agents, and Observability](https://arize.com/blog-course/llmops-operationalizing-llms-at-scale/)
- [Argument for using RL LLMs](https://gist.github.com/yoavg/6bff0fecd65950898eba1bb321cfbd81?utm_source=substack&utm_medium=email)
- [Why Google and OpenAI are loosing against the open-source communities](https://www.semianalysis.com/p/google-we-have-no-moat-and-neither?utm_source=substack&utm_medium=email)
- [You probably don't know how to do Prompt Engineering!](https://gist.github.com/Hellisotherpeople/45c619ee22aac6865ca4bb328eb58faf?utm_source=substack&utm_medium=email)
- [The Full Story of Large Language Models and RLHF](https://www.assemblyai.com/blog/the-full-story-of-large-language-models-and-rlhf/?utm_source=substack&utm_medium=email)
- [Understanding OpenAI's Evals](https://arize.com/blog-course/evals-openai-simplifying-llm-evaluation/)
- [What We Know About LLMs  (Primer)](https://willthompson.name/what-we-know-about-llms-primer?utm_source=substack&utm_medium=email)
- [F**k You, Show Me The Prompt.](https://hamel.dev/blog/posts/prompt/?utm_source=substack&utm_medium=email)
- [How to avoid LLM Security risks (Fiddler AI)](https://www.fiddler.ai/blog/how-to-avoid-llm-security-risks)
- [Developing Agentic Workflows With Safety And Accuracy (Fiddler AI)](https://www.fiddler.ai/blog/developing-agentic-ai-workflows-with-safety-and-accuracy)
***

## Know their limitations!
- [ChatGPT and Wolfram|Alpha](https://writings.stephenwolfram.com/2023/01/wolframalpha-as-the-way-to-bring-computational-knowledge-superpowers-to-chatgpt/)
- [Numbers every LLM Developer should know](https://github.com/ray-project/llm-numbers?utm_source=substack&utm_medium=email)
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
- You can then also write it back to GitHub provided permission to Colab is granted. The whole procedure is automated.
***

## Implementations from scratch
- [How to Code BERT Using PyTorch](https://neptune.ai/blog/how-to-code-bert-using-pytorch-tutorial)
- [miniGPT in PyTorch](https://github.com/karpathy/minGPT)
- [nanoGPT in PyTorch](https://github.com/karpathy/nanoGPT)
- [TensorFlow implementation of Attention is all you need](https://github.com/edumunozsala/Transformer-NMT) + [article](https://towardsdatascience.com/attention-is-all-you-need-discovering-the-transformer-paper-73e5ff5e0634) 
***
