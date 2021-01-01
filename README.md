# ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic
<img src="ARBERT_MARBERT.jpg" alt="drawing" width="30%" height="30%" align="right"/>

# What is the repository is about?
This is the repository accompanying our paper [ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic](https://mageed.arts.ubc.ca/files/2020/12/marbert_arxiv_2020.pdf).
In the paper, we:
* introduce ```ARBERT``` and ```MARBERT```, two powerful Transformer-based language models for Arabic;
* introduce ```ArBench```, a multi-domain, multi-variety benchmark for Arabic naturaal language understanding based on 41 datasets across 5 different tasks and task clusters;
* evaluate ARBERT and MARBERT on ArBench and compare against available language models.

Our models establish new state-of-the-art (SOTA) on all 5 tasks/task clusters on 37 out of the 41 datasets.
Our language models are publicaly available for research (see below).
The rest of this repository provides more information about our new language models, benchmark, and experiments.

---

## Table of Contents
- [1 Our Language Models](#1-Our-Language-Models)
  - [1.1 ARBERT & MARBERT](#11-arbert--marbert)
  - [1.2 Training Data and Vocabulary](#12-training-data-and-vocabulary)
- [2. Our Benchmark: ArBench](#2-our-benchmark-arbench)
  - [2.1 Sentiment Analysis](#21-sentiment-analysis)
  - [2.2 Social Meaning](#22-social-meaning)
  - [2.3 Topic Classification](#23-topic-classification)
  - [2.4 Dialect Identification](#24-dialect-identification)
  - [2.5 Named Entity Recogntion (NER)](#25-named-entity-recogntion)
- [3. Model Evaluation](#3-model-evaluation)
  - [3.1 Sentiment Analysis](#31-sentiment-analysis)
  - [3.2 Social Meaning](#32-social-meaning)
  - [3.3 Topic Classification](#33-topic-classification)
  - [3.4 Dialect Identification](#34-Dialect-Identification)
  - [3.5 Named Entity Recogntion (NER)](#35-named-entity-recogntion)
- [4. 4. How to use ARBERT and MARBERT](#4-how-to-use-arbert-and-marbert)
- [5. Ethics](#5-ethics)
- [6. Download ARBERT and MARBERT](#6-download-arbert-and-marbert)
- [7. Citation](#7-citation)
- [8. Acknowledgments](#8-acknowledgments)
---

## 1. Our Language Models

## 1.1 ARBERT & MARBERT
**ARBERT** is a large scale pre-training masked language model focused on Modern Standard Arabic (MSA). To train ARBERT, we use the same architecture as BERT-base: 12 attention layers, each has 12 attention heads and 768 hidden dimensions, a vocabulary of 100K WordPieces, making ∼163M parameters. We  train  ARBERT on a collection of Arabic datasets comprising 61GB of text (6.2B tokens)

**MARBERT** is a large scale pre-training masked language model focused on both Dialectal Arabic (DA) and MSA. Arabic has multiple varieties. To train MARBERT, we randomly sample 1B Arabic tweets from a large in-house dataset of about 6B tweets. We only include tweets with at least 3 Arabic words, based on character string matching, regardless whether the tweet has non-Arabic string or not. That is, we do not remove non-Arabic so long as the tweet meets the 3 Arabic word criterion. The dataset makes up 128GB of text (15.6B tokens). We use the same network architecture as ARBERT (BERT-base), but without the next sentence prediction (NSP) objective since tweets are short. See our [repo](https://github.com/UBC-NLP/LMBERT) for modifying BERT code to remove NSP.

## 1.2 Training Data and Vocabulary

The following table shows a comparison between ARBERT and mBERT, on the one hand, and XLM-R, AraBERT, and MARBERT on the other hand. We compare in terms of pre-training data sources and size, vocabulary size, and model parameter size.

|             | **Data Source**              | **#Tokens(ar/all)** | **Tokanization**  | **Vocab Size(ar/all)** | **Cased** | **Arch.**         | **#Param** |
|---------|---------------------|----------------|---------------|--------------|-------|---------------|--------|
| mBERT   | Wikipedia           | 153M/1.5B      | WordPiece     | 5K/110K      | yes   | base          | 110M   |
| XLM-R<sub>B</sub> | CommonCraw          | l2.9B/295B     | SentencePiece | 14K/250K     | yes   | base          | 270M   |
| XLM-R<sub>L</sub>  | CommonCraw          | l2.9B/295B     | SentencePiece | 14K/250K     | yes   | large         | 550M   |
| AraBERT | Several (3 sources) | 2.5B/2.5B      | SentencePiece | 60K/64K      | no    | base          | 135M   |
| **ARBERT**  | Several (6 sources) | 6.2B/6.2B      | WordPiece     | 100K/100K    | no    | base          | 163M   |
| **MARBERT** | Arabic Twitter      | 15.6B/15.6B    | WordPiece     | 100K/100K    | no    | base          | 163M   |

---

## 2. Our Benchmark: ArBench
To  evaluate  our  models, we  also introduce  **ArBench**,   a new benchmark for multi-dialectal Arabic language understanding.  ***ArBench is built using 41 datasets targeting 5 different tasks/task clusters***, allowing us to offer a series of standardized experiments under rich conditions. The following are the different tasks/task clusers covered by ArBench:

### 2.1 Sentiment Analysis

|**Reference**| **Data  (#classes)**     | **TRAIN**   | **DEV**    | **TEST**   |
|---------|--------|--------|-------|------|
|Alomari et al. (2017)|AJGT (2)      |   1.4K | -      |    361 | 
|Abdul-Mageed et al. (2020b) |AraNET<sub>Sent</sub> (2)      | 100K | 14.3K | 11.8K |
|Al-Twairesh et al. (2017)|AraSenTi (3)          |  11,117 |  1,407 |  1,382 | 
|Al-Twairesh et al. (2017)|ArSarcasm<sub>Sent</sub> (3)   |   8.4K | -      |  2.K | 
|Elmadany et al. (2018)|ArSAS (3)                           |  24.7K | -      |  3.6K | 
|Baly et al. (2019)|ArsenTD-LEV (5)                     |   3.2K | -      |    801 | 
|[Nabil et al. (2015)](https://www.aclweb.org/anthology/D15-1299)|ASTD (3)                            |  24.7K | -      |    664 | 
|[Nabil et al. (2015)](https://www.aclweb.org/anthology/D15-1299)|ASTD-B(2)                           |   1.06K | --     |    267 | 
|[AbdulMageed and Diab, (2012)](https://www.aclweb.org/anthology/L12-1630/)|AWATIF(4)                           |   2.28K |    288 |    284 | 
|[Salameh et al. (2015)](https://www.aclweb.org/anthology/N15-1078)|BBN(3)                              |     960 |    125 |    116 | 
|[Aly and Atiya (2013) ](https://www.springerprofessional.de/en/hotel-arabic-reviews-dataset-construction-for-sentiment-analysis/15234334)|HARD (2)                            |  84.5K | -      | 21.1K | 
|[Nabil et al. (2015)](http://www.aclweb.org/anthology/P/P13/P13-2088.pdf)|LABR (2)                            |  13.1K |        |  3.28K | 
|[AbdulMageed and Diab, (2014)](https://cl.indiana.edu/~skuebler/papers/wassa12.pdf)|SAMAR(5)                            |   2.49K |    310 |    316 | 
||SemEval (3)                         |  24.7K | -      |  6.10K | 
||SYTS(3)                             |     960 |    202 |    199 | 
||Twitter<sub>Saad</sub> (2) |   1.5K |    202 |    190 | |
||Twitter<sub>Abdullah</sub> (2)     |  46k |  5.77k |  5.82k | 

### 2.2 Social Meaning

|**Reference**| **Data  (#classes)**                    | **TRAIN**   | **DEV**    | **TEST**   | 
|-------------------------------------|---------|--------|--------|-------|
|| Arab_Tweet - Age (3)     | 1.28M | 160K | 160K | 
|| Arab_Tweet - Gender (2)   | 1.28M | 160K | 160K |
|| AraNET<sub>Emo</sub> - Emotion (8)   |  189K |    911 |    942 | 
|| AraSarcasm  - Sarcasm (2)   |   8.4K | -      |  2.1K | 
|| Dangerous(2)  |    3.4K |    616 |    664 | 
|| FIRE2019 - Irony (2)    |    3.6K | -      |    404 | 
|| OSACT-A - Offensive (2)  |   10K |   1K |   2K | 
|| OSACT-B - Hate Speech(2) |   10K |   1K |   2K | 


### 2.3 Topic Classification


|   **Reference**                 | **Data  (#classes)**                    | **TRAIN**   | **DEV**    | **TEST**   |
|-------------------------------------|---------|--------|--------|-------|
|                  |  OSAC (10)  | 17.9K | 2.24K | 2.24K | 
|                   | Khallej (4) |  4.55K |  570 |  570 | 
|                   |  ANT(5)  | 25.2K | 31.5K | 31.5K |  


### 2.4 Dialect Identification

|**Reference**| **Data  (#classes)**                    | **TRAIN**   | **DEV**    | **TEST**   |
|-----------|--------------------------|---------|--------|--------|
||              AOC (2)              |      Binary     |  86.5K | 10.8K | 10.8K |
||              AOC (3)              |      Region     |  35.7K |  4.46K |  4.45K |
||              AOC (4)              |      Region     |  86.5K | 10.8K | 10.8K |
|| ArSarcasm<sub>Dia</sub> (5) |      Regoin     |  8.43K |    -   |  2.11K |
||           MADAR-TL (21)           |     Country     | 193K | 26.6K | 43.9K |
||             NADI (21)             |     Country     |  2.1K  |  4.96K |  5K |
||             NADI (100)             |     Province     |  2.1K  |  4.96K |  5K |
||             QADI (18)             |     Country     | 498K |   --   |  3.5K |


### 2.5 Named Entity Recogntion

|**Reference**|  **Dataset**| **#Tokens** | **#PER**  | **#LOC**  | **#ORG**  |
|---------|-----------|-------------|-----------|------------|----------|
|| ANERCorp   | 150K    | 6.50K | 5.01K | 3.43K |
|| ACE-2003BN | 15K     | 832   | 1.22K | 288   |
|| ACE-2003NW | 27K     | 1.14K | 1.14K | 893   |
|| ACE-2004BN | 70K     | 3.20K | 3.92K | 2.23K |
|| TW-NER     | 81K     | 1.25K | 1.30K | 765   |
---

## 3 Model Evaluation
When fine-tuned on ArBench,  ARBERT and MARBERT collectively achieve new SOTA  with sizeable margins compared to all existing models such as mBERT, XLM-R (Base and Large), and  AraBERT on 37 out of 45 classification tasks on the 41 datasets (82.22%). We present our results on the different TEST sets in the subsections below. For performance on DEV sets, please see appendixes in our [paper](https://mageed.arts.ubc.ca/files/2020/12/marbert_arxiv_2020.pdf). 

### 3.1 Sentiment Analysis

| **Dataset (#classes)** |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|--------------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|
| AJGT (2)           | 86.67 |   89.44   |    91.94   |  92.22 | 94.44 |  **96.11** |
| HARD (2)           |95.54 |   95.74   |    95.96   |  95.89 | 96.12 |  **96.17** |
| ArsenTD-LEV (5)    | 50.50 |   55.25   |    **62.00**  |  56.13 | 61.38 |  60.38 |
| LABR (2)           | 91.20 |   91.23   |    92.20   |  91.97 | 92.51 |  **92.49** |
| ASTD-B(2)          |  79.32 |   87.59   |    77.44   |  83.08 | 93.23 |  **96.24** |

***Results reported based on Acc. score***

| **Dataset (#classes)** |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|--------------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|
| ArSAS (3)           | 87.50 | 90.00 | 91.50 | 91.00 | 92.00 | **93.00**|
| ASTD (3)            | 67.00 | 60.67 | 67.67 | 72.00 | 76.50 | **78.00** |
| SemEval (3)          | 57.00 | 64.00 | 67.00 | 62.00 | 69.00 | **71.00** |
| AraNET<sub>Sent</sub> (2)       | 84.00 | 92.00 | 93.00 | 86.50 | 89.00 | **92.00** |
| ArSarcasm<sub>Sent</sub> (3)      | 60.50 | 63.50 | 70.00 | 63.50 | 68.00 | **71.50** |
| AraSenTi (noura) (3) | 89.50 | **92.00** | 93.50 | 91.00 | 90.00 | 90.00 |
| BBN(3)                     | 55.50 | 69.50 | 46.50 | 70.00 | 76.50 | **79.00** |
| SYTS(3)                    | 67.00 | 78.00 | 40.50 | 75.50 | **79.00** | 76.50 |
| Twitter<sub>Saad</sub> (2)               | 79.00 | 95.00 | 95.00 | 81.00 | 90.00 | **96.00** |
| SAMAR(5)                   | 22.50 | 54.00 | **57.00** | 36.50 | 43.50 | 55.50 |
| AWATIF(4)                  | 60.50 | 63.50 | 68.50 | 66.50 | 71.50 | **72.50** |
| Twitter<sub>Abdullah</sub> (2)           | 81.50 | 91.00 | 92.00 | 89.50 | 91.50 | **95.00** |

***Results reported based on F<sub>1</sub><sup>NP</sup> score.***
### 3.2 Social Meaning
| **Task (#classes)**        |   **Dataset**  |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|----------------|---------------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|
| Offensive (2)  | OSACT-A | 84.25 |   85.26  |    88.28  |  86.57 | 90.38 |  **92.41** |
| Hate Speech(2) | OSACT-B | 72.81 |   71.33  |    79.31   |  78.89 | 83.02 |  **84.79**|
| Dangerous (2)  |       Dangerous       | 62.66 |   62.76  |    65.01 |  64.37 | 63.21 |  **67.53** |
| Sarcasm (2)    |       AraSarcasm      | 68.20 |   66.76  |    69.23  |  72.23 | 75.04 |  **76.30** |
| Emotion (8)    |       AraNET<sub>Emo</sub>     | 65.79 |   70.67   |    74.89   |  65.68 | 67.73 |  **75.83** |
| Irony (2)      |     FIRE2019     | 80.96 |   81.97  |    82.52%  |  83.01 | 85.59 |  **85.33** |
| Age (3)        |       Arab_Tweet      | 56.35 |   59.73   |    53.60   |  57.72 | 58.95 |  **62.27** |
| Gender (2)     |       Arab_Tweet      | 68.06|   71.00   |    71.14   |  67.75 | 69.86 |  **72.62** |

***Results reported based on F<sub>1</sub> score.***

### 3.3 Topic Classification
| **Dataset (#classes)**  |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|----------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|
| OSAC (10)                    | 96.84 | 97.15 | 98.20 | 97.03 | **97.50** | 97.23 |
| Khallej (4)                  | 92.81 | 91.87 | 93.56 | 93.83 | **94.53** | 93.63 |
| ANT<sub>Text</sub> (5)       | 84.89 | 85.77 | 86.72 | **88.17** | 86.87 | 85.27 |
| ANT<sub>Title</sub> (5)      | 78.29 | 79.96 | 81.25 | 81.03 | **81.70** | 81.19 |
| ANT<sub>Text+Title</sub> (5) | 84.67 | 86.21 | 86.96 | **87.22** | 87.21 | 85.60 |

***Results reported based on F<sub>1</sub> score.***
### 3.4 Dialect Identification
| **Task  (#classes)**        |   **Dataset**  |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|----------------|---------------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|
| Regoin (5)      | ArSarcasm<sub>Dia</sub> | 43.81 | 41.71 | 41.83 | 47.54 |  51.27 | **54.70**|
| Country (21)    |  MADAR-TL | 34.92 | 35.91 | 35.14 | 34.87 | 37.90 |**40.40** |
| Region (4)      |       AOC      | 77.27 | 77.34 | 78.77 | 79.20 | 81.09 | **82.37**|
| Region (3)      |       AOC      | 85.76 | 86.39 | 87.56 | 87.68 | 89.06 | **90.85** |
| Binary (4)            |       AOC      | 86.19 | 86.85 | 87.30 | 87.76 | 88.46 | **88.59** |
| Country(18)     |      QADI      | 66.57 | 77.00 | 82.73 | 72.23 | 88.63 | **90.89** |
| Country(21)     |      NADI      | 13.32 | 16.36 | 17.17 | 17.46 | 22.56 | **29.14** |
| Province (100 ) |      NADI      |  2.13 |  4.12 |  0.32 |  3.13 |  6.10 |  **6.28** |

***Results reported based on F<sub>1</sub> score.***

### 3.5 Named Entity Recogntion
| **Dataset**  |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|----------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|
| ANERcorp.   | 86.78 | 87.24 | **89.94** | 89.13 | 84.38 | 80.64 |
| ACE 2004 NW | 86.37 | **89.93** | 89.89 | 89.03 | 88.24 | 85.02 |
| ACE 2003BN  | 91.23 | 53.97 | 85.41 | 91.94 | **96.18** | 79.05 |
| ACE 2003NW  | 81.40| 87.24 | 90.62 | 88.09 | **90.09** | 87.76 |
| TW-NER     | 36.83 | 49.16 | 54.44 | 41.26 | 59.17 | **67.39** |

***Results reported based on F<sub>1</sub> score.***

---

## 4. How to use ARBERT and MARBERT

You can use our models by installing torch or tensorflow and Huggingface library transformers. And you can use it directly by initializing it like this:
 
 ```python
    from transformers import AutoTokenizer, AutoModel
    #load AEBERT model from huggingface
    ARBERT_tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/ARBERT")
    ARBERT_model = AutoModel.from_pretrained("UBC-NLP/ARBERT")

    #load MAEBERT model from huggingface
    MARBERT_tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")
    MARBERT_model = AutoModel.from_pretrained("UBC-NLP/MARBERT")
 ```

---

## 5. Ethics

Our models are developed using data from the public domain. 
We provide access to our models to accelerate scientific research with no liability on our part.
Please use our models and benchmark only ethically.
This includes, for example, respect and protection of people's privacy.
We encourage all researchers who decide to use our models to adhere to the highest standards.
For example, if you apply our models on Twitter data, we encourage you to review Twitter policy at [Twitter policy](https://developer.twitter.com/en/developer-terms/agreement-and-policy). For example, Twitter provides the following policy around use of [sensitive information](https://developer.twitter.com/en/developer-terms/more-on-restricted-use-cases): 

### Sensitive information

You should be careful about using Twitter data to derive or infer potentially sensitive characteristics about Twitter users. Never derive or infer, or store derived or inferred, information about a Twitter user’s:

- Health (including pregnancy)
- Negative financial status or condition
- Political affiliation or beliefs
- Racial or ethnic origin
- Religious or philosophical affiliation or beliefs
- Sex life or sexual orientation
- Trade union membership
- Alleged or actual commission of a crime
- Aggregate analysis of Twitter content that does not store any personal data (for example, user IDs, usernames, and other identifiers) is permitted, provided that the analysis also complies with applicable laws and all parts of the Developer Agreement and Policy.

---

## 6. Download ARBERT and MARBERT
ARBERT and MARBERT are available for direct download and use ```exclusively for research```.
`For commercial use, please contact the authors via email @ (*muhammad.mageed[at]ubc[dot]ca*).`
- MARBERT can be downloaded [here]().
- MARBERT can be downloaded [here]().

---

## 7. Citation
If you use our models (ARBERT or MARBERT) for your scientific publication, or if you find the resources in this repository useful, please cite our paper as follows (to be updated):
```
@article{mageed2020marbert,
  title={ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic},
  author={Abdul-Mageed, Muhammad and Elmadany, AbdelRahim, and Nagoudi, El Moatez Billah},
  journal={arXiv preprint},
  year={2020}
}
```

---

## 8. Acknowledgments
We gratefully acknowledge support from the Natural Sciences and Engineering Research Council  of Canada, the  Social  Sciences and  Humanities  Research  Council  of  Canada, Canadian  Foundation  for  Innovation,  [ComputeCanada](www.computecanada.ca) and [UBC ARC-Sockeye](https://doi.org/10.14288/SOCKEYE). We  also  thank  the  [Google TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) program for providing us with free TPU access.
