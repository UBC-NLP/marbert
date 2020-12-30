# ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic
<img src="ARBERT_MARBERT.jpg" alt="drawing" width="30%" height="30%" align="right"/>

**ARBERT** is a large scale pre-training masked language model focused on Modern Standard Arabic(MSA). To train ARBERT, we use the same architecture as BERT-base: 12 attention layers, each has 12 attention heads and 768 hidden dimensions, avocabulary of 100K WordPieces, making ∼163M parameters. We  train  ARBERT on a collection of Arabic datasets comprising 61GB of text (6.2 B tokens)

**MARBERT** is a large scale pre-training masked language model focused on Dialectal Arabic (DA) and Modern Standard Arabic(MSA). Arabic has multiple varieties. To train MARBERT, we randomly sample 1B Arabic tweets from a large in-house dataset of about 6B tweets. We only include tweets with at least 3 Arabic words, based on character string matching, regardless whether the tweet has non-Arabic stringor not.  That is, we do not remove non-Arabic solong as the tweet meets the 3 Arabic word criterion. The dataset makes up 128GB of text (15.6B tokens). We use the same network architecture as ARBERT (BERT-base), but without the next sentenceprediction (NSP) objective since tweets are short. NSP were also shown not to be crucial for model performance. 

We compare with ARBERT with mBERT, XLM-R, AraBERT, and MARBERT in terms of data sources and size, vocabulary size, and model parameter size.

|             | **Data Source**              | **#Tokens(ar/all)** | **Tokanization**  | **Vocab Size(ar/all)** | **Cased** | **Arch.**         | **#Param** |
|---------|---------------------|----------------|---------------|--------------|-------|---------------|--------|
| mBERT   | Wikipedia           | 153M/1.5B      | WordPiece     | 5K/110K      | yes   | base          | 110M   |
| XLM-R<sub>B</sub> | CommonCraw          | l2.9B/295B     | SentencePiece | 14K/250K     | yes   | base          | 270M   |
| XLM-R<sub>L</sub>  | CommonCraw          | l2.9B/295B     | SentencePiece | 14K/250K     | yes   | large         | 550M   |
| AraBERT | Several (3 sources) | 2.5B/2.5B      | SentencePiece | 60K/64K      | no    | base          | 135M   |
| **ARBERT**  | Several (6 sources) | 6.2B/6.2B      | WordPiece     | 100K/100K    | no    | base          | 163M   |
| **MARBERT** | Arabic Twitter      | 15.6B/15.6B    | WordPiece     | 100K/100K    | no    | base          | 163M   |

## Models Evaluation
To  evaluate  our  models,   we  propose  **ArBench**,   a new benchmark for multi-dialectal Arabic language understanding.  ***ArBench is built using 41 datasets targeting 5 different tasks/task clusters***, allowing us to offer a series of standardized experiments under rich conditions. When fine-tuned on ArBench,  ARBERT and MARBERT collectively achieve new SOTA  with sizeable margins compared to all existing models such as mBERT, XLM-R (Base and Large), and  AraBERT on 37 out of 45 classification tasks on the 41 datasets (82.22). 

### (1) Sentiment Analysis

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
|  Twitter<sub>Abdullah</sub> (2)           | 81.50 | 91.00 | 92.00 | 89.50 | 91.50 | **95.00** |

***Results reported based on F<sub>1</sub><sup>NP</sup> score.***
### (2) Social Meaning
| **Task (#classes)**        |   **Dataset **  |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
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

### (3) Topic Classification
| **Dataset (#classes)**  |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|----------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|
| OSAC (10)                    | 96.84 | 97.15 | 98.20 | 97.03 | **97.50** | 97.23 |
| Khallej (4)                  | 92.81 | 91.87 | 93.56 | 93.83 | **94.53** | 93.63 |
| ANT<sub>Text</sub> (5)       | 84.89 | 85.77 | 86.72 | **88.17** | 86.87 | 85.27 |
| ANT<sub>Title</sub> (5)      | 78.29 | 79.96 | 81.25 | 81.03 | **81.70** | 81.19 |
| ANT<sub>Text+Title</sub> (5) | 84.67 | 86.21 | 86.96 | **87.22** | 87.21 | 85.60 |

***Results reported based on F<sub>1</sub> score.***
### (4) Dialect Identification
| **Dataset**        |   **Dataset (#classes)**  |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|----------------|---------------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|
| Regoin (5)      | ArSarcasm<sub>Dia</sub) | 43.81 | 41.71 | 41.83 | 47.54 | 54.70 | 51.27 |
| Country (21)    |  MADAR-TL | 34.92 | 35.91 | 35.14 | 34.87 | 37.90 | 40.40 |
| Region (4)      |       AOC      | 77.27 | 77.34 | 78.77 | 79.20 | 81.09 | 82.37 |
| Region (3)      |       AOC      | 85.76 | 86.39 | 87.56 | 87.68 | 89.06 | 90.85 |
| Binary (4)            |       AOC      | 86.19 | 86.85 | 87.30 | 87.76 | 88.46 | 88.59 |
| Country(18)     |      QADI      | 66.57 | 77.00 | 82.73 | 72.23 | 88.63 | 90.89 |
| Country(21)     |      NADI      | 13.32 | 16.36 | 17.17 | 17.46 | 22.56 | 29.14 |
| Province (100 ) |      NADI      |  2.13 |  4.12 |  0.32 |  3.13 |  6.10 |  6.28 |

***Results reported based on F<sub>1</sub> score.***

### (5) Named Entity Recogntion (NER)
| **Dataset**        |   **Dataset (#classes)**  |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|----------------|---------------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|

***Results reported based on F<sub>1</sub> score.***

## Fine-tuning ARBERT and MARBERT on ArBench datasets
 
 ```python
    import 
 ```
 
## Citation
If you use our models (ARBERT or MARBERT) for your scientific publication, or if you find the resources in this repository useful, please cite one of the following paper:
```
citiation
```

## Acknowledgements
Gratefully   acknowledges   support   fromthe   Natural   Sciences   and   Engineering   Research  Council  of  Canada,  the  Social  Sciencesand  Humanities  Research  Council  of  Canada,Canadian  Foundation  for  Innovation,  [ComputeCanada](www.computecanada.ca) and [UBC ARC-Sockeye](https://doi.org/10.14288/SOCKEYE). We  also  thank  the  [Google TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) program for providing us with free TPU access.