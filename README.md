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

***Results reported based on Acc.***

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

***Results reported based on F<sub>1</sub><sup>NP</sup>.***
### (2) Social Meaning
 **Dataset (#classes)** |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|--------------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|

### (3) Topic Classificatio
 **Dataset (#classes)** |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|--------------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|

### (4) Dialect Identification
 **Dataset (#classes)** |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|--------------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|

### (5) Named Entity Recogntion (NER)
 **Dataset (#classes)** |  **mBERT** | **XLM-R<sub>B</sub>** | **XLM-R<sub>L</sub>** | **AraBERT** | **ARBERT** | **MARBERT** |
|--------------------|:------:|:----------:|:-----------:|:-------:|:------:|:-------:|

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