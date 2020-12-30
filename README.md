# ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic
<img src="ARBERT_MARBERT.jpg" alt="drawing" width="30%" height="305%" align="right"/>

**ARBERT** is a large scale pre-training masked language model focused on Modern Standard Arabic(MSA). To train ARBERT, we use the same architecture as BERT-base: 12 attention layers, each has 12 attention heads and 768 hidden dimensions, avocabulary of 100K WordPieces, making ∼163M parameters. We  train  ARBERT on a collection of Arabic datasets comprising 61GB of text (6.5 B tokens)

**MARBERT** is a large scale pre-training masked language model focused on Dialectal Arabic (DA) and Modern Standard Arabic(MSA). Arabic has multiple varieties. To train MARBERT, we randomly sample 1B Arabic tweets from a large in-house dataset of about 6B tweets. We only include tweets with at least 3 Arabic words, based on character string matching, regardless whether the tweet has non-Arabic stringor not.  That is, we do not remove non-Arabic solong as the tweet meets the 3 Arabic word criterion. The dataset makes up 128GB of text (15.6B tokens). We use the same network architecture as ARBERT (BERT-base), but without the next sentenceprediction (NSP) objective since tweets are short. NSP were also shown not to be crucial for model performance. 

## Compare with other models
As shown in the below table, we compare with ARBERT with mBERT, XLM-R, AraBERT, and MARBERT in terms of data sources and size, vocabulary size, and model parameter size.
<center> <img src="Configuration_ARBERT_MARBERT.png" alt="drawing" width="100%" height="100%"/></center>

## Models Evaluation
To  evaluate  our  models,   we  propose  **ArBench**,   a new benchmark for multi-dialectal Arabic language understanding.  ***ArBench is built using 41 datasets targeting 5 different tasks/task clusters***, allowing us to offer a series of standardized experiments under rich conditions. When fine-tuned on ArBench,  ARBERT and MARBERT collectively achieve new SOTA  with sizeable margins compared to all existing models such as mBERT, XLM-R (Base and Large), and  AraBERT on 37 out of 45 classification tasks on the 41 datasets (%82.22). 

| Dataset (#classes) |  Cluster Task  |    Task   |  mBERT |XLM R<sub>B</sub>            | XLM R<sub>L</sub>         | AraBERT | ARBERT | MARBERT |
|:------------------:|:--------------:|:---------:|:------:|:---------------------:|:-------------------:|:-------:|:------:|:-------:|
|    Offensive (2)   | Soical Meaning | Offensive | 84.25 |         85.26        |        88.28       |  86.57 | 90.38 |  **92.41** |
## Fine-tuning ARBERT and MARBERT on the ArBench datasets
 
 ```python
  
 ```
 
## Citation
If you use our models (ARBERT or MARBERT) for your scientific publication, or if you find the resources in this repository useful, please cite one of the following paper:
```
citiation
```

## Acknowledgements
Gratefully   acknowledges   support   fromthe   Natural   Sciences   and   Engineering   Research  Council  of  Canada,  the  Social  Sciencesand  Humanities  Research  Council  of  Canada,Canadian  Foundation  for  Innovation,  [ComputeCanada](www.computecanada.ca) and [UBC ARC-Sockeye](https://doi.org/10.14288/SOCKEYE). We  also  thank  the  [Google TensorFlow Research Cloud (TFRC)](https://www.tensorflow.org/tfrc) program for providing us with free TPU access.