# ARBERT & MARBERT: Deep Bidirectional Transformers for Arabic
<img src="ARBERT_MARBERT.jpg" alt="drawing" width="50%" height="50%" align="right"/>

<p style='text-align: justify;'> **ARBERT** is a large scale pre-training masked lan-guage model focused on Modern Standard Arabic(MSA). To train ARBERT, we use the same archi-tecture as BERT-Base: 12attention layers, each has 12attention heads and 768 hidden dimensions, avocabulary of100K WordPieces, making ∼163M parameters. We now describe ARBERT’s pre-traindataset, vocabulary, and pre-training setup.</p>
<p style='text-align: justify;'> Arabic has multiple varieties.  Many of these va-rieties are understudied due to rarity of resources. Multilingual models such as mBERT and XLM-Rare trained almost exclusively on MSA data, which is also the case for AraBERT and ARBERT. As such, these models are not best suited for down-stream tasks involving dialectal Arabic.  To treat this issue, we use a large Twitter dataset to pre-train a new model, **MARBERT** , from scratch. For this new model, we also use the BERT-Base architecture as ARBERT. </p>
## How to install
 - Using pip
 ```shell
  pip install git+https://github.com/UBC-NLP/aranet
 ```
 - Clone and install
 ```shell
  git clone https://github.com/UBC-NLP/aranet
  cd aranet
  pip install .
```
