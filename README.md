## Author Identification with `BiLSTM` and `Pretrained` Word Embeddings

### 📘 Project Overview

This project implements an **Author Identification System** trained on literary texts from classic authors — _Edgar Allan Poe, HP Lovecraft,_ and _Mary Shelley_. The model classifies text excerpts to the corresponding author using **Bidirectional LSTM (BiLSTM)** networks with **GloVe pretrained word embeddings** in **PyTorch**.

The notebook and scripts demonstrate a complete, **reproducible machine learning pipeline** from text preprocessing, vocabulary construction, and embedding integration, to model training, evaluation, and inference. This notebook can be found here [`00_AUTHOR_IDENTIFICATION`](/notebooks/00_AUTHOR_IDENTIFICATION.ipynb).

### Dataset Description

| Author          | Abbreviation | Description                                 |
| :-------------- | :----------- | :------------------------------------------ |
| Edgar Allan Poe | EAP          | Horror and mystery prose with poetic rhythm |
| HP Lovecraft    | HPL          | Cosmic horror themes, archaic language      |
| Mary Shelley    | MWS          | Gothic, emotional, and philosophical prose  |

Dataset source: [**Spooky Author Identification (Kaggle)**](https://www.kaggle.com/competitions/spooky-author-identification)  
Each record consists of text and an author label. The dataset is split into **training**, **validation**, and **test** sets.

### 3. Preprocessing Pipeline

| Step                   | Description                                                                         |
| :--------------------- | :---------------------------------------------------------------------------------- |
| Tokenization           | The `str.split()` method was used for word tokenization.                            |
| Text Cleaning          | Removed punctuation, numbers, and stopwords                                         |
| Padding                | Used `[pad]` tokens to equalize sequence lengths                                    |
| Threshold Filtering    | Removed samples with <23 words                                                      |
| Vocabulary Building    | Created `stoi` and `itos` mappings                                                  |
| Embedding Construction | Loaded GloVe embeddings and initialized missing tokens with zeros from google drive |

Threshold filter example:

```python
df = df[df['text'].apply(lambda x: len(x.split()) >= 23)]
```

### Model Architecture

**Class:** `AuthorsBiLSTM`

| Component           | Layer                                     | Description                               |
| :------------------ | :---------------------------------------- | :---------------------------------------- |
| Embedding           | `nn.Embedding(vocab_size, embedding_dim)` | Initialized with pretrained GloVe vectors |
| BiLSTM              | `nn.LSTM(..., bidirectional=True)`        | Extracts bidirectional context            |
| Layer Normalization | `nn.LayerNorm()`                          | Reduces internal covariate shift          |
| Dropout             | `nn.Dropout(0.5)`                         | Prevents overfitting                      |
| Output              | `nn.Linear(hidden_size*2, num_classes)`   | Predicts author label                     |

#### Model Definition Example

```python
author_bilstm = AuthorsBiLSTM(
    vocab_size=len(stoi),
    embedding_size=300,
    hidden_size=128,
    output_size=len(labels_dict),
    num_layers=2,
    bidirectional=True,
    dropout=0.5,
    pad_index=stoi['[pad]']
).to(device)
```

```shell
AuthorsBiLSTM(
  (embedding): Embedding(33626, 300, padding_idx=1)
  (lstm): LSTM(300, 128, num_layers=2, batch_first=True, dropout=0.6321, bidirectional=True)
  (layer_norm): LayerNorm((256,), eps=1e-05, elementwise_affine=True)
  (out): Linear(in_features=256, out_features=3, bias=True)
  (dropout): Dropout(p=0.6321, inplace=False)
)
```

### Optimization and Overfitting Prevention

Optimizer used:

```python
optimizer = torch.optim.Adam(
    author_bilstm.parameters(),
    lr=1e-3,
    weight_decay=1e-5,
    betas=(0.9, 0.999),
    eps=1e-8
)
```

Other regularization strategies:

- **Layer Normalization** after LSTM outputs
- **Dropout (0.5)**
- **Model Checkpointing** during training
- **Learning rate scheduler** (optional)

### Inference Example

Example model output:

```python
response = {
  'top': {
      'label': 1,
      'probability': 0.7819,
      'class': 'HPL',
      'author': 'Edgar Allan Poe'
  },
  'predictions': [
      {'label': 0, 'probability': 0.1901, 'class': 'EAP', 'author': 'Mary Shelley'},
      {'label': 1, 'probability': 0.7819, 'class': 'HPL', 'author': 'Edgar Allan Poe'},
      {'label': 2, 'probability': 0.0279, 'class': 'MWS', 'author': 'HP Lovecraft'}
  ]
}
```

Formatted console output:

```
==================================================
📑 AUTHOR TEXT
==================================================
How mutable are our feelings, and how strange is that clinging love we have of life even in the excess of misery I constructed another sail with a part of my dress and eagerly steered my course towards the land.

==================================================
🔮 TOP PREDICTION
==================================================
 Author       : Mary Shelley
 Class Code   : MWS
 Confidence   : 99.46%

==================================================
📊 ALL PREDICTIONS
==================================================
  - HP Lovecraft    (HPL) → 0.11%
  - Edgar Allan Poe (EAP) → 0.43%
  - Mary Shelley    (MWS) → 99.46%
==================================================

Enter the author text or (q) to quit:
```

### 8. Reproduction Instructions

1. Clone the repository:

```bash
git clone https://github.com/yourusername/author-identification.git
cd author-identification/FANsBot
```

1. Create virtual environment and activate it

```shell
virtualenv venv

#
.\Scripts\venv\activate
```

2. Installing requirements.

```bash
pip install -r requirements.txt
```

3. Run inference:

```bash
python main.py
```

### Acknowledgements

- Dataset: _Kaggle Spooky Author Identification_
- Pretrained Embeddings: _GloVe (6B, 300D)_
- Framework: _PyTorch_
- Project: _FANsBot NLP Research — Author Attribution_

### 🧠 Key Takeaways

- **Pretrained embeddings** improved model convergence and accuracy.
- **Regularization** (dropout + normalization) effectively reduced overfitting.
- The architecture generalizes well across small and large literary datasets.
- Demonstrates practical NLP author classification using **BiLSTM + GloVe**.
