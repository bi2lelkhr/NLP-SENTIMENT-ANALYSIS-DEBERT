# NLP Sentiment Analysis with DeBERTa v3

## Overview
This project implements a **sentiment analysis model for movie reviews** using Microsoft's **DeBERTa v3 base** architecture.  
The model achieves **94.7% accuracy** on the test set.

## Model Details
- **Architecture:** DeBERTa v3 Base (`microsoft/deberta-v3-base`)  
- **Task:** Binary sentiment classification (positive/negative)  
- **Dataset:** Combined IMDB reviews (~50K samples after deduplication)  
  - Original IMDB Dataset (50K reviews)  
  - Synthetic IMDB reviews (5K reviews)  
- **Final Performance:**  
  - Accuracy: **94.68%**  
  - Weighted F1-score: **94.68%**

## Dataset Preprocessing
The preprocessing pipeline includes:
- Removal of **HTML tags, URLs, and emojis**
- **Contraction expansion** (e.g., `"won't"` → `"will not"`)
- **Negation handling** with special tagging (e.g., `"not good"` → `"not good_NEG"`)
- **Lemmatization** using WordNet
- **Keeping stopwords** and sentiment indicators (`!`, `?`)

## Training Configuration
- **Learning rate:** `2e-5`  
- **Batch size:** `16` (with gradient accumulation steps of 2)  
- **Epochs:** `6` (with early stopping patience of 2)  
- **Optimizer:** AdamW with linear learning rate schedule  
- **Warmup ratio:** `0.1`  
- **Weight decay:** `0.01`  
- **Max sequence length:** `256 tokens`

## Results
| Metric        | Score   |
|---------------|---------|
| Accuracy      | 94.68%  |
| F1-Score (weighted) | 94.68%  |

## Model Usage
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_id = "billelkhr/deberta-v3-sentiment-review-movie"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSequenceClassification.from_pretrained(model_id)

text = "This movie was amazing!"
inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)
    prediction = outputs.logits.argmax(dim=-1).item()

sentiment = "positive" if prediction == 1 else "negative"
print(f"Sentiment: {sentiment}")
