# NLP-SENTIMENT-ANALYSIS-DEBERT

Overview
This project implements a sentiment analysis model for movie reviews using Microsoft's DeBERTa v3 base architecture, achieving 94.7% accuracy on the test set.
Model Details

Architecture: DeBERTa v3 Base (microsoft/deberta-v3-base)
Task: Binary sentiment classification (positive/negative)
Dataset: Combined IMDB reviews (~50K samples after deduplication)

Original IMDB Dataset (50K reviews)
Synthetic IMDB reviews (5K reviews)


Final Performance: 94.68% accuracy, 94.68% weighted F1-score

Dataset Preprocessing
The preprocessing pipeline includes:

HTML tag removal
URL and emoji removal
Contraction expansion (e.g., "won't" → "will not")
Negation handling with special tagging (e.g., "not good" → "not good_NEG")
Lemmatization using WordNet
Keeping stopwords and sentiment indicators (!, ?)

Training Configuration
python- Learning rate: 2e-5
- Batch size: 16 (with gradient accumulation steps of 2)
- Epochs: 6 (with early stopping patience of 2)
- Optimizer: AdamW with linear learning rate schedule
- Warmup ratio: 0.1
- Weight decay: 0.01
- Max sequence length: 256 tokens
Results
MetricScoreAccuracy94.68%F1-Score (weighted)94.68%
Model Usage
pythonfrom transformers import AutoTokenizer, AutoModelForSequenceClassification
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
Key Features

Robust text preprocessing with negation handling
Fine-tuned on domain-specific movie review data
Efficient inference with mixed precision training (FP16)
Early stopping to prevent overfitting

Requirements

transformers
torch
datasets
scikit-learn
pandas
nltk

Model Hub
The trained model is available on Hugging Face: billelkhr/deberta-v3-sentiment-review-movie
