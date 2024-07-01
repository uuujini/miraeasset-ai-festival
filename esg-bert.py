from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model_name = "ESGBERT/EnvironmentalBERT-environmental"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# 예시 텍스트
texts = [
    "The company has made significant improvements in its environmental policies.",
    "There have been several reports of pollution caused by the factory.",
]

inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

model.eval()

with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits

probs = torch.nn.functional.softmax(logits, dim=-1)

for text, prob in zip(texts, probs):
    print(f"Text: {text}\nProbabilities: {prob}\n")
