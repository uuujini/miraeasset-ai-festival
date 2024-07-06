import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 모델 및 토크나이저 로드
tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg-9-categories')
model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg-9-categories')

def analyze_esg_report(report):
    inputs = tokenizer(report, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
    categories = ['environment', 'social', 'governance', 'positive', 'negative', 'neutral', 'unknown', 'controversial', 'other']
    return dict(zip(categories, scores[0]))

# 예시 ESG 보고서 텍스트
report = '''Apple announced significant progress towards its goal to decarbonize its value chain, including revealing that more than 320 suppliers – representing 95% of the company’s direct manufacturing spend – have now committed to use 100% renewable energy for Apple production by 2030, up from around 250 suppliers last year.

Emissions from the product manufacturing represents nearly two thirds of Apple’s carbon footprint, with electricity use as the single largest contributor. Addressing these emissions forms a major part of the company’s strategy to achieve its ambition to become carbon neutral across its entire business, manufacturing supply chain, and product life cycle by 2030, a goal set by the company in 2020.

In October 2022, Apple urged its supply chain to decarbonize their entire Apple-related Scope 1 and 2 emissions footprint, and notified suppliers that progress towards these goals would be one of the key criteria considered when awarding business. Since that time, the use of clean energy in Apple’s supply chain has grown rapidly, reaching 16.5 GW currently, up 20% over last year, and more than 55% higher than 2022.

Apple said that its supply chain generated more than 25.5 million MW of clean energy last year, avoiding over 18.5 million metric tons of carbon emissions.'''
esg_scores = analyze_esg_report(report)

# 결과를 텍스트 파일로 저장
with open('esg_scores.txt', 'w') as f:
    for category, score in esg_scores.items():
        f.write(f"{category}: {score}\n")

print("esg_results.txt")
