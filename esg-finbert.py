import fitz
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# PDF 파일에서 텍스트 추출
def extract_text_from_pdf(pdf_path):
    document = fitz.open(pdf_path)
    text = ""
    for page_num in range(len(document)):
        page = document.load_page(page_num)
        text += page.get_text()
    return text

# ESG 보고서 텍스트 분석
def analyze_esg_report(report):
    tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-esg-9-categories')
    model = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-esg-9-categories')
    inputs = tokenizer(report, return_tensors='pt', truncation=True, padding=True, max_length=512)
    outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1).detach().cpu().numpy()
    categories = ['environment', 'social', 'governance', 'positive', 'negative', 'neutral', 'unknown', 'controversial', 'other']
    return dict(zip(categories, scores[0]))

# PDF 파일 경로
pdf_path = 'SamsungESG2024.pdf'

# PDF에서 텍스트 추출
report_text = extract_text_from_pdf(pdf_path)

# 텍스트를 분석하여 ESG 점수 계산
esg_scores = analyze_esg_report(report_text)

# 결과를 텍스트 파일로 저장
with open('esg_scores.txt', 'w') as f:
    for category, score in esg_scores.items():
        f.write(f"{category}: {score}\n")

print("ESG 점수 결과가 esg_scores.txt 파일에 저장되었습니다.")
