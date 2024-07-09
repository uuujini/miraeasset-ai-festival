import yfinance as yf
import pandas as pd

# 삼성전자 주가 데이터 가져오기
data = yf.download('005930.KS', start='2024-01-01', end='2024-06-30')

# 데이터 저장
data.to_csv('samsung_stock_data.csv')

# CSV 파일 읽기
data = pd.read_csv('samsung_stock_data.csv')

# 데이터 확인
print(data.head())
