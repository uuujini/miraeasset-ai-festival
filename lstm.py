import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# 주가 데이터 로드
def load_stock_data(filepath):
    data = pd.read_csv(filepath)
    return data

stock_data = load_stock_data('samsung_stock_data.csv')  # csv 파일로 변경

# 주가 데이터 전처리
scaler = MinMaxScaler(feature_range=(0, 1))
stock_data['scaled_close'] = scaler.fit_transform(stock_data['Close'].values.reshape(-1, 1))

# ESG 점수 로드
def load_esg_scores(filepath):
    esg_scores = {}
    with open(filepath, 'r') as file:
        for line in file:
            key, value = line.strip().split(': ')
            esg_scores[key] = float(value)
    return esg_scores

esg_scores = load_esg_scores('esg-analysis/esg_scores.txt')

# ESG 점수를 데이터프레임으로 변환
esg_df = pd.DataFrame([esg_scores])
esg_values = esg_df.values[0]

# 시퀀스 데이터 생성
def create_sequences(data, esg_data, seq_length):
    sequences = []
    for i in range(len(data) - seq_length):
        sequence = np.concatenate((data[i:i + seq_length], esg_data), axis=None)
        label = data[i + seq_length]
        sequences.append((sequence, label))
    return sequences

seq_length = 5
sequences = create_sequences(stock_data['scaled_close'].values, esg_values, seq_length)

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# 데이터 준비
X = np.array([seq[0] for seq in sequences])
y = np.array([seq[1] for seq in sequences])

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# 모델 초기화
input_size = X.shape[1]  # input_size 변경
hidden_size = 50
output_size = 1
num_layers = 2

model = LSTMModel(input_size, hidden_size, output_size, num_layers)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    outputs = model(X.unsqueeze(1))
    optimizer.zero_grad()
    loss = criterion(outputs, y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 예측
model.eval()
predictions = model(X.unsqueeze(1)).detach().numpy()
predictions = scaler.inverse_transform(predictions)

# 시각화
actual_prices = stock_data['Close'][seq_length:].values

plt.figure(figsize=(10,6))
plt.plot(actual_prices, label='Actual Prices')
plt.plot(predictions, label='Predicted Prices')
plt.title('Stock Price Prediction with ESG Scores')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
