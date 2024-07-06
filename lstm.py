import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# LSTM 모델 정의
class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# ESG 점수 파일 읽기
def load_esg_scores(file_name):
    esg_scores = {}
    with open(file_name, 'r') as f:
        for line in f:
            category, score = line.strip().split(': ')
            esg_scores[category] = float(score)
    return esg_scores

esg_scores = load_esg_scores('esg_scores.txt')

# 주식 데이터 로드 및 전처리
def load_stock_data(file_name):
    stock_data = pd.read_csv(file_name, parse_dates=['date'])
    stock_data.set_index('date', inplace=True)
    return stock_data

# 주식 데이터 파일명 및 정규화
file_name = 'stock_prices.txt'
stock_data = load_stock_data(file_name)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(stock_data[['stock_price']].values)

# ESG 점수를 데이터프레임에 추가
for category, score in esg_scores.items():
    stock_data[category] = score

# 학습 데이터 준비
def create_dataset(data, time_step=1):
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        a = data[i:(i + time_step), :]
        X.append(a)
        Y.append(data[i + time_step, 0])  # 주식 가격 예측
    return np.array(X), np.array(Y)

# 데이터셋 생성
time_step = 1
X, Y = create_dataset(scaled_data, time_step)

# 데이터셋 분할
train_size = int(len(X) * 0.8)
test_size = len(X) - train_size
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(Y)]

# PyTorch Dataset 및 DataLoader
class StockDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.Y[idx], dtype=torch.float32)

# 데이터셋 생성
train_dataset = StockDataset(X_train, Y_train)
test_dataset = StockDataset(X_test, Y_test)

# DataLoader 생성
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

# LSTM 모델 정의 및 초기화
input_dim = X_train.shape[2]  # 여기서 input_dim은 feature의 수를 나타냅니다.
hidden_dim = 50
num_layers = 2
output_dim = 1
model = LSTMModel(input_dim, hidden_dim, num_layers, output_dim)

# 손실 함수 및 옵티마이저
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 모델 학습
model.train()
for epoch in range(1):
    for X_batch, Y_batch in train_loader:
        outputs = model(X_batch)
        optimizer.zero_grad()
        loss = criterion(outputs, Y_batch)
        loss.backward()
        optimizer.step()
    print(f'Epoch [{epoch+1}/1], Loss: {loss.item():.4f}')

# 예측
model.eval()
train_predict = []
test_predict = []
with torch.no_grad():
    for X_batch, _ in train_loader:
        train_predict.append(model(X_batch).item())
    for X_batch, _ in test_loader:
        test_predict.append(model(X_batch).item())

# 예측값 역정규화
train_predict = scaler.inverse_transform(np.array(train_predict).reshape(-1, 1))
test_predict = scaler.inverse_transform(np.array(test_predict).reshape(-1, 1))

# 결과 시각화
plt.figure(figsize=(14, 5))
plt.plot(stock_data.index, stock_data['stock_price'], label='True Price')
plt.plot(stock_data.index[time_step:len(train_predict)+time_step], train_predict[:,0], label='Train Predict')
plt.plot(stock_data.index[len(train_predict)+(time_step*2)+1:len(stock_data)-1], test_predict[:,0], label='Test Predict')
plt.xlabel('Date')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
