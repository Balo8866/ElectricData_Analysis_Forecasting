import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
import sys

# 設定繁體中文字型
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 使用微軟正黑體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題

# 讀取數據
file_path = "2020_2023_electritic.xlsx"
try:
    df = pd.read_excel(file_path)
    print("成功讀取檔案！")
except FileNotFoundError:
    print(f"錯誤：找不到檔案 {file_path}！")
    sys.exit()

# 數據處理
df['時間'] = pd.to_datetime(df['時間'], errors='coerce')
df = df.dropna(subset=['時間'])
df['年份'] = df['時間'].dt.year
df['月份'] = df['時間'].dt.month
fields = df.columns[3:12]  # 場域列的範圍

for field in fields:
    df[field] = pd.to_numeric(df[field], errors='coerce')

df.dropna(subset=fields, how='all', inplace=True)

# 繪製歷年用電情況
unique_years = df['年份'].unique()
for year in unique_years:
    yearly_data = df[df['年份'] == year]
    plt.figure(figsize=(14, 8))
    for field in fields:
        monthly_data = yearly_data.groupby('月份')[field].sum()
        plt.plot(monthly_data.index, monthly_data.values, marker='o', label=field)
    plt.title(f'{year} 年各場域用電情況', fontsize=16)
    plt.xlabel('月份', fontsize=12)
    plt.ylabel('用電量 (kWh)', fontsize=12)
    plt.xticks(range(1, 13))
    plt.legend(title='場域', fontsize=10)
    plt.grid(alpha=0.5)
    plt.tight_layout()
    plt.show()

# 獲取用戶輸入
year = int(input("請輸入年份 (例如 2020): "))
month = int(input("請輸入月份 (1-12): "))
field = input(f"請選擇場域 ({', '.join(fields)}): ")

# 驗證輸入
available_years = df['年份'].unique()
if year not in available_years:
    print(f"目前沒有 {year} 年的資料！")
    sys.exit()

if field not in fields:
    print(f"場域 {field} 不存在！")
    sys.exit()

filtered_df = df[(df['年份'] == year) & (df['月份'] == month)][['時間', field]]
if filtered_df.empty:
    print(f"目前沒有 {year} 年 {month} 月場域 {field} 的資料！")
    sys.exit()

# 輸出該年份場域用電資料
print(f"{year} 年場域 {field} 用電資料：")
print(filtered_df)

# 計算該月份總用電量
actual_total_usage = filtered_df[field].sum()
print(f"{year} 年 {month} 月場域 {field} 的總用電量為：{actual_total_usage:.2f} kWh")

# 繪製該月份用電情況
plt.figure(figsize=(10, 6))
plt.plot(filtered_df['時間'], filtered_df[field], label=f'{year} 年 {month} 月用電量')
plt.title(f'{year} 年 {month} 月場域 {field} 用電情況')
plt.xlabel('日期')
plt.ylabel('用電量 (kWh)')
plt.legend()
plt.grid()
plt.show()

# 構造時間序列數據
data = filtered_df[field].values
sequence_length = 12
X, y = [], []
for i in range(len(data) - sequence_length):
    X.append(data[i:i + sequence_length])
    y.append(data[i + sequence_length])
X = np.array(X)
y = np.array(y)

# 計算原始數據的最小值和最大值
X_min, X_max = df[field].min(), df[field].max()
y_min, y_max = y.min(), y.max()

# 將數據縮放到 0~1
X_scaled = (X - X_min) / (X_max - X_min)
y_scaled = (y - y_min) / (y_max - y_min)

# 分割訓練集與測試集
split = int(len(X_scaled) * 0.8)
X_train, X_test = X_scaled[:split], X_scaled[split:]
y_train, y_test = y_scaled[:split], y_scaled[split:]

# CNN 模型
model = Sequential([
    Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(sequence_length, 1)),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(50, activation='relu'),
    Dropout(0.2),
    Dense(1)
])

# 編譯與訓練模型
model.compile(optimizer='adam', loss='mean_squared_error')
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
train_history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# 繪製訓練過程的折線圖
def plot_train_history(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='訓練損失')
    plt.plot(history.history['val_loss'], label='驗證損失')
    plt.title('模型訓練過程')
    plt.xlabel('世代數')
    plt.ylabel('損失')
    plt.legend()
    plt.grid()
    plt.show()

plot_train_history(train_history)

# 預測下一年該月份用電情況
last_sequence = X_scaled[-1].reshape((1, sequence_length, 1))
scaled_predicted_value = model.predict(last_sequence)[0][0]
# 將預測值還原到原始範圍
predicted_daily_usage = scaled_predicted_value * (y_max - y_min) + y_min
predicted_total_usage = predicted_daily_usage * len(filtered_df)
print(f"預測下一年度 {year+1} 年 {month} 月場域 {field} 的總用電量為：{predicted_total_usage:.2f} kWh")

# 繪製預測結果
plt.figure(figsize=(10, 6))
plt.bar([f'{year} 年 {month} 月', f'{year+1} 年 {month} 月'],
        [actual_total_usage, predicted_total_usage],
        color=['blue', 'orange'], alpha=0.7)
plt.title(f'{field} 場域用電量：實際與預測')
plt.ylabel('總用電量 (kWh)')
plt.show()
