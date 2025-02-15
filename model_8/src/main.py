"""
---------------------------------------------------------------------------------------

Модель:
model = Sequential([
    Dense(128, activation='relu', input_shape=(window_size,)),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dropout(0.15),
    Dense(32, activation='relu'),
    Dense(forecast_horizon)
])

---------------------------------------------------------------------------------------

Использовались данные из столбцов: 'Цена_ЭЭ' и 'Дата_и_Время' 

---------------------------------------------------------------------------------------

Метрики прогноза на тестовой выборке получились:
MAE: 218.66
MSE: 72077.44
RMSE: 268.47
MAPE: 0.16
R2 Score: 0.6553

---------------------------------------------------------------------------------------

Метрики прогноза значений на июнь получились:
MAE: 182.83
RMSE: 242.82
MSE: 58959.87
MAPE: 0.17
R2 Score: 0.7106

---------------------------------------------------------------------------------------
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.losses import MeanAbsolutePercentageError
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Attention
from tensorflow.keras.optimizers import Adam, RMSprop, AdamW, Nadam, Adagrad


# Загрузка данных
data_merged = pd.read_csv("data/май_smoothed_2.csv")
data_merged['Дата_и_Время'] = pd.to_datetime(data_merged['Дата_и_Время'])
data_merged.set_index('Дата_и_Время', inplace=True)

# Подготовка данных
scaler = MinMaxScaler()
data_merged['Цена_ЭЭ_scaled'] = scaler.fit_transform(data_merged[['Цена_ЭЭ']])

# Функция создания последовательностей
def create_sequences(data, window_size, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size:i+window_size+forecast_horizon])
    return np.array(X), np.array(y)

# Параметры
window_size = 48
forecast_horizon = 48

# Создание выборок
data = data_merged['Цена_ЭЭ_scaled'].values
X, y = create_sequences(data, window_size, forecast_horizon)

# Разделение данных на тренировочную и тестовую выборки
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Преобразование для использования в Dense-слоях
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# Создание модели
model = Sequential([
    Dense(128, activation='relu', input_shape=(window_size,)),
    Dropout(0.25),
    Dense(64, activation='relu'),
    Dropout(0.15),
    Dense(32, activation='relu'),
    Dense(forecast_horizon)
])
"""
model = Sequential([
    LSTM(128, activation='tanh', return_sequences=True, input_shape=(window_size, 1)),
    Dropout(0.25),
    LSTM(64, activation='tanh', return_sequences=False),
    Dropout(0.15),
    Dense(32, activation='relu'),
    Dense(forecast_horizon)
])
"""
# Компиляция модели
model.compile(optimizer=RMSprop(learning_rate=0.0005), loss='mape')
# model.compile(optimizer=Adam(learning_rate=0.0005), loss='mae')
# Обучение модели
model.fit(X_train, y_train, epochs=200, batch_size=16, validation_data=(X_test, y_test))


# Прогноз на исторических данных
historical_forecast_scaled = model.predict(X_test)  # Прогнозируем тестовые данные
historical_forecast = scaler.inverse_transform(historical_forecast_scaled)
real_prices_test = scaler.inverse_transform(y_test)

# Метрики качества
mae = mean_absolute_error(real_prices_test, historical_forecast)
mse = mean_squared_error(real_prices_test, historical_forecast)
rmse = np.sqrt(mse)
mape = mean_absolute_percentage_error(real_prices_test, historical_forecast)
r2 = r2_score(real_prices_test, historical_forecast)


print("Метрики качества на тестовой выборке:")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R2 Score: {r2:.4f}")

# График 1: Адекватность модели на историческом интервале
plt.figure(figsize=(16, 8))
plt.plot(real_prices_test.flatten(), label="Фактические данные", color="green")
plt.plot(historical_forecast.flatten(), label="Прогноз модели", color="blue", linestyle="--")
plt.title("Адекватность модели на историческом интервале")
plt.xlabel("Временной шаг")
plt.ylabel("Цена электроэнергии")
plt.legend()
plt.grid()
plt.show()

# Остатки
residuals = real_prices_test.flatten() - historical_forecast.flatten()

# График 2: Остатки vs Прогноз
plt.figure(figsize=(12, 6))
plt.scatter(historical_forecast.flatten(), residuals, alpha=0.5, color="purple")
plt.axhline(0, color='red', linestyle='--')
plt.title("Остатки vs Прогноз")
plt.xlabel("Прогнозируемые значения")
plt.ylabel("Остатки")
plt.grid()
plt.show()

 
# Прогноз
last_window_data = data[-window_size:].reshape(1, -1)
forecast_scaled = model.predict(last_window_data)
forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1))

# Загрузка реальных данных
real_data = pd.read_csv("data/июнь_merged.csv")
real_data['Дата_и_Время'] = pd.to_datetime(real_data['Дата_и_Время'])
real_data_june = real_data[(real_data['Дата_и_Время'] >= '2024-06-01 00:00:00') &
                           (real_data['Дата_и_Время'] < '2024-06-03 00:00:00')]
real_prices = real_data_june['Цена_ЭЭ'].values

# Визуализация
plt.figure(figsize=(16, 8))
# Исторические данные
historical_prices = data_merged['Цена_ЭЭ']
plt.plot(range(len(historical_prices)), historical_prices, label='Исторические данные', color='red')
# Прогноз
x_forecast = range(len(historical_prices), len(historical_prices) + len(forecast))
plt.plot(x_forecast, forecast, label='Прогноз', color='blue')
# Реальные данные
x_real = range(len(historical_prices), len(historical_prices) + len(real_prices))
plt.plot(x_real, real_prices, label='Реальные данные', color='green')
plt.xlabel('Час')
plt.ylabel('Цена электроэнергии')
plt.title('Прогноз цены электроэнергии')
plt.legend()
plt.grid()
plt.show()

# Метрики качества
mae = mean_absolute_error(real_prices, forecast)
rmse = np.sqrt(mean_squared_error(real_prices, forecast))
mse = mean_squared_error(real_prices, forecast)
mape = mean_absolute_percentage_error(real_prices, forecast)
r2 = r2_score(real_prices, forecast)
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MSE: {mse:.2f}")
print(f"MAPE: {mape:.2f}")
print(f"R2 Score: {r2:.4f}")