import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop

# Загрузка и подготовка данных
data_merged = pd.read_csv("data/май_smoothed_2.csv")
data_merged['Дата_и_Время'] = pd.to_datetime(data_merged['Дата_и_Время'])
data_merged.set_index('Дата_и_Время', inplace=True)
# Создание новых столбцов
data_merged['Час'] = data_merged.index.hour
data_merged['День'] = data_merged.index.day
data_merged['Месяц'] = data_merged.index.month

# Корреляционная матрица
correlation_columns = ['Цена_ЭЭ', 'Час', 'День', 'Месяц', 'Цена_газ', 'Цена_нефть', 'sin_час', 'cos_час']
correlation_matrix = data_merged[correlation_columns].corr()

# Визуализация корреляционной матрицы
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Корреляционная матрица признаков')
plt.tight_layout()
plt.show()
# exit()

# Подготовка данных
scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(data_merged[['Цена_ЭЭ', 'sin_час', 'cos_час', 'День', 'Цена_газ', 'Цена_нефть']])
data_merged[['Цена_ЭЭ_scaled', 'sin_час_scaled', 'cos_час_scaled', 'День_scaled', 'Цена_газ_scaled', 'Цена_нефть_scaled']] = scaled_features

# Функция создания последовательностей
def create_sequences(data, window_size, forecast_horizon, feature_columns):
    X, y = [], []
    for i in range(len(data) - window_size - forecast_horizon + 1):
        X.append(data[i:i+window_size, feature_columns])
        y.append(data[i+window_size:i+window_size+forecast_horizon, 0])  # Целевая переменная - Цена_ЭЭ
    return np.array(X), np.array(y)

# Параметры
window_size = 48
forecast_horizon = 48

# Подготовка данных для создания выборок
feature_columns = [0, 1, 2, 3, 4, 5]  # Индексы колонок: Цена_ЭЭ, sin_час, cos_час, День, Цена_газ, Цена_нефть
data = data_merged[['Цена_ЭЭ_scaled', 'sin_час_scaled', 'cos_час_scaled', 'День_scaled', 'Цена_газ_scaled', 'Цена_нефть_scaled']].values
X, y = create_sequences(data, window_size, forecast_horizon, feature_columns)

split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Преобразование для использования в Dense-слоях
X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)


# Модель
model = Sequential([
    LSTM(64, activation='tanh', return_sequences=True, input_shape=(window_size, len(feature_columns))),
    Dropout(0.2),
    LSTM(32, activation='tanh', return_sequences=False),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(forecast_horizon)
])

model.compile(optimizer=RMSprop(learning_rate=0.0001), loss='mape')
model.fit(
    X_train.reshape(X_train.shape[0], window_size, len(feature_columns)),
    y_train,
    epochs=200,
    batch_size=16,
    validation_data=(X_test.reshape(X_test.shape[0], window_size, len(feature_columns)), y_test)
)

# Прогноз на исторических данных (тестовая выборка)
historical_forecast_scaled = model.predict(X_test.reshape(X_test.shape[0], window_size, len(feature_columns)))

# Подготовим матрицу для обратного масштабирования
historical_forecast_full = np.zeros((historical_forecast_scaled.shape[0], 6))
historical_forecast_full[:, 0] = historical_forecast_scaled.reshape(-1)[:historical_forecast_full.shape[0]]

# Обратное масштабирование
historical_forecast = scaler.inverse_transform(historical_forecast_full)[:, 0]


# Восстановление реальных значений для тестовой выборки
real_prices_test = scaler.inverse_transform(
    np.column_stack([y_test.reshape(-1, 1), np.zeros((y_test.reshape(-1, 1).shape[0], 5))])
)[:, 0]

# Метрики качества прогноза на исторических данных
# Усечение данных до минимальной длины
min_length = min(real_prices_test.flatten().shape[0], historical_forecast.flatten().shape[0])

mae_test = mean_absolute_error(
    real_prices_test.flatten()[:min_length], 
    historical_forecast.flatten()[:min_length]
)
rmse_test = np.sqrt(mean_squared_error(
    real_prices_test.flatten()[:min_length], 
    historical_forecast.flatten()[:min_length]
))
mse_test = mean_squared_error(
    real_prices_test.flatten()[:min_length], 
    historical_forecast.flatten()[:min_length]
)
mape_test = mean_absolute_percentage_error(
    real_prices_test.flatten()[:min_length], 
    historical_forecast.flatten()[:min_length]
)

print("Метрики качества прогноза на исторических данных:")
print(f"MAE: {mae_test:.2f}")
print(f"RMSE: {rmse_test:.2f}")
print(f"MSE: {mse_test:.2f}")
print(f"MAPE: {mape_test:.2f}")

# График 1: Адекватность модели на историческом интервале
plt.figure(figsize=(16, 8))
plt.plot(real_prices_test.flatten(), label='Реальные данные', color='green')
plt.plot(historical_forecast.flatten(), label='Прогноз', color='blue')
plt.xlabel('Час')
plt.ylabel('Цена электроэнергии')
plt.title('Адекватность модели на историческом интервале')
plt.legend()
plt.grid()
plt.show()

# Остатки
# Усечение данных до минимальной длины
min_length = min(real_prices_test.flatten().shape[0], historical_forecast.flatten().shape[0])
residuals = real_prices_test.flatten()[:min_length] - historical_forecast.flatten()[:min_length]

# График 2: Остатки
plt.figure(figsize=(16, 8))
plt.plot(residuals, label='Остатки', color='purple')
plt.axhline(0, linestyle='--', color='red', label='Нулевая линия')
plt.xlabel('Час')
plt.ylabel('Остатки')
plt.title('График остатков')
plt.legend()
plt.grid()
plt.show()

# График 3: Остатки vs Прогноз
plt.figure(figsize=(16, 8))
plt.scatter(historical_forecast.flatten(), residuals, alpha=0.6, color='purple')
plt.axhline(0, linestyle='--', color='red', label='Нулевая линия')
plt.xlabel('Прогноз')
plt.ylabel('Остатки')
plt.title('Остатки vs Прогноз')
plt.legend()
plt.grid()
plt.show()


# Прогноз
last_window_data = data[-window_size:, feature_columns].reshape(1, window_size, len(feature_columns))
forecast_scaled = model.predict(last_window_data)
forecast = scaler.inverse_transform(np.concatenate([forecast_scaled.reshape(-1, 1), np.zeros((forecast_horizon, 5))], axis=1))[:, 0]

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
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MSE: {mse:.2f}")
print(f"MAPE: {mape:.2f}")