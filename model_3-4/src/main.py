import pandas as pd
import numpy as np
import warnings
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error


# Отключаем предупреждения
warnings.filterwarnings('ignore')

def load_and_prepare_data(file_path, smooth_outliers=False):
    # Загрузка данных
    data = pd.read_csv(file_path, parse_dates=['Дата'], dayfirst=True)
    data = data.sort_values('Дата')
    data.set_index('Дата', inplace=True)
    data['Цена'] = pd.to_numeric(data['Цена'], errors='coerce')
    data.dropna(subset=['Цена'], inplace=True)
    
    if smooth_outliers:
        # Определяем период выбросов
        outlier_mask = (data.index >= '2022-01-25') & (data.index <= '2022-04-13')
        
        # Находим значения до и после периода выбросов
        value_before = data.loc[data.index < '2022-01-25', 'Цена'].iloc[-1]
        value_after = data.loc[data.index > '2022-04-13', 'Цена'].iloc[0]
        
        # Создаем линейный переход между значениями
        outlier_period = data[outlier_mask].copy()
        num_points = len(outlier_period)
        
        # Создаем плавный переход между значениями до и после выбросов
        smooth_values = np.linspace(value_before, value_after, num_points)
        
        # Заменяем выбросы на сглаженные значения
        data.loc[outlier_mask, 'Цена'] = smooth_values
        
        # Дополнительное сглаживание методом скользящего среднего
        # для более плавного перехода
        # лучше оставить
        window_size = 7
        data.loc[outlier_mask, 'Цена'] = (
            data.loc[outlier_mask, 'Цена']
            .rolling(window=window_size, center=True)
            .mean()
            .fillna(method='ffill')
            .fillna(method='bfill')
        )
        
    return data

def test_stationarity(timeseries, diff_step=0):
    result = adfuller(timeseries)
    print(f'Test Statistic: {result[0]}')
    print(f'P-value: {result[1]}')
    print(f'Lags used: {result[2]}')
    print(f'Number of observations: {result[3]}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'\t{key}: {value}')

    if result[1] > 0.05:
        if diff_step == 0:
            print("Историческая выборка не стационарна.")
        else:
            print(f"Ряд остается нестационарным после {diff_step} раз дифференцирования.")
    else:
        if diff_step == 0:
            print("Историческая выборка стационарна.")
        else:
            print(f"Временный ряд стационарен после {diff_step} раз дифференцирования.")

    print('\n')
    return result[1]

def display_arima_formula(model_fit, order, data):
    p, d, q = order
    
    # Пересчет среднего значения ряда для оценки уровня
    mean_y = data.mean()
    
    # Явное добавление тренда 'c' в модель ARIMA при интеграции
    if 'const' in model_fit.params:
        mu = model_fit.params['const']
    else:
        mu = mean_y  # Используем среднее значение ряда, если 'const' не найден
        
    # Получаем части AR и MA, заполняя отсутствующие коэффициенты нулями
    ar_params = [model_fit.params.get(f'ar.L{i}', 0) for i in range(1, p+1)]
    ma_params = [model_fit.params.get(f'ma.L{i}', 0) for i in range(1, q+1)]

    # Формируем части уравнения с оператором Δ^d, отображая все члены
    ar_terms = " + ".join([f"{ar:.3f}*Δ^{d}Y(t-{i})" for i, ar in enumerate(ar_params, 1)])
    ma_terms = " + ".join([f"{ma:.3f}*Δ^{d}e(t-{i})" for i, ma in enumerate(ma_params, 1)])

    # Добавляем интегрированную часть
    integrated_term = f"Δ^{d}Y(t)" if d > 0 else "Y(t)"

    # Собираем полное уравнение
    equation = f"{integrated_term} = {mu:.3f} + {ar_terms} + {ma_terms} + e(t)"
    
    print(f"ARIMA модель (p={p}, d={d}, q={q}):")
    print(equation)
    print(f"Среднее значение временного ряда: {mean_y:.3f}")



def simple_check_residuals(model_fit):
    residuals = model_fit.resid

    # Проверка на нулевое среднее значение остатков
    mean_resid = np.mean(residuals)
    print(f"Среднее значение остатков: {mean_resid:.5f} (должно быть близко к 0)")

    # Визуальный анализ равномерности разброса
    plt.figure(figsize=(12, 6))
    
    plt.plot(residuals, label='Остатки')
    plt.axhline(y=0, color='red', linestyle='--')
    plt.title("Проверка остатков")
    plt.xlabel("Время")
    plt.ylabel("Остатки")
    plt.legend()
    plt.show()

    # Сообщение о визуальном анализе
    if abs(mean_resid) < 0.05:
        print("Среднее значение остатков близко к 0, всё в порядке.")
    else:
        print("Среднее значение остатков значительно отличается от 0, стоит проверить модель.")

    print("Если остатки выглядят равномерно, без видимых увеличений разброса, то можно считать, что дисперсия постоянна.")

def plot_acf_pacf(data, title):
    #Функция для построения графиков АКФ и ЧАФК
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plot_acf(data, lags=30, ax=plt.gca())
    plt.subplot(122)
    plot_pacf(data, lags=30, ax=plt.gca())
    plt.suptitle(title)
    plt.show()

def make_stationary(data, max_diff=4):
    plot_acf_pacf(data, 'АКФ и ЧАФК для исходного временного ряда')

    p_value = test_stationarity(data)
    diff = 0
    while p_value > 0.05 and diff < max_diff:
        diff += 1
        data = data.diff().dropna()
        p_value = test_stationarity(data, diff)
        
        # Построение АФК и ЧАФК на каждом шаге дифференцирования
        plot_acf_pacf(data, f'АКФ и ЧАФК для дифференцированного ряда (Шаг {diff})')

    return data, diff

def evaluate_arima_model(data, arima_order):
    model = ARIMA(data, order=arima_order)
    model_fit = model.fit()
    aic = model_fit.aic
    print(f'Model: ARIMA{arima_order}, AIC: {aic}')
    return aic

def find_best_arima_params(data, p_values, q_values, d):
    best_aic = np.inf
    best_order = None
    for p in p_values:
        for q in q_values:
            try:
                aic = evaluate_arima_model(data, (p, d, q))
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
            except:
                continue
    return best_order, best_aic



##########################################################################################################################
########################################################## MAIN ##########################################################
##########################################################################################################################
def main():

    # 1. Загрузка и подготовка данных
    # training_data = load_and_prepare_data('data/start2021_end2023.csv')
    training_data = load_and_prepare_data('data/start2021_end2023.csv', smooth_outliers=True)
    # для сглаживания значений с 2022-01-25 по 2022-04-13

    mean_price = training_data['Цена'].mean()
    max_price = training_data['Цена'].max()
    min_price = training_data['Цена'].min()
    print('\n')
    print(f'Среднее значение исторического ряда: {mean_price:.3f}')
    print(f'Максимальное значение исторического ряда: {max_price:.3f}')
    print(f'Минимальное значение исторического ряда: {min_price:.3f}')

    # Визуализация временного ряда
    plt.figure(figsize=(10, 6))
    plt.plot(training_data['Цена'], label='CNY/RUB Exchange Rate')
    plt.title('Колебания валютного курса CNY/RUB')
    plt.xlabel('Дата')
    plt.ylabel('Цена')
    plt.legend()
    plt.show()

    # 2. Проверка на стационарность и выбор параметра d
    stationary_data, d = make_stationary(training_data['Цена'])

    # 3. Поиск оптимальных значений параметров p и q
    # p_values = range(1, 7)
    # q_values = range(0, 4)
    p_values = list(range(1, 20))
    q_values = list(range(0, 10))
    best_order, best_aic = find_best_arima_params(training_data['Цена'], p_values, q_values, d)
    print('\n')
    print(f'Оптимальные параметры ARIMA: {best_order}, AIC: {best_aic}')

    # 4. Обучение модели и диагностика
    model = ARIMA(training_data['Цена'], order=best_order)
    model_fit = model.fit()
    print('\n')
    display_arima_formula(model_fit, best_order, training_data['Цена'])
    simple_check_residuals(model_fit)
    print('\n')

    # Диагностика модели
    print(model_fit.summary())
    model_fit.plot_diagnostics(figsize=(12, 8))
    plt.show()


    # 5. Тестирование модели
    pred_start = training_data.index[0]
    pred_end = training_data.index[-1]
    pred = model_fit.get_prediction(start=pred_start, end=pred_end)
    predicted_values = pred.predicted_mean
    true_values = training_data['Цена']

    # Оценка модели
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(true_values, predicted_values)

    print('\n')
    print("Проверка адекватности модели:")
    print(f'MSE: {mse}')
    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape}')

    # Визуализация прогноза на исторических данных
    plt.figure(figsize=(10, 6))
    plt.plot(true_values, label='Истинные значения')
    plt.plot(predicted_values, label='Прогнозные значения', color='red')
    plt.title('Прогноз на исторических данных')
    plt.xlabel('Дата')
    plt.ylabel('Цена')
    plt.legend()
    plt.show()

    # 6. Прогнозирование на 3 месяца вперыед
    forecast = model_fit.get_forecast(steps=60)  # Прогноз на 60 дней
    forecast_values = forecast.predicted_mean

    # 7. Сохранение результатов в файл
    forecast_df = pd.DataFrame({'Дата': pd.date_range(start='2024-01-01', periods=60, freq='D'),
                                'Прогноз': forecast_values})
    forecast_df.to_csv('results/result.csv', index=False)

    # Загрузка данных для сравнения с прогнозом
    comparison_data = load_and_prepare_data('data/data_for_comparison_with_forecast.csv')

    # Визуализация прогнозов и данных
    plt.figure(figsize=(10, 6))
    plt.plot(training_data['Цена'], label='Исторические данные (CNY/RUB)', color='blue')
    plt.plot(comparison_data.index, comparison_data['Цена'], label='Реальные данные (CNY/RUB)', color='green')
    plt.plot(forecast_df['Дата'], forecast_df['Прогноз'], label='Прогнозные значения', color='red')
    plt.title('Сравнение прогноза с реальными данными')
    plt.xlabel('Дата')
    plt.ylabel('Цена')
    plt.legend()
    plt.show()

    # Оценка прогноза
    mse_forecast = mean_squared_error(comparison_data['Цена'], forecast_values[:len(comparison_data)])
    rmse_forecast = np.sqrt(mse_forecast)
    mape_forecast = mean_absolute_percentage_error(comparison_data['Цена'], forecast_values[:len(comparison_data)])

    print('\n')
    print(f'Ошибка прогноза MSE: {mse_forecast}')
    print(f'Ошибка прогноза RMSE: {rmse_forecast}')
    print(f'Ошибка прогноза MAPE: {mape_forecast}')


if __name__ == "__main__":
    main()