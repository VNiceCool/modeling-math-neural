'''
import pandas as pd
from datetime import datetime, timedelta

# Чтение данных из Excel-файла
excel_file = 'data/май.xlsx'
df = pd.read_excel(excel_file, header=None)  # Указать header=None, если нет заголовков

# Создание CSV-файла
csv_file = 'data/май.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    # Запись заголовков столбцов
    csvfile.write('Дата и Время,Цена\n')

    # Создание списка дат за май 2024 года
    start_date = datetime(2024, 5, 1)
    end_date = datetime(2024, 5, 31) + timedelta(days=1)
    all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days)]

    # Перебор строк (дней) и столбцов (часов)
    for i, date in enumerate(all_dates):  # Перебор дней
        for hour in range(24):  # Перебор часов
            # Создание временного метки
            time = pd.Timestamp(f"{date.strftime('%Y-%m-%d')} {hour:02d}:00:00")
            # Получение цены из текущей строки и столбца
            price = df.iloc[i, hour]
            # Запись даты/времени и цены в CSV-файл
            csvfile.write(f"{time},{price}\n")


import pandas as pd
from datetime import datetime, timedelta

# Чтение данных из Excel-файла
excel_file = 'data/июнь.xlsx'
df = pd.read_excel(excel_file, header=None)  # Указать header=None, если нет заголовков

# Создание CSV-файла
csv_file = 'data/июнь.csv'
with open(csv_file, 'w', newline='', encoding='utf-8') as csvfile:
    # Запись заголовков столбцов
    csvfile.write('Дата и Время,Цена\n')

    # Создание списка дат за июнь 2024 года
    start_date = datetime(2024, 6, 1)  # Начало июня
    end_date = datetime(2024, 6, 30) + timedelta(days=1)  # Конец июня
    all_dates = [start_date + timedelta(days=i) for i in range((end_date - start_date).days)]

    # Перебор строк (дней) и столбцов (часов)
    for i, date in enumerate(all_dates):  # Перебор дней
        for hour in range(24):  # Перебор часов
            # Создание временной метки
            time = pd.Timestamp(f"{date.strftime('%Y-%m-%d')} {hour:02d}:00:00")
            # Получение цены из текущей строки и столбца
            price = df.iloc[i, hour]
            # Запись даты/времени и цены в CSV-файл
            csvfile.write(f"{time},{price}\n")
'''
'''
import pandas as pd
import numpy as np

# Загрузка файлов
electricity_df = pd.read_csv('data/май.csv', parse_dates=['Дата_и_Время'])
gas_df = pd.read_csv('data/природный_газ.csv', parse_dates=['Дата'])
oil_df = pd.read_csv('data/нефть.csv', parse_dates=['Дата'])
usd_df = pd.read_csv('data/usd.csv', parse_dates=['Дата'])
eur_df = pd.read_csv('data/eur.csv', parse_dates=['Дата'])

# Подготовка данных для репликации по часам
gas_df['Дата'] = gas_df['Дата'].dt.date
oil_df['Дата'] = oil_df['Дата'].dt.date
usd_df['Дата'] = usd_df['Дата'].dt.date
eur_df['Дата'] = eur_df['Дата'].dt.date

# Заполнение пропущенных значений средними, округленными до 3 знаков
gas_mean = round(gas_df['Цена'].mean(), 3)
oil_mean = round(oil_df['Цена'].mean(), 3)
usd_mean = round(usd_df['Цена'].mean(), 3)
eur_mean = round(eur_df['Цена'].mean(), 3)

gas_df['Цена'] = gas_df['Цена'].fillna(gas_mean)
oil_df['Цена'] = oil_df['Цена'].fillna(oil_mean)
usd_df['Цена'] = usd_df['Цена'].fillna(usd_mean)
eur_df['Цена'] = eur_df['Цена'].fillna(eur_mean)

# Создание почасового DataFrame
electricity_df['Дата'] = electricity_df['Дата_и_Время'].dt.date
electricity_df['Час'] = electricity_df['Дата_и_Время'].dt.hour

# Слияние DataFrame
result_df = electricity_df.merge(gas_df, left_on='Дата', right_on='Дата', how='left')
result_df = result_df.merge(oil_df, on='Дата', how='left', suffixes=('_газ', '_нефть'))
result_df = result_df.merge(usd_df, on='Дата', how='left', suffixes=('', '_USD'))
result_df = result_df.merge(eur_df, on='Дата', how='left', suffixes=('', '_EUR'))

# Удаление столбца Дата
result_df = result_df.drop(columns=['Дата'])

# Сохранение результата
result_df.to_csv('data/май_merged.csv', index=False)

print("Файл успешно объединен: data/май_merged.csv")
print(f"Средняя цена газа: {gas_mean}")
print(f"Средняя цена нефти: {oil_mean}")
print(f"Средний курс USD: {usd_mean}")
print(f"Средний курс EUR: {eur_mean}")

import pandas as pd
import numpy as np

# Загрузка данных
file_path = "data/май_merged.csv"
df = pd.read_csv(file_path)

# Преобразование столбца "Дата_и_Время" в формат datetime, если это не сделано
df['Дата_и_Время'] = pd.to_datetime(df['Дата_и_Время'])

# Добавление синусоидальных признаков для "Час"
df['sin_час'] = np.sin(2 * np.pi * df['Час'] / 24)
df['cos_час'] = np.cos(2 * np.pi * df['Час'] / 24)

# Удаление столбца "Час"
df = df.drop(columns=['Час'])

# Сохранение изменений в файл
df.to_csv(file_path, index=False)

import pandas as pd

# Загрузка данных
file_path = "data/май_merged.csv"
df = pd.read_csv(file_path)

# Преобразование столбца "Дата_и_Время" в формат datetime
df['Дата_и_Время'] = pd.to_datetime(df['Дата_и_Время'])

# Добавление столбца "День"
df['День'] = df['Дата_и_Время'].dt.day

# Сохранение изменений в файл
df.to_csv(file_path, index=False)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Загрузка данных
df = pd.read_csv('data/май_merged.csv', parse_dates=['Дата_и_Время'])

# Расчет корреляционной матрицы
correlation_matrix = df[['Цена_ЭЭ', 'Цена_газ', 'Цена_нефть', 'Цена_USD', 'Цена_EUR', 'sin_час', 'cos_час', 'День']].corr()

# Визуализация корреляционной матрицы
plt.figure(figsize=(14, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Корреляционная матрица признаков')
plt.tight_layout()
plt.show()

# Вывод корреляции с ценой на электроэнергию
print("\nКорреляция с ценой на электроэнергию:")
features = ['Цена_газ', 'Цена_нефть', 'Цена_USD', 'Цена_EUR', 'sin_час', 'cos_час', 'День']
for column in features:
   correlation = df['Цена_ЭЭ'].corr(df[column])
   print(f"{column}: {correlation:.4f}")


import pandas as pd
import numpy as np

# Загрузка данных за июнь
df_june = pd.read_csv('data/июнь.csv', parse_dates=['Дата_и_Время'])

# Извлечение часа
df_june['Час'] = df_june['Дата_и_Время'].dt.hour

# Добавление циклических признаков времени
df_june['sin_час'] = np.sin(2 * np.pi * df_june['Час'] / 24)
df_june['cos_час'] = np.cos(2 * np.pi * df_june['Час'] / 24)

# Добавление признака дня месяца
df_june['День'] = df_june['Дата_и_Время'].dt.day

# Сохранение обновленного файла
df_june.to_csv('data/июнь_merged.csv', index=False)



import pandas as pd

# Загрузка данных
data_merged = pd.read_csv("data/май_merged.csv")

# Преобразование столбца 'Дата_и_Время' в формат datetime
data_merged['Дата_и_Время'] = pd.to_datetime(data_merged['Дата_и_Время'])

# Создание новых столбцов
data_merged['Час'] = data_merged['Дата_и_Время'].dt.hour
data_merged['День'] = data_merged['Дата_и_Время'].dt.day
data_merged['Месяц'] = data_merged['Дата_и_Время'].dt.month


# Просмотр первых строк
print(data_merged.head())


import pandas as pd
import numpy as np

# Загрузка данных
file_path = 'data/EUR_RUB_240501_240531.csv'
data = pd.read_csv(file_path, delimiter=';')

# Оставим только нужные столбцы
data = data[['DATE', 'TIME', 'CLOSE']]

# Установим временные рамки
start_date = '2024-05-01'
end_date = '2024-05-31'

# Создадим полный диапазон дат и времени
dates = pd.date_range(start=start_date, end=end_date)
times = pd.date_range('00:00:00', '23:00:00', freq='H').time

# Создание полного датафрейма
full_data = pd.DataFrame([(date.date(), time) for date in dates for time in times], 
                         columns=['DATE', 'TIME'])

# Преобразуем DATE и TIME в такой же формат, как в исходных данных
full_data['DATE'] = full_data['DATE'].astype(str)
full_data['TIME'] = full_data['TIME'].astype(str)

# Слияние с исходными данными
merged_data = full_data.merge(data, on=['DATE', 'TIME'], how='left')

# Расчет глобального среднего
global_mean_close = data['CLOSE'].mean()

# Заполнение пропусков
def fill_missing_values(group):
    # Если все значения NaN
    if group['CLOSE'].isna().all():
        group['CLOSE'] = global_mean_close
    else:
        # Сколько непустых значений в группе
        non_null_count = group['CLOSE'].notna().sum()
        
        if non_null_count >= 2:
            # Если 2 и более значений - заполняем средним по дню
            day_mean = group['CLOSE'].mean()
            group['CLOSE'] = group['CLOSE'].fillna(day_mean)
        else:
            # Если меньше 2 значений - заполняем глобальным средним
            group['CLOSE'] = group['CLOSE'].fillna(global_mean_close)
    
    return group

# Применяем функцию заполнения к каждому дню
filled_data = merged_data.groupby('DATE', group_keys=False).apply(fill_missing_values)

# Сортировка по дате и времени
filled_data = filled_data.sort_values(['DATE', 'TIME'])

# Сохраняем результат
output_file = 'data/EURRUB_filled.csv'
filled_data.to_csv(output_file, index=False, sep=';')



import pandas as pd

# Путь к данным
base_file = "data/май_merged.csv"
files_to_merge = {
    "Цена_EUR_NEW": "data/EURRUB_filled.csv",
    "Цена_USD_NEW": "data/USDRUB_filled.csv",
    "Цена_газ_NEW": "data/Газ_filled.csv",
    "Цена_нефть_NEW": "data/Нефть_filled.csv"
}

# Загрузка основного файла
df_main = pd.read_csv(base_file)

# Преобразование даты и времени в основной таблице
df_main["Дата_и_Время"] = pd.to_datetime(df_main["Дата_и_Время"])

# Функция для загрузки и обработки дополнительных данных
def load_and_process(file_path, column_name):
    df = pd.read_csv(file_path, sep=";")
    df["Дата_и_Время"] = pd.to_datetime(df["DATE"] + " " + df["TIME"])
    df = df[["Дата_и_Время", "CLOSE"]].rename(columns={"CLOSE": column_name})
    return df

# Добавление новых столбцов
for new_column, file_path in files_to_merge.items():
    df_to_merge = load_and_process(file_path, new_column)
    df_main = pd.merge(df_main, df_to_merge, on="Дата_и_Время", how="left")

# Удаление старых столбцов
columns_to_drop = ["Цена_газ", "Цена_нефть", "Цена_USD", "Цена_EUR"]
df_main = df_main.drop(columns=columns_to_drop)

# Сохранение результата
output_file = "data/май_merged_updated.csv"
df_main.to_csv(output_file, index=False)

print(f"Обновленный файл сохранен как {output_file}")

'''
import pandas as pd

# Загрузка данных
data_path = "data/май_smoothed.csv"
df = pd.read_csv(data_path)

# Преобразование даты в datetime
df['Дата_и_Время'] = pd.to_datetime(df['Дата_и_Время'])

# Функция для сглаживания выбросов
def smooth_outliers(df, start_hour, end_hour):
    # Получение индексов строк в указанном диапазоне
    start_time = df['Дата_и_Время'].iloc[0] + pd.Timedelta(hours=start_hour)
    end_time = df['Дата_и_Время'].iloc[0] + pd.Timedelta(hours=end_hour)
    mask = (df['Дата_и_Время'] >= start_time) & (df['Дата_и_Время'] <= end_time)
    
    # Рассчет среднего значения
    mean_value = df.loc[mask, 'Цена_ЭЭ'].mean()
    
    # Замена значений в диапазоне на среднее
    df.loc[mask, 'Цена_ЭЭ'] = mean_value

# Применение сглаживания для трех интервалов
smooth_outliers(df, 724, 741)


# Сохранение результата
output_path = "data/май_smoothed_2.csv"
df.to_csv(output_path, index=False)
'''
import pandas as pd

result_df = pd.read_csv("data/май_smoothed.csv")
result_df['Дата_и_Время'] = pd.to_datetime(result_df['Дата_и_Время'])
# Создание новых столбцов
result_df['Час'] = result_df['Дата_и_Время'].dt.hour
result_df['День'] = result_df['Дата_и_Время'].dt.day
result_df['Месяц'] = result_df['Дата_и_Время'].dt.month

result_df.to_csv('data/май_asdfasfd.csv', index=False)
'''