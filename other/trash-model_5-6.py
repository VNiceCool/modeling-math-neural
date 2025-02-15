import pandas as pd
from datetime import datetime, timedelta

def convert_excel_to_timeseries(excel_file, output_file):
    """
    Convert Excel file with hourly electricity prices to a CSV timeseries format.
    """
    # Read Excel file
    df = pd.read_excel(excel_file)
    
    # Print the first row and column names for debugging
    print("\nПервая строка данных из Excel:")
    print(df.iloc[0])
    print("\nНазвания столбцов:")
    print(df.columns.tolist())
    
    # Create empty list to store results
    timeseries_data = []
    
    # Iterate through each row (day)
    for index, row in df.iterrows():
        # Получаем дату из первого столбца
        date = pd.to_datetime(row.iloc[0])
        
        print(f"\nОбработка даты: {date}")  # Отладочная информация
        
        # Process each hour (columns 1 through 24)
        for hour in range(24):
            # Create timestamp for this hour
            timestamp = date.replace(hour=hour)
            
            # Get price value (hour + 1 because first column is date)
            try:
                price = float(row.iloc[hour + 1])
                print(f"Час {hour}: {price}")  # Отладочная информация
            except Exception as e:
                print(f"Ошибка при получении цены для часа {hour}: {str(e)}")
                continue
            
            # Append to results
            timeseries_data.append({
                'Дата_и_время': timestamp,
                'Цена': price
            })
    
    # Convert to DataFrame
    result_df = pd.DataFrame(timeseries_data)
    
    # Sort by timestamp
    result_df = result_df.sort_values('Дата_и_время')
    
    # Set timestamp as index
    result_df.set_index('Дата_и_время', inplace=True)
    
    # Save to CSV with UTF-8 encoding
    result_df.to_csv(output_file, encoding='utf-8')
    
    print(f"\nФайл успешно сохранен: {output_file}")
    return result_df

# Example usage
excel_file = 'data/1.xlsx'
output_file = 'data/123.csv'

try:
    # Сначала просто прочитаем Excel файл и посмотрим на его структуру
    raw_df = pd.read_excel(excel_file)
    print("Структура исходного Excel файла:")
    print("\nРазмеры:")
    print(raw_df.shape)
    print("\nТипы данных:")
    print(raw_df.dtypes)
    print("\nПервые несколько строк:")
    print(raw_df.head())
    
    # Теперь конвертируем данные
    print("\nНачинаем конвертацию...")
    df = convert_excel_to_timeseries(excel_file, output_file)
    
    # Print results
    print("\nРезультат конвертации (первые несколько строк):")
    print(df.head())
    print("\nРазмер итогового набора данных:")
    print(f"Количество строк: {df.shape[0]}")

except Exception as e:
    print(f"\nПроизошла ошибка: {str(e)}")
    print("\nПожалуйста, проверьте:")
    print("1. Правильность пути к файлу")
    print("2. Наличие данных во всех ячейках")
    print("3. Структуру Excel файла (первый столбец - даты, следующие 24 столбца - почасовые цены)")