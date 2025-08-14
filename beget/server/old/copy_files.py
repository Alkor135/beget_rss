import os
import shutil

# Путь к исходной и целевой папкам
source_dir = "/home/user/rss_scraper/db_data_investing"
target_dir = "/home/user/rss_scraper/db_rss_investing"

# Создаем целевую папку, если она не существует
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Получаем список всех файлов в исходной папке
for filename in os.listdir(source_dir):
    source_file = os.path.join(source_dir, filename)
    target_file = os.path.join(target_dir, filename)

    # Проверяем, что это файл, а не папка
    if os.path.isfile(source_file):
        # Копируем файл
        shutil.copy2(source_file, target_file)
        print(f"Скопирован файл: {filename}")

print("Копирование завершено!")