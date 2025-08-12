# C:\Users\Alkor\PycharmProjects\beget_rss\beget\sync_files.ps1

# Скрипт для синхронизации файлов .db и .log с удалённого сервера на локальную машину
# Проверяем, существует ли директория для логов, и создаём её, если отсутствует
$logDir = "C:\Users\Alkor\gd\data_beget_rss\log"
if (-not (Test-Path -Path $logDir)) {
    New-Item -Path $logDir -ItemType Directory -Force
}

# Получаем текущую дату и время для логов
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Выполняем синхронизацию .db файлов
"[$timestamp] Sync .db files" | Out-File -FilePath C:\Users\Alkor\gd\data_beget_rss\log\sync.log -Append -Encoding UTF8
wsl rsync -avz --include="*/" --include="**/*.db" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/db_data/ /mnt/c/Users/Alkor/gd/data_beget_rss/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\data_beget_rss\log\sync.log -Append -Encoding UTF8

# Выполняем синхронизацию .log файлов
"[$timestamp] Sync .log files" | Out-File -FilePath C:\Users\Alkor\gd\data_beget_rss\log\sync.log -Append -Encoding UTF8
wsl rsync -avz --include="*/" --include="**/*.log" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/log/ /mnt/c/Users/Alkor/gd/data_beget_rss/log/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\data_beget_rss\log\sync.log -Append -Encoding UTF8