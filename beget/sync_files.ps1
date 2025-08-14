# C:\Users\Alkor\PycharmProjects\beget_rss\beget\sync_files.ps1

# Скрипт для синхронизации файлов .db и .log с удалённого сервера на локальную машину
# Проверяем, существует ли директория для логов, и создаём её, если отсутствует
$logDir = "C:\Users\Alkor\gd\db_rss_investing\log"
if (-not (Test-Path -Path $logDir)) {
    New-Item -Path $logDir -ItemType Directory -Force
}

# Получаем текущую дату и время для логов
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Выполняем синхронизацию .db файлов investing
"[$timestamp] Sync .db files" | Out-File -FilePath C:\Users\Alkor\gd\db_rss_investing\log\sync.log -Encoding UTF8
wsl rsync -avz --include="*/" --include="**/*.db" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/db_rss_investing/ /mnt/c/Users/Alkor/gd/db_rss_investing/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\db_rss_investing\log\sync.log -Append -Encoding UTF8
# Выполняем синхронизацию .log файлов investing
"[$timestamp] Sync .log files" | Out-File -FilePath C:\Users\Alkor\gd\db_rss_investing\log\sync.log -Append -Encoding UTF8
wsl rsync -avz --include="rss_scraper_investing_month*.log" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/log/ /mnt/c/Users/Alkor/gd/db_rss_investing/log/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\db_rss_investing\log\sync.log -Append -Encoding UTF8

# Выполняем синхронизацию .db файлов interfax
"[$timestamp] Sync .db files" | Out-File -FilePath C:\Users\Alkor\gd\db_rss_interfax\log\sync.log -Encoding UTF8
wsl rsync -avz --include="*/" --include="**/*.db" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/db_rss_interfax/ /mnt/c/Users/Alkor/gd/db_rss_interfax/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\db_rss_interfax\log\sync.log -Append -Encoding UTF8
# Выполняем синхронизацию .log файлов interfax
"[$timestamp] Sync .log files" | Out-File -FilePath C:\Users\Alkor\gd\db_rss_interfax\log\sync.log -Append -Encoding UTF8
wsl rsync -avz --include="rss_scraper_interfax_month*.log" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/log/ /mnt/c/Users/Alkor/gd/db_rss_interfax/log/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\db_rss_interfax\log\sync.log -Append -Encoding UTF8

# Выполняем синхронизацию .db файлов prime
"[$timestamp] Sync .db files" | Out-File -FilePath C:\Users\Alkor\gd\db_rss_prime\log\sync.log -Encoding UTF8
wsl rsync -avz --include="*/" --include="**/*.db" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/db_rss_prime/ /mnt/c/Users/Alkor/gd/db_rss_prime/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\db_rss_prime\log\sync.log -Append -Encoding UTF8
# Выполняем синхронизацию .log файлов prime
"[$timestamp] Sync .log files" | Out-File -FilePath C:\Users\Alkor\gd\db_rss_prime\log\sync.log -Append -Encoding UTF8
wsl rsync -avz --include="rss_scraper_prime_month*.log" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/log/ /mnt/c/Users/Alkor/gd/db_rss_prime/log/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\db_rss_prime\log\sync.log -Append -Encoding UTF8
