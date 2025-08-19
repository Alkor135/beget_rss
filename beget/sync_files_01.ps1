# C:\Users\Alkor\PycharmProjects\beget_rss\beget\sync_files_01.ps1

# Скрипт для синхронизации файлов .db и .log с удалённого сервера на локальную машину
# и запуска Python-скриптов из виртуального окружения

# Строки для планировщика задач
# C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe
# -NoProfile -NonInteractive -WindowStyle Hidden -ExecutionPolicy Bypass -File C:\Users\Alkor\PycharmProjects\beget_rss\beget\sync_files_01.ps1
# C:\Users\Alkor\PycharmProjects\beget_rss\rts

# Получаем текущую дату и время для логов
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"

# Путь к интерпретатору Python в виртуальном окружении
$pythonExe = "C:\Users\Alkor\PycharmProjects\beget_rss\.venv\Scripts\python.exe"

# Проверка существования интерпретатора Python
if (-not (Test-Path -Path $pythonExe)) {
    $errorMsg = "[$timestamp] Ошибка: Интерпретатор Python не найден в $pythonExe"
    Write-Output $errorMsg
    $errorMsg | Out-File -FilePath C:\Users\Alkor\gd\db_rss_investing\log\sync.log -Encoding UTF8
    exit 1
}

# Список Python-скриптов для запуска
$pythonScripts = @(
    "C:\Users\Alkor\PycharmProjects\beget_rss\rts\rts_download_minutes_to_db.py",
    "C:\Users\Alkor\PycharmProjects\beget_rss\rts\rts_21_00_convert_minutes_to_days.py",
    "C:\Users\Alkor\PycharmProjects\beget_rss\rts\rts_21_00_db_investing_month_to_md.py",
    "C:\Users\Alkor\PycharmProjects\beget_rss\rts\predict_next_session_investing_ollama.py",
    "C:\Users\Alkor\PycharmProjects\beget_rss\rts\backtesting_investing_ollama.py"
)

# Выполняем синхронизацию .db файлов investing
# Проверяем, существует ли директория для логов, и создаём её, если отсутствует
$logDir = "C:\Users\Alkor\gd\db_rss_investing\log"
if (-not (Test-Path -Path $logDir)) {New-Item -Path $logDir -ItemType Directory -Force}
"[$timestamp] Sync .db files (investing)" | Out-File -FilePath C:\Users\Alkor\gd\db_rss_investing\log\sync.log -Encoding UTF8
wsl rsync -avz --include="*/" --include="**/*.db" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/db_rss_investing/ /mnt/c/Users/Alkor/gd/db_rss_investing/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\db_rss_investing\log\sync.log -Append -Encoding UTF8
# Выполняем синхронизацию .log файлов investing
"[$timestamp] Sync .log files (investing)" | Out-File -FilePath C:\Users\Alkor\gd\db_rss_investing\log\sync.log -Append -Encoding UTF8
wsl rsync -avz --include="rss_scraper_investing_month*.log" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/log/ /mnt/c/Users/Alkor/gd/db_rss_investing/log/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\db_rss_investing\log\sync.log -Append -Encoding UTF8

# Выполняем синхронизацию .db файлов interfax
# Проверяем, существует ли директория для логов, и создаём её, если отсутствует
$logDir = "C:\Users\Alkor\gd\db_rss_interfax\log"
if (-not (Test-Path -Path $logDir)) {New-Item -Path $logDir -ItemType Directory -Force}
"[$timestamp] Sync .db files (interfax)" | Out-File -FilePath C:\Users\Alkor\gd\db_rss_interfax\log\sync.log -Encoding UTF8
wsl rsync -avz --include="*/" --include="**/*.db" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/db_rss_interfax/ /mnt/c/Users/Alkor/gd/db_rss_interfax/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\db_rss_interfax\log\sync.log -Append -Encoding UTF8
# Выполняем синхронизацию .log файлов interfax
"[$timestamp] Sync .log files (interfax)" | Out-File -FilePath C:\Users\Alkor\gd\db_rss_interfax\log\sync.log -Append -Encoding UTF8
wsl rsync -avz --include="rss_scraper_interfax_month*.log" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/log/ /mnt/c/Users/Alkor/gd/db_rss_interfax/log/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\db_rss_interfax\log\sync.log -Append -Encoding UTF8

# Выполняем синхронизацию .db файлов prime
# Проверяем, существует ли директория для логов, и создаём её, если отсутствует
$logDir = "C:\Users\Alkor\gd\db_rss_prime\log"
if (-not (Test-Path -Path $logDir)) {New-Item -Path $logDir -ItemType Directory -Force}
"[$timestamp] Sync .db files (prime)" | Out-File -FilePath C:\Users\Alkor\gd\db_rss_prime\log\sync.log -Encoding UTF8
wsl rsync -avz --include="*/" --include="**/*.db" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/db_rss_prime/ /mnt/c/Users/Alkor/gd/db_rss_prime/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\db_rss_prime\log\sync.log -Append -Encoding UTF8
# Выполняем синхронизацию .log файлов prime
"[$timestamp] Sync .log files (prime)" | Out-File -FilePath C:\Users\Alkor\gd\db_rss_prime\log\sync.log -Append -Encoding UTF8
wsl rsync -avz --include="rss_scraper_prime_month*.log" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/log/ /mnt/c/Users/Alkor/gd/db_rss_prime/log/ | ForEach-Object { "[$timestamp] $_" } | Out-File -FilePath C:\Users\Alkor\gd\db_rss_prime\log\sync.log -Append -Encoding UTF8

# Запуск Python-скриптов из виртуального окружения
$logDir = "C:\Users\Alkor\gd\predict_ai\rts_investing_ollama\log"
foreach ($script in $pythonScripts) {
    if (-not (Test-Path -Path $script)) {
        $errorMsg = "[$timestamp] Ошибка: Скрипт $script не найден"
        Write-Output $errorMsg
        $errorMsg | Out-File -FilePath $logDir\sync.log -Append -Encoding UTF8
        continue
    }

    $scriptName = [System.IO.Path]::GetFileName($script)
    $scriptLog = "$logDir\$($scriptName -replace '\.py$', '.log')"
    "[$timestamp] Запуск $scriptName" | Out-File -FilePath $logDir\sync.log -Append -Encoding UTF8

    try {
        & $pythonExe $script 2>&1 | ForEach-Object { "[$timestamp] [$scriptName] $_" } | Out-File -FilePath $scriptLog -Encoding UTF8
        $successMsg = "[$timestamp] $scriptName успешно выполнен"
        Write-Output $successMsg
        $successMsg | Out-File -FilePath $logDir\sync.log -Append -Encoding UTF8
    } catch {
        $errorMsg = "[$timestamp] Ошибка при выполнении $scriptName : $_"
        Write-Output $errorMsg
        $errorMsg | Out-File -FilePath $logDir\sync.log -Append -Encoding UTF8
    }
}