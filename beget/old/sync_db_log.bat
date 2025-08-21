@echo off
REM Убедитесь, что у вас установлен rsync и WSL (Windows Subsystem for Linux)
REM Синхронизация файлов БД и логов с удаленного сервера
:: wsl rsync -avz --include="rss_news_investing_*_*.db" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/db_data/ /mnt/c/Users/Alkor/gd/data_beget_rss/

REM Синхронизация всех файлов БД и логов с удаленного сервера
:: wsl rsync -avz --delete root@109.172.46.10:/home/user/rss_scraper/db_data/ /mnt/c/Users/Alkor/gd/data_beget_rss/

REM Если вы хотите синхронизировать только определенные файлы, например, файлы с расширением .db и .log, используйте следующую команду:
:: wsl rsync -avz --delete --include="*/" --include="*.db" --include="*.log" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/db_data/ /mnt/c/Users/Alkor/gd/data_beget_rss/

REM Синхронизация всех файлов БД и логов с удаленного сервера
:: wsl rsync -avz --delete --include="*/" --include="**/*.db" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/db_data/ /mnt/c/Users/Alkor/gd/data_beget_rss/
:: wsl rsync -avz --delete --include="*/" --include="**/*.log" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/log/ /mnt/c/Users/Alkor/gd/data_beget_rss/log/

echo [%date% %time%] Sync .db files >> C:\Users\Alkor\gd\data_beget_rss\log\sync.log
wsl rsync -avz --include="*/" --include="**/*.db" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/db_data/ /mnt/c/Users/Alkor/gd/data_beget_rss/ >> C:\Users\Alkor\gd\data_beget_rss\log\sync.log 2>&1
echo [%date% %time%] Sync .log files >> C:\Users\Alkor\gd\data_beget_rss\log\sync.log
wsl rsync -avz --include="*/" --include="**/*.log" --exclude="*" root@109.172.46.10:/home/user/rss_scraper/log/ /mnt/c/Users/Alkor/gd/data_beget_rss/log/ >> C:\Users\Alkor\gd\data_beget_rss\log\sync.log 2>&1