# beget_rss

## Скрипты для сервера beget
[main_server_beget.py](main_server_beget.py) Скачивание rss лент новостей с investing.com и помещение их в БД SQLite3.  
[quote_download_to_db.py](quote_download_to_db.py) Скачивание котировок фьючерса RTS через MOEX ISS API и помещение их в БД SQLite3.

После обновления скриптов для сервера beget их закачка на сервер:
```PowerShell
scp C:/Users/Alkor/PycharmProjects/beget_rss/main_server_beget.py root@109.172.46.10:/home/user/rss_scraper/
scp C:/Users/Alkor/PycharmProjects/beget_rss/futures_scraper.py root@109.172.46.10:/home/user/rss_scraper/
```

