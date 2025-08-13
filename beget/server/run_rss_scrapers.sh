#!/bin/bash
# Установка прав исполнения chmod +x /home/user/rss_scraper/run_rss_scrapers.sh
# или
# chown root:root /home/user/rss_scraper/run_rss_scrapers.sh
# chmod 755 /home/user/rss_scraper/run_rss_scrapers.sh
/home/user/rss_scraper/venv/bin/python /home/user/rss_scraper/rss_scraper_investing_to_db_month.py
/home/user/rss_scraper/venv/bin/python /home/user/rss_scraper/rss_scraper_investing_to_db_month_msk.py

# Скрипт для запуска RSS-скрапера и записи логов в файл /home/user/rss_scraper/scraper_log.txt
# /home/user/rss_scraper/venv/bin/python /home/user/rss_scraper/rss_scraper_investing_to_db_month.py >> /home/user/rss_scraper/scraper_log.txt 2>&1
# /home/user/rss_scraper/venv/bin/python /home/user/rss_scraper/rss_scraper_investing_to_db_month_msk.py >> /home/user/rss_scraper/scraper_log.txt 2>&1
