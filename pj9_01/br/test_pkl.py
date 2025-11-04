from pathlib import Path
import pandas as pd


# p = Path("partial.pkl")
# p = Path("news_embeds.pkl")
# p = Path("minutes_RTS_processed_p8000.pkl")
p = Path("combine.pkl")

dfp = pd.read_pickle(p)

# Настройки для отображения широкого df pandas
pd.options.display.width = 1200
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100

print(dfp)
