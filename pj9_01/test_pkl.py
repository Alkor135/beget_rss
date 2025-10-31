from pathlib import Path
import pandas as pd


# p = Path("partial.pkl")
p = Path("news_embeds.pkl")
dfp = pd.read_pickle(p)

# Настройки для отображения широкого df pandas
pd.options.display.width = 1200
pd.options.display.max_colwidth = 100
pd.options.display.max_columns = 100

print(dfp)
