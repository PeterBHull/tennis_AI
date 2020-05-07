import pandas as pd
import os
import numpy as np
from datetime import datetime

datadir = '../'
files = os.listdir(datadir)

files = [f for f in files if 'atp_matches_' in f]
df_list = []

for f in files:
    df = pd.read_csv(os.path.join(datadir,f))
    # print(df.head(10))
    print(len(df))
    df_list.append(df)


def find_upsets(df):
    upsets = df.loc[(df['loser_seed']>df['winner_seed'])]
    count = pd.Series(np.arange(1, len(upsets)+1), index=upsets["Date"])
    prev1day = count.index.shift(-1, freq="D")
    prev6month = count.index.shift(-365, freq="D")
    result = count.asof(prev1day).fillna(0).values - count.asof(prev6month).fillna(0).values
    return pd.Series(result, upsets.index)

data = pd.concat(df_list)

data = data.reset_index()

data['Year'] = data['tourney_date']//10000


data['Date'] = data['tourney_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))

print(data['Date'])

# data['Date'] = datetime.strptime(data['tourney_date'], '%Y%m%d').strftime('%m/%d/%Y')

for c in data.columns:
    print(c)

data["UpsetWins"] = data.groupby("winner_id").apply(find_upsets)

print(data['tourney_date'])






