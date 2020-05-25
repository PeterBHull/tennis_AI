import numpy as np
import pandas as pd
import pyarrow
import sklearn
from sklearn import preprocessing
import os

data = pd.read_parquet('./feature_engineering/datav2.parquet')
print(len(data))
year_cutoff = 1995
#choose features
data = data.loc[data['Year']>=year_cutoff]
data = data.sort_values(by=['Date', 'match_num'])
#print(len(data))
print(len(data))

feature_data = data[['Year',
'surface',
'best_of',
'overdog_rank',
'overdog_hand',
'underdog_rank',
'underdog_hand',
'overdog_h2h_wins',
'underdog_h2h_wins',
'overdog_h2h_recent_wins',
'underdog_h2h_recent_wins',
'overdog_h2h_surface_wins',
'underdog_h2h_surface_wins',
'overdog_h2h_surface_recent_wins',
'underdog_h2h_surface_recent_wins',
'overdog_gotupset',
'overdog_notupset',
'overdog_recent_gotupset',
'overdog_recent_notupset',
'underdog_upset',
'underdog_notupset',
'underdog_recent_upset',
'underdog_recent_notupset',
'total_wins_overdog',
'total_losses_overdog',
'recent_wins_overdog',
'recent_losses_overdog',
'total_wins_surface_overdog',
'total_losses_surface_overdog',
'recent_wins_surface_overdog',
'recent_losses_surface_overdog',
'total_wins_underdog',
'total_losses_underdog',
'recent_wins_underdog',
'recent_losses_underdog',
'total_wins_surface_underdog',
'total_losses_surface_underdog',
'recent_wins_surface_underdog',
'recent_losses_surface_underdog',
'overdog_ace_MVA',
'overdog_df_MVA',
'overdog_svpt_MVA',
'overdog_1stIn_MVA',
'overdog_1stWon_MVA',
'overdog_2ndWon_MVA',
'overdog_SvGms_MVA',
'overdog_bpSaved_MVA',
'overdog_bpFaced_MVA',
'underdog_ace_MVA',
'underdog_df_MVA',
'underdog_svpt_MVA',
'underdog_1stIn_MVA',
'underdog_1stWon_MVA',
'underdog_2ndWon_MVA',
'underdog_SvGms_MVA',
'underdog_bpSaved_MVA',
'underdog_bpFaced_MVA',
'Winner']]

feature_data = feature_data.dropna()


feature_data["surface"] = feature_data["surface"].astype('category').cat.codes
feature_data['overdog_hand'] = feature_data['overdog_hand'].astype('category').cat.codes
feature_data['underdog_hand'] = feature_data['underdog_hand'].astype('category').cat.codes
l1 = len(feature_data.loc[feature_data['Winner'] == 0])
l2 = len(feature_data)
print(f'If we just guessed overdog victory everytime, our accuracy would be: {l1/l2}')


#Preprocessing

length = len(feature_data)


test_data = feature_data.loc[feature_data['Year']>=2019] #Make testing data most recent!

feature_data = feature_data.drop(test_data.index)

feature_data.drop('Year', axis=1, inplace=True)
test_data.drop('Year', axis=1, inplace=True)


X_train = feature_data.loc[:, feature_data.columns != 'Winner']
Y_train = feature_data.loc[:, feature_data.columns == 'Winner']
X_test = test_data.loc[:, feature_data.columns != 'Winner']
Y_test = test_data.loc[:, feature_data.columns == 'Winner']

#test_val_percent = .90


column_list = X_train.columns.tolist()



X_train = preprocessing.scale(X_train.values)
X_test = preprocessing.scale(X_test.values)


X_train = pd.DataFrame(X_train, columns = column_list)

X_test = pd.DataFrame(X_test, columns = column_list)


print(len(X_train))
print(len(X_test))


#SAVE TO PARQUETS
X_train.to_parquet('./cleandata/trainx.parquet')
X_test.to_parquet('./cleandata/testx.parquet')
#X_val.to_parquet('./cleandata/valx.parquet')
Y_train.to_parquet('./cleandata/trainy.parquet')
Y_test.to_parquet('./cleandata/testy.parquet')
#Y_val.to_parquet('./cleandata/valy.parquet')

