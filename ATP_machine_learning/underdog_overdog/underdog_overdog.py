import pandas as pd
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
import pyarrow
import multiprocessing as mp
import tqdm
import copy


def one_hot(start,end):
    #start,end specify which rows the core should perform the function on
    df = copy.deepcopy(data.loc[start:end,:])
    col_list = ['winner_id','winner_seed',
               'winner_entry','winner_name','winner_hand',
               'winner_ht','winner_ioc','winner_age',
               'loser_id','loser_seed','loser_entry','loser_name',
               'loser_hand','loser_ht','loser_ioc','loser_age',
               'w_ace','w_df','w_svpt','w_1stIn','w_1stWon',
                'w_2ndWon','w_SvGms','w_bpSaved','w_bpFaced',
            'l_ace','l_df',
    'l_svpt',
    'l_1stIn',
    'l_1stWon',
    'l_2ndWon',
    'l_SvGms',
    'l_bpSaved',
    'l_bpFaced',
    'winner_rank',
    'winner_rank_points',
    'loser_rank',
    'loser_rank_points']

    
    #renaming features based on overdog vs underdog
    new_col_list = []

    for c in col_list:
        if c.startswith('winner'):
            new_col_name = 'overdog'+c.split('winner')[1]
        elif c.startswith('w'):
            new_col_name = 'overdog'+c.split('w')[1]
        elif c.startswith('loser'):
            new_col_name = 'underdog' + c.split('loser')[1]
        elif c.startswith('l'):
            new_col_name = 'underdog' + c.split('l')[1]
        df[new_col_name] = np.nan
        new_col_list.append(new_col_name)



    ind = df.index


    count = 0

    df['Winner'] = np.nan
    bad_count = 0
    for k in ind:
        row = df.loc[k]
        if math.isnan(row['winner_rank']) or math.isnan(row['loser_rank']):
            bad_count += 1
            print("BAD")
            continue
        elif row['winner_rank']>row['loser_rank']:   #upset happened
            for i,val in enumerate(col_list):
                if val.startswith('w'):
                    try:
                        new_col_name = 'underdog'+val.split('winner')[1]
                    except:
                        new_col_name = 'underdog'+val.split('w')[1]
                    row[new_col_name] = row[val]
                else:
                    try:
                        new_col_name = 'overdog'+val.split('loser')[1]
                    except:
                        new_col_name = 'overdog'+val.split('l')[1]
                    row[new_col_name] = row[val]
            row['Winner'] = 1
        else:                               #upset did not happen
            for i,val in enumerate(col_list):
                print(val)
                if val.startswith('l'):
                    try:
                        new_col_name = 'underdog'+val.split('loser')[1]
                    except:
                        new_col_name = 'underdog'+val.split('l')[1]
                    row[new_col_name] = row[val]
                else:
                    try:
                        new_col_name = 'overdog'+val.split('winner')[1]
                    except:
                        new_col_name = 'overdog'+val.split('w')[1]
                    row[new_col_name] = row[val]
            row['Winner'] = 0
        print(row)
        count += 1
        print(f'This many done {count} out of {len(df)}')
        df.loc[k] = row
    return df


if __name__ == "__main__":
    datadir = '../rawdata/'
    files = os.listdir(datadir)

    files = [f for f in files if 'atp_matches_' in f]
    df_list = []


    for f in files:
        df = pd.read_csv(os.path.join(datadir,f))
        # print(df.head(10))
        df_list.append(df)


    data = pd.concat(df_list)

    data['Year'] = data['tourney_date']//10000
    data['Date'] = data['tourney_date'].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
    data.sort_values(by=['Date'],inplace=True)

    data = data.reset_index(drop=True)

    NUM_CORES = 14
    results = []
    pool = mp.Pool(processes=NUM_CORES)
    num_bundles = len(data)
    tasks = []
    current_task = 0
    i = 0
    row_per_core = num_bundles//(NUM_CORES-1)

    #splitting up tasks into different cores
    for i in range(NUM_CORES-1):
        tasks.append((i*row_per_core,(i+1)*row_per_core-1))

    tasks.append(((NUM_CORES-1)*row_per_core,num_bundles-1))
    for task_list in tasks:
        print(task_list)
        results.append( pool.apply_async( one_hot, args=(task_list) ) )
    outputs = [ (p.get()) for p in results ]
    data2 = pd.concat(outputs)

    print(data2)
    data2.astype({'winner_seed': str,'loser_seed': str,'overdog_seed':str,'underdog_seed':str}).to_parquet('data_one_hot.parquet')

    pool.join()
    pool.close()
