import pyarrow
import os
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import math
import pyarrow
import multiprocessing as mp
import tqdm
import copy
import pandas as pd



def underdog_upset_num(df):
    '''
    Get how many times the underdog has successfully and unsuccesfully upset in the underdog position
    both over entire career, and over past year.
    '''
    win_count = 0
    lose_count = 0
    wins = []
    losses = []
    for i,row in df.iterrows():
        wins.append(win_count)
        losses.append(lose_count)
        if row['Winner'] == 1:
            win_count += 1
        elif row['Winner'] == 0:
            lose_count += 1

    #get only previous 365 day6s
    # count_w = pd.Series(wins, index=df["Date"])
    # count_l = pd.Series(losses,index=df["Date"])

    # prev1day = count_w.index.shift(-1, freq="D")
    # prev12month = count_w.index.shift(-365, freq="D")
    # result_w = count_w.asof(prev1day).fillna(0).values - count_w.asof(prev6month).fillna(0).values

    # prev1day = count_l.index.shift(-1, freq="D")
    # prev12month = count_l.index.shift(-365, freq="D")
    # result_l = count_l.asof(prev1day).fillna(0).values - count_l.asof(prev6month).fillna(0).values

    recent_wins = pd.DataFrame(wins,index=df.index)
    recent_losses = pd.DataFrame(losses,index=df.index)

    recent_wins = recent_wins.rolling(10,min_periods=1).max().fillna(0) - recent_wins.rolling(10,min_periods=1).min().fillna(0)
    recent_losses = recent_losses.rolling(10,min_periods=1).max().fillna(0) - recent_losses.rolling(10,min_periods=1).min().fillna(0)


    ret_val = pd.DataFrame({'underdog_upset': wins,
                            'underdog_notupset': losses,
                            'underdog_recent_upset' : recent_wins[0].astype(int),
                            'underdog_recent_notupset': recent_losses[0].astype(int)
                            },index=df.index)
    return ret_val

def overdog_getupset_num(df):
    '''
    Get how many times the overdog has successfully and unsuccesfully prevented an upset in the overdog
    position both over entire career and over past year
    '''
    win_count = 0
    lose_count = 0
    wins = []
    losses = []
    # print(df)
    # data.loc[data['underdog_rank'] == df['overdog_rank']]
    for i,row in df.iterrows():
        wins.append(win_count)
        losses.append(lose_count)
        if row['Winner'] == 0:
            win_count += 1
        elif row['Winner'] == 1:
            lose_count += 1

    #get only previous 365 day6s
    # count_w = pd.Series(wins, index=df["Date"])
    # count_l = pd.Series(losses,index=df["Date"])

    # prev1day = count_w.index.shift(-1, freq="D")
    # prev12month = count_w.index.shift(-365, freq="D")
    # result_w = count_w.asof(prev1day).fillna(0).values - count_w.asof(prev6month).fillna(0).values

    # prev1day = count_l.index.shift(-1, freq="D")
    # prev12month = count_l.index.shift(-365, freq="D")
    # result_l = count_l.asof(prev1day).fillna(0).values - count_l.asof(prev6month).fillna(0).values
    recent_wins = pd.DataFrame(wins,index=df.index)
    recent_losses = pd.DataFrame(losses,index=df.index)
    recent_wins = recent_wins.rolling(10,min_periods=1).max().fillna(0) - recent_wins.rolling(10,min_periods=1).min().fillna(0)
    recent_losses = recent_losses.rolling(10,min_periods=1).max().fillna(0) - recent_losses.rolling(10,min_periods=1).min().fillna(0)
    
    ret_val = pd.DataFrame({'overdog_gotupset': losses,
                            'overdog_notupset': wins, 
                            'overdog_recent_gotupset': recent_losses[0].astype(int),
                            'overdog_recent_notupset': recent_wins[0].astype(int)}
                            ,index=df.index)
    # print('ret_val',ret_val)
    # print('df',df)
    return ret_val



def previous_head2head(df,surface_flag = False):
    '''
    Get previous head2head between two players both in career and over past 365 days
    '''
    overdog_wins = []
    underdog_wins = []
    
    f_wins = [0]
    s_wins = [0]

    # f_wins = 0
    # s_wins = 0
    recent_overdog = []
    recent_underdog = []

    for i,row in df.iterrows():
        if row['winner_id'] == row['overdog_id']:
            if row['winner_rank']>row['loser_rank']:
                underdog_wins.append(f_wins[-1])
                overdog_wins.append(s_wins[-1])
                try:
                    recent_overdog.append(s_wins[-1]-s_wins[-10])
                    recent_underdog.append(f_wins[-1]-f_wins[-10])
                except:
                    recent_overdog.append(s_wins[-1]-s_wins[0])
                    recent_underdog.append(f_wins[-1]-f_wins[0])
            else:
                overdog_wins.append(f_wins[-1])
                underdog_wins.append(s_wins[-1])
                try:
                    recent_overdog.append(f_wins[-1]-f_wins[-10])
                    recent_underdog.append(s_wins[-1]-s_wins[-10])
                except:
                    recent_overdog.append(f_wins[-1]-f_wins[0])
                    recent_underdog.append(s_wins[-1]-s_wins[0])
        elif row['winner_id'] == row['underdog_id']:
            if row['winner_rank']>row['loser_rank']:
                underdog_wins.append(s_wins[-1])
                overdog_wins.append(f_wins[-1])
                try:
                    recent_overdog.append(f_wins[-1]-f_wins[-10])
                    recent_underdog.append(s_wins[-1]-s_wins[-10])
                except:
                    recent_overdog.append(f_wins[-1]-f_wins[0])
                    recent_underdog.append(s_wins[-1]-s_wins[0])
            else:
                overdog_wins.append(s_wins[-1])
                underdog_wins.append(f_wins[-1])
                try:
                    recent_overdog.append(s_wins[-1]-s_wins[-10])
                    recent_underdog.append(f_wins[-1]-f_wins[-10])
                except:
                    recent_overdog.append(s_wins[-1]-s_wins[0])
                    recent_underdog.append(f_wins[-1]-f_wins[0])
        if row['winner_id'] > row['loser_id']:
            # s_wins += 1
            s_wins.append(s_wins[-1]+1)
            f_wins.append(f_wins[-1])
        else:
            # f_wins += 1
            f_wins.append(f_wins[-1]+1)
            s_wins.append(s_wins[-1])

    if surface_flag:
        ret_val = pd.DataFrame({'overdog_h2h_surface_wins': overdog_wins,
                            'underdog_h2h_surface_wins': underdog_wins,
                            'overdog_h2h_surface_recent_wins':recent_overdog,
                            'underdog_h2h_surface_recent_wins':recent_underdog
                            },index=df.index)
    else:
        ret_val = pd.DataFrame({'overdog_h2h_wins': overdog_wins,
                                'underdog_h2h_wins': underdog_wins,
                                'overdog_h2h_recent_wins':recent_overdog,
                                'underdog_h2h_recent_wins':recent_underdog
                                },index=df.index)
    print(df)
    print(ret_val)
    return ret_val

def total_win_loss(df,data,type='overdog',surface_flag=False):
    '''
    Get total win loss record of player and in recent history
    '''
    print('df')
    print(df)
    if type == 'overdog':
        l = 0
        w = 0
        losses = []
        wins = []
        if surface_flag:
            opp = data.loc[(data['underdog_id']==df['overdog_id'].iloc[0]) & (data['surface']==df['surface'].iloc[0])]
        else:
            opp = data.loc[(data['underdog_id']==df['overdog_id'].iloc[0])]
        opp['flag'] = False
        df_ = copy.deepcopy(df)
        df_['flag'] = True

        final_df = pd.concat([opp,df_])
        final_df.sort_values(['Date','match_num'],inplace=True)
        # print(final_df)

        # print('reached')
        for i,row in final_df.iterrows():
            wins.append(w)
            losses.append(l)
            if (row['Winner'] == 0 and row['flag']==True) or (row['Winner'] == 1 and row['flag']==False):
                w += 1
            else:
                l += 1
            # result_l.append(len(data.loc[((row['Date']-data['Date']).days<365) & (data['loser_id']==row['winner_id'])& ((row['Date']-data['Date']).days>0)]))
        
        recent_w = pd.DataFrame(wins,index=final_df.index)
        recent_l = pd.DataFrame(losses,index=final_df.index)
        recent_wins = recent_w.rolling(10,min_periods=1).max().fillna(0) - recent_w.rolling(10,min_periods=1).min().fillna(0)
        recent_losses = recent_l.rolling(10,min_periods=1).max().fillna(0) - recent_l.rolling(10,min_periods=1).min().fillna(0)

        # print('REACHED')
        
        if surface_flag:
            ret_val = pd.DataFrame({'total_wins_surface_overdog': wins,
                                    'total_losses_surface_overdog': losses,
                                    'recent_wins_surface_overdog': recent_wins[0].astype(int),
                                    'recent_losses_surface_overdog':recent_losses[0].astype(int),
                                    'flag':final_df['flag']
                                    },index=final_df.index)
            print('before')
            print(ret_val)
            ret_val = ret_val.loc[(ret_val['flag']==True)]
        else:
            ret_val = pd.DataFrame({'total_wins_overdog': wins,
                                    'total_losses_overdog': losses,
                                    'recent_wins_overdog': recent_wins[0].astype(int),
                                    'recent_losses_overdog': recent_losses[0].astype(int),
                                    'flag':final_df['flag']
                                    },index=final_df.index)
            ret_val = ret_val.loc[(ret_val['flag']==True)]
        print('ret_val')
        print(ret_val)
    elif type =='underdog':
        l = 0
        w = 0
        losses = []
        wins = []
        if surface_flag:
            opp = data.loc[(data['overdog_id']==df['underdog_id'].iloc[0]) & (data['surface']==df['surface'].iloc[0])]
        else:
            opp = data.loc[(data['overdog_id']==df['underdog_id'].iloc[0])]
        opp['flag'] = False
        df_ = copy.deepcopy(df)
        df_['flag'] = True

        final_df = pd.concat([opp,df_])
        final_df.sort_values(['Date','match_num'],inplace=True)
        # print(final_df)

        # print('reached')
        for i,row in final_df.iterrows():
            wins.append(w)
            losses.append(l)
            if (row['Winner'] == 1 and row['flag']==True) or (row['Winner'] == 0 and row['flag']==False):
                w += 1
            else:
                l += 1
            # result_l.append(len(data.loc[((row['Date']-data['Date']).days<365) & (data['loser_id']==row['winner_id'])& ((row['Date']-data['Date']).days>0)]))
        
        recent_w = pd.DataFrame(wins,index=final_df.index)
        recent_l = pd.DataFrame(losses,index=final_df.index)
        recent_wins = recent_w.rolling(10,min_periods=1).max().fillna(0) - recent_w.rolling(10,min_periods=1).min().fillna(0)
        recent_losses = recent_l.rolling(10,min_periods=1).max().fillna(0) - recent_l.rolling(10,min_periods=1).min().fillna(0)

        # print('REACHED')
        
        if surface_flag:
            ret_val = pd.DataFrame({'total_wins_surface_underdog': wins,
                                    'total_losses_surface_underdog': losses,
                                    'recent_wins_surface_underdog': recent_wins[0].astype(int),
                                    'recent_losses_surface_underdog':recent_losses[0].astype(int),
                                    'flag':final_df['flag']
                                    },index=final_df.index)
            print('before')
            print(ret_val)
            ret_val = ret_val.loc[(ret_val['flag']==True)]
        else:
            ret_val = pd.DataFrame({'total_wins_underdog': wins,
                                    'total_losses_underdog': losses,
                                    'recent_wins_underdog': recent_wins[0].astype(int),
                                    'recent_losses_underdog': recent_losses[0].astype(int),
                                    'flag':final_df['flag']
                                    },index=final_df.index)
            ret_val = ret_val.loc[(ret_val['flag']==True)]
        print('ret_val')
        print(ret_val)
    else:
        sys.exit("Wrong type specified")
    ret_val.drop(['flag'],inplace=True,axis=1)
    return ret_val


def prev_stats(df,data,type='overdog',columns = [
'_ace',
'_df',
'_svpt',
'_1stIn',
'_1stWon',
'_2ndWon',
'_SvGms',
'_bpSaved',
'_bpFaced']):
    if type == 'overdog':
        cols = copy.deepcopy(columns)
        acc_cols = ['underdog'+s for s in columns]
        acc_cols.insert(0, 'Date')
        acc_cols.insert(0, 'match_num')
        # columns.append('Date')
        opp = data.loc[(data['underdog_id'] == df.loc[:,'overdog_id'].iloc[0])]
        opp = opp[opp.columns.intersection(acc_cols)]

        df_cols = ['overdog'+s for s in columns]
        df_cols.insert(0,'Date')
        df_cols.insert(0,'match_num')
        df_ = df[df.columns.intersection(df_cols)]

        cols.insert(0, 'Date')
        cols.insert(0,'match_num')

        opp.columns = cols
        df_.columns = cols

        opp['flag'] = False
        df_['flag'] = True
        final_df = pd.concat([opp,df_])
        df_cols.append('flag')
        final_df.columns = df_cols
        final_df.sort_values(['Date','match_num'],inplace=True)
        final_df.drop(['Date','match_num'],axis=1,inplace=True)
        df_cols.remove('Date')
        df_cols.remove('match_num')
        df_cols.remove('flag')
        for i in df_cols:
            final_df[i+'_MVA']= final_df[i].shift().rolling(10,min_periods=1).mean().fillna(0)
        ret = final_df.loc[(final_df['flag']==True)]
        # print('ret:',ret)
        # print('df:',final_df)
        ret.drop(['flag'],axis = 1,inplace=True)
        ret.drop(df_cols,axis=1,inplace=True)
        return ret
    elif type == 'underdog':
        cols = copy.deepcopy(columns)
        acc_cols = ['overdog'+s for s in columns]
        acc_cols.insert(0, 'Date')
        acc_cols.insert(0,'match_num')
        # columns.append('Date')
        opp = data.loc[(data['overdog_id'] == df.loc[:,'underdog_id'].iloc[0])]
        opp = opp[opp.columns.intersection(acc_cols)]

        df_cols = ['underdog'+s for s in columns]
        df_cols.insert(0,'Date')
        df_cols.insert(0,'match_num')
        df_ = df[df.columns.intersection(df_cols)]

        cols.insert(0, 'Date')
        cols.insert(0,'match_num')

        opp.columns = cols
        df_.columns = cols

        opp['flag'] = False
        df_['flag'] = True
        final_df = pd.concat([opp,df_])
        df_cols.append('flag')
        final_df.columns = df_cols
        final_df.sort_values(['Date','match_num'],inplace=True)
        final_df.drop(['Date','match_num'],axis=1,inplace=True)
        df_cols.remove('Date')
        df_cols.remove('match_num')
        df_cols.remove('flag')
        for i in df_cols:
            final_df[i+'_MVA']= final_df[i].shift().rolling(10,min_periods=1).mean().fillna(0)
        ret = final_df.loc[(final_df['flag']==True)]
        # print('ret:',ret)
        # print('df:',final_df)
        ret.drop(['flag'],axis = 1,inplace=True)
        ret.drop(df_cols,axis=1,inplace=True)
        return ret
    else:
        sys.exit('wrong type specified')

    



    

    