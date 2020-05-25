from features import underdog_upset_num,overdog_getupset_num,previous_head2head, total_win_loss, prev_stats
import pandas as pd
import numpy as np
import copy

if __name__ == "__main__":

    data = pd.read_parquet('./../underdog_overdog/data_one_hot.parquet')

    data = data[data['underdog_rank'].notna()]
    data = data[data['surface'].notna()]

    data = data.loc[data['Year']>=1991]

    data.sort_values(by=['Date','match_num'],inplace=True)

    #EXTRACT INTERESTING FEATURES
    # HEAD2HEAD
    head2head = copy.deepcopy(data[['overdog_id','underdog_id','winner_id','loser_id','winner_rank','loser_rank','surface']])
    head2head[['overdog_id','underdog_id']]=np.sort(head2head[['overdog_id', 'underdog_id']], axis=1)
    head2head_total = head2head.groupby(['overdog_id','underdog_id']).apply(previous_head2head)
    print('first', head2head_total)
    #bysurface
    surface_head2head = head2head.groupby(['overdog_id','underdog_id','surface']).apply(previous_head2head,surface_flag = True)
    print('second',surface_head2head)
    #UPSET POTENTIAL
    overdog_gets_upsets = data.groupby("overdog_id").apply(overdog_getupset_num)
    underdog_upsets = data.groupby("underdog_id").apply(underdog_upset_num)
    print('third',overdog_gets_upsets)
    print('fourth',underdog_upsets)

    # #TOTAL WIN_LOSS
    win_loss_overdog = data.groupby("overdog_id").apply(total_win_loss,data,type='overdog')
    win_loss_underdog = data.groupby("underdog_id").apply(total_win_loss,data,type='underdog')
    print('fifth',win_loss_overdog)
    print('sixth',win_loss_underdog)
    #By surface
    win_loss_overdog_surface = data.groupby(["overdog_id","surface"]).apply(total_win_loss,data,type='overdog',surface_flag=True)
    win_loss_underdog_surface = data.groupby(["underdog_id","surface"]).apply(total_win_loss,data,type='underdog',surface_flag=True)
    print('seventh',win_loss_overdog_surface)
    print('eighth',win_loss_underdog_surface)
    #STATS
    stats_overdog = data.groupby("overdog_id").apply(prev_stats,data,type='overdog')
    stats_underdog = data.groupby("underdog_id").apply(prev_stats,data,type='underdog')
    print('ninth',stats_overdog)
    print('tenth',stats_underdog)


    if overdog_gets_upsets.index.nlevels > 1: 
        overdog_gets_upsets.reset_index(level=0, drop=True, inplace=True)
    if underdog_upsets.index.nlevels > 1:
        underdog_upsets.reset_index(level=0, drop=True, inplace=True)
    if head2head_total.index.nlevels > 1:
        head2head_total.reset_index(level=0, drop=True, inplace=True)
    if surface_head2head.index.nlevels > 1:
        surface_head2head.reset_index(level=[0,1], drop=True, inplace=True)
    if win_loss_overdog.index.nlevels > 1:
        win_loss_overdog.reset_index(level=0, drop=True, inplace=True)
    if win_loss_underdog.index.nlevels > 1:
        win_loss_underdog.reset_index(level=0, drop=True, inplace=True)
    if win_loss_overdog_surface.index.nlevels > 1:
        win_loss_overdog_surface.reset_index(level=[0,1], drop=True, inplace=True)
    if win_loss_underdog_surface.index.nlevels > 1:
        win_loss_underdog_surface.reset_index(level=[0,1], drop=True, inplace=True)
    if stats_overdog.index.nlevels > 1:
        stats_overdog.reset_index(level=0, drop=True, inplace=True)
    if stats_underdog.index.nlevels > 1:
        stats_underdog.reset_index(level=0, drop=True, inplace=True)

    dfs = [head2head_total,surface_head2head,overdog_gets_upsets,underdog_upsets, win_loss_overdog, win_loss_overdog_surface,win_loss_underdog, win_loss_underdog_surface,stats_overdog,stats_underdog]
    count = 0
    for i in dfs:
        print(count)
        print(i)
        count += 1

    
    data = data.join(dfs)
    data.astype({'winner_seed': str,'loser_seed': str,'overdog_seed':str,'underdog_seed':str}).to_parquet('datav2.parquet')



    
