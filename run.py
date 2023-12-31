import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import calendar
from datetime import timedelta, datetime

COLS = [
        'Date',
        'holiday',
        'Block_duration',
        'Programme_duration',
        'Block_Programme_ratio',
        'month',
        # 'hour_break_start',
        'part_of_day_break_start',
        # 'hour_programme_start',
        'part_of_day_programme_start',
        # 'hour_programme_end',
        'part_of_day_programme_end',
        'mean_break_hours',
        'median_break_hours',
        'count_break_hours',
        # 'is_cristmas',
        # 'is_23_feb',
        # 'is_8_march',
        # 'is_1_may',
        # 'is_9_may',
        # 'is_12_june',
        # 'is_4_nov',
        'is_weekend'

    
    
        

    
       ]

def preprocessing(train, test, side_data):
    
    #Feature engeneering

    train = train.sort_values('Date').copy()
    test = test.copy()
    side_data = side_data.copy()
    

    train = train[train['TVR Index'] != 0]
    # train['TVR Index'] = np.log(train['TVR Index'] + 1)
    
    #side data  TV Viewing
    
    # side_data.index = range(1, 11)
    # print(side_data)
    # side_data.loc[len(side_data) + 1] = pd.Series({2021: np.nan, 2022: np.nan, 2023: np.nan})
    # print(side_data)
    # side_data = pd.DataFrame(side_data.interpolate(method='cubicspline').apply(lambda x: x.mean(), axis=1), columns=['mean_viewing'])

    
    # Date
    
    def date_features(df):
        
        df['month'] = df['Date'].apply(lambda x: x.month).astype('category')
        ##############
        # df['is_cristmas'] = df['Date'].apply(lambda x: 0 < x.day_of_year <=  8).astype(int)
        # df['is_23_feb'] = df['Date'].apply(lambda x: x.day_of_year == 54).astype(int)
        # df['is_8_march'] = df['Date'].apply(lambda x: x.day_of_year == 68).astype(int)   
        # df['is_1_may'] = df['Date'].apply(lambda x: x.day_of_year == 122).astype(int)
        # df['is_9_may'] = df['Date'].apply(lambda x: x.day_of_year == 130).astype(int)
        # df['is_12_june'] = df['Date'].apply(lambda x: x.day_of_year == 164).astype(int)
        # df['is_4_nov'] = df['Date'].apply(lambda x: x.day_of_year == 308).astype(int)
        df['is_weekend'] = df['Date'].apply(lambda x: 5 <= x.day_of_week <= 6).astype(int)

        ############
        df['holiday'] = 'Common'

        df.loc[(0 < df['Date'].dt.day_of_year) & (df['Date'].dt.day_of_year <= 8), 'holiday'] = 'Christmas'
        df.loc[df['Date'].dt.day_of_year == 54, 'holiday'] = '23rd February'
        df.loc[df['Date'].dt.day_of_year == 68, 'holiday'] = '8th March'
        df.loc[df['Date'].dt.day_of_year == 122, 'holiday'] = '1st May'
        df.loc[df['Date'].dt.day_of_year == 130, 'holiday'] = '9th May'
        df.loc[df['Date'].dt.day_of_year == 164, 'holiday'] = '12th June'
        df.loc[df['Date'].dt.day_of_year == 308, 'holiday'] = '4th November'
        df.loc[(5 <= df['Date'].dt.day_of_week) & (df['Date'].dt.day_of_week <= 6), 'holiday'] = 'Weekend'
        df['holiday'] = df['holiday'].astype('category')

        ###########
        
        df['hour_break_start'] = df['Break flight start'].apply(lambda x: x.hour)
        df['part_of_day_break_start'] = pd.cut(df['hour_break_start'],
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['Night',
                                           'Morning',
                                           'Afternoon',       
                                           'Evening'],
                                   include_lowest=True,
                                   right=False)
        ##########
        daypart_break_stats = df.groupby('part_of_day_break_start')['hour_break_start'].agg(mean_break_hours='mean', 
                                                                                            median_break_hours='median', 
                                                                                            count_break_hours='count')

        df = df.merge(right=daypart_break_stats, how='inner', left_on='part_of_day_break_start', right_on=daypart_break_stats.index)

        ##########

        df['hour_programme_start'] = df['Programme flight start'].apply(lambda x: x.hour)
        df['part_of_day_programme_start'] = pd.cut(df['hour_programme_start'],
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['Night',
                                           'Morning',
                                           'Afternoon',       
                                           'Evening'],
                                   include_lowest=True,
                                   right=False)

        
        df['hour_programme_end'] = df['Programme flight end'].apply(lambda x: x.hour)
        df['part_of_day_programme_end'] = pd.cut(df['hour_programme_end'],
                                   bins=[0, 6, 12, 18, 24], 
                                   labels=['Night',
                                           'Morning',
                                           'Afternoon',       
                                           'Evening'],
                                   include_lowest=True,
                                   right=False)
        # df['isprogramme_between_part_of_day'] = 
        
        
        return df

    
    #break_flight
    
    def break_flight_features(df):
    
        def to_sec(break_flight_time):
            res = int(break_flight_time.hour) * 3600 + int(break_flight_time.minute) * 60 + int(break_flight_time.second)
            return res
        
        df['Break flight start_in_sec'] = df['Break flight start'].apply(lambda x: to_sec(x))
        df['Break flight end_in_sec'] = df['Break flight end'].apply(lambda x: to_sec(x))
        df['Block_duration'] = df['Break flight end_in_sec'] - df['Break flight start_in_sec']
        df.loc[df['Block_duration'] < 0, 'Block_duration'] += 24 * 3600  
            
        df['Programme flight start_in_sec'] = df['Programme flight start'].apply(lambda x: to_sec(x))
        df['Programme flight end_in_sec'] = df['Programme flight end'].apply(lambda x: to_sec(x))
        df['Programme_duration'] = df['Programme flight end_in_sec'] - df['Programme flight start_in_sec']
        df.loc[df['Programme_duration'] < 0, 'Programme_duration'] += 24 * 3600  
    
        df['Block_Programme_ratio'] = df['Block_duration'] / df['Programme_duration']

        return df
    
    train = date_features(train)
    test = date_features(test)
    
    train = break_flight_features(train)
    test = break_flight_features(test)    
    # train = pd.merge(left=train, right=side_data, how='inner', left_on='month', right_on='month')
    # test = pd.merge(left=test, right=side_data, how='inner', left_on='month', right_on='month')

    # train = train.merge(right=side_data, how='inner', left_on='month', right_on=side_data)
    # test = test.merge(right=side_data, how='inner', left_on='month', right_on=side_data)
    train = train[COLS + ['TVR Index']]
    test = test[COLS]
    return train, test

    
