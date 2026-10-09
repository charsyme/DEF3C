
import os
import pandas as pd
import numpy as np
from astral import LocationInfo
from astral.sun import sun
import pickle
import argparse
import zipfile

def get_time_of_day(instance):
   instance = pd.to_datetime(instance)
   location = LocationInfo("Athens", "Greece").observer
   phases = sun(location, date=instance)
   phases = {k:pd.to_datetime(phases[k]).tz_convert(None) for k in phases.keys()}
   if phases['dawn'] < instance <= phases['sunrise']:
      return 'sunrise'
   elif phases['sunrise'] < instance <= phases['noon']:
      return 'morning'
   elif phases['noon'] < instance <= phases['sunset']:
      return 'noon'
   elif phases['sunset'] < instance <= phases['dusk']:
      return 'sunset'
   else:
      return 'night'

def season_calc(month):
   if month in [6,7,8,9,10]:
     return "summer"
   else:
     return "winter"

def differenciate(data:np.array):
    data_rolled = np.roll(data, 1, axis=1) # shift from one period
    data_rolled[:, 0] = 0
    data_diff = data - data_rolled
    data_diff[data_diff < 0] = 0
    return data_diff


def build_inputs(path_dir: str):
    df_inputs = pd.read_csv(path_dir, parse_dates=True, index_col=1)
    df_inputs['ZONE_1'] = 0
    df_inputs['ZONE_2'] = 0
    df_inputs['ZONE_3'] = 0
    df_inputs['ZONE_4'] = 0
    df_inputs['ZONE_5'] = 0
    df_inputs['ZONE_6'] = 0
    df_inputs['ZONE_7'] = 0
    df_inputs['ZONE_8'] = 0
    df_inputs['ZONE_9'] = 0
    df_inputs['ZONE_10'] = 0
    df_inputs.loc[df_inputs.ZONEID == 1, 'ZONE_1'] = 1
    df_inputs.loc[df_inputs.ZONEID == 2, 'ZONE_2'] = 1
    df_inputs.loc[df_inputs.ZONEID == 3, 'ZONE_3'] = 1
    df_inputs.loc[df_inputs.ZONEID == 4, 'ZONE_4'] = 1
    df_inputs.loc[df_inputs.ZONEID == 5, 'ZONE_5'] = 1
    df_inputs.loc[df_inputs.ZONEID == 6, 'ZONE_6'] = 1
    df_inputs.loc[df_inputs.ZONEID == 7, 'ZONE_7'] = 1
    df_inputs.loc[df_inputs.ZONEID == 8, 'ZONE_8'] = 1
    df_inputs.loc[df_inputs.ZONEID == 9, 'ZONE_9'] = 1
    df_inputs.loc[df_inputs.ZONEID == 10, 'ZONE_10'] = 1
    df_inputs = df_inputs.drop('ZONEID', axis=1)
    return df_inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    args = parser.parse_args()
    for zone in range(1, 16):
        with zipfile.ZipFile(os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'raw', 'Task ' + str(zone),
                                          'Task' + str(zone) + '_W_Zone1_10.zip'),'r') as zip_ref:
            zip_ref.extractall(os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'raw', 'Task '+ str(zone)))
        with zipfile.ZipFile(os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'raw', 'Task ' + str(zone),
                                          'TaskExpVars' + str(zone) + '_W_Zone1_10.zip'),'r') as zip_ref:
            zip_ref.extractall(os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'raw', 'Task '+ str(zone)))

    en_part_2 = build_inputs(path_dir=os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'raw', 'Solution to Task 15',
                                                 'solution15_W.csv'))
    en_part_1 = pd.DataFrame(columns=['TARGETVAR', 'ZONE_1', 'ZONE_2', 'ZONE_3', 'ZONE_4', 'ZONE_5', 'ZONE_6', 'ZONE_7',
                                      'ZONE_8', 'ZONE_9', 'ZONE_10'])
    for zone in range(1, 11):
        en_part_1_zone = build_inputs(path_dir=os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'raw',
                                                            'Task 15',
                                                            'Task15_W_Zone1_10',
                                                            'Task15_W_Zone' + str(zone) + '.csv'))[['TARGETVAR', 'ZONE_1', 'ZONE_2', 'ZONE_3', 'ZONE_4',
                                     'ZONE_5', 'ZONE_6', 'ZONE_7', 'ZONE_8', 'ZONE_9', 'ZONE_10']]
        en_part_1 = pd.concat([en_part_1, en_part_1_zone])
    en_data = pd.concat([en_part_1, en_part_2])

    for type in range(1, 16):
        df_train = pd.DataFrame(columns=['TARGETVAR', 'U10', 'V10', 'U100', 'V100', 'ZONE_1', 'ZONE_2', 'ZONE_3', 'ZONE_4',
                              'ZONE_5', 'ZONE_6', 'ZONE_7', 'ZONE_8', 'ZONE_9', 'ZONE_10'])
        df_test = pd.DataFrame(columns=['TARGETVAR', 'U10', 'V10', 'U100', 'V100', 'ZONE_1', 'ZONE_2', 'ZONE_3', 'ZONE_4',
                              'ZONE_5', 'ZONE_6', 'ZONE_7', 'ZONE_8', 'ZONE_9', 'ZONE_10'])
        for zone in range(1, 11):
            df_train = pd.concat([df_train, build_inputs(path_dir=os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'raw', 'Task ' + str(type),
                                                       'Task' + str(type) + '_W_Zone1_10', 'Task' + str(type) +
                                                       '_W_Zone' + str(zone) + '.csv'))])
            df_test = pd.concat([df_test, build_inputs(path_dir=os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'raw', 'Task ' + str(type),
                                                       'TaskExpVars' + str(type) + '_W_Zone1_10', 'TaskExpVars' + str(type) +
                                                       '_W_Zone' + str(zone) + '.csv'))])
            df_test.loc[df_test['ZONE_' + str(zone)] == 1, 'TARGETVAR'] = en_data.loc[en_data['ZONE_' + str(zone)] ==1, 'TARGETVAR'].copy()

        df_train = df_train.rename(columns={'TARGETVAR': 'energy'})
        df_test = df_test.rename(columns={'TARGETVAR': 'energy'})
        stamp_val = df_train.index[-1] + pd.DateOffset(hours=1) - pd.DateOffset(months=1)
        stamp_test = df_test.index[0]
        wind_features = ['U10', 'U100', 'V10', 'V100']
        wind_zones = ['ZONE_1', 'ZONE_2', 'ZONE_3','ZONE_4', 'ZONE_5', 'ZONE_6', 'ZONE_7', 'ZONE_8', 'ZONE_9', 'ZONE_10']

        data = {}
        data['dataset'] = 'GEFCOM2014_WIND'
        data['energy'] = []
        data['weather'] = []

        for i, zone in enumerate(wind_zones):
            df_en_train_tmp = df_train.loc[df_train[zone] == 1].copy()[['energy']]
            df_en_train_tmp['energy'] = df_en_train_tmp['energy'].astype(float)
            df_en_test_tmp = df_test.loc[df_test[zone] == 1].copy()[['energy']]
            df_en_test_tmp['energy'] = df_en_test_tmp['energy'].astype(float)
            data['energy'].append(pd.concat([df_en_train_tmp, df_en_test_tmp]))
            data['energy'][i] = data['energy'][i].iloc[:-24].copy()
            data['energy'][i] = data['energy'][i].ffill()
            df_weather_train_tmp = df_train.loc[df_train[zone] == 1].copy()[wind_features]
            df_weather_test_tmp = df_test.loc[df_test[zone] == 1].copy()[wind_features]
            data['weather'].append(pd.concat([df_weather_train_tmp, df_weather_test_tmp]))
            data['weather'][i] = data['weather'][i].iloc[:-24].copy()
            data['weather'][i] = data['weather'][i].ffill()

        data['time'] = data['energy'][0].index.to_frame()
        data['split'] = [stamp_val, stamp_test]

        data['time']['day'] = data['time'].index.day
        data['time']['datetime'] = data['time'].index
        data['time']['timeofday'] = data['time']['datetime'].apply(get_time_of_day)
        data['time'] = data['time'].drop(columns=['datetime'])
        data['time']['year'] = data['time'].index.year - 2011
        data['time']['hour'] = data['time'].index.hour
        data['time']['dayofyear'] = data['time'].index.dayofyear
        data['time']['month'] = data['time'].index.month
        data['time']['season'] = data['time']['month'].apply(season_calc)
        data['time'] = data['time'][['day', 'timeofday', 'year', 'hour', 'dayofyear', 'season', 'month']]
        if not os.path.isdir(os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'processed')):
            os.mkdir(os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'processed'))
        if not os.path.isdir(os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'processed', 'Task ' + str(type))):
            os.mkdir(os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'processed', 'Task ' + str(type)))
        with open(os.path.join(args.data_dir, 'GEFCOM2014_WIND', 'processed', 'Task ' + str(type), 'data.pickle'), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
