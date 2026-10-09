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
    df_inputs.loc[df_inputs.ZONEID == 1, 'ZONE_1'] = 1
    df_inputs.loc[df_inputs.ZONEID == 2, 'ZONE_2'] = 1
    df_inputs.loc[df_inputs.ZONEID == 3, 'ZONE_3'] = 1
    df_inputs = df_inputs.drop('ZONEID', axis=1)
    return df_inputs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    args = parser.parse_args()


    en_part_2 = pd.read_csv(os.path.join(args.data_dir, 'GEFCOM2014_SOLAR', 'raw', 'Solution to Task 15',
                                         'Solution to Task 15.csv'), parse_dates=True, index_col=1)
    en_part_1 = pd.read_csv(os.path.join(args.data_dir, 'GEFCOM2014_SOLAR', 'raw', 'Task 15', 'train15.csv'),
                            parse_dates=True, index_col=1)
    en_data = pd.concat([en_part_1, en_part_2])

    for type in range(1, 16):
        df_weather_data = build_inputs(path_dir=os.path.join(args.data_dir, 'GEFCOM2014_SOLAR', 'raw', 'Task ' + str(type),
                                                           'predictors' + str(type) + '.csv'))
        df_en_train = pd.read_csv(os.path.join(args.data_dir, 'GEFCOM2014_SOLAR', 'raw', 'Task ' + str(type),
                                               'train' + str(type) + '.csv'), parse_dates=True, index_col=1)
        df_en_train = df_en_train.rename(columns={'POWER': 'energy'})
        stamp_val = df_en_train.index[-1] + pd.DateOffset(hours=1) - pd.DateOffset(months=1)
        stamp_test = df_en_train.index[-1] + pd.DateOffset(hours=1)
        df_en_test = en_data[en_data.index.isin(df_weather_data.index)]
        df_en_test = df_en_test[df_en_test.index >= stamp_test]
        df_en_train.ffill(inplace=True)
        df_en_test.ffill(inplace=True)

        df_en_test = df_en_test.rename(columns={'POWER': 'energy'})
        solar_FEATURES = ['VAR78', 'VAR79', 'VAR134', 'VAR157', 'VAR164', 'VAR165', 'VAR166', 'VAR167', 'VAR169',
                          'VAR175', 'VAR178', 'VAR228', 'ZONE_1', 'ZONE_2', 'ZONE_3']
        solar_ZONES = ['ZONE_1', 'ZONE_2', 'ZONE_3']

        ssrd = df_weather_data['VAR169'].values.reshape(-1,24) / 3600 # from J/m2 to W/m2
        ssrd_diff = differenciate(data=ssrd)
        strd = df_weather_data['VAR175'].values.reshape(-1,24) / 3600 # from J/m2 to W/m2
        strd_diff = differenciate(data=strd)

        tsr = df_weather_data['VAR178'].values.reshape(-1,24) / 3600 # from J/m2 to W/m2
        tsr_diff = differenciate(data=tsr)

        tp = df_weather_data['VAR228'].values.reshape(-1,24)
        tp_diff = differenciate(data=tp)

        df_pv_new = df_weather_data.copy()
        df_pv_new['VAR169'] = ssrd_diff.reshape(-1)
        df_pv_new['VAR175'] = strd_diff.reshape(-1)
        df_pv_new['VAR178'] = tsr_diff.reshape(-1)
        df_pv_new['VAR228'] = tp_diff.reshape(-1)
        df_pv_new = df_pv_new.rename(columns={'VAR78': 'TCLW', 'VAR79': 'TCIW', 'VAR134': 'SP', 'VAR157': 'R',
                                              'VAR164': 'TCC', 'VAR165': '10U', 'VAR166': '10V', 'VAR167': '2T',
                                              'VAR169': 'SSRD', 'VAR175': 'STRD', 'VAR178': 'TSR', 'VAR228': 'TP'})
        ZONES = ['ZONE_1', 'ZONE_2', 'ZONE_3']

        # Shift PV dataset from 10 periods for each zone and drop the first day
        data = {}
        data['dataset'] = 'GEFCOM2014_SOLAR'
        data['energy'] = []
        data['weather'] = []

        for i, zone in enumerate(ZONES):
            df_en_train_tmp = df_en_train.loc[df_en_train['ZONEID'] == i + 1].copy()[['energy']]
            df_en_test_tmp = df_en_test.loc[df_en_test['ZONEID'] == i + 1].copy()[['energy']]
            data['energy'].append(pd.concat([df_en_train_tmp, df_en_test_tmp]))
            data['energy'][i] = data['energy'][i].shift(periods=10).copy().iloc[24:].copy()
            data['weather'].append(df_pv_new[df_pv_new[zone] == 1].shift(periods=10).iloc[24:].copy()[['TCLW', 'TCIW', 'SP', 'R', 'TCC', '10U', '10V', '2T', 'SSRD', 'STRD', 'TSR', 'TP']])
        data['time'] = df_pv_new[df_pv_new['ZONE_1'] == 1].shift(periods=10)['2012-04-02 01':].copy().index.to_frame()
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
        if not os.path.isdir(os.path.join(args.data_dir, 'GEFCOM2014_SOLAR', 'processed')):
            os.mkdir(os.path.join(args.data_dir, 'GEFCOM2014_SOLAR', 'processed'))
        if not os.path.isdir(os.path.join(args.data_dir, 'GEFCOM2014_SOLAR', 'processed', 'Task ' + str(type))):
            os.mkdir(os.path.join(args.data_dir, 'GEFCOM2014_SOLAR', 'processed', 'Task ' + str(type)))
        with open(os.path.join(args.data_dir, 'GEFCOM2014_SOLAR', 'processed', 'Task ' + str(type), 'data.pickle'), 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
