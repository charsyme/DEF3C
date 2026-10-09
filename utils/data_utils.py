import os
import pickle
import pandas as pd
import numpy as np
import torch

class DataLoader():
    def __init__(self, dataset=None, split='train', t_h=336, t_f=24, data_dir='./data/', device='cpu',
                 plant_locs_sel=None, weather_locs_sel=None):
        fln = None
        self.device = device
        self.t_h = t_h
        self.t_f = t_f
        self.plant_locs_sel = plant_locs_sel
        self.weather_locs_sel = weather_locs_sel
        self.min_max_train = []
        self.min_max_all = []
        if 'GEFCOM2014' in dataset:
            task = dataset.split('_')[2]
            fln = os.path.join(data_dir, 'processed', 'Task ' + task, 'data.pickle')
        elif 'load' in dataset:
            fln = os.path.join(data_dir, 'load.pkl')
        with open(fln, 'rb') as file:
            data = pickle.load(file)
        if weather_locs_sel is None:
            self.weather_locs_sel = []
            for i in range(len(data['weather'])):
                self.weather_locs_sel.append(i)
        if plant_locs_sel is None:
            self.plant_locs_sel = []
            for i in range(len(data['energy'])):
                self.plant_locs_sel.append(i)
        if 'load' in dataset:
            ind_split = [np.where(data['energy'][0].index == (data['split'][0].tz_localize('UTC')))[0][0],
                         np.where(data['energy'][0].index == data['split'][1].tz_localize('UTC'))[0][0]]
        else:
            ind_split = [np.where(data['energy'][0].index == (data['split'][0]))[0][0],
                         np.where(data['energy'][0].index == data['split'][1])[0][0]]
        for i in range(len(data['weather'])):
            cols = data['weather'][i].columns
            for col in cols:
                if ('deg' in col) or ('dir' in col):
                    sin_col = 'sin_' + col
                    cos_col = 'cos_' + col
                    data['weather'][i][sin_col] = np.sin(data['weather'][i][col] * 2 * np.pi / 360.0)
                    data['weather'][i][cos_col] = np.cos(data['weather'][i][col] * 2 * np.pi / 360.0)
                    data['weather'][i] = data['weather'][i].drop([col], axis=1)
        self.mm_data = self.get_mm(data, ind_split[0])

        self.size = data['time'].shape[0] - (self.t_h + self.t_f)
        self.data = {}
        self.data['energy'] = []
        self.data['weather'] = []

        for i in range(len(data['energy'])):
            self.data['energy'].append(data['energy'][i])
        for i in range(len(data['weather'])):
            self.data['weather'].append(data['weather'][i])

        data['time'] = convert_hour(data['time'])
        data['time'].drop('hour', inplace=True, axis=1)

        data['time'] = convert_month(data['time'])
        data['time'].drop('month', inplace=True, axis=1)
        if 'timeofday' in data['time'].columns:
        #    data['time'] = convert_timeofday(data['time'])
            data['time'].drop('timeofday', inplace=True, axis=1)
        #data['time'] = convert_season(data['time'])
        data['time'].drop('season', inplace=True, axis=1)
        if 'is_weekend' in data['time'].columns:
            data['time']['is_weekend'] =  data['time']['is_weekend'].astype(int)
        if 'weekday' in data['time'].columns:
            data['time']['weekday_sin'] = np.sin(2 * np.pi * data['time']['weekday'] / 7)
            data['time']['weekday_cos'] = np.cos(2 * np.pi * data['time']['weekday'] / 7)
            data['time'].drop('weekday', inplace=True, axis=1)

        if 'dayofyear' in data['time'].columns:
            data['time'].drop('dayofyear', inplace=True, axis=1)
        data['time'].drop('year', inplace=True, axis=1)
        data['time'].drop('day', inplace=True, axis=1)
        if 'minute' in data['time'].columns:
            data['time'].drop('minute', inplace=True, axis=1)
        self.data['time'] = [data['time']]
        hours = np.concatenate([np.expand_dims(np.arange(start=0, stop=24), axis=-1),
                                np.expand_dims(np.arange(start=1, stop=25), axis=-1)], axis=-1)
        hours[-1, -1] = 0
        if split == 'test' or split == 'val':
            hours_selected = hours[hours[:, 0] <= (24 - self.t_f)]
        else:
            hours_selected = hours
        self.ids = []
        self.data['timestamps'] = self.data['time'][0].index
        for i in range(hours_selected.shape[0]):
            self.ids.extend(np.where(self.data['timestamps'].hour == hours_selected[i, 1])[0].tolist())
        self.ids.sort()
        self.ids = np.asarray(self.ids)
        if split == 'train':
            self.ids = self.ids[(self.ids >= 0) & (self.ids < (ind_split[0] - self.t_h - self.t_f))]
            for i in range(len(self.data['energy'])):
                self.data['energy'][i] = self.data['energy'][i].head(ind_split[0])
            for i in range(len(self.data['weather'])):
                self.data['weather'][i] = self.data['weather'][i].head(ind_split[0])
            for i in range(len(self.data['time'])):
                self.data['time'][i] = self.data['time'][i].head(ind_split[0])
            self.data['timestamps'] = self.data['time'][0].index
        elif split == 'val':
            self.ids = self.ids[self.ids >= (ind_split[0] - self.t_h)]
            self.ids = self.ids[self.ids <= (ind_split[1] - self.t_h - self.t_f)]
            self.ids = self.ids - self.ids[0]
            for i in range(len(self.data['energy'])):
                self.data['energy'][i] = self.data['energy'][i].iloc[(ind_split[0] - self.t_h):ind_split[1]]
            for i in range(len(self.data['weather'])):
                self.data['weather'][i] = self.data['weather'][i].iloc[
                                          (ind_split[0] - self.t_h):ind_split[1]]
            for i in range(len(self.data['time'])):
                self.data['time'][i] = self.data['time'][i].iloc[(ind_split[0] - self.t_h):ind_split[1]]
            self.data['timestamps'] = self.data['time'][0].index
        elif split == 'test':
            for i in range(len(self.data['energy'])):
                self.data['energy'][i] = self.data['energy'][i].tail(
                    self.data['energy'][i].shape[0] - ind_split[1] + self.t_h)
            for i in range(len(self.data['weather'])):
                self.data['weather'][i] = self.data['weather'][i].tail(
                    self.data['weather'][i].shape[0] - ind_split[1] + self.t_h)
            for i in range(len(self.data['time'])):
                self.data['time'][i] = self.data['time'][i].tail(
                    self.data['time'][i].shape[0] - ind_split[1] + self.t_h)
            self.data['timestamps'] = self.data['time'][0].index
            self.ids = self.ids[self.ids >= (ind_split[1] - self.t_h)]
            self.ids = self.ids - self.ids[0]
            self.ids = self.ids[self.ids <= (self.data['energy'][0].shape[0] - self.t_h - self.t_f)]
        self.size = len(self.ids)
        self.filter_locs()
        self.to_tensor()
        #self.data['energy'] = self.normalize_energy(self.data['energy'])
        self.data['weather'] = self.normalize_weather(self.data['weather'])


    def get_batch_data(self, ids):
        h_ids = []
        f_ids = []
        for id in ids:
            h_ids.append(np.arange(id, id + self.t_h))
            f_ids.append(np.arange(id + self.t_h, id + self.t_h + self.t_f))
        h_ids = np.array(h_ids)
        f_ids = np.array(f_ids)
        e_h = self.data['energy'][:, h_ids, :].transpose(0, 1).transpose(-1, -2).transpose(-2, -3).to(self.device)
        e_f = self.data['energy'][:, f_ids, :].transpose(0, 1).transpose(-1, -2).transpose(-2, -3).to(self.device)
        w_h = self.data['weather'][:, h_ids, :].transpose(0, 1).transpose(-1, -2).transpose(-2, -3).to(self.device)
        w_f = self.data['weather'][:, f_ids, :].transpose(0, 1).transpose(-1, -2).transpose(-2, -3).to(self.device)
        t_h = self.data['time'][:, h_ids, :].transpose(0, 1).transpose(-1, -2).transpose(-2, -3).to(self.device)
        t_f = self.data['time'][:, f_ids, :].transpose(0, 1).transpose(-1, -2).transpose(-2, -3).to(self.device)
        time_stamp = [self.data['timestamps'][ids[0] + self.t_h: ids[0] + self.t_h + self.t_f]]
        for i in range(1, ids.shape[0]):
            time_stamp.append(
                self.data['timestamps'][ids[i] + self.t_h: ids[i] + self.t_h + self.t_f])
        return [e_h, w_h, t_h], [e_f, w_f, t_f], time_stamp

    def filter_locs(self):
        d_e = []
        for i in range(len(self.plant_locs_sel)):
            d_e.append(self.data['energy'][self.plant_locs_sel[i]])
        self.data['energy'] = d_e
        del d_e
        d_w = []
        for i in range(len(self.weather_locs_sel)):
            d_w.append(self.data['weather'][self.weather_locs_sel[i]])
        self.data['weather'] = d_w
        del d_w

    def to_tensor(self):
        for i in range(len(self.data['energy'])):
            self.data['energy'][i] = torch.tensor(np.array(self.data['energy'][i])).float().unsqueeze(0)
        self.data['energy'] = torch.cat(self.data['energy'], dim=0)
        for i in range(len(self.data['weather'])):
            self.data['weather'][i] = torch.tensor(
    np.asarray(self.data['weather'][i], dtype=np.float32)
).unsqueeze(0)
        self.data['weather'] = torch.cat(self.data['weather'], dim=0)
        self.data['time'] = torch.tensor(np.array(self.data['time'][0])).float().unsqueeze(0)

    def get_mm(self, data, ind_slit):
        mm_data = {'energy': [], 'weather': []}
        min = data['energy'][0].head(ind_slit).min().to_frame().transpose()
        max = data['energy'][0].head(ind_slit).max().to_frame().transpose()
        mean = data['energy'][0].head(ind_slit).mean().to_frame().transpose()
        std = data['energy'][0].head(ind_slit).std().to_frame().transpose()

        for i in range(1, len(data['energy'])):
            min = pd.concat(
                [min, data['energy'][i].head(ind_slit).min().to_frame().transpose()])
            max = pd.concat(
                [max, data['energy'][i].head(ind_slit).max().to_frame().transpose()])
            mean = pd.concat(
                [mean, data['energy'][i].head(ind_slit).mean().to_frame().transpose()])
            std = pd.concat(
                [std, data['energy'][i].head(ind_slit).std().to_frame().transpose()])
        mm_data['energy'] = {'min': min.to_numpy(), 'max': max.to_numpy(), 'mean': mean.to_numpy(), 'std': std.to_numpy()}

        min = data['weather'][0].head(ind_slit).min().to_frame().transpose()
        max = data['weather'][0].head(ind_slit).max().to_frame().transpose()
        for i in range(len(data['weather'])):
            min = pd.concat(
                [min, data['weather'][i].head(ind_slit).min().to_frame().transpose()]).min().to_frame().transpose()
            max = pd.concat(
                [max, data['weather'][i].head(ind_slit).max().to_frame().transpose()]).max().to_frame().transpose()
        mm_data['weather'] = {'min': min.values[0], 'max': max.values[0]}
        return mm_data

    def get_size(self):
        return self.size

    def get_data_ids(self):
        return self.ids

    def get_data(self, id):
        h_ids = np.arange(id, id + self.t_h)
        f_ids = np.arange(id + self.t_h, id + self.t_h + self.t_f)
        e_h = self.data['energy'][:, np.array([h_ids]), :].transpose(0, 1).transpose(-1, -2).transpose(-2, -3).to(self.device)
        e_f = self.data['energy'][:, np.array([f_ids]), :].transpose(0, 1).transpose(-1, -2).transpose(-2, -3).to(self.device)
        w_h = self.data['weather'][:, np.array([h_ids]), :].transpose(0, 1).transpose(-1, -2).transpose(-2, -3).to(self.device)
        w_f = self.data['weather'][:, np.array([f_ids]), :].transpose(0, 1).transpose(-1, -2).transpose(-2, -3).to(self.device)
        t_h = self.data['time'][:, np.array([h_ids]), :].transpose(0, 1).transpose(-1, -2).transpose(-2, -3).to(self.device)
        t_f = self.data['time'][:, np.array([f_ids]), :].transpose(0, 1).transpose(-1, -2).transpose(-2, -3).to(self.device)
        time_stamp = self.data['timestamps'][id + self.t_h: id + self.t_h + self.t_f]
        return [e_h, w_h, t_h], [e_f, w_f, t_f], time_stamp

    def normalize_energy(self, e, device='cpu'):
        e_n = e.clone()
        for i in range(len(e_n)):
            #e_n[i] = (e[i] - torch.tensor(self.mm_data['energy']['min'][i], device=device)) / torch.tensor(
            #    (self.mm_data['energy']['max'][i] - self.mm_data['energy']['min'][i]), device=device)
            e_n[i] = (e[i] - torch.tensor(self.mm_data['energy']['mean'][i], device=device)) / torch.tensor(
                (self.mm_data['energy']['std'][i]), device=device)
        return e_n

    def denormalize_energy(self, e_n, device='cpu'):
        if isinstance(e_n, list):
            e = e_n.clone()
            for i in range(len(e)):
                #e[i] = e_n[i] * torch.tensor((self.mm_data['energy']['max'][i] - self.mm_data['energy']['min'][i]),
                #                             device=device) + torch.tensor(self.mm_data['energy']['min'][i], device=device)
                e[i] = e_n[i] * torch.tensor((self.mm_data['energy']['std'][i]),
                                             device=device) + torch.tensor(self.mm_data['energy']['mean'][i], device=device)
        else:
            e = e_n * torch.tensor((self.mm_data['energy']['std']),
                                         device=device) + torch.tensor(self.mm_data['energy']['mean'], device=device)
        return e

    def normalize_weather(self, w, device='cpu'):
        w_n = w.clone()
        for i in range(len(w_n)):
            w_min = torch.as_tensor(np.asarray(self.mm_data['weather']['min'], dtype=np.float32), device=device)
            w_max = torch.as_tensor(np.asarray(self.mm_data['weather']['max'], dtype=np.float32), device=device)
            w_n[i] = (w[i] - w_min) / (w_max - w_min)
        return w_n


def convert_timeofday(df):
    df.loc[df['timeofday'] == 'sunrise', 'timeofday'] = 1.0
    df.loc[df['timeofday'] == 'morning', 'timeofday'] = 2.0
    df.loc[df['timeofday'] == 'noon', 'timeofday'] = 3.0
    df.loc[df['timeofday'] == 'sunset', 'timeofday'] = 4.0
    df.loc[df['timeofday'] == 'night', 'timeofday'] = 5.0
    df['timeofday'] = pd.to_numeric(df['timeofday'])
    df['timeofday_sin'] = np.sin(2 * np.pi * df['timeofday'] / (df['timeofday'].max() - df['timeofday'].min() + 1))
    df['timeofday_cos'] = np.cos(2 * np.pi * df['timeofday'] / (df['timeofday'].max() - df['timeofday'].min() + 1))
    return df


def convert_season(df):
    # 1=winter, 2=spring, 3=summer, 4=autumn
    df.loc[df.index.month <= 2, 'season'] = 1
    df.loc[df.index.month == 12, 'season'] = 1
    df.loc[(df.index.month >= 3) & (df.index.month <= 5), 'season'] = 2
    df.loc[(df.index.month >= 6) & (df.index.month <= 8), 'season'] = 3
    df.loc[(df.index.month >= 9) & (df.index.month <= 11), 'season'] = 4
    df['season'] = pd.to_numeric(df['season'])
    df['season_sin'] = np.sin(2 * np.pi * df['season'] / (df['season'].max() - df['season'].min() + 1))
    df['season_cos'] = np.cos(2 * np.pi * df['season'] / (df['season'].max() - df['season'].min() + 1))
    return df


def convert_hour(df):
    df['hour_sin'] = np.sin(2 * np.pi * (df['hour']) / (df['hour'].max() - df['hour'].min() + 1))
    df['hour_cos'] = np.cos(2 * np.pi * (df['hour']) / (df['hour'].max() - df['hour'].min() + 1))
    return df


def convert_minute(df):
    df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / (df['minute'].max() - df['minute'].min() + 1))
    df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / (df['minute'].max() - df['minute'].min() + 1))
    return df


def convert_month(df):
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / (df['month'].max() - df['month'].min() + 1))
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / (df['month'].max() - df['month'].min() + 1))
    return df
