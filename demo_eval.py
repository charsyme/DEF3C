from def3c_learner import DEF_3C
import numpy as np
import argparse
import os
import json
from types import SimpleNamespace
from tqdm import tqdm
import csv


parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, default='./json/gefcom_w_1.json')
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--sample_chunk', type=int, default=8)
params = parser.parse_args()


with open(params.json, 'r') as j:
    params_json = json.loads(j.read(), object_hook=lambda d: SimpleNamespace(**d))
params = SimpleNamespace(**params.__dict__, **params_json.__dict__)

if not os.path.isdir(params.results_dir):
    os.makedirs(params.results_dir, exist_ok=True)

def write_metric_csv(path, values, precision=5):
    """Save [num_exps, num_series] metrics."""
    with open(path, 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        for exp_id in range(values.shape[0]):
            writer.writerow([f"{values[exp_id, s]:.{precision}f}" for s in range(values.shape[1])])


def3c = DEF_3C(params)
id = int(params.exp.checkpoints_dir.split('_')[-1])
best_epochs = []
loss_min = 1e+10
ep_min = 1
checkpoints_dir = '_'.join(params.exp.checkpoints_dir.split('_')[:-1]) + '_' + str(j)
filepath = checkpoints_dir +  '/checkpoint_epoch_ema_' + str(1)
num_epochs = 0
for name in os.listdir(checkpoints_dir):
    full_path = os.path.join(checkpoints_dir, name)
    if os.path.isfile(full_path) and 'ema' in full_path:
        num_epochs += 1
for ep in range(1, num_epochs+1):
    loss_val = 0
    filepath = checkpoints_dir + '/checkpoint_epoch_ema_' + str(ep) + '.zip'
    def3c.load(filepath)
    for seed in params.eval.seeds:
        losses_val, _, _, _  = def3c.eval_probabilistic(split='val', seed=seed, num_samples=params.num_samples,
                                                        sample_chunk=params.sample_chunk, return_samples=False,
                                                        compute_quantiles=False, verbose=False,)
        if params.training.use_huber:
            loss_val = loss_val + np.mean(losses_val['huber'])
        else:
            loss_val = loss_val + np.mean(losses_val['rmse'])
    loss_val = loss_val / len(params.eval.seeds)
    if loss_val < loss_min:
        loss_min = loss_val
        ep_min = ep
best_epochs.append(ep_min)

val_mae = np.zeros([1, params.dataset.n_e])
test_mae = np.zeros([1, params.dataset.n_e])
val_rmse = np.zeros([1, params.dataset.n_e])
test_rmse = np.zeros([1, params.dataset.n_e])
test_crps = np.zeros([1, params.dataset.n_e])

val_huber = np.zeros([1, params.dataset.n_e])
test_huber = np.zeros([1, params.dataset.n_e])
val_r2 = np.zeros([1, params.dataset.n_e])
test_r2 = np.zeros([1, params.dataset.n_e])
val_crps = np.zeros([1, params.dataset.n_e])

cnt = 0
checkpoints_dir = '_'.join(params.exp.checkpoints_dir.split('_')[:-1]) + '_' + str(j)
filepath = checkpoints_dir + '/checkpoint_epoch_ema_' + str(best_epochs[cnt])
def3c.load(filepath)
for seed, i in enumerate(params.eval.seeds):
    val_losses, _, _, _  = def3c.eval_probabilistic(split='val', seed=seed, num_samples=params.num_samples,
                                                    sample_chunk=params.sample_chunk, return_samples=False,
                                                    compute_quantiles=False, verbose=False,)
    test_losses, _, _, _  = def3c.eval_probabilistic(split='test', seed=seed,
                                                     sample_chunk=params.sample_chunk, return_samples=False,
                                                     compute_quantiles=False, verbose=False, )

    test_mae[cnt] += test_losses['mae']
    test_rmse[cnt] += test_losses['rmse']
    test_huber[cnt] += test_losses['huber']
    test_r2[cnt] += test_losses['r2']
    test_crps[cnt] += test_losses['crps_per_series']


n_seeds = len(params.eval.seeds)
test_mae[cnt] /= n_seeds
test_rmse[cnt] /= n_seeds
test_huber[cnt] /= n_seeds
test_r2[cnt] /= n_seeds
test_crps[cnt] /= n_seeds
print(
    'Exp: ' + str(j) + ', best epoch: ' + str(best_epochs[cnt]) +
    ', test. av. MAE: ' + f"{np.mean(test_mae[cnt]):.5f}" +
    ', test. av. RMSE: ' + f"{np.mean(test_rmse[cnt]):.5f}" +
    ', test. av. Huber: ' + f"{np.mean(test_huber[cnt]):.5f}" +
    ', test. av. R2: ' + f"{np.mean(test_r2[cnt]):.5f}" +
    ', test. av. CRPS: ' + f"{np.mean(test_crps[cnt]):.4f}" + '\n'
)


print('\nTest. av. MAE ' + f"{np.mean(test_mae):.5f}")
print('Test. av. RMSE ' + f"{np.mean(test_rmse):.5f}")
print('Test. av. Huber ' + f"{np.mean(test_huber):.5f}")
print('Test. av. R2 ' + f"{np.mean(test_r2):.5f}")
print('Test. av. CRPS ' + f"{np.mean(test_crps):.4f}")

prefix = os.path.join(params.results_dir, params.dataset.name)
write_metric_csv(prefix + '_test_mae.csv', test_mae)
write_metric_csv(prefix + '_test_rmse.csv', test_rmse)
write_metric_csv(prefix + '_test_r2.csv', test_r2)
write_metric_csv(prefix + '_test_crps.csv', test_crps)


out_best_epochs = prefix + '_best_epochs.csv'
with open(out_best_epochs, 'w', encoding='UTF8', newline='') as file:
    writer = csv.writer(file)
    for j in range(cnt):
        writer.writerow([f"{best_epochs[j]}"])
