from def3c_learner import DEF_3C
import numpy as np
import argparse
import os
import json
from types import SimpleNamespace
from tqdm import tqdm
import csv
import sys


def write_metric_csv(path, values, precision=5):
    """Save [num_exps, num_series] metrics."""
    with open(path, 'w', encoding='UTF8', newline='') as file:
        writer = csv.writer(file)
        for exp_id in range(values.shape[0]):
            writer.writerow([f"{values[exp_id, s]:.{precision}f}" for s in range(values.shape[1])])

def update_progress(completed, n_windows, batch_id, num_batches, generated, num_samples,):
    if interactive:
        if pbar.total is None:
            pbar.total = n_evaluations * n_windows * num_samples
        pbar.set_postfix_str(f"E{cnt + 1}/{params.num_exps} " f"S{seed_idx + 1}/{n_seeds} "
                             f"B{batch_id}/{num_batches} " f"K{generated}/{num_samples}", refresh=False,)
        pbar.update(completed)
    elif generated == num_samples:
        processed = min(batch_id * params.batch_size, n_windows)
        print(f"Exp {cnt + 1}/{params.num_exps}, " f"seed {seed_idx + 1}/{n_seeds}: "
              f"{processed}/{n_windows} windows completed", flush=True,)

parser = argparse.ArgumentParser()
parser.add_argument('--json', type=str, default='./json/gefcom_w_1.json')
parser.add_argument('--results_dir', type=str, default='./results')
parser.add_argument('--num_exps', type=int, default=4)
parser.add_argument('--best_epochs', type=int, nargs="+", default=[19, 17, 12, 16], )
parser.add_argument('--num_samples', type=int, default=100)
parser.add_argument('--sample_chunk', type=int, default=20)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--progress_mode', choices=['auto', 'bar', 'log'], default='auto',
                    help='auto: detect terminal; bar: force live progress; log: one line per completed batch')

params = parser.parse_args()
with open(params.json, 'r') as j:
    params_json = json.loads(j.read(), object_hook=lambda d: SimpleNamespace(**d))
params = SimpleNamespace(**params.__dict__, **params_json.__dict__)

if not os.path.isdir(params.results_dir):
    os.makedirs(params.results_dir, exist_ok=True)

def3c = DEF_3C(params)
start_id = int(params.exp.checkpoints_dir.split('_')[-1])

test_mae = np.zeros((params.num_exps, params.dataset.n_e))
test_rmse = np.zeros((params.num_exps, params.dataset.n_e))
test_huber = np.zeros((params.num_exps, params.dataset.n_e))
test_r2 = np.zeros((params.num_exps, params.dataset.n_e))
test_crps = np.zeros((params.num_exps, params.dataset.n_e))

interactive = sys.stderr.isatty() if params.progress_mode == "auto" else params.progress_mode == "bar"
n_seeds = len(params.eval.seeds)
if n_seeds == 0:
    raise ValueError("At least one evaluation seed is required.")
if len(params.best_epochs) < params.num_exps:
    raise ValueError("Provide a best epoch for every experiment.")

n_evaluations = params.num_exps * n_seeds

with tqdm(total=None, desc="Forecasts", unit="pred", position=0, leave=True, dynamic_ncols=True,
          mininterval=0.5, file=sys.stderr, disable=not interactive,) as pbar:

    for cnt, j in enumerate(range(start_id, start_id + params.num_exps)):
        checkpoints_dir = '_'.join(params.exp.checkpoints_dir.split('_')[:-1]) + '_' + str(j)
        best_epoch = params.best_epochs[cnt]
        filepath = checkpoints_dir + '/checkpoint_epoch_ema_' + str(best_epoch) + '.zip'
        def3c.load(filepath)
        for seed_idx, eval_seed in enumerate(params.eval.seeds):
            test_losses, test_stats, gt, _ = def3c.eval_probabilistic(split="test", seed=eval_seed,
                                                                      batch_size=params.batch_size,
                                                                      num_samples=params.num_samples,
                                                                      sample_chunk=params.sample_chunk,
                                                                      return_samples=False,
                                                                      verbose=False,
                                                                      progress_callback=update_progress,)

            test_mae[cnt] += test_losses["mae"]
            test_rmse[cnt] += test_losses["rmse"]
            test_huber[cnt] += test_losses["huber"]
            test_r2[cnt] += test_losses["r2"]
            test_crps[cnt] += test_stats["crps_per_series"]

        # Average across seeds for this experiment.
        test_mae[cnt] /= n_seeds
        test_rmse[cnt] /= n_seeds
        test_huber[cnt] /= n_seeds
        test_r2[cnt] /= n_seeds
        test_crps[cnt] /= n_seeds

        summary = (f"Exp {j}, epoch {best_epoch} | " f"MAE={np.mean(test_mae[cnt]):.5f}, "
                   f"RMSE={np.mean(test_rmse[cnt]):.5f}, " f"Huber={np.mean(test_huber[cnt]):.5f}, "
                   f"R2={np.mean(test_r2[cnt]):.5f}, " f"CRPS={np.mean(test_crps[cnt]):.4f}")
        if interactive:
            tqdm.write(summary, file=sys.stderr)
        else:
            print(summary, flush=True)

print('\nTest. av. MAE (across ' + str(params.num_exps) + ' experiments): ' + f"{np.mean(test_mae):.5f}")
print('Test. av. RMSE (across ' + str(params.num_exps) + ' experiments): ' + f"{np.mean(test_rmse):.5f}")
print('Test. av. Huber (across ' + str(params.num_exps) + ' experiments): ' + f"{np.mean(test_huber):.5f}")
print('Test. av. R2 (across ' + str(params.num_exps) + ' experiments): ' + f"{np.mean(test_r2):.5f}")
print('Test. av. CRPS (across ' + str(params.num_exps) + ' experiments): ' + f"{np.mean(test_crps):.4f}")


prefix = os.path.join(params.results_dir, params.dataset.name +'_' +str(start_id))
write_metric_csv(prefix + '_test_mae.csv', test_mae)
write_metric_csv(prefix + '_test_rmse.csv', test_rmse)
write_metric_csv(prefix + '_test_r2.csv', test_r2)
write_metric_csv(prefix + '_test_crps.csv', test_crps)


out_best_epochs = prefix + '_best_epochs.csv'
with open(out_best_epochs, 'w', encoding='UTF8', newline='') as file:
    writer = csv.writer(file)
    for j in range(params.num_exps):
        writer.writerow([f"{params.best_epochs[j]}"])