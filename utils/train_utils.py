import torch


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def losses_cmpt(pred, gt, l_func=None, training=False):
    """
    pred: Batch_size x Num_of_energy_locs x Num_of_timesteps x Vector_size
    gt: Batch_size x Num_of_energy_locs x Num_of_timesteps x Vector_size
    """
    mse = torch.nn.MSELoss(reduction='none')
    mae = torch.nn.L1Loss(reduction='none')
    mae_ar = torch.zeros(pred.shape[1])
    mse_ar = torch.zeros(pred.shape[1])
    rmse_ar = torch.zeros(pred.shape[1])

    for i in range(pred.shape[1]):
        mse_l = mse(pred[:, i, :, :], gt[:, i, :, :])
        mae_l = mae(pred[:, i, :, :], gt[:, i, :, :])
        mse_l = torch.sum(mse_l)
        mae_l = torch.sum(mae_l)
        mae_ar[i] = mae_l / (pred.shape[0] * pred.shape[2])
        mse_ar[i] = mse_l / (pred.shape[0] * pred.shape[2])
        rmse_ar[i] = torch.sqrt(mse_l / (pred.shape[0] * pred.shape[2]))

    if l_func == 'rmse':
        loss_batch = torch.mean(rmse_ar)
    elif l_func == 'mse':
        loss_batch = torch.mean(mse_ar)
    elif l_func == 'mae':
        loss_batch = torch.mean(mae_ar)
    else:
        loss_batch = None
    return loss_batch, mae_ar, mse_ar, rmse_ar
