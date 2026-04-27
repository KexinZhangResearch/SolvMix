import torch


def pearson_correlation(x, y):
    x = x.float()
    y = y.float()
    x_mean = x.mean()
    y_mean = y.mean()
    x_std = x.std(unbiased=False)
    y_std = y.std(unbiased=False)
    cov = ((x - x_mean) * (y - y_mean)).mean()
    return cov / (x_std * y_std + 1e-8)


def spearman_correlation(x, y):
    x_rank = torch.argsort(torch.argsort(x)).float()
    y_rank = torch.argsort(torch.argsort(y)).float()
    return pearson_correlation(x_rank, y_rank)


def r2_score(pred, target):
    pred = pred.float()
    target = target.float()
    ss_res = torch.sum((target - pred) ** 2)
    ss_tot = torch.sum((target - target.mean()) ** 2)
    return 1.0 - ss_res / (ss_tot + 1e-12)


def regression_metrics(pred, target):
    mse = torch.nn.functional.mse_loss(pred.float(), target.float())
    mae = torch.nn.functional.l1_loss(pred.float(), target.float())
    return {
        "r2": r2_score(pred, target),
        "mae": mae,
        "mse": mse,
        "rmse": torch.sqrt(mse + 1e-12),
        "spearman": spearman_correlation(pred, target),
        "pearson": pearson_correlation(pred, target),
    }
