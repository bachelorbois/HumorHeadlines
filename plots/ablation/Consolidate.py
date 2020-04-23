import csv
import numpy as np

N_configs = 13
N_runs = 10

all_runs_tl = [None]*N_configs
all_runs_tr = [None]*N_configs
all_runs_dl = [None]*N_configs
all_runs_dr = [None]*N_configs

train_loss = [0]*N_configs
train_RMSE = [0]*N_configs
dev_loss = [0]*N_configs
dev_RMSE = [0]*N_configs

train_loss_sd = [0]*N_configs
train_RMSE_sd = [0]*N_configs
dev_loss_sd = [0]*N_configs
dev_RMSE_sd = [0]*N_configs

for config in range(N_configs):
    runs_tl = [0]*N_runs
    runs_tr = [0]*N_runs
    runs_dl = [0]*N_runs
    runs_dr = [0]*N_runs
    for run in range(N_runs):
        with open(f"data/test-{config}-{run}.csv", "r") as fd:
            reader = csv.reader(fd, delimiter=",")
            data = []
            for row in reader:
                data.append(row)
        row = data[-1]
        assert row[0] == "24"
        runs_tl[run] = float(row[1])
        runs_tr[run] = float(row[2])
        runs_dl[run] = float(row[3])
        runs_dr[run] = float(row[4])

    all_runs_tl[config] = runs_tl.copy()
    all_runs_tr[config] = runs_tr.copy()
    all_runs_dl[config] = runs_dl.copy()
    all_runs_dr[config] = runs_dr.copy()

    train_loss[config] = np.mean(runs_tl)
    train_RMSE[config] = np.mean(runs_tr)
    dev_loss[config] = np.mean(runs_dl)
    dev_RMSE[config] = np.mean(runs_dr)

    train_loss_sd[config] = np.std(runs_tl)
    train_RMSE_sd[config] = np.std(runs_tr)
    dev_loss_sd[config] = np.std(runs_dl)
    dev_RMSE_sd[config] = np.std(runs_dr)


with open("all_points.csv", "w") as fd:
    writer = csv.writer(fd)
    writer.writerow(
        [
            "Group",
            "Kind",
            "Set",
            "Value"
        ]
    )

    for config in range(N_configs):
        for run in range(N_runs):
            writer.writerow([
                config,
                "Loss",
                "Train",
                all_runs_tl[config][run]
            ])
            writer.writerow([
                config,
                "RMSE",
                "Train",
                all_runs_tr[config][run]
            ])
            writer.writerow([
                config,
                "Loss",
                "Dev",
                all_runs_dl[config][run]
            ])
            writer.writerow([
                config,
                "RMSE",
                "Dev",
                all_runs_dr[config][run]
            ])

with open("summary.csv", "w") as fd:
    writer = csv.writer(fd)
    writer.writerow(
        [
            "Group",
            "Loss",
            "Loss_SD",
            "RMSE",
            "RMSE_SD",
            "Dev_Loss",
            "Dev_Loss_SD",
            "Dev_RMSE",
            "Dev_RMSE_SD"
        ]
    )

    for i in range(N_configs):
        writer.writerow(
            [
                i,
                train_loss[i],
                train_loss_sd[i],
                train_RMSE[i],
                train_RMSE_sd[i],
                dev_loss[i],
                dev_loss_sd[i],
                dev_RMSE[i],
                dev_RMSE_sd[i]
            ]
        )
