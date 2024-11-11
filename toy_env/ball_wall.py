import torch
from torch.distributions.normal import Normal
from torch.autograd import Variable
import matplotlib.pyplot as plt
from tqdm import tqdm
from mlp import mlp, SimNorm
from torch.optim import Adam
import torch.nn as nn
import numpy as np

import sys

relu_optimality_gap = []
simnorm_optimality_gap = []


for i in range(10):
    # gravity, height, width
    g, h, w = -9.81, 0.0, 5


    def f(x, v, th, a, t):
        ty = (-v * torch.cos(th) + (v**2 * torch.cos(th) ** 2 + a * w) ** 0.5) / a
        y = v * torch.sin(th) * ty + g / 2 * ty**2
        out = x + v * torch.cos(th) * t + 1 / 2 * a * t**2
        out = torch.where((h > y) & (ty < t), w, out)
        return out


    def f2(x, v, th, a, t):
        th1 = th[:, 0]  # First column (xx)
        th2 = th[:, 1]  # Second column (xx2)

        # Calculate intermediate values
        ty1 = (-v * torch.cos(th1) + (v**2 * torch.cos(th1)**2 + a * w) ** 0.5) / a
        y1 = v * torch.sin(th1) * ty1 + g / 2 * ty1**2

        out1 = x + v * torch.cos(th1) * t + 1 / 2 * a * t**2
        out1 = torch.where((h > y1) & (ty1 < t), w, out1)


        # For second column, use th2
        # Calculate intermediate values
        ty1 = (-v * torch.cos(th2) + (v**2 * torch.cos(th2)**2 + a * w) ** 0.5) / a
        y1 = v * torch.sin(th2) * ty1 + g / 2 * ty1**2

        out2 = x + v * torch.cos(th2) * t + 1 / 2 * a * t**2
        out2 = torch.where((h > y1) & (ty1 < t), w, out2)
        # out2 = x + v * torch.cos(th2) * t + 1 / 2 * a * t**2

        # Combine the outputs
        return torch.stack([out1, out2], dim=1)  # Shape: (1000, 2)


    # simulation variables
    samples = 1000
    xx = torch.linspace(-torch.pi, torch.pi, samples)
    xx2 = torch.linspace(-torch.pi, torch.pi, samples)
    xx_combined = torch.stack([xx, xx2], dim=1)  # Shape: (2, 1000)

    x, v, a, t = 0, 10, 1, 2
    yy = -f(x, v, xx, a, t)
    yy2 = -f2(x, v, xx_combined, a, t)
    std = 0.1  # noise for policy
    N = 5000  # data samples
    epochs = 100  # for optimization
    batch_size = 56
    lr = 2e-3

    # train simply MLP
    model0 = mlp(2, [32, 32], 2, last_layer="linear", last_layer_kwargs={})
    opt = Adam(model0.parameters(), lr=lr)
    steps = samples // batch_size
    print("Training...")
    model0.train()
    losses0 = []
    with tqdm(range(epochs), unit="epoch", total=epochs) as tepoch:
        for epoch in tepoch:
            epoch_loss = 0
            for step in range(steps):
                idx = torch.randint(0, samples, (batch_size,))
                _xx = xx_combined[idx].unsqueeze(1)
                _yy = yy2[idx].unsqueeze(1)
                pred = model0(_xx)
                loss = torch.mean((pred - _yy) ** 2)
                model0.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                losses0.append(loss.item())
            epoch_loss /= steps
            tepoch.set_postfix(loss=epoch_loss)


    # train TDMPC model
    model = mlp(
        2,
        [32],
        32,
        last_layer="normedlinear",
        last_layer_kwargs={"act": SimNorm(8)},
    )
    losses1 = []
    decoder = mlp(32, [], 2, last_layer="linear", last_layer_kwargs={})
    opt = Adam([{"params": model.parameters()}, {"params": decoder.parameters()}], lr=lr)
    print("Training...")
    model.train()

    with tqdm(range(epochs), unit="epoch", total=epochs) as tepoch:
        for epoch in tepoch:
            epoch_loss = 0
            for step in range(steps):
                idx = torch.randint(0, samples, (batch_size,))
                _xx = xx_combined[idx].unsqueeze(1)
                _yy = yy2[idx].unsqueeze(1)
                pred = decoder(model(_xx))
                loss = torch.mean((pred - _yy) ** 2)
                model.zero_grad()
                loss.backward()
                opt.step()
                epoch_loss += loss.item()
                losses1.append(loss.item())
            epoch_loss /= steps
            tepoch.set_postfix(loss=epoch_loss)

    model1 = lambda x: decoder(model(x))

    # fig, ax1 = plt.subplots(1, 1, figsize=(3, 2.6))

    # print("Plotting the problem landscape")
    # ax1.plot(xx, -f(x, v, xx, a, t), label=r"$J(\theta)$")
    models = {0: "ReLU", 1: "SimNorm", 2: "Spectral MLP"}
    predictions = {}
    for i, m in enumerate([model0, model1]):
        est = m(xx_combined.unsqueeze(1))
        # ax1.plot(xx, est.detach(), label=models[i])
        error = torch.mean((est - yy2) ** 2).item() ** 0.5
        print(f"Model has {error:.3f} approx error")
        predictions[i] = est

    opt_value = torch.min(yy2, dim=0)[0]
    # plt.plot(xx[0], yy[0], "x", color="black")
    # ii = 328
    # plt.plot(xx[ii], yy[ii], "x", color="tab:blue")
    # print(f"Opt error GT {yy2[ii]-opt_value}")
    est = predictions[0].squeeze(0)
    min_value = torch.min(est, dim=0)[0]

    # plt.plot(xx[argmin], est.detach()[argmin], color="tab:orange", marker="x")
    # plt.plot(xx[ii], yy[ii], "x", color="tab:blue")
    relu_opt_error = (min_value-opt_value).norm(dim=1, p=2).item()

    # print(f"Opt error MLP {(min_value-opt_value).norm(dim=1, p=2).item()}")
    est = predictions[1].squeeze(0)
    min_value = torch.min(est, dim=0)[0]
    # plt.plot(xx[argmin], est.detach()[argmin], color="tab:orange", marker="x")
    # plt.plot(xx[ii], yy[ii], "x", color="tab:blue")
    simnorm_opt_error = (min_value-opt_value).norm(dim=1, p=2).item()

    relu_optimality_gap.append(relu_opt_error)
    simnorm_optimality_gap.append(simnorm_opt_error)
    # print(f"Opt error MLP SimNorm {(min_value-opt_value).norm(dim=1, p=2).item()}")

    # ax1.set_xlabel(r"$\theta$")
    # ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3))

    # plt.tight_layout()
    # plt.savefig("ball_wall.pdf", bbox_inches="tight", pad_inches=0)


    # fig, ax = plt.subplots(1, 1, figsize=(3, 2.2))
    # cutoff = 1000
    # ax.plot(np.array(losses0)[:cutoff], label="ReLU", color="tab:orange")
    # ax.plot(np.array(losses1)[:cutoff], label="SimNorm", color="tab:green")
    # ax.set_xlabel("Iteration")
    # ax.set_ylabel("Loss")
    # plt.savefig("ball_wall_losses.pdf", bbox_inches="tight", pad_inches=0)

print("relu")
print(np.mean(relu_optimality_gap), np.std(relu_optimality_gap))
print("simnorm")
print(np.mean(simnorm_optimality_gap), np.std(simnorm_optimality_gap))
