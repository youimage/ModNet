import torch
import random
import numpy as np
import matplotlib.pyplot as plt

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

def plot_loss_curve(losses):
    plt.plot(losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve (mod N)")
    plt.grid()
    plt.savefig("modnet_pytorch/loss_curve.png")
    plt.show()
