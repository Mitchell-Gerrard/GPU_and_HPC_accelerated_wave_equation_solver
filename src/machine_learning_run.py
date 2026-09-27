import numpy as np
import torch
import torch.nn as nn

INPUT_COLS = ['sigma', 'c', 'x0_frac', 'y0_frac']
OUTPUT_COLS = ['peak_amplitude', 'peak_time', 'final_energy',
               'energy_half_time', 'spread_x', 'spread_y']

class SurrogateMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dims: list, activation):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), activation()]
            prev = h
        layers.append(nn.Linear(prev, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Load checkpoint and normalization
ckpt = torch.load("results/surrogate_model.pt", map_location="cpu")
norm = np.load("results/surrogate_norm.npz")

model = SurrogateMLP(
    in_dim=ckpt["in_dim"],
    out_dim=ckpt["out_dim"],
    hidden_dims=ckpt["hidden_dims"],
    activation=nn.SiLU,
)
model.load_state_dict(ckpt["model_state"])
model.eval()

def predict_one(sigma, c, x0_frac, y0_frac):
    x = np.array([[sigma, c, x0_frac, y0_frac]], dtype=np.float32)
    x_n = (x - norm["x_mean"]) / norm["x_std"]

    with torch.no_grad():
        y_n = model(torch.tensor(x_n, dtype=torch.float32)).numpy()

    y = y_n * norm["y_std"] + norm["y_mean"]
    return dict(zip(OUTPUT_COLS, y[0]))
def plot_energy():
    INPUT_COLS  = ['sigma', 'c', 'x0_frac', 'y0_frac']
    OUTPUT_COLS = ['peak_amplitude', 'peak_time', 'final_energy',
               'energy_half_time', 'spread_x', 'spread_y']
pred = predict_one(6.0, 0.7, 0.5, 0.5)
print(pred)