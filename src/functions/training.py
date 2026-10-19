import torch

def preprocess(dataset):
    X = dataset.data.float().div(255).view(-1, 784)   # flatten 28x28 -> 784 and scale pixels to [0, 1]
    return X, dataset.targets

def init_params(hidden_size=128, weight_init=0.01):
    """784 -> hidden_size -> 10. Small random weights, zero biases."""
    W1 = torch.randn(784, hidden_size) * weight_init
    b1 = torch.zeros(hidden_size)
    W2 = torch.randn(hidden_size, 10) * weight_init
    b2 = torch.zeros(10)
    params = [W1, b1, W2, b2]
    for p in params:
        p.requires_grad_()          # so loss.backward() fills in p.grad
    return params


def forward(X, params, activation="relu", dropout=0.0, training=False):
    """x -> z1 = x W1 + b1 -> h = act(z1) -> (dropout) -> logits = h W2 + b2"""
    W1, b1, W2, b2 = params
    h = ACTIVATIONS[activation](X @ W1 + b1)
    if training and dropout > 0:
        mask = (torch.rand_like(h) > dropout).float()
        h = h * mask / (1 - dropout)    # "inverted" dropout: rescale so the expected activation is unchanged
    return h @ W2 + b2


def loss_fn(logits, y, params, l1=0.0, l2=0.0):
    """Cross-entropy + optional L1 / L2 penalties on the weight matrices (biases are not penalized)."""
    W1, _, W2, _ = params
    loss = F.cross_entropy(logits, y)
    if l2 > 0:
        loss = loss + l2 * ((W1 ** 2).sum() + (W2 ** 2).sum())
    if l1 > 0:
        loss = loss + l1 * (W1.abs().sum() + W2.abs().sum())
    return loss


def make_optimizer(name, params, lr, momentum_beta=0.9, adam_beta2=0.999, adam_eps=1e-8):
    """Returns step(): one parameter update from each p.grad (then zeroes the grads).
    "batch_gd" and "sgd" are the same update rule; they differ only in batch size (see run_experiment)."""
    velocity = [torch.zeros_like(p) for p in params]    # momentum's running velocity
    m = [torch.zeros_like(p) for p in params]           # adam: running mean of gradients
    v = [torch.zeros_like(p) for p in params]           # adam: running mean of squared gradients
    t = 0                                               # adam: number of updates so far

    def step():
        nonlocal t
        t += 1
        with torch.no_grad():
            for i, p in enumerate(params):
                g = p.grad                                              # dL/dp from loss.backward()
                if name in ("batch_gd", "sgd"):
                    p -= lr * g                                         # theta = theta - lr * g
                elif name == "momentum":
                    velocity[i] = momentum_beta * velocity[i] + g       # v = beta * v + g
                    p -= lr * velocity[i]                               # theta = theta - lr * v
                elif name == "adam":
                    m[i] = momentum_beta * m[i] + (1 - momentum_beta) * g
                    v[i] = adam_beta2 * v[i] + (1 - adam_beta2) * g * g
                    m_hat = m[i] / (1 - momentum_beta ** t)             # bias correction
                    v_hat = v[i] / (1 - adam_beta2 ** t)
                    p -= lr * m_hat / (torch.sqrt(v_hat) + adam_eps)
                p.grad.zero_()

    return step


@torch.no_grad()
def evaluate(params, X, y, activation="relu"):
    """Cross-entropy and accuracy with dropout off and no penalty terms."""
    logits = forward(X, params, activation)
    return F.cross_entropy(logits, y).item(), (logits.argmax(dim=1) == y).float().mean().item()


def weight_stats(params):
    """How big / how sparse are the weights? (shows the effect of L1 vs L2)"""
    W = torch.cat([params[0].detach().flatten(), params[2].detach().flatten()])
    return {
        "L1 norm": W.abs().sum().item(),
        "L2 norm": W.norm().item(),
        "% near-zero": 100 * (W.abs() < 1e-3).float().mean().item(),
    }


def run_many(configs, **shared):
    """configs: {label: {overrides}}; shared: settings common to every run.
    Example: run_many({"sgd": dict(optimizer="sgd"), "adam": dict(optimizer="adam", lr=0.001)}, activation="relu")"""
    results = {}
    for label, overrides in configs.items():
        print(f"running {label} ...", end=" ")
        r = run_experiment(name=label, verbose=False, **{**shared, **overrides})
        print(f"val acc {r['history']['val_acc'].iloc[-1]:.4f} ({r['seconds']:.0f}s)")
        results[label] = r
    return results