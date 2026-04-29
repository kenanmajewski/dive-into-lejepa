"""
Dive into LeJEPA.

Standalone script that trains three encoders on concentric circles data
and saves a side-by-side embedding comparison as comparison.png.

Based on: Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised
Learning Without the Heuristics" (arXiv:2511.08544)

This implementation is written from scratch in JAX/Equinox by Kenan Majewski.
The original LeJEPA paper and official implementation are by Randall Balestriero
and Yann LeCun: https://github.com/rbalestr-lab/lejepa
"""

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import optax

matplotlib.rcParams.update(
    {
        "font.family": "monospace",
        "figure.dpi": 150,
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 16,
    }
)


@dataclass
class Config:
    seed: int = 43

    n_samples: int = 500
    noise_std: float = 0.05
    n_classes: int = 3

    hidden_dim: int = 32
    out_dim: int = 2
    lr: float = 5e-4
    aug_noise: float = 0.1
    num_slices: int = 10
    num_eval_points: int = 17

    n_steps: int = 500
    lambda_sigreg: float = 5e-6
    gamma_vcreg: float = 1.0
    mu_vcreg: float = 0.1

    red: str = "#e74c3c"
    blue: str = "#3498db"
    green: str = "#2ecc71"


def generate_circles(
    n_samples: int = 1000,
    noise_std: float = 0.05,
    n_classes: int = 3,
    seed: int = 41,
    n_dim: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.RandomState(seed)
    pts_per_class = n_samples // n_classes
    X = np.zeros((n_samples, n_dim))
    y = np.zeros(n_samples, dtype=int)
    for i in range(n_classes):
        radius = (i + 1) * 6
        angles = rng.uniform(0, 2 * np.pi, pts_per_class)
        noise = rng.normal(0, noise_std, pts_per_class)
        idx = slice(i * pts_per_class, (i + 1) * pts_per_class)
        X[idx, 0] = (radius + noise) * np.cos(angles)
        X[idx, 1] = (radius + noise) * np.sin(angles)
        if n_dim > 2:
            extra = rng.normal(0, 0.3, (pts_per_class, n_dim - 2))
            X[idx, 2:] = extra
        y[idx] = i
    return X, y


class SmallEncoder(eqx.Module):
    layers: list

    def __init__(self, key, in_dim=2, hidden_dim=32, out_dim=2):
        keys = jrand.split(key, 3)
        self.layers = [
            eqx.nn.Linear(in_dim, hidden_dim, key=keys[0]),
            eqx.nn.Linear(hidden_dim, hidden_dim, key=keys[1]),
            eqx.nn.Linear(hidden_dim, out_dim, key=keys[2]),
        ]

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = jax.nn.gelu(layer(x))
        return self.layers[-1](x)


def sigreg_loss_with_target(
    embeddings: jnp.ndarray,
    key: jax.Array,
    target_cf_fn,
    num_slices: int = 64,
    num_eval_points: int = 17,
) -> jnp.ndarray:
    N, K = embeddings.shape
    key, subkey = jrand.split(key)

    # Project to 1D
    directions = jrand.normal(subkey, (num_slices, K))
    directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
    projections = directions @ embeddings.T # Shape: (M, N)

    # CF computation over the [-5, 5] domain as in the paper
    t_points = jnp.linspace(-5.0, 5.0, num_eval_points)

    # projections[:, None, :] -> (M, 1, N)
    # t_points[None, :, None] -> (1, T, 1)
    # angles -> (M, T, N)
    angles = projections[:, None, :] * t_points[None, :, None]

    # Complex ECF: (1/N) * sum(cos) + i * (1/N) * sum(sin)
    ecf_real = jnp.mean(jnp.cos(angles), axis=-1) # (M, T)
    ecf_imag = jnp.mean(jnp.sin(angles), axis=-1)

    # Match against real-valued target CF
    target_cf = target_cf_fn(t_points) # (T, )

    # Distance: |ECF - target|^2 = (Real - target)^2 + Imag^2
    diff_sq = (ecf_real - target_cf[None, :]) ** 2 + ecf_imag**2

    # Apply the Epps-Pulley Gaussian weighting function w(t)
    weight = jnp.exp(-0.5 * t_points**2)
    err = diff_sq * weight[None, :]

    # Integrate using Trapezoidal rule and scale by N
    integral = jnp.trapezoid(err, x=t_points, axis=1) * N # Shape: (M,)

    return jnp.mean(integral) # Average over the M slices


def sigreg_loss(
    embeddings: jnp.ndarray,
    key: jax.Array,
    num_slices: int = 64,
    num_eval_points: int = 17,
) -> jnp.ndarray:
    # Default LeJEPA targets the Standard Normal distribution
    return sigreg_loss_with_target(
        embeddings,
        key,
        lambda t: jnp.exp(-(t**2) / 2),
        num_slices=num_slices,
        num_eval_points=num_eval_points,
    )


def vcreg_loss(
    embeddings: jnp.ndarray, gamma: float | jax.Array = 1.0, mu: float | jax.Array = 1.0
) -> jnp.ndarray:
    z_centered = embeddings - jnp.mean(embeddings, axis=0, keepdims=True)

    # Variance loss (hinge loss on standard deviation)
    std = jnp.sqrt(jnp.mean(z_centered**2, axis=0) + 1e-4)
    std_loss = jnp.mean(jnp.maximum(0.0, 1.0 - std))

    # Covariance loss (decorrelation)
    cov = (z_centered.T @ z_centered) / (z_centered.shape[0] - 1)
    off_diag = cov - jnp.diag(jnp.diag(cov))
    cov_loss = jnp.sum(off_diag**2) / cov.shape[0]

    return gamma * std_loss + mu * cov_loss


def make_total_loss(
    encoder,
    x: jnp.ndarray,
    key: jax.Array,
    lam: jnp.ndarray,
    gamma: jnp.ndarray,
    mu: jnp.ndarray,
    method: str,
    num_slices: int = 10,
    num_views: int = 2,
    aug_noise: float = 0.1,
) -> jnp.ndarray:
    noise_key, sigreg_key = jrand.split(key)

    # Generate V noisy views (num_views)
    noise = jrand.normal(noise_key, (num_views,) + x.shape) * aug_noise

    # Broadcasting x to all V views
    v_views = x[None, ...] + noise

    # Apply the encoder to ALL views at once
    # jax.vmap(encoder) processes a batch (N, D) -> (N, K)
    # jax.vmap(jax.vmap(encoder)) processes the views (V, N, D) -> (V, N, K)
    z_views = jax.vmap(jax.vmap(encoder))(v_views)

    # Predictive loss (Invariance)
    centers = jnp.mean(z_views, axis=0)
    pred_loss = jnp.mean((centers[None, ...] - z_views) ** 2)

    # Regularization loss
    if method == "sigreg":
        # Apply SIGReg to all V views
        vmap_sigreg = jax.vmap(
            lambda z: sigreg_loss(z, sigreg_key, num_slices=num_slices)
        )
        reg_loss = jnp.mean(vmap_sigreg(z_views))
        return (1.0 - lam) * pred_loss + lam * reg_loss
    elif method == "vcreg":
        vmap_vcreg = jax.vmap(lambda z: vcreg_loss(z, gamma, mu))
        reg_loss = jnp.mean(vmap_vcreg(z_views))
        return pred_loss + reg_loss
    else:
        return pred_loss


def train_one_method(
    method: str,
    key: jax.Array,
    x: jnp.ndarray,
    steps: int,
    lam: float,
    gamma: float,
    mu: float,
    num_slices: int,
    cfg: Config,
) -> np.ndarray:
    model = SmallEncoder(key, in_dim=x.shape[1], out_dim=cfg.out_dim)
    optimizer = optax.adam(cfg.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
    tkey = jrand.PRNGKey(cfg.seed)

    lam_jnp = jnp.array(lam)
    gamma_jnp = jnp.array(gamma)
    mu_jnp = jnp.array(mu)

    @eqx.filter_jit
    def train_step(m, os, k, l_val, g_val, m_val):
        loss_v, grads = eqx.filter_value_and_grad(make_total_loss)(
            m, x, k, l_val, g_val, m_val, method, num_slices, num_views=2,
            aug_noise=cfg.aug_noise,
        )
        updates, os = optimizer.update(grads, os, m)
        m = eqx.apply_updates(m, updates)
        return m, os

    for _ in range(steps):
        tkey, skey = jrand.split(tkey)
        model, opt_state = train_step(model, opt_state, skey, lam_jnp, gamma_jnp, mu_jnp)

    return np.array(jax.vmap(model)(x))


def main():
    cfg = Config()

    X_np, y_np = generate_circles(
        n_samples=cfg.n_samples, 
        noise_std=cfg.noise_std,
        n_classes=cfg.n_classes, 
        seed=cfg.seed,
    )
    X = jnp.array(X_np)

    methods = [
        ("none", "No Reg (Collapse)"),
        ("vcreg", "VCReg-style"),
        ("sigreg", "LeJEPA (SIGReg)"),
    ]

    print("Training encoders")
    results = {}
    for method_key, method_label in methods:
        print(f"> {method_label}")
        results[method_key] = train_one_method(
            method_key,
            jrand.PRNGKey(7),
            X,
            steps=cfg.n_steps,
            lam=cfg.lambda_sigreg,
            gamma=cfg.gamma_vcreg,
            mu=cfg.mu_vcreg,
            num_slices=cfg.num_slices,
            cfg=cfg,
        )

    class_colors = [cfg.red, cfg.blue, cfg.green]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for ax, (mkey, lbl) in zip(axes, methods):
        emb = results[mkey]
        for ci in range(cfg.n_classes):
            mask = y_np == ci
            ax.scatter(
                emb[mask, 0], emb[mask, 1],
                c=class_colors[ci], s=8, alpha=1.0, lw=0.3, ec="black",
            )
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_title(lbl, fontsize=16)
        ax.set_xlabel("Latent Dim 1")
        ax.set_ylabel("Latent Dim 2")
        ax.set_aspect("equal")
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)

    fig.suptitle(
        f"Step {cfg.n_steps}, "
        r"$\lambda$" + f"={cfg.lambda_sigreg:.1e}, "
        r"$\gamma$" + f"={cfg.gamma_vcreg:.1f}, "
        r"$\mu$" + f"={cfg.mu_vcreg:.1f}, "
        f"M={cfg.num_slices}",
        fontsize=16, y=1.05,
    )
    plt.tight_layout()
    plt.savefig("comparison.png", dpi=150, bbox_inches="tight")
    print("Saved comparison.png")


if __name__ == "__main__":
    main()
