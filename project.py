import marimo

__generated_with = "0.22.5"
app = marimo.App(width="wide")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <div style="text-align: left; padding: 20px 0;">
        <h1 style="font-family: monospace; font-size: 2.5em; margin: 0;">
            Dive into LeJEPA: Provable Self-Supervised Learning Without the Heuristics
        </h1>
        <p style="font-family: monospace; color: #555; margin-top: 10px; font-style: italic;">
            Implemented from scratch in JAX and Equinox
        </p></div>
    </div>

    **[arXiv:2511.08544](https://arxiv.org/abs/2511.08544)** - Randall Balestriero & Yann LeCun

    Notebook by [Kenan Majewski](https://github.com/kenanmajewski)

    ---


    Joint-Embedding Predictive Architectures (JEPAs) learn representations by predicting the embedding of one view from another. But existing methods rely on fragile heuristics, top-gradients [[Chen & He, 2021](https://arxiv.org/abs/2011.10566)], teacher-student networks, and EMA schedules [[Grill et al., 2020](https://arxiv.org/abs/2006.07733);[Caron et al., 2021](https://arxiv.org/abs/2104.14294)], to prevent **representation collapse**.

    **LeJEPA** replaces these heuristics with a single, principled regularizer:
    **SIGReg** (Sketched Isotropic Gaussian Regularization). The key insight:

    1. **Theory**: Embeddings should follow an **isotropic Gaussian** distribution
       $\mathcal{N}(\mathbf{0}, \mathbf{I})$ to minimize downstream prediction risk.
    2. **Practice**: SIGReg enforces this via **random projections** and
       **characteristic function** matching - $O(NMK)$ complexity, no heuristics.

    ## This notebook

    Navigate through the cells to explore each concept **interactively**:

    - **The Collapse Problem** - why prediction alone fails.
    - **SIGReg from Scratch** - a clean, readable implementation.
    - **Why Isotropic Gaussian?** - bias-variance proofs.
    - **LeJEPA in Action** - side-by-side comparison on 2D data.
    - **Our Extension** - SIGReg with adaptive distribution targets beyond $\mathcal{N}(0, I)$ based on task priors.
    """)
    return


@app.cell(hide_code=True)
def _():
    import marimo as mo
    import jax
    import jax.numpy as jnp
    import jax.random as jrand
    import equinox as eqx
    import optax
    import numpy as np
    import matplotlib
    import matplotlib.pyplot as plt
    from sklearn.linear_model import Lasso

    matplotlib.rcParams.update(
        {
            "font.family": "monospace",
            "figure.dpi": 110,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "figure.figsize": (6, 4),
        }
    )

    class Config:
        """Centralized Configuration for the Notebook."""

        n_samples = 500
        noise_std = 0.05
        n_classes = 3
        seed = 41

        # Model
        hidden_dim = 32
        out_dim = 2
        out_dim_quant = 128

        # Training
        lr = 5e-4
        aug_noise = 0.1

        # SIGReg
        num_slices = 64
        num_eval_points = 17

        # Colors
        red = "#e74c3c"
        blue = "#3498db"
        green = "#2ecc71"

    def generate_circles(n_samples=1000, noise_std=0.05, n_classes=3, seed=41, n_dim=2):
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

    return (
        Config,
        Lasso,
        SmallEncoder,
        eqx,
        generate_circles,
        jax,
        jnp,
        jrand,
        mo,
        np,
        optax,
        plt,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The Collapse Problem

    JEPAs train an encoder $f_\theta$ to make the embeddings of two views of the same data point predictable from each other:

    $$\mathcal{L}_{\text{pred}} = \|f_\theta(x_{v1}) - f_\theta(x_{v2})\|^2$$

    But this objective has a trivial solution: **collapse**. If $f_\theta$ maps every input to the same point, $\mathcal{L}_{\text{pred}} = 0$.

    We train a small MLP on 2D data (three concentric circles) with two views created by adding Gaussian noise. Adjust the steps and click "Train" to watch the embeddings collapse into a single cluster.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    n_steps_collapse_slider = mo.ui.slider(
        10, 50, step=10, value=20, label="Training Steps"
    )
    run_collapse_btn = mo.ui.run_button(label="Train Model")
    collapse_controls = mo.vstack([n_steps_collapse_slider, run_collapse_btn])
    return collapse_controls, n_steps_collapse_slider, run_collapse_btn


@app.cell(hide_code=True)
def _(
    Config,
    SmallEncoder,
    collapse_controls,
    eqx,
    generate_circles,
    jax,
    jnp,
    jrand,
    mo,
    n_steps_collapse_slider,
    np,
    optax,
    plt,
    run_collapse_btn,
):
    mo.stop(
        not run_collapse_btn.value,
        mo.hstack(
            [
                collapse_controls,
                mo.md("**<- Select steps and click Train Model to start.**"),
            ]
        ),
    )

    def build_collapse_ui():
        def pred_loss(model, x, key):
            k1, k2 = jrand.split(key)
            v1 = x + jrand.normal(k1, x.shape) * Config.aug_noise
            v2 = x + jrand.normal(k2, x.shape) * Config.aug_noise
            z1 = jax.vmap(model)(v1)
            z2 = jax.vmap(model)(v2)
            return jnp.mean((z1 - z2) ** 2)

        with mo.status.spinner("Training model to demonstrate collapse"):
            X_np, y_np = generate_circles(n_samples=Config.n_samples)
            X = jnp.array(X_np)

            model = SmallEncoder(jrand.PRNGKey(7))
            opt = optax.adam(Config.lr)
            opt_st = opt.init(eqx.filter(model, eqx.is_inexact_array))

            n_steps = n_steps_collapse_slider.value
            snap_steps = sorted({0, 10, 20} | {min(n_steps, s) for s in [30, 40, 50]})
            tkey = jrand.PRNGKey(123)

            snaps = [(0, np.array(jax.vmap(model)(X)))]

            @eqx.filter_jit
            def train_step(m, os, k):
                loss_v, grads = eqx.filter_value_and_grad(pred_loss)(m, X, k)
                updates, os = opt.update(grads, os, m)
                m = eqx.apply_updates(m, updates)
                return m, os

            for step in range(1, n_steps + 1):
                tkey, skey = jrand.split(tkey)
                model, opt_st = train_step(model, opt_st, skey)

                if step in snap_steps:
                    snaps.append((step, np.array(jax.vmap(model)(X))))

            if snaps[-1][0] != n_steps:
                snaps.append((n_steps, np.array(jax.vmap(model)(X))))

            fig, axes = plt.subplots(1, len(snaps), figsize=(3.5 * len(snaps), 3.5))
            if len(snaps) == 1:
                axes = [axes]

            colors = [Config.red, Config.blue, Config.green]
            for ax_item, (s_step, emb_s) in zip(axes, snaps):
                ax_item.grid(True, linestyle=":", alpha=0.6)
                for ci in range(3):
                    mask = y_np == ci
                    ax_item.scatter(
                        emb_s[mask, 0],
                        emb_s[mask, 1],
                        c=colors[ci],
                        s=8,
                        alpha=1.0,
                        lw=0.3,
                        ec="black",
                    )
                ax_item.set_title(f"Step {s_step}", fontsize=11)
                ax_item.set_xlabel("Latent Dim 1")
                ax_item.set_ylabel("Latent Dim 2")
                ax_item.set_xlim(-2, 2)
                ax_item.set_ylim(-2, 2)
                ax_item.set_aspect("equal")

            fig.suptitle(
                "Prediction-Only Training -> Complete Collapse", fontsize=13, y=1.05
            )
            plt.tight_layout()

        return mo.vstack([collapse_controls, fig])

    build_collapse_ui()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why do we care? And why don't we just simply use a decoder?

    Representation collapse means the encoder has found a "cheat code" to perfectly satisfy the prediction objective without learning any meaningful information about the input. If all images map to the exact same vector, predicting view B from view A is perfectly accurate (error = 0), but the vector is completely useless for any downstream task.

    **Why not just use a decoder?**
    Reconstruction-based methods (like Autoencoders or Masked Autoencoders) avoid collapse by forcing the network to decode the representation back into the original pixel/input space. However, decoding forces the representation to memorize *EVERYTHING*, including low-level, irrelevant details (like the exact texture of background grass or high-frequency noise). By dropping the decoder and doing prediction *only in latent space*, the model can learn abstract, high-level semantics and ignore irrelevant details-but we need a way to prevent the representation from collapsing into a single point.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SIGReg: From Scratch

    SIGReg (Sketched Isotropic Gaussian Regularization) is the core innovation
    of LeJEPA. It enforces that embeddings follow $\mathcal{N}(\mathbf{0}, \mathbf{I})$
    by combining two key ideas:

    ### 1. The Cramér-Wold Principle

    A distribution is uniquely determined by **all its 1D projections**.
    Instead of testing the full $K$-dimensional distribution, we project along
    $M$ random directions and compare each 1D distribution to $\mathcal{N}(0, 1)$.

    ### 2. Characteristic Function Matching

    > **What is a Characteristic Function (CF)?**
    > It is the Fourier transform of a probability distribution. Every distribution has a unique CF. For a Standard Normal distribution, its CF is exactly $e^{-t^2/2}$. By forcing the empirical CF of our projected data to match $e^{-t^2/2}$, we force the data itself to become a Standard Normal distribution.

    For each projection $a_m^\top z$, we compare its empirical characteristic
    function to the theoretical CF of $\mathcal{N}(0, 1)$:

    $$\hat{\varphi}(t) = \frac{1}{N}\sum_{n=1}^{N} e^{i t \cdot a_m^\top z_n}
    \quad\text{vs}\quad
    \varphi_{\mathcal{N}(0,1)}(t) = e^{-t^2/2}$$

    Below, visualize how random projections project 2D data to 1D distributions.
    Drag the angle to sweep through projection directions:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    sigreg_angle = mo.ui.slider(
        0, 180, step=1, value=45, label="Projection Angle (degrees)"
    )
    return (sigreg_angle,)


@app.cell(hide_code=True)
def _(Config, mo, np, plt, sigreg_angle):
    def build_sigreg_ui():
        angle_rad = np.deg2rad(sigreg_angle.value)
        direction = np.array([np.cos(angle_rad), np.sin(angle_rad)])
        rng_proj = np.random.RandomState(42)

        X_demo = rng_proj.randn(500, 2) * np.array([1.5, 0.5])
        X_demo = X_demo @ np.array(
            [[np.cos(0.5), -np.sin(0.5)], [np.sin(0.5), np.cos(0.5)]]
        )

        pvals = X_demo @ direction
        t_vals = np.linspace(0.1, 5.0, 50)
        emp_cos_demo = np.mean(np.cos(np.outer(t_vals, pvals)), axis=1)
        emp_sin_demo = np.mean(np.sin(np.outer(t_vals, pvals)), axis=1)
        emp_magnitude = np.sqrt(emp_cos_demo**2 + emp_sin_demo**2)
        target_cf_vals = np.exp(-(t_vals**2) / 2)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        ax_data, ax_hist, ax_cf = axes

        ax_data.scatter(
            X_demo[:, 0],
            X_demo[:, 1],
            s=5,
            alpha=0.8,
            c=Config.blue,
            lw=0.3,
            ec="black",
        )
        ax_data.annotate(
            "",
            xy=(direction[0] * 3, direction[1] * 3),
            xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color=Config.red, lw=2.5),
        )
        ax_data.grid(True, linestyle=":", alpha=0.6)
        ax_data.set_xlabel("Dimension 1")
        ax_data.set_ylabel("Dimension 2")
        ax_data.set_xlim(-5, 5)
        ax_data.set_ylim(-5, 5)
        ax_data.set_aspect("equal")
        ax_data.set_title("2D Data + Projection Direction")

        proj_std = np.std(pvals)
        proj_mean = np.mean(pvals)
        tr = np.linspace(proj_mean - 4 * proj_std, proj_mean + 4 * proj_std, 100)

        ax_hist.grid(True, linestyle=":", alpha=0.6)
        ax_hist.hist(
            pvals,
            bins=40,
            density=True,
            alpha=0.6,
            color=Config.blue,
            label="Projected data",
            ec="black",
        )
        ax_hist.plot(
            tr,
            np.exp(-0.5 * ((tr - proj_mean) / proj_std) ** 2)
            / (proj_std * np.sqrt(2 * np.pi)),
            Config.red,
            lw=2,
            label=r"$\mathcal{N}(0,1)$ reference",
        )
        ax_hist.set_xlabel("Projection Value")
        ax_hist.set_ylabel("Density")
        ax_hist.set_title(r"1D Projection ($\theta$" + f"={sigreg_angle.value}"+r"$^{\circ}$)")
        ax_hist.legend(fontsize=9)

        ax_cf.grid(True, linestyle=":", alpha=0.6)
        ax_cf.plot(
            t_vals, emp_magnitude, Config.blue, lw=2, label=r"$|\hat{\varphi}(t)|$"
        )
        ax_cf.plot(
            t_vals, target_cf_vals, Config.red, lw=2, label=r"$e^{-t^2/2}$ (target)"
        )
        ax_cf.set_xlabel("t")
        ax_cf.set_ylabel("|CF|")
        ax_cf.set_title("Characteristic Function Matching")
        ax_cf.legend(fontsize=9)

        plt.tight_layout()
        return mo.vstack([mo.vstack([sigreg_angle]), fig])

    build_sigreg_ui()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What are Projections doing?

    Think of random projections as taking **"shadows"** of your high-dimensional embeddings.

    - **The Challenge:** Directly checking if a 64-dimensional blob of data is "Gaussian" is mathematically and computationally nearly impossible (the *curse of dimensionality*).
    - **The Solution (Cramér-Wold Principle):** We don't need to check the whole 64D shape at once. The math tells us that if we look at enough 1D "shadows" from random angles, we can determine the original shape.
    - **How it works:**
        1. We pick a random direction (a "projection angle").
        2. We rotate our 64D data to look at it from that angle (the dot product).
        3. We measure how "Gaussian" that 1D shadow looks.
        4. We repeat this for $M$ different angles.

    By forcing **every single projection** to be a standard normal bell curve, we mathematically guarantee that the original high-dimensional object must also be a standard normal isotropic Gaussian. It turns a high-dimensional problem into a simple 1D matching task.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How Many Projections Do You Need?

    The Cramér-Wold principle says a distribution is uniquely determined by **all** its 1D projections. SIGReg samples **M** random projection directions and matches each one's empirical characteristic function to $\mathcal{N}(0, I)$.

    **The "X" Challenge:** To test if this works, we deliberately **corrupt the data** by forcing the first two dimensions into an "X" shape. The remaining dimensions are standard Gaussian.

    Drag the slider to observe how effectively even a small number of projections can reconstruct the target Gaussian distribution.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    m_slider = mo.ui.slider(1, 256, step=1, value=16, label="Number of Projections (M)")
    n_slider = mo.ui.slider(100, 5000, step=100, value=1000, label="Number of Samples (N)")
    k_slider = mo.ui.slider(2, 512, step=1, value=2, label="Number of Dimensions (K)")

    controls = mo.vstack([m_slider, n_slider, k_slider])
    return controls, k_slider, m_slider, n_slider


@app.cell(hide_code=True)
def _(Config, controls, jnp, jrand, k_slider, m_slider, mo, n_slider, np, plt):
    def build_proj_ui():
        M = m_slider.value
        N = n_slider.value
        K = k_slider.value
        key = jrand.PRNGKey(0)

        k1, k2, k3 = jrand.split(key, 3)
        labels = jrand.bernoulli(k1, 0.5, (N,))
        x = jrand.uniform(k2, (N,), minval=-3, maxval=3)
        y = jnp.where(labels, x, -x)
        noise = jrand.normal(k1, (N, 2)) * 0.1
        x_2d = jnp.stack([x, y], axis=1) + noise


        x_rest = jrand.normal(k3, (N, K - 2))
        data = jnp.concatenate([x_2d, x_rest], axis=1)

        directions = jrand.normal(k2, (M, K))
        directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

        projections = jnp.array(np.array(directions) @ np.array(data).T)

        t_points = jnp.linspace(0.1, 5.0, 50)
        emp_cos = jnp.mean(jnp.cos(projections[:, None, :] * t_points[None, :, None]), axis=-1)
        emp_sin = jnp.mean(jnp.sin(projections[:, None, :] * t_points[None, :, None]), axis=-1)
        emp_mag = jnp.sqrt(emp_cos**2 + emp_sin**2)
        avg_mag = jnp.mean(emp_mag, axis=0)

        target_cf = jnp.exp(-(t_points**2) / 2)

        loss_values = []
        m_range = list(range(1, M + 1))
        for m in m_range:
            diff_sq = (jnp.mean(emp_cos[:m], axis=0) - target_cf) ** 2 + jnp.mean(emp_sin[:m], axis=0) ** 2
            loss_values.append(float(jnp.mean(diff_sq)))

        projections_np = np.array(projections)
        target_cf_np = np.array(target_cf)
        avg_mag_np = np.array(avg_mag)
        emp_mag_np = np.array(emp_mag)
        t_np = np.array(t_points)

        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

        n_show = min(M, 6)
        t_density = np.linspace(-4, 4, 200)
        target_dens = np.exp(-t_density**2 / 2) / np.sqrt(2 * np.pi)

        ax_hist = axes[0]
        colors_proj = plt.cm.Set2(np.linspace(0, 1, max(n_show, 1)))
        for i in range(n_show):
            ax_hist.hist(
                projections_np[i],
                bins=40,
                density=True,
                alpha=0.3,
                color=colors_proj[i],
                label=f"Proj {i+1}" if M <= 6 else None,
            )
        ax_hist.plot(t_density, target_dens, "k-", lw=2.5, label=r"$\mathcal{N}(0,1)$")
        ax_hist.set_title(f"1D Projections ({n_show}/{M} shown)")
        ax_hist.set_xlabel("Projection value")
        ax_hist.set_ylabel("Density")
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, linestyle=":", alpha=0.6)

        ax_cf = axes[1]
        for i in range(min(M, 8)):
            ax_cf.plot(t_np, emp_mag_np[i], color=Config.blue, alpha=0.15, lw=0.7)
        ax_cf.plot(t_np, avg_mag_np, color=Config.blue, lw=2.5, label=r"$|\hat{\varphi}(t)|$" + f" (avg over {M})")
        ax_cf.plot(t_np, target_cf_np, color=Config.red, lw=2.5, ls="-", label=r"Target $e^{-t^2/2}$")
        ax_cf.set_title("Characteristic Function Matching")
        ax_cf.set_xlabel("t")
        ax_cf.set_ylabel("|CF|")
        ax_cf.legend(fontsize=9)
        ax_cf.grid(True, linestyle=":", alpha=0.6)

        ax_loss = axes[2]
        ax_loss.plot(m_range, loss_values, Config.blue, lw=2)
        ax_loss.axvline(M, color=Config.red, ls="--", lw=1.5, alpha=0.7, label=f"M = {M}")
        ax_loss.set_title("SIGReg Loss vs. Number of Projections")
        ax_loss.set_xlabel("M (number of projections)")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend(fontsize=12, loc="upper center", frameon=False)
        ax_loss.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()

        return mo.vstack([controls, fig])

    build_proj_ui()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Why Isotropic Gaussian? The Bias-Variance Argument

    LeJEPA proves that the isotropic Gaussian $\mathcal{N}(\mathbf{0}, \mathbf{I})$ is the **optimal embedding distribution** for minimizing worst-case risk on arbitrary downstream linear probes (Lemmas 1–2, Theorem 1).

    **Anisotropy amplifies both bias and variance.** When embeddings have anisotropic covariance (eigenvalues $\lambda_1 \neq \lambda_2$), the linear probe estimator has higher variance across training sets.

    Adjust the anisotropy and see how the variance of learned decision boundaries changes:
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    aniso_slider = mo.ui.slider(
        1.0,
        20.0,
        step=1.0,
        value=10.0,
        label=r"Anisotropy Ratio ($\lambda_{max}$ / $\lambda_{min}$)",
    )
    return (aniso_slider,)


@app.cell(hide_code=True)
def _(Config, aniso_slider, mo, np, plt):
    def build_aniso_ui():
        scale = 0.5
        ratio = aniso_slider.value
        theta = np.deg2rad(-45)
        R = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        cov_base = np.array([[1.0, 0], [0, ratio]])
        cov_normalized = cov_base * (2.0 / np.trace(cov_base))
        cov_aniso = scale * R @ cov_normalized @ R.T
        cov_iso = scale * np.eye(2)

        np.random.seed(42)
        fig, (ax_a, ax_i) = plt.subplots(1, 2, figsize=(10, 5))

        for ax_item, cov_val, title in [
            (ax_a, cov_aniso, f"Anisotropic (ratio={ratio:.1f})"),
            (ax_i, cov_iso, f"Isotropic (ratio=1.0)"),
        ]:
            ax_item.grid(True, linestyle=":", alpha=0.6)
            # Generate the data points: draw from Gaussian and splits at x1+x2=0
            X_data = np.random.multivariate_normal([0, 0], cov_val, 900)
            y_data = (X_data[:, 0] + X_data[:, 1] > 0).astype(float)

            # Plot the scatter points
            ax_item.scatter(
                X_data[y_data == 0, 0],
                X_data[y_data == 0, 1],
                c=Config.red,
                s=15,
                alpha=1.0,
                lw=0.3,
                ec="black",
            )
            ax_item.scatter(
                X_data[y_data == 1, 0],
                X_data[y_data == 1, 1],
                c=Config.blue,
                s=15,
                alpha=1.0,
                lw=0.3,
                ec="black",
            )

            # The True Green Boundary (Bayes optimal for this rule is x2 = -x1)
            x_line = np.linspace(-6, 6, 50)
            ax_item.plot(
                x_line, -x_line, color="green", lw=3, label="True boundary", zorder=10
            )

            beta_list = []

            for trial in range(50):
                X_probe = np.random.multivariate_normal([0, 0], cov_val, 30)
                y_probe = (X_probe[:, 0] + X_probe[:, 1] > 0).astype(float) - 0.5

                Xd = np.column_stack([X_probe, np.ones(30)])
                beta = np.linalg.lstsq(Xd, y_probe, rcond=None)[0]
                beta_list.append(beta)

                ax_item.plot(
                    x_line,
                    -(beta[0] * x_line + beta[2]) / (beta[1] + 1e-10),
                    "purple",
                    alpha=0.2,
                    lw=0.8,
                )

            # Calculate the variance of beta estimates
            beta_array = np.array(beta_list)  # Shape: (50, 3)
            beta_cov = np.cov(beta_array, rowvar=False)  # (3, 3) matrix
            total_variance = np.trace(beta_cov)

            ax_item.text(
                0.65,
                0.95,
                r"Var($\beta$):" + f"{total_variance:.4f}",
                transform=ax_item.transAxes,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
            )

            ax_item.set_xlim(-3, 3)
            ax_item.set_ylim(-3, 3)
            ax_item.set_xlabel("Dimension 1")
            ax_item.set_ylabel("Dimension 2")
            ax_item.set_aspect("equal")
            ax_item.set_title(title, size=12)

        fig.suptitle(
            "Linear Probe Decision Boundaries (Illustration of Lemma 2)", fontsize=12
        )
        plt.tight_layout()
        return mo.vstack([mo.vstack([aniso_slider]), fig])

    build_aniso_ui()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What does \~isotropic\~ mean?

    Imagine drawing a circle (or a sphere in higher dimensions). An **isotropic** distribution expands equally in all directions, like a perfectly round balloon. An **anisotropic** distribution is stretched out like a cigar or a pancake.

    If your data representation is stretched (anisotropic), some features (directions) have huge variance, while others are squished. When you train a simple linear classifier (a probe) on top of this, the squished features are harder to rely on, making the classifier's decision boundary highly sensitive to the specific training samples it saw. **An isotropic Gaussian ensures all features are equally spread out and uncorrelated**, leading to more stable, reliable classifiers across different random subsets of data.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## LeJEPA in Action: Method Comparison

    Now let's compare three methods side-by-side on 2D data:

    1. **Prediction Only** - collapses to a single point.
    2. **VCReg-style** - Variance-Covariance Regularization (2 hyperparameters, $\gamma$ and $\mu$).
    3. **LeJEPA** - SIGReg regularization (1 hyperparameter $\lambda$).

    Adjust the parameters and click "Compare Methods" to see how they structure the latent space.
    """)
    return


@app.cell
def _(Config, jax, jnp, jrand):
    def sigreg_loss_with_target(
            embeddings, key, target_cf_fn, num_slices=64, num_eval_points=17
        ):
        N, K = embeddings.shape
        key, subkey = jrand.split(key)

        # Project to 1D
        directions = jrand.normal(subkey, (num_slices, K))
        directions = directions / jnp.linalg.norm(directions, axis=-1, keepdims=True)
        projections = directions @ embeddings.T # Shape: (M, N)

        # CF computation over the [-5, 5] domain
        t_points = jnp.linspace(-5.0, 5.0, num_eval_points)

        # projections[:, None, :] -> (M, 1, N)
        # t_points[None, :, None] -> (1, T, 1)
        # angles -> (M, T, N)
        angles = projections[:, None, :] * t_points[None, :, None]

        # Complex ECF: (1/N) * sum(cos) + i * (1/N) * sum(sin)
        ecf_real = jnp.mean(jnp.cos(angles), axis=-1) # (M, T)
        ecf_imag = jnp.mean(jnp.sin(angles), axis=-1)

        # Match against real-valued target CF
        target_cf = target_cf_fn(t_points) # (T,)

        # Distance: |ECF - target|^2 = (Real - target)^2 + Imag^2
        diff_sq = (ecf_real - target_cf[None, :])**2 + ecf_imag**2

        # Apply the Epps-Pulley Gaussian weighting function w(t)
        weight = jnp.exp(-0.5 * t_points**2)
        err = diff_sq * weight[None, :]

        # Integrate using Trapezoidal rule and scale by N
        integral = jnp.trapezoid(err, x=t_points, axis=1) * N # Shape: (M,)

        return jnp.mean(integral) # Average over the M slices

    def sigreg_loss(embeddings, key, num_slices=64, num_eval_points=17):
        # Default LeJEPA targets the Standard Normal distribution
        return sigreg_loss_with_target(
            embeddings,
            key,
            lambda t: jnp.exp(-(t**2) / 2),
            num_slices=num_slices,
            num_eval_points=num_eval_points,
        )

    # Theoretical Target Characteristic Functions
    sqrt3 = jnp.sqrt(3.0)
    target_normal_cf = lambda t: jnp.exp(-(t**2) / 2)
    target_laplace_cf = lambda t: 1.0 / (1.0 + (t**2) / 2.0)
    target_uniform_cf = lambda t: jnp.sinc(
        sqrt3 * t / jnp.pi
    )  # sinc(x) computes sin(pi*x)/(pi*x)

    def vcreg_loss(embeddings, gamma=1.0, mu=1.0):
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
        encoder, x, key, lam, gamma, mu, method, sigreg_fn, num_slices=17, num_views=2
    ):
        noise_key, sigreg_key = jrand.split(key)

        # Generate V noisy views (num_views)
        noise = jrand.normal(noise_key, (num_views,) + x.shape) * Config.aug_noise

        # Broadcasting x to all V views
        v_views = x[None, ...] + noise # (V, N, D)

        # Apply the encoder to ALL views at once
        # jax.vmap(encoder) processes a batch (N, D) -> (N, K)
        # jax.vmap(jax.vmap(encoder)) processes the views (V, N, D) -> (V, N, K)
        z_views = jax.vmap(jax.vmap(encoder))(v_views)

        # Predictive loss (Invariance)
        centers = jnp.mean(z_views, axis=0) # (N, K)
        pred_loss = jnp.mean((centers[None, ...] - z_views) ** 2)

        # Regularization loss
        if method == "sigreg":
            # Apply SIGReg to all V views
            def single_view_sigreg(z):
                return sigreg_fn(z, sigreg_key, num_slices=num_slices)

            # Vectorize over the V views
            vmap_sigreg = jax.vmap(single_view_sigreg)
            reg_loss = jnp.mean(vmap_sigreg(z_views))

            # Convex combination weighting
            loss = (1.0 - lam) * pred_loss + lam * reg_loss
        elif method == "vcreg":
            vmap_vcreg = jax.vmap(lambda z: vcreg_loss(z, gamma, mu))
            reg_loss = jnp.mean(vmap_vcreg(z_views))
            loss = pred_loss + reg_loss
        else:
            loss = pred_loss

        return loss

    return (
        make_total_loss,
        sigreg_loss,
        sigreg_loss_with_target,
        target_laplace_cf,
        target_normal_cf,
        target_uniform_cf,
    )


@app.cell(hide_code=True)
def _(mo):
    steps_slider = mo.ui.slider(0, 1000, step=10, value=50, label="Training Steps")
    vcreg_mu = mo.ui.slider(
        0.0, 10.0, step=0.1, value=0.1, label=r"VCReg Weight ($\mu$)"
    )
    vcreg_gamma = mo.ui.slider(
        0.0, 10.0, step=0.1, value=1.0, label=r"VCReg Weight ($\gamma$)"
    )
    lambda_slider = mo.ui.slider(
        0.00, 5e-5, step=1e-6, value=5e-6, label=r"SIGReg Weight ($\lambda$)"
    )
    slices_slider = mo.ui.slider(1, 256, step=1, value=10, label="Projections (M)")

    in_dim_dropdown = mo.ui.dropdown(
        options={"2": 2, "4": 4, "8": 8, "16": 16, "32": 32, "64": 64},
        value="2",
        label="Input Dimension",
    )
    out_dim_dropdown = mo.ui.dropdown(
        options={"2": 2, "4": 4, "8": 8, "16": 16, "32": 32, "64": 64, "128": 128, "256": 256, "512": 512},
        value="2",
        label="Output Dimension",
    )

    run_cmp_btn = mo.ui.run_button(label="Compare Methods")

    cmp_controls = mo.vstack([
            mo.hstack([steps_slider, in_dim_dropdown, out_dim_dropdown]),
            mo.hstack([vcreg_gamma, vcreg_mu]),
            mo.hstack([lambda_slider, slices_slider]),
            mo.hstack([run_cmp_btn, mo.md("**<- Click 'Compare Methods' to run training.**")]),
        ])
    return (
        cmp_controls,
        in_dim_dropdown,
        lambda_slider,
        out_dim_dropdown,
        run_cmp_btn,
        slices_slider,
        steps_slider,
        vcreg_gamma,
        vcreg_mu,
    )


@app.cell(hide_code=True)
def _(
    Config,
    SmallEncoder,
    cmp_controls,
    eqx,
    generate_circles,
    in_dim_dropdown,
    jax,
    jnp,
    jrand,
    lambda_slider,
    make_total_loss,
    mo,
    np,
    optax,
    out_dim_dropdown,
    plt,
    run_cmp_btn,
    sigreg_loss,
    slices_slider,
    steps_slider,
    vcreg_gamma,
    vcreg_mu,
):
    mo.stop(
        not run_cmp_btn.value,
        cmp_controls,
    )

    def build_cmp_ui():
        in_dim_val = in_dim_dropdown.value
        X_comp, y_comp = generate_circles(n_samples=Config.n_samples, n_dim=in_dim_val)
        X = jnp.array(X_comp)
        n_steps = steps_slider.value
        lam = lambda_slider.value
        gamma = vcreg_gamma.value
        mu = vcreg_mu.value
        n_slices = slices_slider.value
        out_dim_val = out_dim_dropdown.value

        def train_one_method(method_key_str, key, x, steps, lam_val, gamma_val, mu_val, slices, in_dim, out_dim):
            model = SmallEncoder(key, in_dim=in_dim, out_dim=out_dim)
            optimizer = optax.adam(Config.lr)
            opt_state = optimizer.init(eqx.filter(model, eqx.is_inexact_array))
            tkey = jrand.PRNGKey(123)

            lam_jnp = jnp.array(lam_val)
            gamma_jnp = jnp.array(gamma_val)
            mu_jnp = jnp.array(mu_val)

            @eqx.filter_jit
            def train_step(m, os, k, l_val, g_val, m_val):
                loss_v, grads = eqx.filter_value_and_grad(make_total_loss)(
                    m, x, k, l_val, g_val, m_val, method_key_str, sigreg_loss, slices
                )
                updates, os = optimizer.update(grads, os, m)
                m = eqx.apply_updates(m, updates)
                return m, os

            for _ in range(steps):
                tkey, skey = jrand.split(tkey)
                model, opt_state = train_step(model, opt_state, skey, lam_jnp, gamma_jnp, mu_jnp)

            return np.array(jax.vmap(model)(x))

        with mo.status.spinner("Training models for comparison"):
            results = {}
            for m_key in ["none", "vcreg", "sigreg"]:
                results[m_key] = train_one_method(
                    m_key, jrand.PRNGKey(7), X, n_steps, lam, gamma, mu, n_slices, in_dim_val, out_dim_val
                )

            labels = [
                ("none", "No Reg (Collapse)"),
                ("vcreg", "VCReg-style"),
                ("sigreg", "LeJEPA (SIGReg)"),
            ]
            class_colors = [Config.red, Config.blue, Config.green]

            fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
            for idx, (mkey, lbl) in enumerate(labels):
                ax = axes[idx]
                emb = results[mkey]
                for ci in range(3):
                    mask = y_comp == ci
                    ax.scatter(
                        emb[mask, 0],
                        emb[mask, 1],
                        c=class_colors[ci],
                        s=8,
                        alpha=1.0,
                        lw=0.3,
                        ec="black",
                    )
                ax.grid(True, linestyle=":", alpha=0.6)
                ax.set_title(lbl, fontsize=11)
                ax.set_xlabel("Latent Dim 1")
                ax.set_ylabel("Latent Dim 2")
                ax.set_aspect("equal")
                ax.set_xlim(-4, 4)
                ax.set_ylim(-4, 4)

            fig.suptitle(
                f"Step {n_steps}, " + r"$\lambda$" + f"={lam:.1e}, " + r" $\gamma$" + f"={gamma:.1f}, " + r" $\mu$" + f"={mu:.1f}, " + f"M={n_slices}, in={in_dim_val}, out={out_dim_val}",
                fontsize=12,
                y=1.05,
            )
            plt.tight_layout()

        return mo.vstack([cmp_controls, fig])

    build_cmp_ui()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why is having only one hyperparameter a big deal?

    In self-supervised learning, you don't have labels. If your method requires tuning multiple interacting hyperparameters (like VCReg's variance $\gamma$ and covariance $\mu$ weights), you have a catch-22: how do you tune them without a labeled validation set?

    By reducing the regularization to a single hyperparameter $\lambda$ that just balances "prediction" vs "being a Gaussian", LeJEPA is much easier to apply to new, unlabeled datasets without needing a massive hyperparameter sweep or a proxy labeled task.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Extension: Adaptive Distribution Targets

    While the paper focuses on the isotropic Gaussian, SIGReg can actually target **ANY** distribution simply by swapping the characteristic function $\varphi_{\text{target}}(t)$:

    | Target | CF | Why? |
    |--------|-----|------|
    | $\mathcal{N}(0, I)$ | $e^{-t^2/2}$ | Optimal worst-case risk for unknown tasks |
    | Laplace$(0, 1/\sqrt{2})$ | $\frac{1}{1 + t^2/2}$ | Heavier tails → sparser L1 probes |
    | Uniform$(-\sqrt{3}, \sqrt{3})$ | $\frac{\sin(\sqrt{3}\,t)}{\sqrt{3}\,t}$ | Bounded support → bounded downstream tasks |

    Matching the embedding distribution to the probe regularizer yields **sparser, more efficient solutions** at the same accuracy. Sparsity is measured via the **Gini coefficient** of L1 probe weights (0 = uniform, 1 = single non-zero).

    Below: 2D embeddings for intuition, and high-dim L1-probe evaluation (Accuracy + Gini Sparsity).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    target_dropdown = mo.ui.dropdown(
        options={
            "Laplace(0, 1/√2)": "laplace",
            "Uniform(-√3, √3)": "uniform",
        },
        value="Laplace(0, 1/√2)",
        label="Custom Target",
    )
    ext_steps_slider = mo.ui.slider(1, 5000, step=100, value=1000, label="Training Steps")
    ext_lambda = mo.ui.slider(
        0.00, 5e-5, step=1e-6, value=2e-5, label=r"SIGReg Weight ($\lambda$)"
    )
    ext_seeds = mo.ui.slider(1, 100, step=1, value=1, label="Seeds")
    ext_in_dim = mo.ui.dropdown(
        options={"2": 2, "4": 4, "8": 8, "16": 16, "32": 32, "64": 64, "128": 128, "256": 256, "512": 512},
        value="2",
        label="Input Dimension",
    )
    ext_out_dim = mo.ui.dropdown(
        options={"2": 2, "4": 4, "8": 8, "16": 16, "32": 32, "64": 64, "128": 128, "256": 256, "512": 512},
        value="128",
        label="Output Dimension",
    )
    run_ext_btn = mo.ui.run_button(label="Train & Evaluate")

    ext_controls = mo.vstack([
        mo.hstack([target_dropdown, ext_steps_slider, ext_seeds]),
        mo.hstack([ext_lambda, ext_in_dim, ext_out_dim]),
        run_ext_btn,
    ])
    return (
        ext_controls,
        ext_in_dim,
        ext_lambda,
        ext_out_dim,
        ext_seeds,
        ext_steps_slider,
        run_ext_btn,
        target_dropdown,
    )


@app.cell(hide_code=True)
def _(
    Config,
    Lasso,
    SmallEncoder,
    eqx,
    ext_controls,
    ext_in_dim,
    ext_lambda,
    ext_out_dim,
    ext_seeds,
    ext_steps_slider,
    generate_circles,
    jax,
    jnp,
    jrand,
    mo,
    np,
    optax,
    plt,
    run_ext_btn,
    sigreg_loss_with_target,
    target_dropdown,
    target_laplace_cf,
    target_normal_cf,
    target_uniform_cf,
):
    mo.stop(
        not run_ext_btn.value, ext_controls,
    )

    def build_extension_ui():
        target_cfs = {
            "normal": target_normal_cf,
            "laplace": target_laplace_cf,
            "uniform": target_uniform_cf,
        }
        target_labels = {
            "normal": "N(0,I)",
            "laplace": "Laplace",
            "uniform": "Uniform",
        }

        in_dim_val = ext_in_dim.value
        X_np, y_np = generate_circles(n_dim=in_dim_val)
        X = jnp.array(X_np)

        custom_target_key = target_dropdown.value
        ext_lam = ext_lambda.value
        n_ext_steps = ext_steps_slider.value
        n_seeds = ext_seeds.value
        out_dim_q = ext_out_dim.value    

        def make_loss_fn(target_fn):
            def loss_fn(m, x, k, l_val):
                k1, k2, k3 = jrand.split(k, 3)
                v1 = x + jrand.normal(k1, x.shape) * Config.aug_noise
                v2 = x + jrand.normal(k2, x.shape) * Config.aug_noise
                z1 = jax.vmap(m)(v1)
                z2 = jax.vmap(m)(v2)
                p_loss = jnp.mean((z1 - z2) ** 2)
                reg_loss = sigreg_loss_with_target(
                    z1, k3, target_fn, num_slices=Config.num_slices
                )
                return p_loss + l_val * reg_loss

            return loss_fn

        def train_model(target_key, seed_val, in_dim, out_dim=2):
            model = SmallEncoder(jrand.PRNGKey(seed_val), in_dim=in_dim, out_dim=out_dim)
            opt = optax.adamw(Config.lr)
            opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
            tkey = jrand.PRNGKey(seed_val-1)
            target_fn = target_cfs[target_key]
            lam_jnp = jnp.array(ext_lam)
            loss_fn = make_loss_fn(target_fn)

            @eqx.filter_jit
            def train_step(m, os, k, l_val):
                loss_v, grads = eqx.filter_value_and_grad(loss_fn)(m, X, k, l_val)
                updates, os = opt.update(grads, os, m)
                m = eqx.apply_updates(m, updates)
                return m, os

            for _ in range(n_ext_steps):
                tkey, skey = jrand.split(tkey)
                model, opt_state = train_step(model, opt_state, skey, lam_jnp)

            return np.array(jax.vmap(model)(X))

        def evaluate_probe_l1(embeds):
            embeds_centered = embeds - embeds.mean(axis=0)
            y_one_hot = np.eye(3)[y_np]
            X_probe = np.column_stack([embeds_centered, np.ones(len(embeds_centered))])
            lasso = Lasso(alpha=0.01, fit_intercept=False, max_iter=5000)
            lasso.fit(X_probe, y_one_hot)
            beta = lasso.coef_.T
            preds = X_probe @ beta
            accuracy = float(np.mean(np.argmax(preds, axis=1) == y_np))
            w = np.abs(beta[:-1, :]).ravel()
            sorted_w = np.sort(w)
            n = len(sorted_w)
            cum = np.cumsum(sorted_w)
            gini = (n + 1 - 2 * np.sum(cum) / (cum[-1] + 1e-12)) / n
            sparsity = float(max(0.0, gini))
            return accuracy, sparsity

        with mo.status.spinner(f"Training 2D + {n_seeds}x IN:{in_dim_val}D OUT:{out_dim_q}D models"):
            embeds_2d = {}
            for tkey in ["normal", custom_target_key]:
                embeds_2d[tkey] = train_model(tkey, 7, in_dim=in_dim_val, out_dim=2)

            accs = {"normal": [], custom_target_key: []}
            sparsities = {"normal": [], custom_target_key: []}

            for seed_val in range(7, 7 + n_seeds):
                for tkey in ["normal", custom_target_key]:
                    emb = train_model(tkey, seed_val, in_dim=in_dim_val, out_dim=out_dim_q)
                    a, s = evaluate_probe_l1(emb)
                    accs[tkey].append(a)
                    sparsities[tkey].append(s)

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        colors_scatter = [Config.red, Config.blue, Config.green]

        for ax, (tkey, title) in zip(
            axes[:2],
            [
                ("normal", "Target: N(0,I)"),
                (custom_target_key, f"Target: {target_labels[custom_target_key]}"),
            ],
        ):
            emb = embeds_2d[tkey]
            for ci in range(3):
                mask = y_np == ci
                ax.scatter(
                    emb[mask, 0],
                    emb[mask, 1],
                    c=colors_scatter[ci],
                    s=10,
                    alpha=1.0,
                    lw=0.3,
                    ec="black",
                )
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.set_title(title)
            ax.set_xlabel("Dim 1")
            ax.set_ylabel("Dim 2")
            ax.set_aspect("equal")
            ax.set_xlim(-4, 4)
            ax.set_ylim(-4, 4)

        ax_bar = axes[2]
        ax_gini = ax_bar.twinx() 
        targets_info =[
            ("normal", "N(0,I)", Config.red),
            (custom_target_key, target_labels[custom_target_key], Config.blue),
        ]
        bar_width = 0.35

        for tidx, (tkey, tlabel, tcolor) in enumerate(targets_info):
            acc_vals = np.array(accs[tkey]) * 100
            gini_vals = np.array(sparsities[tkey])

            acc_mean = np.mean(acc_vals)
            acc_err = np.std(acc_vals)
            gini_mean = np.mean(gini_vals)
            gini_err = np.std(gini_vals)

            ax_bar.bar(
                0 + tidx * bar_width - bar_width / 2,
                acc_mean,
                bar_width,
                yerr=acc_err,
                color=tcolor,
                alpha=0.85,
                capsize=4,
                label=tlabel,
                ec="black",
                zorder=1,
            )
            ax_gini.bar(
                1.4 + tidx * bar_width - bar_width / 2,
                gini_mean,
                bar_width,
                yerr=gini_err,
                color=tcolor,
                alpha=0.85,
                capsize=4,
                ec="black",
                zorder=1,
            )

            ax_bar.text(
                0 + tidx * bar_width - bar_width / 2,
                8,
                f"{acc_mean:.1f}%",
                ha="center",
                va="bottom",              
                fontsize=7,
                fontweight="bold",
                zorder=3,
            )

            ax_gini.text(
                1.4 + tidx * bar_width - bar_width / 2,
                0.08
                , 
                f"{gini_mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
                zorder=3,
            )

        ax_bar.set_xticks([0, 1.4])
        ax_bar.set_xticklabels(["Accuracy", "Gini"])
        ax_bar.set_ylabel("Accuracy (%)")
        ax_gini.set_ylabel("Gini Index")

        ax_bar.set_ylim([0, 110]) 
        ax_gini.set_ylim([0, 1.1])

        ax_bar.legend(fontsize=8, loc="upper left")
        ax_bar.set_title(f"L1 Probe (in={in_dim_val}, out={out_dim_q}, {n_seeds} seed{"s" if n_seeds > 1 else ""})")
        ax_bar.grid(True, linestyle=":", alpha=0.6)

        fig.suptitle(
            f"Adaptive Targets | {n_ext_steps} step{"s" if n_ext_steps > 1 else ""}, "+r"$\lambda$"+f"={ext_lam:.1e}",
            fontsize=13,
            y=1.03,
        )
        plt.tight_layout()

        return mo.vstack([ext_controls, fig])

    build_extension_ui()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Key Takeaways

    **LeJEPA** replaces the heuristics of current JEPA methods with a single, principled regularizer. Here's what we demonstrated:

    1. **The Collapse Problem** - Prediction-only training collapses embeddings to trivial solutions. Some regularizer is essential.

    2. **Why Isotropic Gaussian?** - Anisotropic embeddings amplify both bias and variance of downstream linear probes. The isotropic Gaussian $\mathcal{N}(\mathbf{0}, \mathbf{I})$ uniquely minimizes worst-case risk.

    3. **SIGReg** - Enforces $\mathcal{N}(\mathbf{0}, \mathbf{I})$ via random projections and characteristic function matching. $O(NMK)$ complexity. Single hyperparameter $\lambda$. Clean code. No stop-gradient, no EMA, no teacher-student.

    4. **LeJEPA = Prediction + SIGReg** - Combines the predictive loss with SIGReg to produce well-structured, non-collapsed embeddings.

    5. **Extension** - SIGReg can target *any* distribution, not just $\mathcal{N}(\mathbf{0}, \mathbf{I})$. Matching the target distribution to the downstream probe regularizer yields **sparser solutions** at the same accuracy - Laplace-like embeddings give L1 probes sparser representations, meaning fewer active features for the same predictive performance.

    ---

    **References:**
    - Balestriero, R. & LeCun, Y. (2025). *LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics.* [arXiv:2511.08544](https://arxiv.org/abs/2511.08544)
    - Chen & He, 2021. *Exploring Simple Siamese Representation Learning.* [arxiv:2011.10566](https://arxiv.org/abs/2011.10566)
    - Grill et al., 2020. *Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning.* [arxiv:2006.07733](https://arxiv.org/abs/2006.07733)
    - Caron et al., 2021. *Emerging Properties in Self-Supervised Vision Transformers.* [arxiv:2104.14294](https://arxiv.org/abs/2104.14294)
    - This notebook implements core ideas from scratch using **JAX** + **Equinox**. The official implementation is at [github.com/rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa).
    """)
    return


if __name__ == "__main__":
    app.run()
