"""Firefly algorithm vs PSO on the 2-D Michalewicz function (m = 10).

Scenes
  1. PSO (Eqs. 1-2, alpha = beta = 2, no inertia, velocity clamped)
  2. Actual brightness vs brightness "in the eye of the beholder"
  3. FA (Eq. 9, alpha = 0.2, gamma = 1, beta0 = 1)
  4. FA with gamma -> 0 (constant attractiveness: the PSO-like special case)

The surface is drawn as brightness I(x) = -f(x), so the global minimum
f* = -1.801 at (2.20, 1.57) is the highest peak.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = sys.argv[1] if len(sys.argv) > 1 else "frames"
os.makedirs(OUT, exist_ok=True)

M = 10
LO, HI = 0.0, np.pi
GBEST = np.array([2.20319, 1.57049])


def mich(x, y):
    return -(np.sin(x) * np.sin(x**2 / np.pi) ** (2 * M)
             + np.sin(y) * np.sin(2 * y**2 / np.pi) ** (2 * M))


def bright(p):
    return -mich(p[:, 0], p[:, 1])


g = np.linspace(LO, HI, 70)
GX, GY = np.meshgrid(g, g)
GZ = -mich(GX, GY)

NAVY = "#14284B"
FIG_W, FIG_H, DPI = 8.0, 4.8, 100
frame_no = [0]


def new_axes(fig, rect, azim, title=None):
    ax = fig.add_axes(rect, projection="3d", computed_zorder=False)
    ax.plot_surface(GX, GY, GZ, cmap="viridis", alpha=0.55, linewidth=0,
                    antialiased=True, rstride=1, cstride=1, zorder=1)
    ax.set_xlim(LO, HI); ax.set_ylim(LO, HI); ax.set_zlim(0, 2.0)
    ax.view_init(elev=38, azim=azim)
    ax.set_xlabel("$x_1$", labelpad=-8, fontsize=9)
    ax.set_ylabel("$x_2$", labelpad=-8, fontsize=9)
    ax.set_zlabel("brightness $I=-f$", labelpad=-6, fontsize=8)
    ax.tick_params(labelsize=6, pad=-3)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_alpha(0.0)
    if title:
        ax.set_title(title, fontsize=10, color=NAVY, pad=-2)
    return ax


def header(fig, title, sub, foot):
    fig.text(0.5, 0.955, title, ha="center", fontsize=14, weight="bold", color=NAVY)
    fig.text(0.5, 0.905, sub, ha="center", fontsize=9.5, color="0.25")
    fig.text(0.5, 0.025, foot, ha="center", fontsize=8.5, color="0.35", style="italic")


def save(fig, repeat=1):
    for _ in range(repeat):
        fig.savefig(f"{OUT}/f{frame_no[0]:04d}.png", dpi=DPI)
        frame_no[0] += 1
    plt.close(fig)


def dots(ax, p, color, size=26, edge="k", z_off=0.05, **kw):
    ax.scatter(p[:, 0], p[:, 1], bright(p) + z_off, s=size, c=color,
               edgecolors=edge, linewidths=0.5, depthshade=False, zorder=5, **kw)


def star(ax, p, label=None):
    ax.scatter([p[0]], [p[1]], [bright(p[None])[0] + 0.12], marker="*", s=180,
               c="red", edgecolors="k", linewidths=0.6, depthshade=False, zorder=6)


def title_card(title, lines, repeat=20):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.62, title, ha="center", fontsize=20, weight="bold", color=NAVY)
    for k, ln in enumerate(lines):
        fig.text(0.5, 0.48 - 0.07 * k, ln, ha="center", fontsize=11, color="0.25")
    save(fig, repeat)


# ---------------------------------------------------------------- setup
rng = np.random.default_rng(7)
N = 25
X0 = rng.uniform(LO, HI, size=(N, 2))

# ---------------------------------------------------------------- intro
title_card("Firefly Algorithm vs PSO",
           ["on the 2-D Michalewicz function, $m=10$",
            r"surface height = brightness $I(\mathbf{x})=-f(\mathbf{x})$;  "
            r"global optimum $f_*\approx-1.801$ at $(2.20,\,1.57)$",
            "Yang (2009) - COMP 599 - Eliza Kishan"], repeat=16)

for k in range(14):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = new_axes(fig, [0.05, 0.07, 0.9, 0.8], azim=-125 + 10 * np.sin(k / 4))
    star(ax, GBEST)
    header(fig, "The landscape", "Michalewicz, Eq. (10): narrow ridges, one global peak (red star)",
           "Sharp ridges make it easy to get stuck on a local peak")
    save(fig)

# ---------------------------------------------------------------- 1. PSO
title_card("1. Particle Swarm Optimization",
           [r"$v_i \leftarrow v_i + \alpha\,\epsilon_1(g^*-x_i) + \beta\,\epsilon_2(x_i^*-x_i)$,   "
            r"$x_i \leftarrow x_i + v_i$",
            r"$\alpha=\beta=2$, no inertia, $|v|\leq 0.3$ ; same 25 starting points in every scene"])

x = X0.copy(); v = np.zeros_like(x)
pbest = x.copy(); pval = bright(x)
gi = np.argmax(pval); gbest = pbest[gi].copy()
for t in range(32):
    fig = plt.figure(figsize=(FIG_W, FIG_H))
    ax = new_axes(fig, [0.05, 0.07, 0.9, 0.8], azim=-125 + 10 * np.sin(t / 9))
    dots(ax, x, "#3b82f6")
    star(ax, gbest)
    header(fig, "1. PSO explores the landscape",
           f"iteration {t:2d}   -   everyone is pulled toward one global best $g^*$ (red star)",
           "One shared target: the swarm collapses onto a single peak")
    save(fig)
    e1, e2 = rng.random(x.shape), rng.random(x.shape)
    v = v + 2 * e1 * (gbest - x) + 2 * e2 * (pbest - x)
    v = np.clip(v, -0.3, 0.3)
    x = np.clip(x + v, LO, HI)
    b = bright(x)
    better = b > pval
    pbest[better] = x[better]; pval[better] = b[better]
    gi = np.argmax(pval); gbest = pbest[gi].copy()
save_last = None

# ---------------------------------------------------------------- 2. beholder
title_card("2. Brightness is in the eye of the beholder",
           [r"actual brightness:  $I_j = -f(\mathbf{x}_j)$",
            r"what firefly $i$ sees:  $I_j(r_{ij}) = I_j\, e^{-\gamma r_{ij}^2}$  (Eq. 3),   $\gamma = 1$",
            "a bright firefly far away can look dimmer than a modest one nearby"])

xb = X0.copy()
Ib = bright(xb)
observers = [int(np.argmin(np.linalg.norm(xb - c, axis=1)))
             for c in ([0.4, 0.5], [2.8, 2.8], [1.2, 2.6], [2.6, 0.6])]
for oi, o in enumerate(observers):
    r2 = np.sum((xb - xb[o]) ** 2, axis=1)
    seen = Ib * np.exp(-1.0 * r2)
    seen[o] = 0
    act = Ib.copy(); act[o] = 0
    true_best = int(np.argmax(act)); seen_best = int(np.argmax(seen))
    for k in range(10):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        azim = -125 + 10 * np.sin((oi * 10 + k) / 9)
        ax1 = new_axes(fig, [0.0, 0.08, 0.5, 0.78], azim, "Actual brightness (global view)")
        ax2 = new_axes(fig, [0.5, 0.08, 0.5, 0.78], azim, f"As seen by the red firefly ($\\gamma=1$)")
        s_act = 15 + 160 * (Ib / Ib.max())
        s_seen = 15 + 160 * (seen / max(seen.max(), 1e-9))
        for ax, s, best in ((ax1, s_act, true_best), (ax2, s_seen, seen_best)):
            mask = np.arange(N) != o
            ax.scatter(xb[mask, 0], xb[mask, 1], Ib[mask] + 0.05, s=s[mask], c="#facc15",
                       edgecolors="k", linewidths=0.5, depthshade=False, zorder=5)
            ax.scatter([xb[o, 0]], [xb[o, 1]], [Ib[o] + 0.05], s=70, c="red",
                       edgecolors="k", linewidths=0.8, depthshade=False, zorder=6)
            ax.plot([xb[o, 0], xb[best, 0]], [xb[o, 1], xb[best, 1]],
                    [Ib[o] + 0.05, Ib[best] + 0.05], color="red", lw=1.4, zorder=7)
        same = "the same firefly" if seen_best == true_best else "a different, closer firefly"
        header(fig, "2. Actual brightness vs. the eye of the beholder",
               f"observer {oi + 1}/4 - dot size = brightness; red line = brightest firefly in that view",
               f"Left: brightest overall.  Right: most attractive to the observer is {same}")
        save(fig)

# ---------------------------------------------------------------- FA core
def run_fa(gamma, alpha=0.2, beta0=1.0, iters=32, seed=11):
    r = np.random.default_rng(seed)
    x = X0.copy(); hist = [x.copy()]
    for t in range(iters):
        I = bright(x)
        for i in range(N):
            for j in range(N):
                if I[j] > I[i]:
                    r2 = np.sum((x[i] - x[j]) ** 2)
                    x[i] = x[i] + beta0 * np.exp(-gamma * r2) * (x[j] - x[i]) \
                        + alpha * (r.random(2) - 0.5)
                    x[i] = np.clip(x[i], LO, HI)
                    I[i] = bright(x[i][None])[0]
        hist.append(x.copy())
    return hist


def fa_scene(hist, color, title, sub_fmt, foot):
    for t, x in enumerate(hist[:-1]):
        fig = plt.figure(figsize=(FIG_W, FIG_H))
        ax = new_axes(fig, [0.05, 0.07, 0.9, 0.8], azim=-125 + 10 * np.sin(t / 9))
        dots(ax, x, color)
        b = bright(x); star(ax, x[np.argmax(b)])
        header(fig, title, sub_fmt.format(t=t), foot)
        save(fig)


# ---------------------------------------------------------------- 3. FA
title_card("3. Firefly Algorithm",
           [r"$x_i \leftarrow x_i + \beta_0 e^{-\gamma r_{ij}^2}(x_j - x_i) + \alpha(\mathrm{rand}-\frac{1}{2})$",
            r"$\alpha = 0.2,\ \beta_0 = 1$;  first $\gamma = 1$ (the paper's setting), then $\gamma = 5$",
            "each firefly follows the brighter fireflies it can actually see"])
fa_scene(run_fa(gamma=1.0), "#facc15", r"3a. FA, $\gamma = 1$ (paper's settings)",
         "iteration {t:2d}   -   sight range $\\Gamma = 1/\\sqrt{{\\gamma}} = 1$, comparable to this small domain",
         "Attraction fades with distance, but every firefly still reaches the global peak (red star)")
title_card(r"3b. Shorter sight: $\gamma = 5$",
           [r"$\Gamma = 1/\sqrt{5} \approx 0.45$, shorter than the gap between the two peaks ($\approx 1.15$)",
            "fireflies near the second peak cannot see the global one well",
            r"(Sec. 3.5: $\gamma$ should match the scale of the problem)"], repeat=20)
fa_scene(run_fa(gamma=5.0), "#facc15", r"3b. FA, $\gamma = 5$: several optima at once",
         "iteration {t:2d}   -   local groups form around each peak they can see",
         "One group holds the global peak (red star), another holds the local peak at (2.20, 2.72)")

# ---------------------------------------------------------------- 4. FA gamma -> 0
title_card(r"4. Special case: $\gamma \to 0$",
           [r"$e^{-\gamma r^2} \to 1$: attractiveness is $\beta_0$ at every distance",
            "every firefly sees the true brightness of every other (a perfectly clear sky)",
            "the swarm chases the single brightest firefly: a special case of PSO"])
fa_scene(run_fa(gamma=0.0), "#f97316", r"4. FA with $\gamma \to 0$ behaves like PSO",
         "iteration {t:2d}   -   no distance fading: all fireflies are pulled to the same best one",
         "Like PSO, the whole swarm collapses onto one peak; the local optima are no longer kept")

title_card("Summary",
           ["PSO: one global best pulls every particle, so the swarm converges to one peak",
            r"FA: attraction fades with distance; with short sight ($\gamma=5$) it keeps several optima",
            r"FA ($\gamma\to 0$): attraction is global, which recovers the PSO-like behaviour",
            r"($\gamma\to\infty$ would give pure random search)"], repeat=22)
print("frames:", frame_no[0])