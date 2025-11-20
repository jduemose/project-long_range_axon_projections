import matplotlib.pyplot as plt
import numpy as np

"""
ESTIMATE CURVATURE / BEND RADIUS FROM SIGMOID WIDTH PARAMETERS FROM

    COTTAAR (2019). A gyral coordinate system predictive of fibre orientations

AS FOUND IN FIG. 8.

NOTE
Eq. 3 seems to be wrong; they used a logistic function which is apparent from
their code in https://git.fmrib.ox.ac.uk/ndcn0236/gyralcoord

Let

    sigma = logistic function

then

    y' = (sigma[t], 1-sigma[t]) # radial, tangential

y' defines the mixing of radial and tangential vectors and therefore the
direction of a fiber at a given point. Consequently, it defines the unit
tangent vector of another function

    y = (integral{ sigma[t] }, integral{ 1-sigma[t] })

which describes the actual curve/path of a fiber/axon/whatever.

From

    https://en.wikipedia.org/wiki/Curvature
    > Plane Curves
    >> In terms of arc-length parametrization

we have

    k(t) is given by the norm of y''

    k(t) = ||y''(t)||
    r(t) = 1/k(t)
    C(t) = y(t) + 1/k(t)**2 * y''(t) # the center of the oscillating circle

where

    y'' = (diff{ sigma[t] }, diff{ 1-sigma[t] })

i.e., the diff of the original logistic function.
"""


def logistic(x, mu=0.0, sigma=1.0):
    x_ = (x - mu) / sigma
    return 1 / (1 + np.exp(-x_))


def d_logistic(x, mu=0.0, var=1.0):
    s = logistic(x, mu, var)
    return (s - s**2) / var


def d2_logistic(x, mu=0.0, sigma=1.0):
    ds = d_logistic(x, mu, sigma)
    return ds - 2.0 * ds * logistic(x, mu, sigma)


def int_logistic(x, mu=0.0, smooth=1.0):
    return (x - mu) + smooth * np.log(1 + np.exp(-(x - mu) / smooth))


mu = 0.0
sigma = 0.5

x = np.linspace(-3.0, 3.0, 101)
stepsize = x[2] - x[1]

fig, ax = plt.subplots(1, 1)
for sigma in (0.1, 0.25, 0.5, 0.75, 1.0, 2.0):
    y = logistic(x, mu, sigma)
    dy = d_logistic(x, mu, sigma)
    ddy = d2_logistic(x, mu, sigma)

    vx = np.array([[stepsize, 0]])
    vy = np.array([[0, stepsize]])
    r = vx * y[:, None] + vy * (1 - y[:, None])
    r1 = vx * dy[:, None] + vy * (1 - dy[:, None])
    r2 = vx * ddy[:, None] - vy * ddy[:, None]

    v = np.cumsum(r, axis=0)
    # v1 = np.cumsum(r2, axis=0)
    # v2 = np.cumsum(r3, axis=0)

    dy = np.stack((dy, -dy), -1)

    i = 50
    vi = v - v[i]
    ax.plot(*vi.T, label=f"sigma = {sigma:4.2f} mm")
    # ax.plot(*v.T)
    # r = 1.0 / (2 * dy[50])
    # c = np.sqrt(r**2 / 2)
    # circle = plt.Circle([c, -c], r, edgecolor="b", facecolor="none")
    # ax.add_artist(circle)
    # curv = np.sqrt(np.sum(dy[i] ** 2))
    curv = np.linalg.norm(dy, axis=1)
    # c = np.sqrt(r**2 / 2)
    c = vi[i] + 1 / curv[i] ** 2 * dy[i]
    r = 1.0 / curv[i]
    print(f"sigma = {sigma:4.2f} >> radius = {r:4.2f} mm")
    # circle = plt.Circle(c, r, edgecolor="r", facecolor="none")
    # ax.add_artist(circle)
    # ax.set_xlim([-0.5, 0.5])
    # ax.set_ylim([-0.5, 0.5])
    ax.set_aspect("equal")
plt.grid(alpha=0.2)
plt.legend()
plt.xlabel("Radial (mm)")
plt.ylabel("Tangential (mm)")


# plt.plot(x, logistic(x, 0.0, 0.1))

# plt.quiver(x, y, *r.T, angles="xy", color="r")
# plt.quiver(*v.T, *r.T, angles="xy", color="r")

# plt.plot(x, 1 / (1 + np.exp(-x)))
# plt.plot(x, x + np.log(1 + np.exp(-x)))

# plt.plot(x, logistic(x, mu, sigma))
# plt.plot(x, int_logistic(x, mu, sigma))

# plt.plot(x, 1 - logistic(x, mu, sigma))
# plt.plot(x, x - int_logistic(x, mu, sigma))

# plt.figure()
# for w in [0.01, 0.1, 0.5, 1.0, 5.0]:
#     plt.plot(x, 1 / (1 + np.exp(-x / w)))
#     plt.plot(x, x + w * np.log(1 + np.exp(-x / w)))
# plt.grid(alpha=0.2)
