import numpy as np

class StackedVectorPolynomial:
    def __init__(self, coef=None, deg: int | None = None):
        """Class for fitting, evaluating, and differentiating multiple
        polynomials.

        Parameters
        ----------
        coef : _type_, optional
            _description_, by default None
        deg : int | None, optional
            _description_, by default None
        """
        self.coef = coef
        if coef is not None:
            self.deg = self.coef.shape[-2] - 1
        elif deg is not None:
            self.deg = deg
        else:
            self.deg = None

    def get_design_matrix(self, x, deg=None):
        x = x[:, None] if x.ndim == 1 else x
        if deg is None:
            assert self.deg is not None
        deg = deg if deg is not None else self.deg
        return x[..., None] ** np.arange(self.deg + 1)[None]

    @staticmethod
    def _diff(coef, n: int):
        # assume coefs = [
        #    coefs[0] * x**0, coefs[1] * x**1, coefs[2] * x**2, ..., coefs[n-1] * x ** (n-1)
        # ]
        # and calculate
        #    1 * coefs[1], 2 * coefs[2], ...
        #
        return coef[..., 1:, :] * np.arange(1, n)[:, None]

    def fit(self, x, y, deg: int):
        """_summary_

        E.g., x = (3,) and y = (100, 3, 3) [where n_poly = 100,
        n_points_per_poly = 3, n_outputs = (e.g., x,y,z)].

        Parameters
        ----------
        x : _type_
            (m, )
        y : _type_
            (n, m, k) will fit n polynomials to m data points each with k outputs.
        deg : int
            Degree of the polynomial fit.
        """
        assert deg > 0
        self.deg = deg
        y = np.atleast_2d(y)
        self.A = self.get_design_matrix(x)
        if self.A.shape[-2] == self.A.shape[-1]:  # (..., n, n)
            # seems to be faster that svd
            self.coef = np.linalg.solve(self.A, y)
        else:  # (..., m, n)
            # least squares
            U, S, Vt = np.linalg.svd(self.A, full_matrices=False)
            self.coef = Vt.swapaxes(1, 2) @ (U.swapaxes(1, 2) @ y / S[..., None])

    def deriv(self, m: int = 1, coef=None):
        # assume coefs = [coefs[0] * x**0, coefs[1] * x**1, coefs[2] * x**2, ...]
        coef = self.coef if coef is None else coef
        assert self.coef is not None
        n = coef.shape[-2]
        assert m >= 1 and n >= m
        if m == 1:
            return (
                StackedVectorPolynomial(np.array([0.0]))
                if n == 1
                else StackedVectorPolynomial(self._diff(coef, n))
            )
        else:
            return self.deriv(m - 1, self._diff(coef, n))

    def __call__(self, x):  # , squeeze: bool = True
        """_summary_

        Parameters
        ----------
        x : array of shape (n_poly, ) or (n_poly, m)
            Points at which to evaluate the polynomials

        Returns
        -------
        solution
            (n_poly,[ m,] k_outputs)
        """
        A = self.get_design_matrix(x)
        y = A @ self.coef
        return np.squeeze(y)  # if squeeze else y

    def compute_curvature(self, x):
        """_summary_

        Parameters
        ----------
        x : _type_
            (n,) will broadcast (n, m)

        Returns
        -------
        _type_
            _description_
        """
        d1 = self.deriv(1)(x)
        d2 = self.deriv(2)(x)
        d1_norm2 = np.sum(d1**2, -1)
        d2_norm2 = np.sum(d2**2, -1)
        num = np.sqrt(d1_norm2 * d2_norm2 - np.sum(d1 * d2, -1) ** 2)
        denom = d1_norm2 ** (3 / 2)
        # R = 1 / k
        return num / denom

    # def compute_curvature_3d(d1, d2):
    #     num = np.linalg.norm(np.cross(d1, d2), axis=-1)
    #     denom = np.linalg.norm(d1, axis=-1) ** 3
    #     # R = 1 / k
    #     return num / denom




# valid_radii = radii[n_iter > min_number_of_points]
# min_radii = np.nan_to_num(valid_radii, nan=np.inf).min(1)

# fig, ax = plt.subplots()
# ax.scatter(*yw[:, 1].T, marker=".", c=np.log(k), cmap="viridis")
# fig.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# sc = ax.scatter(
#     *yw[:, 1].T, marker=".", c=1e3 * r, cmap="viridis", vmin=0.0, vmax=5000.0
# )
# fig.colorbar(sc, ax=ax, label="radius (um)")
# fig.show()


# x[:, 1]

# d1 = p1(x[:, [1]])
# d2 = p2(x[:, [1]])


# x = np.linspace(0, 1, 3)
# x = sampled_p[20:41:10, 0, 0]
# x = (x - x.min()) / (x.max() - x.min())

# y = sampled_p[20:41:10, 0, 1][:, None]
# y = sampled_p[20:41:10, 0, :2]
# y = sampled_p[20:41:10, 0]
# y = np.stack((sampled_p[20:41:10, 0], sampled_p[35:56:10, 0]))

# mp = StackedVectorPolynomial()
# mp.fit(x, y, 2)
# mp1 = mp.deriv(1)
# mp2 = mp.deriv(2)

# eval_x = np.linspace(0, 1, 100)
# # eval_x = np.linspace(x.min(), x.max(), 100)

# d12 = mp1(eval_x)
# d22 = mp2(eval_x)
# x1 = np.ones_like(y1)
# x2 = np.zeros_like(y2)

# d1 = np.stack([x1, y1], axis=1)
# d2 = np.stack([x2, y2], axis=1)

# k = num / denom
# r = 1 / k


# d1_norm2 = np.sum(d12**2, -1)
# d2_norm2 = np.sum(d22**2, -1)
# num = np.sqrt(d1_norm2 * d2_norm2 - np.sum(d12 * d22, -1) ** 2)
# denom = d1_norm2 ** (3 / 2)
# # R = 1 / k
# k2 = num / denom


# plt.figure()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid(alpha=0.25)
# # plt.scatter(x, y)
# plt.scatter(*y.T)
# # plt.scatter(eval_x, mp(eval_x), marker=".", c=k2, cmap="viridis")
# plt.scatter(*mp(eval_x).T, marker=".", c=k2, cmap="viridis")
# plt.colorbar()

# plt.plot(eval_x, y[1] + mp1(x[1]) * (eval_x - x[1]))
# plt.quiver(x[1], y[1], 1.0, mp1(x[1]), angles="xy", scale=15.0, color="r")
# # plt.plot(eval_x, mp1(eval_x))
# # plt.plot(eval_x, mp2(eval_x))


# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(*y[1].T)
# ax.scatter(*mp(eval_x)[1].T, marker=".", c=k2[1], cmap="viridis")
# fig.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(*sampled_p[:200, 0].T)
# fig.show()


# plt.figure()
# plt.xlabel("x")
# plt.ylabel("y")
# plt.grid(alpha=0.25)
# plt.scatter(*y.T)
# plt.plot(*mp(eval_x).T)
# plt.plot(x, yy[20] + v[0] + v[1] * eval_x)

# plt.quiver(pos[0], pos[1], v[0], v[1] * t, scale=10, color="r")

# plt.plot(q, mp1(eval_x).T[1])

# plt.plot(*mp2(eval_x)[3].T)

# # EXAMPLE
# x = np.arange(-1, 1 + 1)
# y = np.random.rand(10, 3, 2)
# y[..., 0] = x
# mp = MultiPolynomial()
# mp.fit(x, y, 2)
# assert np.allclose(np.squeeze(y), mp(x))  # should fit all points perfectly
# k = mp.compute_curvature(x)
# r = 1.0 / k

# eval_x = np.arange(-2, 2, 0.1)

# plt.figure()
# plt.scatter(x, y[3, :, 1])
# plt.plot(*mp(eval_x)[3].T)
# plt.plot(*mp1(eval_x)[3].T)
# plt.plot(*mp2(eval_x)[3].T)

# k = mp.compute_curvature(eval_x)


# # check that it is correct
# mp1 = mp.deriv(1)
# # mp1(x)  # n, m, (grad_x, grad_y, grad_z)
# mp2 = mp.deriv(2)
# # mp2(x)  # n, m, (grad2_x, grad2_y, grad2_z)

# # only fits a single poly at a time
# p = np.polynomial.Polynomial.fit(x, y[5, :, 0], deg=2)
# p1 = p.deriv(1)
# p2 = p.deriv(2)
# assert np.allclose(p.coef, mp.coef[5, :, 0])
# assert np.allclose(p1.coef, mp1.coef[5, :, 0])
# assert np.allclose(p2.coef, mp2.coef[5, :, 0])
