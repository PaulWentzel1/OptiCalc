import numpy as np
from scipy.stats import multivariate_normal

class BinomialFormulas:
    @staticmethod
    def peizer_pratt_inversion_1(x: float, n: int) -> float:
        """
        Return the Preizer-Pratt inversion method 1 on the variable x.
        Required for Leisen-Reimer binomial tree.
        """

        if x == 0: # Extra check since np.sign returns 0 if x is 0
            sign = 1
        else:
            sign = np.sign(x)

        return 0.5 + 0.5 * sign * np.sqrt(1 - np.exp(-((x / (n + 1/3)) ** 2) * (n + 1/6)))

    @staticmethod
    def peizer_pratt_inversion_2(x: float, n: int) -> float:
        """
        Return the Preizer-Pratt inversion method 2 on the variable x.
        Required for Leisen-Reimer binomial tree.
        """

        if x == 0: # Extra check since np.sign returns 0 if x is 0
            sign = 1
        else:
            sign = np.sign(x)

        return 0.5 + sign * 0.5 * np.sqrt(1 - np.exp(-((x / (n + 1/3 + 0.1 / (n + 1))) ** 2) * (n + 1/6)))

class BjerksundStenslandFormulas:
    @staticmethod
    def std_bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
        """
        Return the values of the Cumulative Bivariate normal distribution.

        Computes P(x <= a, y <= b) where x & y follows a standardized bivariate
        normal distribution with the correlation coefficient rho.

        Parameters
        -----------
        a : float
            Upper limit for first variable.

        b : float
            Upper limit for second variable.

        rho : float
            The correlation between a and b.

        Returns
        -----------
        float
            The cumulative probability P(x <= a, y <= b).
        """

        mean: list[float] = [0, 0]
        cov: list[list[float]] = [[1, rho], [rho, 1]]

        return multivariate_normal.cdf([a, b], mean = mean, cov = cov, allow_singular = True)