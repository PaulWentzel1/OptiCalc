from typing import cast

import numpy as np
from scipy.stats import multivariate_normal, norm

from opticalc.core.enums import OptionType
from opticalc.pricing.base import PricingBase
from opticalc.utils.exceptions import InvalidOptionTypeException


class BjerksundStenslandPricing(PricingBase):
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

        return cast(float, multivariate_normal.cdf([a, b], mean = mean, cov = cov, allow_singular = True))

    def _phi(self, s: float, t: float, gamma: float, h: float, i: float, r: float, b: float) -> float:
        """
        Calculate the value of Phi function, an important component of the Bjerksund-Stensland model(s)

        Parameters
        -----------
        b : float
            The cost of carry rate, which is determined by the given pricing model.

        gamma : float
            ...

        h : float
            ...

        i : float
            The flat boundary (trigger price) used in the Bjerksund-Stensland model(s)

        Returns
        -----------
        float
            The value of Phi, used in the Bjerksund-Stensland 1993 and 2002 models
        """
        _lambda = (-r + gamma * b + 0.5 * gamma * (gamma - 1) * self.sigma ** 2) * t

        d = -(np.log(s / h) + (b + (gamma - 0.5) * self.sigma ** 2) * t) / (self.sigma * np.sqrt(t))

        kappa = (2 * b) / (self.sigma ** 2) + (2 * gamma - 1)

        return np.exp(_lambda) * (s ** gamma) * (norm.cdf(d) - (i / s) ** kappa * norm.cdf(d - 2 * np.log(i / s) / (self.sigma * np.sqrt(t))))

    def _psi(self, s: float, t2: float, gamma: float, h: float, i2: float, i1: float, t1: float, r: float, b: float) -> float:
        """
        Calculate the value of the Psi function, an important component of the Bjerksund-Stensland (2002) model.

        Parameters
        -----------
        s : float
            The current spot price of the underlying.

        t2 : float
            ...

        gamma : float
            ...

        h : float
            ...

        i2 : float
            ...

        i1 : float
            ...

        t1 : float
            ...

        b : float
            The cost of carry rate, which is determined by the given pricing model.

        r : float
            The risk-free rate.

        Returns
        -----------
        float
            The value of Psi, used in the Bjerksund-Stensland 2002 model.
        """

        e1 = (np.log(s/ i1) + (b + (gamma - 0.5) * self.sigma ** 2) * t1 ) / (self.sigma * np.sqrt(t1))
        e2 = (np.log(i2 ** 2 / (s * i1))  + (b + (gamma - 0.5) * self.sigma ** 2) * t1) / (self.sigma * np.sqrt(t1))
        e3 = (np.log(s / i1) - (b + (gamma - 0.5) * self.sigma ** 2) *t1) / (self.sigma * np.sqrt(t1))
        e4 = (np.log(i2 ** 2 / (s * i1)) - (b + (gamma - 0.5) * self.sigma ** 2) * t1) / (self.sigma * np.sqrt(t1))

        f1 = (np.log(s / h) + (b + (gamma - 0.5) * self.sigma ** 2) * t2) / (self.sigma * np.sqrt(t2))
        f2 = (np.log(i2 ** 2 / (s * h)) + (b + (gamma - 0.5) * self.sigma ** 2) * t2) / (self.sigma * np.sqrt(t2))
        f3 = (np.log(i1 ** 2 / (s * h)) + (b + (gamma - 0.5) * self.sigma ** 2) * t2) / (self.sigma * np.sqrt(t2))
        f4 = (np.log( (s * i1 ** 2) / (h * i2 ** 2)) + (b + (gamma - 0.5) * self.sigma ** 2) * t2) / (self.sigma * np.sqrt(t2))

        rho = np.sqrt(t1 / t2)
        _lambda = - r + gamma * b + 0.5 * gamma * (gamma -1) * self.sigma ** 2
        kappa = (2 * b) / (self.sigma ** 2) + (2 * gamma -1)

        return (np.exp(_lambda * t2) * s ** gamma
                * (self.std_bivariate_normal_cdf(-e1, -f1, rho)
                - (i2 / s) ** kappa * self.std_bivariate_normal_cdf(-e2, -f2, rho)
                - (i1 / s) ** kappa * self.std_bivariate_normal_cdf(-e3, -f3, -rho)
                + (i1 / i2) ** kappa * self.std_bivariate_normal_cdf(-e4, -f4, -rho)))

    def _bjerksund_stensland_call_1993(self, s: float, k: float, r: float, b: float) -> float:
        """
        Return the theoretical value of an american call option using the Bjerksund-Stensland approximation model (1993).
        By changing the inputs, the method returns the theoretical price of a put of same characteristics (Bjerksund-Stendland put-call transformation):
        P(s, k, t, r, b, sigma) = C(k, s, t, r - b, -b, sigma)

        Parameters
        -----------
        s : float
            The current spot price of the underlying.

        k : float
            The strike of the option.

        r : float
            The risk-free rate.

        b : float
            The cost of carry rate.

        Returns
        -----------
        float
            The theoretical option value
        """

        if b >= r:
            return self._cost_of_carry_black_scholes(b)

        else:
            beta = (1 / 2 - b / self.sigma ** 2) + np.sqrt((b / self.sigma ** 2 - 1 / 2) ** 2 + 2 * r / self.sigma ** 2)
            b_infinity = beta / (beta - 1) * k
            b_0 = max(k, r / (r - b) * k)
            ht = - (b * self.t + 2 * self.sigma * np.sqrt(self.t)) * b_0 / (b_infinity - b_0)
            i = b_0 + (b_infinity - b_0) * (1 - np.exp(ht))

            if s >= i:
                # The immediate exercise is more advantageous
                if self.option_type == OptionType.Call:
                    return np.maximum(self.s - self.k, 0)
                else:
                    return np.maximum(self.k - self.s, 0)

            else:
                alpha = (i - k) * i ** (-beta)

                return (alpha * s ** beta
                    - alpha * self._phi(b = b, gamma = beta, h = i, i = i, s = s, r = r, t = self.t)
                    + self._phi(b = b, gamma = 1, h = i, i = i, s = s, r = r, t = self.t)
                    - self._phi(b = b, gamma = 1, h = k, i = i, s = s, r = r, t = self.t)
                    - k * self._phi(b = b, gamma = 0, h = i, i = i, s = s, r = r, t = self.t)
                    + k * self._phi(b = b, gamma = 0, h = k, i = i, s = s, r = r, t = self.t))

    def bjerksund_stensland_1993(self) -> float:
        """
        Return the theoretical value of an american option using the Bjerksund-Stensland approximation model (1993).
        Assumes constant volatility, risk-free rate and allows for a continous dividend yield.
        While not as accurate as numerical methods like the Binomial pricing model, it is a faster alternative.

        Raises
        -----------
        InvalidOptionTypeException
            Raised when the option type is something else than "call" and "put".

        Returns
        -----------
        float
            The theoretical option value
        """

        if self.option_type == OptionType.Call:
            s = self.s
            k = self.k
            r = self.r
            b = self.b
            return self._bjerksund_stensland_call_1993(s, k , r, b)

        elif self.option_type == OptionType.Put:
            # Bjerksund-Stendland put-call transformation
            s = self.k
            k = self.s
            r = self.r - self.b
            b = -self.b
            return self._bjerksund_stensland_call_1993(s, k , r, b)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")

    def _bjerksund_stensland_call_2002(self, s: float, k: float, r: float, b: float) -> float:
        """
        Return the theoretical value of an american call option using the Bjerksund-Stensland approximation model (2002).
        By changing the inputs, the method returns the theoretical price of a put of same characteristics (Bjerksund-Stendland put-call transformation):
        P(s, k, t, r, b, sigma) = C(k, s, t, r - b, -b, sigma)

        Parameters
        -----------
        s : float
            The current spot price of the underlying.

        k : float
            The strike of the option.

        r : float
            The risk-free rate.

        b : float
            The cost of carry rate.

        Returns
        -----------
        float
            The theoretical option value
        """

        if b >= r:
            return self._cost_of_carry_black_scholes(b)

        else:
            t1 = 1 / 2 * (np.sqrt(5) -1) * self.t
            beta =  (1 / 2 - b / self.sigma ** 2) + np.sqrt((b / self.sigma ** 2 -1/2) ** 2 + 2 * r / self.sigma ** 2)
            b_infinity = beta / (beta - 1) * k
            b_0 = max(k, r / (r - b) * k)

            ht1 = -(b * (self.t - t1) + 2 * self.sigma * np.sqrt(self.t - t1)) * k ** 2 / ((b_infinity - b_0) * b_0) # (self.t - t1) follows the original paper, it is a deviatio from Haug's book.
            ht2 = -(b * self.t + 2 * self.sigma * np.sqrt(self.t)) * k ** 2 / ((b_infinity - b_0) * b_0)
            i1 = b_0 + (b_infinity - b_0) * (1 - np.exp(ht1))
            i2 = b_0 + (b_infinity - b_0) * (1 - np.exp(ht2))

            if s >= i2:
                return s - k
            else:
                alpha_1 = (i1 - k) * i1 ** (-beta)
                alpha_2 = (i2 - k) * i2 ** (-beta)

                return (
                    alpha_2 * s ** beta    - alpha_2 * self._phi(s=s, t= t1, gamma= beta,h = i2, i = i2, r = r, b = b)
                    + self._phi(s = s, t = t1, gamma = 1,h = i2, i = i2, r = r, b = b) - self._phi(s = s, t = t1, gamma = 1, h = i1, i = i2, r = r, b = b)
                    - k * self._phi(s = s, t = t1, gamma = 0, h = i2, i = i2, r = r, b = b) + k * self._phi(s = s, t = t1, gamma = 0, h = i1, i = i2, r = r, b = b)
                    + alpha_1 * self._phi(s = s, t = t1, gamma = beta, h = i1, i = i2, r = r, b = b) - alpha_1 * self._psi(s = s, t2 = self.t, gamma = beta, h = i1, i2 = i2, i1 = i1, t1 = t1, r = r, b = b)
                    + self._psi(s = s, t2 = self.t, gamma = 1, h = i1, i2 = i2, i1 = i1, t1 = t1, r = r, b = b) - self._psi(s = s, t2 = self.t, gamma = 1, h = k, i2 = i2, i1 = i1, t1 = t1, r = r, b = b)
                    - k * self._psi(s = s, t2 = self.t, gamma = 0, h = i1, i2 = i2, i1 = i1, t1 = t1, r = r, b = b) + k * self._psi(s = s, t2 = self.t, gamma = 0, h = k, i2 = i2, i1 = i1, t1 = t1, r = r, b = b)
                )

    def bjerksund_stensland_2002(self) -> float:
        """
        Return the theoretical value of an american option using the Bjerksund-Stensland approximation model (2002).
        Assumes constant volatility, risk-free rate and allows for a continous dividend yield.
        While not as accurate as numerical methods like the Binomial pricing model, it is a faster alternative.

        Raises
        -----------
        InvalidOptionTypeException
            Raised when the option's exercise style is not recognized or supported.

        Returns
        -----------
        float
            The theoretical option value
        """

        if self.option_type == OptionType.Call:
            s = self.s
            k = self.k
            r = self.r
            b = self.b
            return self._bjerksund_stensland_call_2002(s, k, r, b)

        elif self.option_type == OptionType.Put:
            # Bjerksund-Stendland put-call transformation
            s = self.k
            k = self.s
            r = self.r - self.b
            b = -self.b
            return self._bjerksund_stensland_call_2002(s, k , r, b)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")

    def bjerksund_stensland_combined(self) -> float:
        """
        Return the theoretical value of an american option using both Bjerksund-Stensland approximation models (1993 and 2002).
        Assumes constant volatility, risk-free rate and allows for a continous dividend yield.
        While not as accurate as numerical methods like the Binomial pricing model, it is a faster alternative.

        This approach combines both a flat and two-step boundary result to calculate the value,
        as shown in the paper.

        Raises
        -----------
        InvalidOptionTypeException
            Raised when the option type is something else than "call" and "put".

        Returns
        -----------
        float
            The theoretical value of the option.
        """

        if self.option_type == OptionType.Call:
            s = self.s
            k = self.k
            r = self.r
            b = self.b
            flat_boundary = self._bjerksund_stensland_call_1993(s, k, r, b)
            two_step_boundary = self._bjerksund_stensland_call_2002(s, k, r, b)

            return 2 * two_step_boundary - flat_boundary

        elif self.option_type == OptionType.Put:
            # Bjerksund-Stendland put-call transformation
            s = self.k
            k = self.s
            r = self.r - self.b
            b = -self.b
            flat_boundary = self._bjerksund_stensland_call_1993(s, k, r, b)
            two_step_boundary = self._bjerksund_stensland_call_2002(s, k, r, b)

            return 2 * two_step_boundary - flat_boundary
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
