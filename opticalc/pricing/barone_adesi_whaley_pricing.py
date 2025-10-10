import numpy as np
from scipy.stats import norm

from opticalc.core.enums import OptionType
from opticalc.pricing.base import PricingBase
from opticalc.utils.exceptions import InvalidOptionTypeException


class BaroneAdesiWhaleyPricing(PricingBase):
    def _kc(self, tolerance: float, max_iterations: int) -> float:
        """
        Return the underlying price for american call options above which early exercise is optimal.
        This method is part of the Barone-Adesi and Whaley approximation formula.

        Parameters
        -----------
        tolerance : float
            The amount of deviation tolerated in the stopping condition for the Newton-Raphson algorithm.

        max_iterations : int
            The maximum amount of iterations allowed in the Newton-Raphson algorithm.

        Returns
        -----------
        float
            The underlying price above which early exercise is optimal.
        """
        n = 2 * self.b / self.sigma ** 2
        m = 2 * self.r / self.sigma ** 2
        q2u = (-(n - 1) + np.sqrt((n - 1) ** 2 + 4 * m)) / 2
        su = self.k / (1 - 1 / q2u)
        h2 = - (self.b * self.t + 2 * self.sigma * np.sqrt(self.t)) * self.k /(su - self.k)
        si = self.k + (su - self.k) * (1 - np.exp(h2))
        k = 2 * self.r / (self.sigma ** 2 * (1 - np.exp(-self.r * self.t)))
        d1 = (np.log(si / self.k) + (self.b + self.sigma ** 2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))
        q2 = (-(n - 1) + np.sqrt((n - 1) ** 2 + 4 * k)) / 2

        lhs: float = si - self.k
        rhs: float = self._parameterized_cost_of_carry_black_scholes(si, self.k, self.t, self.r, self.b, self.sigma, OptionType.Call) + (1 - np.exp((self.b - self.r) * self.t) * norm.cdf(d1)) * si / q2
        bi = np.exp((self.b - self.r) * self.t) * norm.cdf(d1) * (1 - 1 / q2) + (1 - np.exp((self.b - self.r) * self.t) * norm.pdf(d1) / (self.sigma * np.sqrt(self.t))) / q2

        while abs(float(lhs - rhs)) / self.k > tolerance:
            si = (self.k + rhs - bi * si) / (1 - bi)
            d1 = (np.log(si / self.k) + (self.b + self.sigma**2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))
            lhs = si - self.k
            rhs = self._parameterized_cost_of_carry_black_scholes(si, self.k, self.t, self.r, self.b, self.sigma, OptionType.Call) + (1 - np.exp((self.b - self.r) * self.t) * norm.cdf(d1)) * si / q2
            bi = np.exp((self.b - self.r) * self.t) * norm.cdf(d1) * (1- 1 / q2) + (1 - np.exp((self.b - self.r) * self.t) * norm.cdf(d1) / (self.sigma * np.sqrt(self.t))) / q2

            max_iterations -= 1

            if max_iterations <= 0:
                break

        return si

    def _kp(self, tolerance: float, max_iterations: int) -> float:
        """
        Return the underlying price for american put options below which early exercise is optimal.
        This method is part of the Barone-Adesi and Whaley approximation formula.

        Parameters
        -----------
        tolerance : float
            The amount of deviation tolerated in the stopping condition for the Newton-Raphson algorithm.

        max_iterations : int
            The maximum amount of iterations allowed in the Newton-Raphson algorithm.

        Returns
        -----------
        float
            The underlying price under which early exercise is optimal.
        """
        n = 2 * self.b / self.sigma ** 2
        m = 2 * self.r / self.sigma ** 2
        q1u = (-(n - 1) - np.sqrt((n - 1) ** 2 + 4 * m)) / 2
        su = self.k / (1 - 1 / q1u)
        h1 = (self.b * self.t - 2 * self.sigma * np.sqrt(self.t)) * self.k / (self.k - su)
        si = su + (self.k - su) * np.exp(h1)
        k = 2 * self.r / (self.sigma ** 2 * (1 - np.exp(-self.r * self.t)))
        d1 = (np.log(si / self.k) + (self.b + self.sigma**2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))
        q1 = (-(n - 1) - np.sqrt((n - 1) ** 2 + 4 * k)) / 2

        lhs = self.k - si
        rhs = self._parameterized_cost_of_carry_black_scholes(si, self.k, self.t, self.r, self.b, self.sigma, OptionType.Put) - (1 - np.exp((self.b - self.r) * self.t) * norm.cdf(-d1)) * si / q1
        bi = -np.exp((self.b - self.r) * self.t) * norm.cdf(-1 * d1) * (1 - 1 / q1) - (1 + np.exp((self.b - self.r) * self.t) * norm.pdf(-d1) / (self.sigma * np.sqrt(self.t))) / q1

        while abs(lhs - rhs) / self.k > tolerance:
            si = (self.k - rhs + bi * si) / (1 + bi)
            d1 = (np.log(si / self.k) + (self.b + self.sigma**2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))
            lhs = self.k - si
            rhs =  self._parameterized_cost_of_carry_black_scholes(si, self.k, self.t, self.r, self.b, self.sigma, OptionType.Put) - (1 - np.exp((self.b - self.r) * self.t) * norm.cdf(-d1)) * si / q1
            bi = -np.exp((self.b - self.r) * self.t) * norm.cdf(-d1) * (1 - 1 / q1) - (1 + np.exp((self.b - self.r) * self.t) * norm.cdf(-d1) / (self.sigma * np.sqrt(self.t))) / q1

            max_iterations -= 1

            if max_iterations <= 0:
                break

        return si

    def _barone_adesi_whaley_call(self, tolerance: float = 1e-10, max_iterations: int = 250) -> float:
        """
        Return the theoretical value of a call option using the Barone-Adesi and Whaley approximation method.

        Parameters
        -----------
        tolerance : float
            The amount of deviation tolerated in the stopping condition for the Newton-Raphson algorithm.

        max_iterations : int
            The maximum amount of iterations allowed in the Newton-Raphson algorithm.

        Returns
        -----------
        float
            The theoretical value of the option.
        """

        if self.b >= self.r:
            return self._cost_of_carry_black_scholes(self.b)

        else:
            sk = self._kc(tolerance, max_iterations)
            if self.s < sk:
                n = 2 * self.b / self.sigma ** 2
                k = 2 * self.r / (self.sigma ** 2 * (1 - np.exp(-self.r * self.t)))
                d1 = (np.log(sk / self.k) + (self.b + self.sigma ** 2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))
                q2 = (-(n -1) + np.sqrt((n -1) ** 2 + 4 * k)) / 2
                a2 = (sk / q2) * (1 - np.exp((self.b - self.r) * self.t) * norm.cdf(d1))

                return self._cost_of_carry_black_scholes(self.b) + a2 * (self.s /sk) ** q2
            else:
                return max(self.s - self.k, 0)


    def _barone_adesi_whaley_put(self, tolerance: float = 1e-10, max_iterations: int = 250) -> float:
        """
        Return the theoretical value of a put option using the Barone-Adesi and Whaley approximation method.

        Parameters
        -----------
        tolerance : float
            The amount of deviation tolerated in the stopping condition for the Newton-Raphson algorithm.

        max_iterations : int
            The maximum amount of iterations allowed in the Newton-Raphson algorithm.

        Returns
        -----------
        float
            The theoretical value of the option.
        """

        sk = self._kp(tolerance, max_iterations)

        if self.s > sk:
            n = 2 * self.b / self.sigma ** 2
            k = 2 * self.r / (self.sigma ** 2 * (1 - np.exp(-self.r * self.t)))
            q1 = (-(n - 1) - np.sqrt((n - 1) ** 2 + 4 * k)) / 2
            d1 = (np.log(sk / self.k) + (self.b + self.sigma ** 2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))
            a1 = -(sk / q1) * (1 - np.exp((self.b - self.r) * self.t) * norm.cdf(-d1))

            return self._cost_of_carry_black_scholes(self.b) + a1 * (self.s / sk)  ** q1
        else:

            return max(self.k - self.s, 0)

    def barone_adesi_whaley(self) -> float:
        """
        Return the theoretical value of an option using the Barone-Adesi and Whaley approximation method.

        The method gives a closed-form approximation for american option prices by finding the optimal early exercise boundary and computing the option value.

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
            return self._barone_adesi_whaley_call()

        elif self.option_type == OptionType.Put:
            return self._barone_adesi_whaley_put()
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")