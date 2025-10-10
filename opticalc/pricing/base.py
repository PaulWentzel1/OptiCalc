from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from opticalc.core.enums import OptionType
from opticalc.core.params import OptionParams
from opticalc.utils.exceptions import InvalidOptionTypeException


class PricingBase:
    def __init__(self, params: OptionParams):
        self.params = params
        self.s = params.s
        self.k = params.k
        self.t = params.t
        self.r = params.r
        self.q = params.q
        self.sigma = params.sigma
        self.option_type = params.option_type
        self.exercise_style = params.exercise_style
        self.b = params.b
        self.rf = params.rf
        self.premium = params.premium
        self.transaction_costs = params.transaction_costs
        self.underlying_type = params.underlying_type
        self.direction = params.direction
        self.underlying_contracts = params.underlying_contracts
        self.experimental = params.experimental

    def d1(self, b: float) -> float:
        """
        Return the d1 parameter used in the Black-Scholes formula, given a specific cost of carry b.

        Parameters
        -----------
        b : float
            The cost of carry rate, which is determined by the given pricing model.

        Returns
        -----------
        float
            The d1 parameter based on the given cost of carry of from a model.
        """
        return (np.log(self.s / self.k) + (b + self.sigma ** 2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))

    def d2(self, b: float) -> float:
        """
        Return the d2 parameter used in the Black-Scholes formula, given a specific cost of carry b.

        Parameters
        -----------
        b : float
            The cost of carry rate, which is determined by the given pricing model.

        Returns
        -----------
        float
            The d2 parameter based on the given cost of carry of from a model.
        """
        return self.d1(b) - self.sigma * np.sqrt(self.t)

    def _cost_of_carry_black_scholes(self, b: float) -> float:
        """
        Return the theoretical price of a european option using a generalized Black-Scholes Formula with the cost of carry b.

        Parameters
        -----------
        b : float
            The cost of carry rate, which is determined by the given pricing model.

        Returns
        -----------
        float
            The theoretical price of the option based on the given cost of carry of from a model.

        Raises
        -----------
        InvalidOptionTypeException
            Raised when the option type is something else than "call" and "put".
        """
        d1 = self.d1(b)
        d2 = self.d2(b)

        if self.option_type == OptionType.Call:
            return self.s * norm.cdf(d1) * np.exp((b - self.r) * self.t) - self.k * np.exp(-self.r * self.t) * norm.cdf(d2)
        elif self.option_type == OptionType.Put:
            return self.k * np.exp(-self.r * self.t) * norm.cdf(-d2) - self.s * np.exp((b - self.r) * self.t) * norm.cdf(-d1)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")

    def _parameterized_cost_of_carry_black_scholes(self, s: float, k: float, t: float, r: float, b: float, sigma: float, option_type: OptionType) -> float:
        """
        Return the theoretical price of a european option using a generalized Black-Scholes Formula with the cost of carry b.
        This method doesn't use any instance attributes, rather it relies on method parameters for input.

        Parameters
        -----------
        s : float
            The current spot price of the underlying.

        k : float
            The strike of the option.

        t : float
            The time left until the option expires.

        r : float
            The risk-free rate.

        b : float
            The cost of carry rate.

        sigma : float
            The volatility of the underlying.

        option_type : OptionType
            The Option type. Valid: OptionType.Call, OptionType.Put

        Returns
        -----------
        float
            The theoretical price of the option based on the given cost of carry of from a model.

        Raises
        -----------
        InvalidOptionTypeException
            Raised when the option type is something else than "call" and "put".
        """

        d1 = (np.log(s / k) + (b + sigma ** 2 / 2) * t) / (sigma * np.sqrt(t))
        d2 = d1 - sigma * np.sqrt(t)

        if option_type == OptionType.Call:
            return s * norm.cdf(d1) * np.exp((b - r) * t) - k * np.exp(-r * t) * norm.cdf(d2)
        elif option_type == OptionType.Put:
            return k * np.exp(-r * t) * norm.cdf(-d2) - s * np.exp((b - r) * t) * norm.cdf(-d1)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")

    def vanilla_intrinsic_value_variable(self, s: float | NDArray[Any] | None = None) -> float | NDArray[Any]:
        """
        Return the intrinsic of a vanilla option (American or European Exercise), given a specific underlying price.

        Raises
        -----------
        InvalidOptionTypeException
            Raised when the option type is something else than "call" and "put".

        Returns
        -----------
        float
            The option's intrinsic value, given its strike and the underlying's price.
        """

        if s is None:
            s = self.s

        if self.option_type == OptionType.Call:
            return np.maximum(s - self.k, 0)
        elif self.option_type == OptionType.Put:
            return np.maximum(self.k - s, 0)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")