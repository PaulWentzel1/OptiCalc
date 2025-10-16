from typing import Any

import numpy as np
from scipy.stats import norm  # type: ignore

from opticalc.core.enums import OptionType, OptionExerciseStyle
from opticalc.core.params import OptionParams
from opticalc.utils.exceptions import InvalidOptionTypeException, InvalidOptionExerciseException, UnsupportedModelException


class PricingBase(OptionParams):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def validate_pricing_model(self, method_name: str):
        """
        Used to validate if a specific model pricing can be used on a option.

        Parameters
        -----------
        function_name : str
            Name of the method to validate.

        Raises
        -----------
        UnsupportedModelException
            Raised if the intended model doesn't support the option's exercise.

        InvalidOptionExerciseException
            Raised if the option's exercise style isn't supported.
        """

        valid_european = [
            "black_scholes_adaptive",
            "black_scholes",
            "black_scholes_merton",
            "black_76",

            "bachelier",
            "bachelier_modified",

            "universal_binomial_tree",
            "binomial_cox_ross_rubinstein",
            "binomial_cox_ross_rubinstein_drift",
            "binomial_rendleman_bartter",
            "binomial_leisen_reimer",
            "binomial_jarrow_rudd",
            "binomial_jarrow_rudd_risk_neutral",
            "binomial_tian"]

        if self.rf is not None:
            valid_european.append("garman_kohlhagen")

        valid_american = [
            "universal_binomial_tree"
            "binomial_cox_ross_rubinstein",
            "binomial_cox_ross_rubinstein_drift",
            "binomial_rendleman_bartter",
            "binomial_leisen_reimer",
            "binomial_jarrow_rudd",
            "binomial_jarrow_rudd_risk_neutral",
            "binomial_tian",

            "bjerksund_stensland_call_1993",
            "bjerksund_stensland_1993",
            "bjerksund_stensland_call_2002",
            "bjerksund_stensland_2002",
            "bjerksund_stensland_combined",

            "barone_adesi_whaley"]

        valid_bermuda = []

        valid_asian = []

        exercise = self.exercise_style.value if isinstance(self.exercise_style, OptionExerciseStyle) else self.exercise_style

        if self.b == 0:
            valid_european.append("vega_black_76_max_time")

        if self.exercise_style == OptionExerciseStyle.European:
            if method_name not in valid_european:
                raise UnsupportedModelException(
                    f"{method_name} is not usable for European-style options. "
                    f"The current option has a {exercise}-style exercise.")

        elif self.exercise_style == OptionExerciseStyle.American:
            if method_name not in valid_american:
                raise UnsupportedModelException(
                    f"{method_name} is not usable for American-style options. "
                    f"The current option has a {exercise}-style exercise.")

        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            if method_name not in valid_bermuda:
                raise UnsupportedModelException(
                    f"{method_name} is not usable for Bermuda-style options. "
                    f"The current option has a {exercise}-style exercise.")

        elif self.exercise_style == OptionExerciseStyle.Asian:
            if method_name not in valid_asian:
                raise UnsupportedModelException(
                    f"{method_name} is not usable for Asian-style options. "
                    f"The current option has a {exercise}-style exercise.")

        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{exercise}' is not valid.")

    def method_class(self, method_name: str) -> str | None:
        """
        Return the class in which a specific method was defined. Used to verify pricing models on a group/class-basis.

        Parameters
        -----------
        method_name : str
            Name of the method to verify.

        Returns:
            The class where the method is defined
        """

        for base_class in self.__class__.__mro__:
            if method_name in base_class.__dict__:
                return base_class.__name__
        return None

    @property
    def d1(self) -> float:
        """
        Return the d1 parameter used in the Black-Scholes formula among others.

        Parameters
        -----------
        b : float
            The cost of carry rate, which is determined by the given pricing model.

        Returns
        -----------
        float
            The d1 parameter based on the given cost of carry of from a model.
        """
        return (np.log(self.s / self.k) + (self.b + self.sigma ** 2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))

    @property
    def d2(self) -> float:
        """
        Return the d2 parameter used in the Black-Scholes formula among others.

        Parameters
        -----------
        b : float
            The cost of carry rate, which is determined by the given pricing model.

        Returns
        -----------
        float
            The d2 parameter based on the given cost of carry of from a model.
        """
        return self.d1 - self.sigma * np.sqrt(self.t)

    def d1_cost_of_carry(self, b: float) -> float:
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

    def d2_cost_of_carry(self, b: float) -> float:
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
        return self.d1_cost_of_carry(b) - self.sigma * np.sqrt(self.t)

    def black_scholes_cost_of_carry(self, b: float) -> float:
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
        d1 = self.d1_cost_of_carry(b)
        d2 = self.d2_cost_of_carry(b)

        if self.option_type == OptionType.Call:
            return self.s * norm.cdf(d1) * np.exp((b - self.r) * self.t) - self.k * np.exp(-self.r * self.t) * norm.cdf(d2)
        elif self.option_type == OptionType.Put:
            return self.k * np.exp(-self.r * self.t) * norm.cdf(-d2) - self.s * np.exp((b - self.r) * self.t) * norm.cdf(-d1)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
