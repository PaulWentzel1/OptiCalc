from typing import TypeVar, cast, Any, ParamSpec
from collections.abc import Callable

import numpy as np
from scipy.stats import norm  # type: ignore

from opticalc.core.enums import OptionType, ExerciseStyle
from opticalc.core.params import OptionParams
from opticalc.utils.exceptions import InvalidOptionTypeException, UnsupportedModelException

T = TypeVar('T', bound='PricingBase')
P = ParamSpec('P')
R = TypeVar('R')


class PricingBase(OptionParams):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    @staticmethod
    def _european_only(func: Callable[P, R]) -> Callable[P, R]:
        """Decorator used to denote pricing models or methods only applicable to european exercise-style option."""
        def wrapper(self: T, *args: Any, **kwargs: Any) -> R:  # type: ignore
            if not self.exercise_style == ExerciseStyle.European:
                exercise = (self.exercise_style.value if isinstance(self.exercise_style, ExerciseStyle)
                            else self.exercise_style)

                raise UnsupportedModelException(
                    f"{func.__name__} is only usable for European-style options. "
                    f"The current option has a {exercise}-style exercise.")
            return func(self, *args, **kwargs)  # type: ignore
        return cast(Callable[..., R], wrapper)

    @staticmethod
    def _american_only(func: Callable[P, R]) -> Callable[P, R]:
        """Decorator used to denote pricing models or methods only applicable to american exercise-style option."""
        def wrapper(self: T, *args: Any, **kwargs: Any) -> R:  # type: ignore
            if not self.exercise_style == ExerciseStyle.American:
                exercise = (self.exercise_style.value if isinstance(self.exercise_style, ExerciseStyle)
                            else self.exercise_style)

                raise UnsupportedModelException(
                    f"{func.__name__} is only usable for American-style options. "
                    f"The current option has a {exercise}-style exercise.")
            return func(self, *args, **kwargs)  # type: ignore
        return cast(Callable[..., R], wrapper)

    @staticmethod
    def _exercises_only(exercises_allowed: list[ExerciseStyle]):  # type: ignore
        """Decorator used to denote pricing models or methods only applicable to specificed exercise-style options."""
        @staticmethod
        def decorator(func: Callable[P, R]) -> Callable[P, R]:
            def wrapper(self: T, *args: Any, **kwargs: Any) -> R:  # type: ignore
                if self.exercise_style not in exercises_allowed:
                    exercise = (self.exercise_style.value if isinstance(self.exercise_style, ExerciseStyle)
                                else self.exercise_style)
                    reformat_exercise_list = list(map(lambda x: x.value, exercises_allowed))

                    raise UnsupportedModelException(
                        f"{func.__name__} is only usable for {', '.join(reformat_exercise_list)}-style options. "
                        f"The current option has a {exercise}-style exercise.")
                return func(self, *args, **kwargs)  # type: ignore
            return cast(Callable[..., R], wrapper)
        return decorator

    @property
    def d1(self) -> float:
        """
        Return the d1 parameter used in the Black-Scholes formula among others.

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
