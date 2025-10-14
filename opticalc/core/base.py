from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from opticalc.core.enums import Direction, OptionExerciseStyle, OptionType, Underlying
from opticalc.core.params import OptionParams
from opticalc.core.constants import ATM_THRESHOLD, AT_FORWARD_THRESHOLD
from opticalc.utils.exceptions import (InvalidDirectionException,
                                       InvalidOptionExerciseException,
                                       InvalidOptionTypeException,
                                       InvalidUnderlyingException)


class OptionBase(OptionParams, ABC):
    """Core methods with validation and cost of carry logic"""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

        # Input validation and processing
        self._process_and_validate_inputs()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(s={self.s}, k={self.k}, t={self.t}, r={self.r}, q={self.q}, sigma={self.sigma}, option_type={self.option_type}, exercise_style={self.exercise_style}, b={self.b}, rf={self.rf}, premium={self.premium},transaction_costs={self.transaction_costs}, underlying_type={self.underlying_type}, direction={self.direction})"

    def __str__(self) -> str:
        """
        Return a string containing general information about the option.

        Returns
        -----------
        str
            Basic information about the option and its given parameters.
        """

        _option_direction = str(self.direction.value) + " " if isinstance(self.direction, Direction) else ""
        _exercise = self.exercise_style.value if isinstance(self.exercise_style, OptionExerciseStyle) else self.exercise_style
        _type = self.option_type.value if isinstance(self.option_type, OptionType) else self.option_type
        _contracts = f" with {str(self.underlying_contracts)} underlying contracts." if self.underlying_contracts is not None else "."
        _underlying = f"The option's underlying is of type {self.underlying_type.value if isinstance(self.underlying_type, Underlying) else self.underlying_type}.\n" if self.underlying_type is not None else ""
        _rf = f"Interest rate (foreign): {str(self.rf)}\n" if self.rf is not None else ""
        _premium = f"The option currently trades at a premium of {self.premium}" if self.premium is not None else ""
        _transaction = f"The transaction costs associated with trading: {self.transaction_costs}" if self.transaction_costs is not None else ""

        return (
            f"This is a {_option_direction}{_exercise}-style {_type} option{_contracts}\n"
            f"{_underlying}"
            f"Spot price of the underlying asset: {self.s}\n"
            f"Strike price: {self.k}\n"
            f"Time to expiry: {self.t}\n"
            f"Interest rate (domestic): {self.r}\n"
            f"{_rf}"
            f"Dividend yield: {self.q}\n"
            f"Volatility: {self.sigma}\n"
            f"The option's cost of carry rate is {self.b}\n"
            f"{_premium}"
            f"{_transaction}"
        )

    def _process_and_validate_inputs(self) -> None:
        """
        Validates specific inputs of an option and converts some to enums if necessary.
        The method validates the underlying price, strike, time to expiry, volatility type, exercise, underlying, direction,

        Raises
        -----------
        InvalidOptionTypeException
            Raised if the option's type is invalid.

        InvalidOptionExerciseException
            Raised if the option's exercise is invalid.

        InvalidUnderlyingException
            Raised if the option's underlying asset is invalid.

        InvalidDirectionException
            Raised if the option's direction is invalid.

        NameError
            Raised if a specific variable is not defined or is None. (In this case if the Underlying is FX and rf is None).

        ValueError
            Raised if any of the inputs seem unreasonable or would result in faulty calculations.
        """

        # Input validation for option_type
        if isinstance(self.option_type, str):
            try:
                self.option_type = OptionType(self.option_type.lower())
            except ValueError as e:
                raise InvalidOptionTypeException(f"Invalid input '{self.option_type}'. Valid inputs for option_type are: "
                                                 f"{[element.value for element in OptionType]}") from e

        # Input validation for exercise_style
        if isinstance(self.exercise_style, str):
            try:
                self.exercise_style = OptionExerciseStyle(self.exercise_style.lower())
            except ValueError as e:
                raise InvalidOptionExerciseException(f"Invalid input '{self.exercise_style}'. Valid inputs for exercise_style"
                                                     f" are: {[element.value for element in OptionExerciseStyle]}") from e

        # Input validation for underlying_type
        if self.underlying_type is not None:
            if isinstance(self.underlying_type, str):
                try:
                    self.underlying_type = Underlying(self.underlying_type.lower())
                except ValueError as e:
                    raise InvalidUnderlyingException(f"Invalid input '{self.underlying_type}'. Valid inputs "
                                                     f"for underlying_type are: "
                                                     f"{[element.value for element in Underlying]}") from e

        # Input validation for direction
        if self.direction is not None:
            if isinstance(self.direction, str):
                try:
                    self.direction = Direction(self.direction.lower())
                except ValueError as e:
                    raise InvalidDirectionException(f"Invalid input '{self.direction}'. Valid inputs for direction are: "
                                                    f"{[element.value for element in Direction]}") from e

        # Specific Input validation for FX Options
        if self.underlying_type == Underlying.FX:
            if not self.rf:
                raise NameError("The foreign interest rate (rf) must be defined for FX Options.")

        if self.s <= 0:
            raise ValueError(f"The underlying's price cannot be negative. Input was {self.s}.")

        if self.k <= 0:
            raise ValueError(f"The option's strike cannot be negative or zero. Input was {self.k}.")

        if self.t <= 0:
            raise ValueError(f"The option's time to expiry cannot be negative or 0. Input was {self.t}.")

        if self.sigma <= 0:
            raise ValueError(f"The underlying's volatility cannot be negative or 0. Input was {self.sigma}.")

    @property
    @abstractmethod
    def intrinsic_value(self) -> float:
        """
        Return the intrinsic value of the option, given the current underlying price.

        Raises
        -----------
        InvalidOptionTypeException
            Raised when the option type is something else than "call" and "put".

        Returns
        -----------
        float
            The option's intrinsic value, given its strike and the underlying's current price.
        """
        if self.option_type == OptionType.Call:
            return np.maximum(self.s - self.k, 0)
        elif self.option_type == OptionType.Put:
            return np.maximum(self.k - self.s, 0)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")

    @abstractmethod
    def intrinsic_value_variable(self, s: float | None = None, k: float | None = None) -> float:
        """
        Return the intrinsic value of the option, given the current underlying price.

        Raises
        -----------
        InvalidOptionTypeException
            Raised when the option type is something else than "call" and "put".

        Returns
        -----------
        float
            The option's intrinsic value, given its strike and the underlying's current price.
        """
        s = s if s is not None else self.s
        k = k if k is not None else self.k

        if self.option_type == OptionType.Call:
            return np.maximum(s - k, 0)
        elif self.option_type == OptionType.Put:
            return np.maximum(k - s, 0)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")

    @property
    @abstractmethod
    def extrinsic_value(self) -> float:
        """
        Return the extrinsic value (Time value) of the option, given its intrinsic value.

        Raises
        -----------
        NameError
            Raised when the options premium isn't defined.

        Returns
        -----------
        float
            The option's extrinsic value given its intrinsic value.
        """

        if self.premium is not None:
            return max(self.premium - self.intrinsic_value, 0)
        else:
            raise NameError("The option's premium must be defined.")

    @abstractmethod
    def profit_at_expiry_variable(self, s: float | None = None, premium: float | None = None, transaction_costs: float | None = None) -> float:
        """
        Return the profit or loss of an option, regardless of exercise type, at expiry.
        This method takes several inputs, allowing for multiple calculations with different values,
        which is useful when constructing payoff diagrams.

        Parameters
        -----------
        s : float or None, default to None
            The current spot price of the underlying.

        premium : float or None, default to None
            The current price of the option. Used to derive implied volatility and calculate P&L.

        transaction_costs : float or None, default to None
            The transaction costs associated with trading the option.

        Returns
        -----------
        The profit or loss as expiry, given the input parameters.
        """
        s = s if s is not None else self.s
        premium = premium if premium is not None else self.premium

        if premium is None:
            if self.premium is None:
                raise ValueError("Premium must be provided")
            premium = self.premium

        if transaction_costs is None:
            transaction_costs = 0

        if self.direction == Direction.Long:
            return self.intrinsic_value_variable(s, self.k) - premium - transaction_costs
        else:
            return premium - self.intrinsic_value_variable(s, self.k) - transaction_costs

    @property
    @abstractmethod
    def moneyness(self) -> str:
        """
        Return the option's current level of moneyness in string format.

        Returns
        -----------
        str
            The option's moneyness.
        """
        if np.absolute(self.s - self.k) < ATM_THRESHOLD:
            return "At the money."
        elif self.at_the_forward:
            return "At the forward."
        elif self.intrinsic_value > 0:
            return "In the money."
        else:
            return "Out of the money."

    @property
    @abstractmethod
    def at_the_forward(self) -> bool:
        """
        Return True if the Option is currently at the forward, else False.

        Returns
        -----------
        bool
            The boolean condition if the option is at the forward or not.
        """
        return abs(self.k - self.s * np.exp(self.b * self.t)) < AT_FORWARD_THRESHOLD

    @property
    @abstractmethod
    def at_the_forward_underlying(self) -> float:
        """
        Return an approximation of the underlying price where the option will trade at-the-forward.

        Returns
        -----------
        float
            The underlying price where the option trades at-the-forward.
        """
        return self.k * np.exp(-self.b * self.t)

    @property
    @abstractmethod
    def at_the_forward_strike(self) -> float:
        """
        Return an approximation of the strike price where the option will trade at-the-forward.

        Returns
        -----------
        float
            The strike price where the option trades at-the-forward.
        """
        return self.s * np.exp(self.b * self.t)
