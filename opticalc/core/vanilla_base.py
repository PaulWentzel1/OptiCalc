from abc import ABC, abstractmethod
from typing import Any, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np

from opticalc.core.enums import Direction, ExerciseStyle, OptionType, Underlying
from opticalc.core.params import OptionParams
from opticalc.utils.constants import (ATM_THRESHOLD,
                                      AT_FORWARD_THRESHOLD,
                                      PLOT_LOWER_THRESHOLD_MULTIPLIER,
                                      PLOT_UPPER_THRESHOLD_MULTIPLIER,
                                      INDICATOR_LINE_LINESTYLE,
                                      INDICATOR_LINE_ALPHA,
                                      INDICATOR_LINE_LINEWIDTH,
                                      PLOT_FONT_SIZE_MAIN,
                                      PLOT_FONT_WEIGHT_MAIN,
                                      PLOT_FONT_SIZE_LEGEND,
                                      PLOT_LINE_WIDTH)
from opticalc.utils.exceptions import (InvalidDirectionException,
                                       InvalidExerciseException,
                                       InvalidOptionTypeException,
                                       InvalidUnderlyingException,
                                       MissingParameterException)


class VanillaOptionBase(OptionParams, ABC):
    """
    OptionBase contains the core methods and logic for a vanilla Option-type object, such as input validation and
    dunder methods. The class inherits from OptionParams to accommodate the option's parameters or "input".

    Some methods are defined as abstractmethods, this is to accommodate for them being overwritten or modified in child
    objects like EuropeanOption, AmericanOption and others as these might contain special cases where the current,
    vanilla implementation in VanillaOptionBase is invalid.
    """
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(s={self.s}, k={self.k}, t={self.t}, r={self.r}, q={self.q}, sigma={self.sigma}, "
                f"option_type={self.option_type}, exercise_style={self.exercise_style}, b={self.b}, rf={self.rf}, "
                f"premium={self.premium},transaction_costs={self.transaction_costs}, underlying_type={self.underlying_type},"
                f" direction={self.direction})")

    def __str__(self) -> str:
        """
        Return a string containing general information about the option.

        Returns
        -----------
        str
            Basic information about the option and its given parameters.
        """
        return_msg = ""

        direction = str(self.direction.value) + " " if isinstance(self.direction, Direction) else ""
        exercise = self.exercise_style.value if isinstance(self.exercise_style, ExerciseStyle) else self.exercise_style
        option_type = self.option_type.value if isinstance(self.option_type, OptionType) else self.option_type
        contracts = f" with {str(self.underlying_contracts)} underlying contracts." if self.underlying_contracts is not None else "."
        underlying = f"The option's underlying is of type " f"{self.underlying_type.value if isinstance(self.underlying_type, Underlying) else self.underlying_type}.\n" if self.underlying_type is not None else ""
        foreign_interest_rate = f"Interest rate (foreign): {str(self.rf)}\n" if self.rf is not None else ""
        premium = f"\nThe option currently trades at a premium of {self.premium}" if self.premium is not None else ""
        transaction = f"\nTransaction costs: {self.transaction_costs}" if self.transaction_costs is not None else ""

        return_msg += f"This is a {direction}{exercise}-exercise {option_type} option{contracts}\n"
        return_msg += underlying
        return_msg += f"Spot price of the underlying: {self.s}\n"
        return_msg += f"Strike price: {self.k}\n"
        return_msg += f"Time to expiry: {self.t}\n"
        return_msg += f"Volatility: {self.sigma}\n"
        return_msg += f"Interest rate (domestic): {self.r}\n"
        return_msg += foreign_interest_rate
        return_msg += f"Dividend yield: {self.q}\n"
        return_msg += f"Cost of carry rate: {self.b}"
        return_msg += premium
        return_msg += transaction

        return return_msg

    def __setattr__(self, name: str, value: Any):
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

        MissingParameterException
            Raised if a specific variable is not defined or is None. (In this case if the Underlying is FX and rf is None).

        ValueError
            Raised if any of the inputs seem unreasonable or would result in faulty calculations.
        """
        if name in ("s", "k", "t", "sigma"):
            if value <= 0:
                raise ValueError(f"{name} must be greater than 0.")

        elif name == "option_type":
            if isinstance(value, str):
                try:
                    value = OptionType(value.lower())
                except ValueError as e:
                    raise InvalidOptionTypeException(f"Invalid input '{value}'. Valid inputs for option_type are: "
                                                    f"{[element.value for element in OptionType]}") from e
            elif not isinstance(value, OptionType):
                raise InvalidOptionTypeException(f"Invalid input '{value}'. Valid inputs for option_type are: "
                                                    f"{[element.value for element in OptionType]}")
        elif name == "exercise_style":
            if isinstance(value, str):
                try:
                    value = ExerciseStyle(value.lower())
                except ValueError as e:
                    raise InvalidExerciseException(f"Invalid input '{value}'. Valid inputs for exercise_style"
                                                     f" are: {[element.value for element in ExerciseStyle]}") from e
            elif not isinstance(value, ExerciseStyle):
                raise InvalidExerciseException(f"Invalid input '{value}'. Valid inputs for exercise_style"
                                                     f" are: {[element.value for element in ExerciseStyle]}")

        elif name == "underlying_type":
            if not value:
                pass
            else:
                if isinstance(value, str):
                    try:
                        value = Underlying(value.lower())
                    except ValueError as e:
                        raise MissingParameterException(f"Invalid input '{value}'. Valid inputs for underlying_type"
                                                        f" are: {[element.value for element in Underlying]}") from e
                elif not isinstance(value, Underlying):
                    raise MissingParameterException(f"Invalid input '{value}'. Valid inputs for underlying_type"
                                                        f" are: {[element.value for element in Underlying]}")
                if value == Underlying.FX:
                    if not hasattr(self, "rf") or self.rf is None:
                        raise MissingParameterException("The foreign interest rate (rf) must be defined for FX Options.")

        elif name == "direction":
            if not value:
                pass
            else:
                if isinstance(value, str):
                    try:
                        value = Direction(value.lower())
                    except ValueError as e:
                        raise InvalidDirectionException(f"Invalid input '{value}'. Valid inputs for direction"
                                                        f" are: {[element.value for element in Direction]}") from e
                elif not isinstance(value, Direction):
                    raise InvalidDirectionException(f"Invalid input '{value}'. Valid inputs for direction"
                                                        f" are: {[element.value for element in Direction]}")
        super().__setattr__(name, value)

    def _process_and_validate_inputs(self) -> None:
        # ! Currently not used
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

        MissingParameterException
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
                self.exercise_style = ExerciseStyle(self.exercise_style.lower())
            except ValueError as e:
                raise InvalidExerciseException(f"Invalid input '{self.exercise_style}'. Valid inputs for exercise_style"
                                                     f" are: {[element.value for element in ExerciseStyle]}") from e

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
                raise MissingParameterException("The foreign interest rate (rf) must be defined for FX Options.")

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

    def plot_payoff(
            self,
            x_range: tuple[float, float] | None = None,
            ) -> Tuple[matplotlib.figure.Figure, Any]:
        """
        Plot the option's payoff at expiry, alongside key data such as breakeven, current spot price, strike and P&L.

        Parameters
        -----------
        x_range : tuple[float, float] or None, default None.
            The minimum and maximum spot prices to be displayed. If None is passed, the values will be autocalculated.

        Raises
        -----------
        MissingParameterException
            Raised if the option's premium or direction is not defined.

        Returns
        -----------
        fix, ax : Tuple[matplotlib.figure.Figure, Any]
            A tuple containing a matplotlib Figure and Axes object.
        """

        if (not self.premium) or (not self.direction):
            raise MissingParameterException("The option's premium and direction must to be defined")

        if x_range is not None:
            x_lower = x_range[0]
            x_upper = x_range[1]
        else:
            x_lower = self.k * PLOT_LOWER_THRESHOLD_MULTIPLIER
            x_upper = self.k * PLOT_UPPER_THRESHOLD_MULTIPLIER

        spot_prices = np.linspace(x_lower, x_upper, 100)

        if self.option_type == OptionType.Call:
            payoff = np.maximum(spot_prices - self.k, 0)
            breakeven = self.k + self.premium
        else:
            payoff = np.maximum(self.k - spot_prices, 0)
            breakeven = self.k - self.premium

        payoff = payoff * -1 if self.direction == Direction.Short else payoff
        pnl = payoff - self.premium if self.direction == Direction.Long else payoff + self.premium

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot payoff and P&L
        ax.axhline(0, color="black", linewidth=PLOT_LINE_WIDTH, alpha=0.7) # 0-line
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=PLOT_LINE_WIDTH)

        # Indicator lines
        ax.axvline(self.s,
                   color="orange",
                   linestyle=INDICATOR_LINE_LINESTYLE,
                   alpha=INDICATOR_LINE_ALPHA,
                   linewidth=INDICATOR_LINE_LINEWIDTH,
                   label=f"Current Spot: {self.s:.2f}")

        ax.axvline(self.k,
                   color="red",
                   linestyle=INDICATOR_LINE_LINESTYLE,
                   alpha=INDICATOR_LINE_ALPHA,
                   linewidth=INDICATOR_LINE_LINEWIDTH,
                   label=f'Strike: {self.k:.2f}')

        if x_lower <= breakeven <= x_upper:
            ax.axvline(breakeven,
                       color="purple",
                       linestyle=INDICATOR_LINE_LINESTYLE,
                       alpha=INDICATOR_LINE_ALPHA,
                       linewidth=INDICATOR_LINE_LINEWIDTH,
                       label=f"Breakeven: {breakeven:.2f}")

        # Payoff & P&L lines
        ax.plot(spot_prices, payoff, linewidth=1, color="steelblue", label="Option payoff at expiry", linestyle="--")
        ax.plot(spot_prices, pnl, linewidth=1, color="black",label="Profit & Loss")
    
        ax.set_xlabel("Spot Price at Expiry", fontsize=PLOT_FONT_SIZE_MAIN, fontweight=PLOT_FONT_WEIGHT_MAIN)
        ax.set_ylabel("Profit & Loss", fontsize=PLOT_FONT_SIZE_MAIN, fontweight=PLOT_FONT_WEIGHT_MAIN)

        direction = self.direction.name if isinstance(self.direction, Direction) else self.direction
        option_type = self.option_type.name if isinstance(self.option_type, OptionType) else self.option_type
        title = f"{direction} {option_type} option payoff at expiry"

        ax.set_title(title, fontsize=PLOT_FONT_SIZE_MAIN, fontweight=PLOT_FONT_WEIGHT_MAIN, pad=15)
        ax.legend(fontsize=PLOT_FONT_SIZE_LEGEND, loc="best")

        plt.tight_layout()
        plt.show()

        return fig, ax

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
        MissingParameterException
            Raised when the options premium isn't defined.

        Returns
        -----------
        float
            The option's extrinsic value given its intrinsic value.
        """

        if self.premium is not None:
            return max(self.premium - self.intrinsic_value, 0)
        else:
            raise MissingParameterException("The option's premium must be defined.")

    @abstractmethod
    def profit_at_expiry_variable(self, s: float | None = None,
                                  premium: float | None = None, transaction_costs: float | None = None) -> float:
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
