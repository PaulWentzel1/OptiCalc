import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
from enum import Enum
from numpy.typing import NDArray
from typing import Any, cast

from utils._exceptions import InvalidOptionTypeException, UnsupportedModelException, InvalidOptionExerciseException, InvalidUnderlyingException, InvalidDirectionException
from utils._formulas import BinomialFormulas, BjerksundStenslandFormulas


class OptionType(Enum):
    """Option type"""
    Call = "call"
    Put = "put"


class OptionExerciseStyle(Enum):
    """Option exercise style"""
    European = "european"
    American = "american"
    Bermuda = "bermuda"
    Asian = "asian"


class OptionUnderlyingType(Enum):
    """Type of underlying"""
    Equity = "equity"
    Stock_index = "index"
    Future = "future"
    FX = "fx" # If defined, make rf required
    Interest_rate = "interest_rate" # Bonds
    Commodity = "commodity" # Only for spot commodity


class OptionDirection(Enum):
    """The direction of the option"""
    Long = "long"
    Short = "short"


class Option:
    """
    The main option class. Represents a financial option (Gives the holder the right, but not the obligation, to buy/sell the specific underlying asset).

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

    q : float
        A continuous dividend yield.

    sigma : float
        The volatility of the underlying.

    option_type : OptionType or str
        The Option type. Valid: call, put, OptionType.Call, OptionType.Put

    exercise_style : OptionExerciseStyle or str
        The exercise style of the option. Valid: european, american, bermuda, asian,
        OptionExerciseStyle.European, OptionExerciseStyle.American etc.

    b : float or None, default None
        The cost of carry rate.

    rf : float or None, default None
        The foreign interest rate. Used for FX Options, in which case r is the domestic interest rate.

    premium : float or None, default None
        The current price of the option. Used to derive implied volatility and calculate P&L.

    transaction_costs : float or None, default to None
        The transaction costs associated with trading the option.

    underlying_type: OptionUnderlyingType, str or None, default None.
        The type of underlying asset the option tracks.

    direction: OptionDirection, str or None, default None.
        The direction of the option, if it is sold or bought.

    experimental: bool, default False
        Used to enter experimental mode, where certain input validations and checks aren't made.
    """

    def __init__(
        self,
        s: float,
        k: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
        option_type: OptionType | str,
        exercise_style: OptionExerciseStyle | str,
        b: float | None = None,
        rf: float | None = None,
        premium: float | None = None,
        transaction_costs: float | None = None,
        underlying_type: OptionUnderlyingType | str | None = None,
        direction: OptionDirection | str | None = None,
        experimental: bool = False
    ) -> None:

        # Input validation for option_type
        if isinstance(option_type, str):
            try:
                self.option_type = OptionType(option_type.lower())
            except ValueError as e:
                raise InvalidOptionTypeException(f"Invalid input '{option_type}'. Valid inputs for option_type are: {[element.value for element in OptionType]}") from e
        else:
            self.option_type = option_type

        # Input validation for exercise_style
        if isinstance(exercise_style, str):
            try:
                self.exercise_style = OptionExerciseStyle(exercise_style.lower())
            except ValueError as e:
               raise InvalidOptionExerciseException(f"Invalid input '{exercise_style}'. Valid inputs for option_type are: {[element.value for element in OptionExerciseStyle]}") from e
        else:
            self.exercise_style = exercise_style

        # Input validation for underlying_type
        if underlying_type is not None:
            if isinstance(underlying_type, str):
                try:
                    self.underlying_type = OptionUnderlyingType(underlying_type.lower())
                except ValueError as e:
                    raise InvalidUnderlyingException(f"Invalid input '{underlying_type}'. Valid inputs for option_type are: {[element.value for element in OptionUnderlyingType]}") from e
            else:
                self.underlying_type = underlying_type
        else:
            self.underlying_type = underlying_type

        # Input validation for direction
        if direction is not None:
            if isinstance(direction, str):
                try:
                    self.direction = OptionDirection(direction.lower())
                except ValueError as e:
                    raise InvalidDirectionException(f"Invalid input '{direction}'. Valid inputs for direction are: {[element.value for element in OptionDirection]}") from e
            else:
                self.direction = direction
        else:
            self.direction = direction

        # Input validation for FX Options
        if self.underlying_type == OptionUnderlyingType.FX:
            if not rf:
                raise NameError("The foreign interest rate (rf) must be defined for FX Options.")

        self.s = s
        self.k = k
        self.t = t
        self.r = r
        self.q = q
        self.sigma = sigma
        self._override_b = b
        self.rf = rf
        self.premium = premium
        self.transaction_costs = transaction_costs
        self.experimental = experimental

        # Input validation for s, k, t, sigma
        if not self.experimental:
            self._validate_inputs()


    @property
    def b(self) -> float:
        """
        Autocalculates the cost of carry rate (b) based on the option's underlying type or uses overrides provided by the user.
        """

        if self._override_b is not None:
            return self._override_b
        else:
            if self.underlying_type == OptionUnderlyingType.Future:
                return 0

            elif self.underlying_type == OptionUnderlyingType.FX:
                if self.rf is None:
                    raise NameError("The foreign interest rate (rf) must be defined.")
                return self.r - self.rf # fx options
            else:
                return self.r - self.q # equity, index, commodity (spot) options


    def modify_cost_of_carry(self, b: float | None = None) -> None:
        """
        Modify the cost of carry with a custom value. Useful for exotic options or plotting changes in cost of carry.

        """
        if b is not None:
            self._override_b = b
        else:
            self._override_b = None


    def _validate_inputs(self) -> None:
        """
        Used to validate the inputs of an option.

        Raises
        -----------
        ValueError
            If any of the inputs seem unreasonable or would result in faulty calculations.
        """

        if self.s <= 0:
            raise ValueError(f"The underlying's price cannot be negative. Input was {self.s}.")

        if self.k <= 0:
            raise ValueError(f"The option's strike cannot be negative or zero. Input was {self.k}.")

        if self.t <= 0:
            raise ValueError(f"The option's time to expiry cannot be negative or 0. Input was {self.t}.")

        if self.sigma <= 0:
            raise ValueError(f"The underlying's volatility cannot be negative or 0. Input was {self.sigma}.")


    def _validate_model(self, function_name: str) -> None:
        """
        Used to validate if a specific model can be used on a option.

        Parameters
        -----------
        function_name : str
            The name of the current function being used.

        Raises
        -----------
        UnsupportedModelException
            Raised if the intended model doesn't support the option's exercise.

        InvalidOptionExerciseException
            Raised if the option's exercise style isn't supported.
        """

        valid_european = ["black_scholes", "black_scholes_merton", "black_76",
                          "binomial_leisen_reimer", "binomial_jarrow_rudd",
                          "binomial_rendleman_bartter", "binomial_cox_ross_rubinstein",
                          "binomial_jarrow_rudd_risk_neutral","binomial_tian", "black_scholes_adaptive", "bachelier",
                          "bachelier_modified"]

        if self.rf is not None:
            valid_european.append("garman_kohlhagen")
        valid_american = ["bjerksund_stensland_1993", "bjerksund_stensland_2002", "bjerksund_stensland_combined",
                           "binomial_leisen_reimer", "binomial_jarrow_rudd", "binomial_rendleman_bartter",
                           "binomial_cox_ross_rubinstein", "binomial_jarrow_rudd_risk_neutral", "binomial_tian",
                           "barone_adesi_whaley"]

        valid_bermuda = []

        valid_asian = []

        if self.b == 0:
            valid_european.append("vega_black_76_max_time")

        if self.exercise_style == OptionExerciseStyle.European:
            if function_name not in valid_european:
                raise UnsupportedModelException(
                f"{function_name} is not usable for European-style options. "
                f"The current option has a {self.exercise_style.value}-style exercise")

        elif self.exercise_style == OptionExerciseStyle.American:
            if function_name not in valid_american:
                raise UnsupportedModelException(
                f"{function_name} is not usable for American-style options."
                f"The current option has a {self.exercise_style.value}-style exercise")

        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            if function_name not in valid_bermuda:
                raise UnsupportedModelException(
                f"{function_name} is not usable for Bermuda-style options."
                f"The current option has a {self.exercise_style.value}-style exercise")

        elif self.exercise_style == OptionExerciseStyle.Asian:
            if function_name not in valid_asian:
                raise UnsupportedModelException(
                f"{function_name} is not usable for Asian-style options."
                f"The current option has a {self.exercise_style.value}-style exercise")

        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def __str__(self) -> str:
        """
        Return a string containing general information about the option.

        Returns
        -----------
        str
            Basic information about the option and its given parameters.
        """

        line_length = 63
        return (
            f"{'='*line_length}\n"
           #f"This is a {self.direction.value if self.direction is not None else ''} {self.option_type.value} option with a {self.exercise_style.value}-style exercise.\n"
            f"Option-specific details:\n"
            f"Current spot price of the underlying: {self.s}\n"
            f"Strike price: {self.k}\n"
            f"Time to expiry: {self.t}\n"
            f"Volatility of the underlying: {self.sigma}\n"
            f"Dividend yield of the underlying: {self.q}\n"
            f"Cost of carry of the underlying: {self.b}\n"
            f"{'='*line_length}"
        )


    def __repr__(self) -> str:
        return (f"Option({self.s}, {self.k}, {self.t}, {self.r}, {self.q}, "
                f"{self.sigma}, {self.option_type}, {self.exercise_style}, {self.b if self.b else None}, {self.premium if self.premium else None}, "
                f"{self.rf if self.rf else None}, {self.transaction_costs if self.transaction_costs else None},"
                f"{self.underlying_type if self.underlying_type else None}, {self.experimental})")


    @property
    def available_models(self) -> list[str]:
        """
        Returns the available pricing models for a specific option.

        Returns
        -----------
        list[str]
            A list of all valid pricing models.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            valid = ["black_scholes", "black_scholes_merton",
                     "black_76", "binomial_leisen_reimer",
                     "binomial_jarrow_rudd", "binomial_rendleman_bartter",
                    "binomial_cox_ross_rubinstein"]

            if self.rf is not None:
                valid.append("garman_kohlhagen")
            return valid

        elif self.exercise_style == OptionExerciseStyle.American:
            valid = ["bjerksund_stensland_1993","bjerksund_stensland_2002",
                     "bjerksund_stensland_combined", "binomial_leisen_reimer",
                     "binomial_jarrow_rudd", "binomial_rendleman_bartter",
                     "binomial_cox_ross_rubinstein"]

            return valid

        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            valid = [""] #Placeholder
            return valid
        elif self.exercise_style == OptionExerciseStyle.Asian:
            valid = [""] # Placeholder
            return valid

        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def intrinsic_value(self) -> float:
        """
        Return the intrinsic value of the option, given the current underlying price.

        Returns
        -----------
        float
            The option's intrinsic value given its strike and the underlying's current price.
        """

        return cast(float, self.intrinsic_value_variable())


    def intrinsic_value_variable(self, s: float | NDArray[Any] | None = None) -> float | NDArray[Any]:
        """
        Return the intrinsic of the option, given a specific underlying price.

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


    @property
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
            return np.maximum(self.premium - self.intrinsic_value, 0)
        else:
            raise NameError("The option's premium must be defined.")


    @property
    def moneyness(self) -> str:
        """
        Return the current moneyness of the option.

        Returns
        -----------
        str
            The option's moneyness.
        """

        if np.absolute(self.s - self.k) < 0.05:
            return "At the money."
        elif self.at_the_forward():
            return "At the forward."
        elif self.intrinsic_value > 0:
            return "In the money."
        else:
            return "Out of the money."


    def at_the_forward(self, tolerance: float = 1e-5) -> bool:
        """
        Return True if the Option is currently at the forward, else False

        Parameters
        -----------
        tolerance : float, defaults to 1e-5
            The tolerance level of the evaluation.

        Returns
        -----------
        bool
            The boolean condition if the option is at the forward or not.
        """

        return abs(self.k - self.s * np.exp(self.b * self.t)) < tolerance


    @property
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
    def at_the_forward_strike(self) -> float:
        """
        Return an approximation of the strike price where the option will trade at-the-forward.

        Returns
        -----------
        float
            The strike price where the option trades at-the-forward.
        """
        return self.s * np.exp(self.b * self.t)


    def profit(self) -> float:
        raise NotImplementedError()


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

        if s is None:
            s = self.s

        if premium is None:
            if self.premium is not None:
                premium = self.premium
            else:
                premium = 0

        if transaction_costs is None:
            transaction_costs = 0

        if self.direction == OptionDirection.Long:
            return cast(float, self.intrinsic_value_variable(s) - premium - transaction_costs)

        else: # Short
            return cast(float, premium - self.intrinsic_value_variable(s) - transaction_costs)


    @property
    def plot(self):
        raise NotImplementedError()


    @property
    def delta(self) -> float:
        """
        Return the greek Delta of the option.

        Delta is also known as Spot Delta or DvalueDspot and is a first-order partial derivative.

        Delta represents the sensitivity of an option's price to changes in the price
        of the underlying asset. It also can be interpreted as an option's hedge ratio,
        e.g. how many of the underlying must be bought or sold as to become delta-neutral.

        For options on underlying assets that pay no dividend yield, the delta
        of a call option is always between 0 and 1, whereas the delta of a put option
        is between -1 and 0. Should the cost of carry (When the underlying pays dividends)
        exceed the interest rate, deep ITM options can be greater than 1 or less than -1.

        Returns
        -----------
        float
            The delta of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised if the option's exercise style isn't supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._delta_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")

    def _delta_european(self) -> float:
        """Return the Delta / Spot Delta / DvalueDspot of a european call or put option."""
        if self.option_type == OptionType.Call:
            return np.exp((self.b - self.r) * self.t) * norm.cdf(self.d1_adaptive)
        elif self.option_type == OptionType.Put:
            return -np.exp((self.b - self.r) * self.t) * norm.cdf(- self.d1_adaptive)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def delta_mirror(self) -> float:
        """
        Return the Delta Mirror strikes of an option.


        If the current option is a call, the strike of a put with the same parameters will be returned, whereas if the
        current option is a put, the strikes of a call with otherwise same parameters will be returned.

        The two strikes equalize the absolute values of the Deltas of the call and put options. The formula can be used
        when determining strikes for Delta-neutral strategies. The major issue is that this only applies to symmetric
        volatility smiles.

        Returns
        -----------
        float
            The strike of the Delta Mirror Call or Put, depending on the option's type.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised if the option's exercise style isn't supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._delta_mirror_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _delta_mirror_european(self) -> float:
        """ Delta mirror of a european call or put option."""
        if self.option_type == OptionType.Call:
            return (self.s ** 2 / self.k) * np.exp((2 * self.b + self.sigma ** 2) * self.t)
        elif self.option_type == OptionType.Put:
            return (self.s ** 2 / self.k) * np.exp((2 * self.b + self.sigma ** 2) * self.t)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def straddle_symmetric_underlying(self) -> float:
        """
        Return the underlying price that yields the same absolute Delta value for identical strikes on a call and put.

        Returns
        -----------
        float
            The symmetric Delta price.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised if the option's exercise style isn't supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.k * np.exp((- self.b - self.sigma ** 2 / 2) * self.t)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def strike_from_spot_delta(self, delta: float) -> float:
        """
        Return the option's strike price for a given spot Delta.

        Spot Delta refers to the Delta of the option in terms of the current underlying price.
        Quotation by Delta rather than strike is common in some OTC markets. This method takes the option's parameters and
        a specified value for Delta and returns the corresponding strike.

        Parameters
        -----------
        delta : float
            The Delta of the option.

        Returns
        -----------
        float
            The strike of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._strike_from_spot_delta_european(delta)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _strike_from_spot_delta_european(self, delta: float) -> float:
        """Return the strike from spot Delta of a european call or put option."""
        if self.option_type == OptionType.Call:
            return self.s * np.exp(-norm.ppf(delta * np.exp((self.r - self.b) * self.t)) * self.sigma *np.sqrt(self.t) + (self.b + self.sigma ** 2 / 2) * self.t)
        elif self.option_type == OptionType.Put:
            return self.s * np.exp(norm.ppf(-delta * np.exp((self.r - self.b) * self.t)) * self.sigma *np.sqrt(self.t) + (self.b + self.sigma ** 2 / 2) * self.t)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    def strike_from_futures_delta(self, delta: float) -> float:
        """
        Return the option's strike price for a given futures Delta.

        Futures Delta is used when hedging with futures.
        Quotation by Delta rather than strike is common in some OTC markets. This method takes the option's parameters and
        a specified value for Delta and returns the corresponding strike.

        Parameters
        -----------
        delta : float
            The Delta of the option.

        Returns
        -----------
        float
            The strike of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._strike_from_futures_european(delta)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _strike_from_futures_european(self, delta: float) -> float:
        """Return the strike from futures Delta of a european call or put option."""
        delta = delta * np.exp(-self.b * self.t)
        if self.option_type == OptionType.Call:
            return self.s * np.exp(-norm.ppf(delta * np.exp((self.r - self.b) * self.t)) * self.sigma *np.sqrt(self.t) + (self.b + self.sigma ** 2 / 2) * self.t)
        elif self.option_type == OptionType.Put:
            return self.s * np.exp(norm.ppf(-delta * np.exp((self.r - self.b) * self.t)) * self.sigma *np.sqrt(self.t) + (self.b + self.sigma ** 2 / 2) * self.t)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def vanna(self) -> float:
        """
        Return the greek Vanna of an Option.

        Vanna is also known as Ddeltavol or DvegaDspot and is a second-order
        partial derivative. Vanna measures how much Delta will change due
        to changes in the volatility and how much Vega will change due to
        changes in the underlying's price. In other words, Vanna represents
        the sensitivity of an option's Delta to changes in the volatility of
        the underlying asset.

        Vanna can be a useful metric when monitoring a Delta- or Vega-hedged portfolio,
        as it helps to predict the effectiveness of a Delta-hedge when vol changes.
        Alternatively it can be used to predict the effectiveness of a Vega-hedge when
        the underlying price changes.

        Returns
        -----------
        float
            The Vanna of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._vanna_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _vanna_european(self) -> float:
        """Return the Vanna / DdeltaDvol / DvegaDspot of a european call or put option."""
        return ((-np.exp((self.b - self.r) * self.t) * self.d2_adaptive) / (self.sigma)) * norm.pdf(self.d1_adaptive)


    @property
    def max_vanna(self) -> float:
        """
        Return the maximal value of the option's Vanna / DdeltaDvol / DvegaDspot.

        Returns
        -----------
        float
            The maximal value of the option's Vanna.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """

        if self.exercise_style == OptionExerciseStyle.European:
            return self.k * np.exp(-self.b * self.t - self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def min_vanna(self) -> float:
        """
        Return the minimal value of the option's Vanna / DdeltaDvol / DvegaDspot.

        Returns
        -----------
        float
            The minimal value of the option's Vanna.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """

        if self.exercise_style == OptionExerciseStyle.European:
            return self.k * np.exp(-self.b * self.t + self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def vanna_min_strike(self) -> float:
        """
        Return the strike price where the Vanna attains its minimum.

        Returns
        -----------
        float
            The strike price of the option where Vanna is the lowest.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.s * np.exp(self.b * self.t - self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def vanna_max_strike(self) -> float:
        """
        Return the strike price where the Vanna attains its maximum.

        Returns
        -----------
        float
            The strike price of the option where Vanna is the highest.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.s * np.exp(self.b * self.t + self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def yanna(self) -> float:
        """
        Return the greek Yanna of an option.

        Yanna is also known as DvannaDvol or DvommaDspot and is a third-order partial derivative. It measures how much Vanna
        will change in respect to volatility. Yanna can be a useful metric when monitoring a Vanna-hedged portfolio,
        as it helps to predict the effectiveness of the Vanna-hedge when vol changes.
        To get the value for a one point change in volatility, one has to divide the Yanna by 10.000.

        Returns
        -----------
        float
            The Yanna of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._yanna_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _yanna_european(self) -> float:
        """Return the Yanna / DvannaDvol / DvommaDspot of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.vanna * (1/self.sigma) * (self.d1_adaptive * self.d2_adaptive - (self.d1_adaptive /self.d2_adaptive) - 1)


    @property
    def charm(self) -> float:
        """
        Return the greek Charm of the option.

        Charm is also known as DdeltaDtime, Delta bleed or Delta decay and is a second-order partial derivative.
        It measures the sensitivity of Delta to changes in time.
        Charm is expressed on a year-basis, so by dividing with the number of days in a year one can get the daily charm.

        Returns
        -----------
        float
            The Charm of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._charm_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _charm_european(self) -> float:
        """Return the Charm / DdeltaDtime / Delta bleed / Delta decay of a european call or put option."""
        if self.option_type == OptionType.Call:
            return -np.exp((self.b - self.r) * self.t) * (norm.pdf(self.d1_adaptive) * ( (self.b) / (self.sigma * np.sqrt(self.t))   - (self.d2_adaptive) / (2 * self.t)) + (self.b - self.r) * norm.cdf(self.d1_adaptive))
        elif self.option_type == OptionType.Put:
            return -np.exp((self.b - self.r) * self.t) * (norm.pdf(self.d1_adaptive) * ( (self.b) / (self.sigma * np.sqrt(self.t))   - (self.d2_adaptive) / (2 * self.t)) - (self.b - self.r) * norm.cdf(-self.d1_adaptive))
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def elasticity(self) -> float:
        """
        Return the greek Elasticity of an option

        Elasticity is also known as Lambda, Omega or leverage. It measures the sensitivity of the option in percent to a
        percent change in the underlying's price.


        Returns
        -----------
        float
            The Elasticity of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._elasticity_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _elasticity_european(self) -> float:
        """Return the Elasticity / Lambda / Omega / leverage of a european call or put option."""
        if self.option_type == OptionType.Call:
            return np.exp((self.b - self.r) * self.t) * norm.cdf(self.d1_adaptive) * (self.s / self.black_scholes_adaptive)
        elif self.option_type == OptionType.Put:
            return np.exp((self.b - self.r) * self.t) * (norm.cdf(self.d1_adaptive) - 1) * (self.s / self.black_scholes_adaptive)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def short_term_option_volatility(self) -> float:
        """
        Return the volatility over a short time period of an option.

        The option's short term volatility is approximately equal to the Elasticity of the option multiplied by the
        volatility of the underlying asset. The Elasticity of the option changes with the underlying price and time,

        Returns
        -----------
        float
            The Elasticity of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.sigma * np.abs(self._elasticity_european())
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def gamma(self) -> float:
        """
        Return the greek Gamma of an Option.

        Gamma is also known as Convexity or DdeltaDspot and is a second-order partial derivative.
        Gamma measures the Delta's sensitivity to changes in the prices of the underlying asset.

        When Delta-hedging

        Behaviour: As the option moves towards ATM, the Gamma increases. ATM options have the highest Gamma at their strike,
        given that the Delta is most sensitive to changes in the underlying price here.

        Returns
        -----------
        float
            The Gamma of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._gamma_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _gamma_european(self) -> float:
        """Return the Gamma / Convexity / DdeltaDspot of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return (norm.pdf(self.d1_adaptive) * np.exp((self.b - self.r) * self.t)) / (self.s * self.sigma * np.sqrt(self.t))


    @property
    def gamma_percent(self) -> float:
        """
        Return the greek Gamma Percent of an option.

        Gamma Percent is also known as GammaP and expresses the percentage changes in Delta for percentage changes in the
        underlying asset. Its used as an alternative or complementary definition for traditional Gamma.

        Returns
        -----------
        float
            The Gamma Percent of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """

        if self.exercise_style == OptionExerciseStyle.European:
            return self._gamma_percent_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _gamma_percent_european(self) -> float:
        """Return the Gamma Percent / GammaP of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return (self.s * (norm.pdf(self.d1_adaptive) * np.exp((self.b - self.r) * self.t)) / (self.s * self.sigma * np.sqrt(self.t))) / 100


    @property
    def gamma_percent_max_underlying(self) -> float:
        """
        Return the underlying price where the Gamma Percent attains its maximum.

        Returns
        -----------
        float
            The underlying price of the option where Gamma Percent is greatest.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.k * np.exp((- self.b -  self.sigma ** 2 / 2) * self.t)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def gamma_percent_max_strike(self) -> float:
        """
        Return the strike where the Gamma Percent attains its maximum.

        Returns
        -----------
        float
            The strike of the option where Gamma Percent is greatest.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.s * np.exp((self.b + self.sigma ** 2 / 2) * self.t)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def gamma_max_underlying(self) -> float:
        """
        Return the underlying price where the Gamma attains its maximum.

        Returns
        -----------
        float
            The underlying price of the option where Gamma is greatest.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.k * np.exp((- self.b - 3 * self.sigma ** 2 / 2) * self.t)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def gamma_max_strike(self) -> float:
        """
        Return the strike where the Gamma attains its maximum.

        Returns
        -----------
        float
            The strike of the option where Gamma is greatest.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.s * np.exp((self.b + self.sigma ** 2 / 2) * self.t)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def gamma_saddle_time(self) -> float:
        """
        Return the time point where the Gamma attains its saddle point.

        Returns
        -----------
        float
            The point in time (In years) where the option's Gamma has a saddle point.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """

        if not self.r < self.sigma ** 2 + 2 * self.b:
            raise ValueError("The returned value must be greater than 0 for the saddle point to exist. Therefore r < sigma ** 2 + 2b")

        if self.exercise_style == OptionExerciseStyle.European:
            return 1 / (2 * (self.sigma ** 2 + 2 * self.b - self.r))
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def gamma_saddle_underlying(self) -> float:
        """
        Return the underlying price where the Gamma attains its saddle point.

        Returns
        -----------
        float
           The underlying price of the option where Gamma has a saddle point.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.k * np.exp((-self.b - 3 * self.sigma ** 2 / 2) * self.gamma_saddle_time)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def gamma_saddle(self) -> float:
        """
        Return the Gamma value found at the Option's Gamma saddle point.

        Returns
        -----------
        float
            The Gamma found at the saddle point.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return (np.sqrt(np.exp(1) / np.pi) * np.sqrt((2 * self.b - self.r) / self.sigma ** 2 + 1)) / self.k
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def zomma(self) -> float:
        """
        Return the greek Zomma of an option.

        Zomma is also known as DgammaDvol and describes the sensitivity of Gamma with respect to changes in the
        implied volatility.

        To get Zomma for a one-unit volatility change (15% -> 16%), one simply divides the Zomma by 100.

        Returns
        -----------
        float
            The Zomma of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._zomma_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _zomma_european(self) -> float:
        """Return the Zomma / DgammaDvol of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.gamma * ((self.d1_adaptive * self.d2_adaptive - 1) / self.sigma)


    @property
    def zomma_range_underlying(self) -> tuple[float, float, str, str]:
        """
        Return the range of underlying prices between which the greek Zomma and Zomma Percent is negative.

        These values are approximations and might not always hold true.

        Returns
        -----------
        tuple
            - lower_boundary (float): Lower bound of the range where Zomma is negative.
            - upper_boundary (float): Upper bound of the range where Zomma is negative.
            - text (str): Description of the negative Zomma or Zomma Percent range.
            - text_alternative (str): Description of the positive Zomma or Zomma Percent range.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            lower_boundary = cast(float, self.k * np.exp(- self.b * self.t - self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2))
            upper_boundary = cast(float, self.k * np.exp(- self.b * self.t + self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2))
            text = f"Zomma and Zomma Percent are negative for the underlying prices between {lower_boundary:.4f} and {upper_boundary:.4f}"
            text_alternative = f"Zomma and Zomma Percent are positive for the underlying prices outside of {lower_boundary:.4f} and {upper_boundary:.4f}"
            return lower_boundary, upper_boundary, text, text_alternative

        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def zomma_range_strikes(self) -> tuple[float, float, str, str]:
        """
        Return the range of strike prices between which the greek Zomma and Zomma Percent is negative.

        These values are approximations and might not always hold true.

        Returns
        -----------
        tuple
            - lower_boundary (float): Lower bound of the range where Zomma is negative.
            - upper_boundary (float): Upper bound of the range where Zomma is negative.
            - text (str): Description of the negative Zomma or Zomma Percent range.
            - text_alternative (str): Description of the positive Zomma or Zomma Percent range.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            lower_boundary = cast(float, self.s * np.exp(self.b * self.t) - self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2)
            upper_boundary = cast(float, self.s * np.exp(self.b * self.t) + self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2)
            text = f"Zomma and Zomma Percent are negative for the strike prices between {lower_boundary:.4f} and {upper_boundary:.4f}"
            text_alternative = f"Zomma and Zomma Percent are positive for the strike prices outside of {lower_boundary:.4f} and {upper_boundary:.4f}"
            return lower_boundary, upper_boundary, text, text_alternative

        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def zomma_percent(self) -> float:
        """
        Return the greek Zomma Percent of an option.

        Zomma Percent also known as ZommaP or DgammaPDvol and expresses the percentage changes in Gamma for percentage
        changes in the implied volatility.

        Returns
        -----------
        float
            The Zomma Percent of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._zomma_percent_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _zomma_percent_european(self) -> float:
        """Return the Zomma Percent / ZommaP / DgammaPDvol of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.gamma_percent * ((self.d1_adaptive * self.d2_adaptive - 1) / self.sigma)


    @property
    def speed(self) -> float:
        """
        Return the greek Speed of an option.

        Speed is also known as DgammaDspot or "Gamma of Gamma" and measures the rate of change in Gamma with respect to
        changes in the underlying price.
        A high value for speed means that the Gamma is very sensitive to changes in the price of the underlying.

        Returns
        -----------
        float
            The Speed of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._speed_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _speed_european(self) -> float:
        """Return the Speed / DgammaDspot / "Gamma of Gamma" of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return -((self.gamma * (1 + self.d1_adaptive / (self.sigma * np.sqrt(self.t))   )) / self.s)


    @property
    def speed_percent(self) -> float:
        """
        Return the greek Speed Percent of an option.

        Speed Percent is also known as DgammaPDspot and measures the percentage changes in Gamma for percentage changes
        in the price of the underlying.
        A high value for speed means that the Gamma is very sensitive to changes in the price of the underlying.

        Returns
        -----------
        float
            The Speed Percent of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._speed_percent_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _speed_percent_european(self) -> float:
        """Return the Speed Percent / DgammaPDspot of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return - self.gamma * (self.d1_adaptive / (100 * self.sigma * np.sqrt(self.t)))


    @property
    def colour(self) -> float:
        """
        Return the greek Colour of an option.

        Colour Percent is also known as ColourP, ColorP, DgammaPDtime and Gamma Percent Bleed represents the change in Gamma
        with respect to changes in time to maturity.

        Colour represents the change in terms of a year. By dividing Colour by 365 we get the sensitivity for a one-day
        move.

        Returns
        -----------
        float
            The Colour of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._colour_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _colour_european(self) -> float:
        """Return the Colour / Color / DgammaDtime / Gamma Bleed / GammaTheta of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.gamma * (self.r - self.b + (self.b * self.d1_adaptive / self.sigma * np.sqrt(self.t))  + ( (1 - self.d1_adaptive * self.d2_adaptive) / 2 * self.t))


    @property
    def colour_percent(self) -> float:
        """
        Return the greek Colour Percent of an option.

        Colour is also known as Color, DgammaDtime, Gamma Bleed or GammaTheta and represents the percentage change in Gamma
        for percentage changes in the time to maturity.


        Returns
        -----------
        float
            The Colour Percent of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._colour_percent_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _colour_percent_european(self) -> float:
        """Return the Colour Percent / Color Percent / DgammaPDtime / Gamma Percent Bleed of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.gamma_percent * (self.r - self.b + (self.b * self.d1_adaptive / self.sigma * np.sqrt(self.t))  + ( (1 - self.d1_adaptive * self.d2_adaptive) / 2 * self.t))


    @property
    def vega(self) -> float:
        """
        Return the greek Vega of an option.

        Vega represents the sensitivity of the option's price sensitivity to changes in the
        underlying's volatility.

        Returns
        -----------
        float
            The Vega of the Option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._vega_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _vega_european(self) -> float:
        """Return the Vega of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.s * np.exp((self.b - self.r) * self.t) * norm.pdf(self.d1_adaptive) * np.sqrt(self.t)


    @property
    def vega_percent(self) -> float:
        """
        Return the greek Vega Percent of an option.

        Vega Percent also known as VegaP and represents the sensitivity of the option's price sensitivity to changes in the
        underlying's volatility.

        Returns
        -----------
        float
            The Vega of the Option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._vega_percent_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _vega_percent_european(self) -> float:
        """Return the Vega Percentage / VegaP of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return (self.sigma / 10) * self.s * np.exp((self.b - self.r) * self.t) * norm.pdf(self.d1_adaptive) * np.sqrt(self.t)


    @property
    def vega_max_underlying_local(self) -> float:
        """
        Return the underlying price where Vega attains its local maximum.

        Returns
        -----------
        float
            The underlying price of the option where Vega is greatest.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.k * np.exp((- self.b + self.sigma ** 2 / 2) * self.t)

        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def vega_max_strike_local(self) -> float:
        """
        Return the strike price where Vega attains its maximum local.

        Returns
        -----------
        float
            The strike price of the option where Vega is greatest.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.s * np.exp((self.b + self.sigma ** 2 / 2) * self.t)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def vega_black_76_max_time(self) -> float:
        """
        Return the time to maturity where Vega attains its maximum for the Black-76 pricing model.

        Returns
        -----------
        float
            The time to maturity of the option where Vega is greatest.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        self._validate_model("vega_black_76_max_time")

        if self.exercise_style == OptionExerciseStyle.European:
            return (2 * (1 + np.sqrt(1 + (8 * self.r * (1/self.sigma ** 2) + 1) * np.log(self.s / self.k) ** 2))) / (8 * self.r + self.sigma ** 2)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def vega_max_time_global(self) -> float:
        """
        Return the time to maturity where Vega attains its global maximum.

        Returns
        -----------
        float
            The time to maturity of the option where Vega is greatest.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return 1 / (2* self.r)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def vega_max_underlying_global(self) -> float:
        """
        Return the underlying price where Vega attains its global maximum.

        Returns
        -----------
        float
            The underlying price of the option where Vega is greatest.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option type is something else than "call" and "put".
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.k * np.exp( (-self.b + self.sigma ** 2 / 2)  / (2 * self.r))
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def vega_max(self) -> float:
        """
        Return the value Vega attains at its global maximum.

        Returns
        -----------
        float
            The value of Vega at its global maximum.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.k / (2 * np.sqrt(self.r * np.e * np.pi))
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def vega_elasticity(self) -> float:
        """
        Return the Vega Elasticity of the option.

        Vega Elasticity is also known as Vega Leverage and describes the percentage change in option value in regard to
        percentage point changes in the volatility of the underlying.

        The Vega Elasticity is the greatest for OTM options.

        Returns
        -----------
        float
            The Vega Elasticity of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.vega * self.sigma / self.black_scholes_adaptive
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
           raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
           raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def vomma(self) -> float:
        """
        Return the greek Vomma of the option.

        Vomma is also known as DvegaDvol, Volga or Vega Convexity measures the sensitivity of vega to changes in the
        implied volatility. Vomma is typically expressed as the change per one percentage point (1% = 100 percentage points).
        By dividing the Vomma by 10.000 one gets this value.

        Positive Vomma meams one will earn more for every percentage point increase in volatility and if IV is fallong

        If one believes implied volatility will be volatile in the short term, one should typically try to find options with
        a high Vega and go long these.

        Returns
        -----------
        float
            The Vomma of the option.

        Examples
        -----------
        >>> option = Option(...)  # European option setup
        >>> vomma_value = option.vomma
        >>> pp_vomma = raw_vomma / 10000  # Convert to a percentage point-change basis

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._vomma_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _vomma_european(self) -> float:
        """Return the Vomma / DvegaDvol / Volga / Vega Convexity of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self._vega_european() * ((self.d1_adaptive * self.d2_adaptive) / self.sigma)


    @property
    def vomma_range_strike(self) -> tuple[float, float, str, str]:
        """
        Return the range of strikes prices between which the greek Vomma and Vomma Percent is negative.

        These values are approximations and might not always hold true.

        Returns
        -----------
        tuple
            - lower_boundary (float): Lower bound of the range where Vomma is negative.
            - upper_boundary (float): Upper bound of the range where Vomma is negative.
            - text (str): Description of the negative Vomma or Vomma Percent range.
            - text_alternative (str): Description of the positive Vomma or Vomma Percent range.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            lower_boundary = self.k * np.exp((- self.b - self.sigma ** 2 / 2) * self.t)
            upper_boundary = self.k * np.exp((- self.b + self.sigma ** 2 / 2) * self.t)
            text = f"Vomma and Vomma Percent are negative for the underlying prices between {lower_boundary:.4f} and {upper_boundary:.4f}"
            text_alternative = f"Vomma and Vomma Percent are positive for the underlying prices outside of {lower_boundary:.4f} and {upper_boundary:.4f}"
            return lower_boundary, upper_boundary, text, text_alternative

        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def vomma_range_underlying(self) -> tuple[float, float, str, str]:
        """
        Return the range of underlying prices between which the greek Vomma and Vomma Percent is negative.

        These values are approximations and might not always hold true.

        Returns
        -----------
        tuple
            - lower_boundary (float): Lower bound of the range where Vomma is negative.
            - upper_boundary (float): Upper bound of the range where Vomma is negative.
            - text (str): Description of the negative Vomma or Vomma Percent range.
            - text_alternative (str): Description of the positive Vomma or Vomma Percent range.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            lower_boundary = self.s * np.exp((self.b - self.sigma ** 2 / 2) * self.t)
            upper_boundary = self.s * np.exp((self.b + self.sigma ** 2 / 2) * self.t)
            text = f"Vomma and Vomma Percent are negative for the underlying prices between {lower_boundary:.4f} and {upper_boundary:.4f}"
            text_alternative = f"Vomma and Vomma Percent are positive for the underlying prices outside of {lower_boundary:.4f} and {upper_boundary:.4f}"
            return lower_boundary, upper_boundary, text, text_alternative

        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def vomma_percent(self) -> float:
        """
        Return the greek Vomma Percent of the option.

        Vomma Percent is also known as VommaP, DvegaPDvol or VolgaP measures the percent change in sensitivity of vega to
        percent changes in the implied volatility.

        Returns
        -----------
        float
            The Vomma Percent of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._vomma_percent_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _vomma_percent_european(self) -> float:
        """Return the Vomma Percent / VommaP / DvegaPDvol / VolgaP of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self._vega_percent_european() * ((self.d1_adaptive * self.d2_adaptive) / self.sigma)


    @property
    def ultima(self) -> float:
        """
        Return the greek Ultima of an option.

        Ultima is also known as DvommaDvol and measures the Vomma's sensitivity to a change in volatility.

        To get Ultima in terms of a one volatility point move, it must be divided by 1.000.000.

        Returns
        -----------
        float
            The Ultima of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._ultima_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _ultima_european(self) -> float:
        """Return the Ultima / DvommaDvol of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.vomma * (1 / self.sigma) * (self.d1_adaptive * self.d2_adaptive - (self.d1_adaptive / self.d2_adaptive) - (self.d2_adaptive / self.d1_adaptive) - 1)


    @property
    def veta(self) -> float:
        """
        Return the greek Veta of an option.

        Veta is also known as DvegaDtime, Vega decay or Vega Bleed and measures the change in Vega with respect to changes in
        the time to expiry.

        To get a one-percentage point change in volatility to a one day change in time, one divides the Veta by 36500 or
        25200, depending on if one looks at trading days only.

        Returns
        -----------
        float
            The Veta of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._veta_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _veta_european(self) -> float:
        """Return the Veta / DvegaDtime of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.vega * (self.r - self.b + self.b * self.d1_adaptive / self.sigma * np.sqrt(self.t) - ((1 + self.d1_adaptive * self.d2_adaptive) / 2 * self.t))


    @property
    def variance_vega(self) -> float:
        """
        Return the Variance of the greek Vega.

        Variance Vega describes the Black-Scholes Merton formula's sensitivity to a small change in the variance of the
        underlying asset's instantaneous rate of return.
        Variance Vega is equal to the option's vega divided by 2 * the underlying's volatility.

        Returns
        -----------
        float
            The Variance Vega of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.s * np.exp((self.b - self.r) * self.t) * norm.pdf(self.d1_adaptive) * (np.sqrt(self.t) /  (2 * self.sigma))
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def ddeltadvar(self) -> float:
        """
        Return the DdeltaDvar of the option.

        DdeltaDvar is the change in Delta for a change in the variance (Variance Vanna).

        Returns
        -----------
        float
            The Variance Vega of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return - self.s * np.exp((self.b - self.r) * self.t) * norm.pdf(self.d1_adaptive) * (self.d2_adaptive / (2 * self.sigma ** 2))
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def variance_vomma(self) -> float:
        """
        Return the Variance Vomma of the option.

        Variance Vomma describes the Variance Vega's sensitivity to a small change in the variance.

        Returns
        -----------
        float
            The Variance Vomma of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return ((self.s * np.exp((self.b - self.r)  * self.t) * np.sqrt(self.t)) / (4 * self.sigma ** 3)) * norm.pdf(self.d1_adaptive) * (self.d1_adaptive * self.d2_adaptive - 1)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def variance_ultima(self) -> float:
        """
        Return the Variance Ultima of the option.

        The Variance Ultima is the Black-Scholes-Merton formulas third derivative with respect to variance.

        Returns
        -----------
        float
            The Variance Ultima of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return ((self.s * np.exp((self.b - self.r)  * self.t) * np.sqrt(self.t)) / (8 * self.sigma ** 5)) * norm.pdf(self.d1_adaptive) * ( (self.d1_adaptive * self.d2_adaptive - 1) * (self.d1_adaptive * self.d2_adaptive - 3) - (self.d1_adaptive ** 2 + self.d2_adaptive ** 2))
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def theta(self) -> float:
        """
        Return the Theta of the option.

        Theta is also known as Expected Bleed, time bleed or time decay and measures the sensitivity of the option's value to
        changes in the time to expiry.

        Returns
        -----------
        float
            The Theta of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._theta_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _theta_european(self) -> float:
        """Return the Theta / Expected Bleed / Time bleed / Time decay of a european call or put option."""
        if self.option_type == OptionType.Call:
            return -((self.s * np.exp((self.b - self.r) * self.t) * norm.pdf(self.d1_adaptive) * self.sigma) / (2 * np.sqrt(self.t))) - (self.b - self.r) * self.s * np.exp((self.b - self.r) * self.t) * norm.cdf(self.d1_adaptive) - self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2_adaptive)
        elif self.option_type == OptionType.Put:
            return -((self.s * np.exp((self.b - self.r) * self.t) * norm.pdf(self.d1_adaptive) * self.sigma) / (2 * np.sqrt(self.t))) + (self.b - self.r) * self.s * np.exp((self.b - self.r) * self.t) * norm.cdf(-self.d1_adaptive) + self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(-self.d2_adaptive)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    def theta_daily(self, trading_days: int = 365) -> float:
        """
        Return the Daily Theta of the option.

        Theta is also known as Expected Bleed, time bleed or time decay and measures the sensitivity of the option's value to
        changes in the time to expiry.
        Daily Theta is the Theta of the option divided by the number of days in one year.

        Parameters
        -----------
        trading_days : int, default to 365
            The amount of trading days. Set to 365 for the daily Theta, or 52 for weekly Theta etc.

        Returns
        -----------
        float
            The Theta of the option.

        Raises
        -----------
        ValueError
            Raised when the trading days are negative

        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if trading_days < 0:
            raise ValueError(f"The option's trading days '{trading_days}' is not valid, it has to be positive.")

        if self.exercise_style == OptionExerciseStyle.European:
            return self._theta_european() / trading_days
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def driftless_theta(self) -> float:
        """
        Return the Driftless Theta of an option.
        Driftless Theta is also known as Pure bleed and isolates the time decay effect, as assumes r = 0 and b = 0.
        It measures the effect time decay has uncertainty assuming constant volatility.

        Returns
        -----------
        float
            The Driftless Theta of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self.driftless_theta_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def driftless_theta_european(self) -> float:
        """Return the Driftless Theta / Pure Bleed of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return -(self.s * norm.pdf(self.d1_adaptive) * self.sigma) / (2 * np.sqrt(self.t))


    def driftless_theta_daily(self, trading_days: int) -> float:
        """
        Return the Driftless Theta of an option.

        Driftless Theta is also known as Pure bleed and isolates the time decay effect, as assumes r = 0 and b = 0.
        It measures the effect time decay has uncertainty assuming constant volatility.

        Parameters
        -----------
        trading_days : int, default to 365
            The amount of trading days. Set to 365 for the daily Theta, or 52 for weekly Theta etc.

        Returns
        -----------
        float
            The Driftless Theta Daily of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if trading_days < 0:
            raise ValueError(f"The option's trading days '{trading_days}' is not valid, it has to be positive.")

        if self.exercise_style == OptionExerciseStyle.European:
            return self.driftless_theta_european() / trading_days
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")



    def bleed_offset_volatility(self, trading_days: int = 365) -> float:
        """
        Return the Bleed-Offset Volatility of the option.

        The Bleed-Offset Volatility descries how much the volatility has to increase to offset the Time decay effect.

        Parameters
        -----------
        trading_days : int, default to 365
            The amount of trading days. Set to 365 for the daily Theta, or 52 for weekly Theta etc.

        Returns
        -----------
        float
            The Driftless Theta Daily of the option.

        Raises
        -----------
        ValueError
            Raised when the trading days are negative

        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """

        if trading_days < 0:
            raise ValueError(f"The option's trading days '{trading_days}' is not valid, it has to be positive.")

        if self.exercise_style == OptionExerciseStyle.European:
            return (self._theta_european() / trading_days) / self._vega_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    @property
    def rho(self) -> float:
        """
        Return the Rho of an option.

        Rho describes the options value's sensitivity to changes in the risk-free interest rate.
        To get the Rho for a 1% change, one divides the Rho by 100.

        Returns
        -----------
        float
            The Rho of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._rho_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _rho_european(self) -> float:
        """Return the Rho of a european call or put option."""
        if self.underlying_type and self.underlying_type == OptionUnderlyingType.Future:
            return -self.t * self.black_scholes_adaptive
        elif self.option_type == OptionType.Call:
            return self.t * self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2_adaptive)
        elif self.option_type == OptionType.Put:
            return -self.t * self.k * np.exp(-self.r * self.t) * norm.cdf(-self.d2_adaptive)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def phi(self) -> float:
        """
        Return the Phi of the option.

        Phi is also known as Rho-2 and represents the option value's sensitivity to a change in the dividend yield of the
        underlying asset or the foreign interest rate, should the option be an FX option.

        Returns
        -----------
        float
            The Phi of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._phi_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _phi_european(self) -> float:
        """Return the Phi / Rho-2 of a european call or put option."""
        if self.option_type == OptionType.Call:
            return - self.t * self.s * np.exp((self.b - self.r) * self.t) * norm.cdf(self.d1_adaptive)
        elif self.option_type == OptionType.Put:
            return self.t * self.s * np.exp((self.b - self.r) * self.t) * norm.cdf(-self.d1_adaptive)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def carry_rho(self) -> float:
        """
        Return Carry Rho of the option.

        The Carry Rho or Cost-of-carry Rho reflects the option value's sensitivity to changes in the cost-of-carry rate.

        Returns
        -----------
        float
            The Carry Rho of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._carry_rho_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _carry_rho_european(self) -> float:
        """Return the Cost-of-carry Rho of a european call or put option."""
        if self.option_type == OptionType.Call:
            return self.t * self.s * np.exp((self.b - self.r) * self.t) * norm.cdf(self.d1_adaptive)
        elif self.option_type == OptionType.Put:
            return -self.t * self.s * np.exp((self.b - self.r) * self.t) * norm.cdf(-self.d1_adaptive)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def zeta(self) -> float:
        """
        Return the Zeta of the option.

        Zeta is also known as the In-the-money probability and measures the risk-neutral probability of the option expiring
        in-the-money (ITM).

        Returns
        -----------
        float
            The Zeta of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._zeta_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _zeta_european(self) -> float:
        """Return the Zeta / In-the-money probability of a european call or put option."""
        if self.option_type == OptionType.Call:
            return norm.cdf(self.d2_adaptive)
        elif self.option_type == OptionType.Put:
            return norm.cdf(-self.d2_adaptive)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def strike_delta(self) -> float:
        """
        Return the Strike Delta of the option.

        Strike Delta is also known as the Discounted probability and measures discounted risk-neutral probability of the
        option ending up in-the-money (ITM). It is basically a discounted version of the option's Zeta.

        Returns
        -----------
        float
            The Strike Delta of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._strike_delta_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _strike_delta_european(self) -> float:
        """Return the Strike Delta / Discounted probability of a european call or put option."""
        if self.option_type == OptionType.Call:
            return - np.exp(- self.r * self.t) * norm.cdf(self.d2_adaptive)
        elif self.option_type == OptionType.Put:
            return np.exp(-self.r * self.t) * norm.cdf(-self.d2_adaptive)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def probability_mirror_strike(self) -> float:
        """
        Return the Probability Mirror Strike of the option.

        The Probability Mirror Strike is the strike, where a call and a put (Identical parameters except strike) have the
        same risk-neutral probability of expiring in-the-money (ITM).
        This method will return the strike of the other option, meaning if Option(...) is a Call, the returned value will be
        the strike of the Put, whereas if Option(...) is a Put, the return value will be the strike of the Call.

        Returns
        -----------
        float
            The strike of the risk-neutral mirror option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._probability_mirror_strike_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _probability_mirror_strike_european(self) -> float:
        """Return the Probability Mirror Strike of a european call or put option."""
        if self.option_type == OptionType.Call:
            return self.s ** 2 / self.k * np.exp((2 * self.b - self.sigma ** 2) * self.t)
        elif self.option_type == OptionType.Put:
            return self.s ** 2 / self.k * np.exp((2 * self.b - self.sigma ** 2) * self.t)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    def strike_probability(self, p: float) -> float:
        """
        Return the strike of an option with the given risk-neutral probability p of finishing in-the-money (ITM).

        Parameters
        -----------
        p : float
            The risk-neutral probability of the option expiring in-the-money (ITM).

        Returns
        -----------
        float
            The strike where the option expires in-the-money given p.

        Raises
        -----------
        ValueError
            Raised when the risk-neutral probability p is invalid.

        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """

        if p > 1 or p < 0:
            raise ValueError(f"The Risk-neutral probability p must be between 0 and 1. Input was: {p}")

        if self.exercise_style == OptionExerciseStyle.European:
            return self._probability_mirror_strike_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _strike_probability_european(self, p: float) -> float:
        """Return the Strike Probability of a european call or put option."""
        if self.option_type == OptionType.Call:
            return self.s * np.exp(-norm.ppf(p) * self.sigma * np.sqrt(self.t) + (self.b - self.sigma ** 2 / 2) * self.t)
        elif self.option_type == OptionType.Put:
            return self.s * np.exp(norm.ppf(p) * self.sigma * np.sqrt(self.t) + (self.b - self.sigma ** 2 / 2) * self.t)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def dzetadvol(self) -> float:
        """
        Return the DzetaDvol of an option.

        The DzetaDvol measures the Zeta's (ITM risk-neutral probability) sensitivity to changes in the implied volatiliy of the underlying.

        Returns
        -----------
        The DzetaDvol of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._dzetadvol_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _dzetadvol_european(self) -> float:
        """ Return the DzetaDvol of a european call or put option."""
        if self.option_type == OptionType.Call:
            return -norm.pdf(self.d2_adaptive) * (self.d1_adaptive / self.sigma)
        elif self.option_type == OptionType.Put:
            return norm.pdf(self.d2_adaptive) * (self.d1_adaptive / self.sigma)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def dzetadtime(self) -> float:
        """
        Return the DzetaDtime of the option.

        The DzetaDtime measures the Zeta's (ITM risk-neutral probability) sensitivity to changes in the time to expiry.

        Dividing the DzetaDtime by 365 or 252 results in the Daily DzetaDvol, depending if only trading days are counted.

        Returns
        -----------
        The DzetaDtime of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._dzetadtime_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _dzetadtime_european(self) -> float:
        """ Return the DzetaDtime of a european call or put option."""
        if self.option_type == OptionType.Call:
            return norm.pdf(self.d2_adaptive) * (self.b / (self.sigma * np.sqrt(self.t)) - self.d1_adaptive / 2 * self.t)
        elif self.option_type == OptionType.Put:
            return - norm.pdf(self.d2_adaptive) * (self.b / (self.sigma * np.sqrt(self.t)) - self.d1_adaptive / 2 * self.t)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def strike_gamma(self) -> float:
        """
        Return the Strike Gamma of the option.

        Strike Gamma is also known as the Risk-Neutral Probability Density or RND.
        The Strike Gamma is a key part of the Breeden-Litzenberger formula.

        Returns
        -----------
        The Strike Gamma of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._strike_gamma_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _strike_gamma_european(self):
        """ Return the Strike Gamma / Risk-Neutral Probability Density / RND of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return (norm.pdf(self.d2_adaptive) * np.exp(-self.r * self.t) ) / (self.k * self.sigma * np.sqrt(self.t))


    def strike_gamma_probability(self, p: float) -> float:
        """
        Return the Strike Gamma of the option, given a specific ITM risk-neutral probability.

        Strike Gamma is also known as the Risk-Neutral Probability Density or RND.
        The Strike Gamma is a key part of the Breeden-Litzenberger formula.

        Parameters
        -----------
        p : float
            The risk-neutral probability of the option expiring in-the-money (ITM).


        Returns
        -----------
        float
            The Strike Gamma of the option, based on the given value for the risk-neutral probability.

        Raises
        -----------
        ValueError
            Raised when the risk-neutral probability p is invalid.

        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """

        if p > 1 or p < 0:
            raise ValueError(f"The Risk-neutral probability p must be between 0 and 1. Input was: {p}")

        if self.exercise_style == OptionExerciseStyle.European:
            return self._strike_gamma_probability_european(p)
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _strike_gamma_probability_european(self, p: float) -> float:
        """ Return the Strike Gamma / Risk-Neutral Probability Density / RND of a european call or put option."""
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return (np.exp(-self.r * self.t) * norm.pdf(norm.ppf(p))) / (self.k * self.sigma * np.sqrt(self.t))


    @property
    def otm_to_itm_probability(self) -> float:
        """
        Return the risk-neutral probability of a Out-of-the-money (OTM) option expiring In-the-money (ITM).

        Returns
        -----------
        float
            The risk-neutral probability a OTM option expiring ITM.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._otm_to_itm_probability_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _otm_to_itm_probability_european(self) -> float:
        """ Return risk-neutral probability the option goes from OTM to ITM of a european call or put option."""
        z = np.log(self.k / self.s) / self.sigma * np.sqrt(self.t)
        mu = (self.b - self.sigma ** 2 / 2) / self.sigma ** 2
        _lambda = np.sqrt(mu ** 2 + 2 * self.r / self.sigma ** 2)
        if self.option_type == OptionType.Call:
            return (self.k / self.s) ** (mu + _lambda) * norm.cdf(-z) + (self.k / self.s) ** (mu - _lambda) * norm.cdf(-z + 2 * _lambda * self.sigma * np.sqrt(self.t))
        elif self.option_type == OptionType.Put:
            return - norm.pdf(self.d2_adaptive) * (self.b / (self.sigma * np.sqrt(self.t)) - self.d1_adaptive / 2 * self.t)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def epsilon(self) -> float:
        """
        Return the greek Epsilon.
        Epsilon represents dividend risk, e.g. the percentage change in option value per percentage change in the underlying's dividend yield.

        Returns
        -----------
        The greek .

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            return self._epsilon_european()
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")


    def _epsilon_european(self):
        if self.option_type == OptionType.Call:
            return  -self.s * self.t * np.exp((self.b - self.r) * self.t) * norm.cdf(self.d1_adaptive)
        elif self.option_type == OptionType.Put:
            return self.s * self.t * np.exp((self.b - self.r) * self.t) * norm.cdf(-self.d1_adaptive)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def vera(self) -> float:
        """
        Return the

        Returns
        -----------
        The greek .

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise style is not recognized or supported.
        """
        if self.exercise_style == OptionExerciseStyle.European:
            ...
        elif self.exercise_style == OptionExerciseStyle.American:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Bermuda:
            raise NotImplementedError()
        elif self.exercise_style == OptionExerciseStyle.Asian:
            raise NotImplementedError()
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_style}' is not valid.")






    @property
    def first_order_greeks(self) -> float:
        raise NotImplementedError()

    @property
    def second_order_greeks(self) -> float:
        raise NotImplementedError()

    @property
    def third_order_greeks(self) -> float:
        raise NotImplementedError()


    def d1(self, b: float) -> float:
        """
        Return the d1 parameter used in the Black-Scholes formula.

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
        Return the d2 parameter used in the Black-Scholes formula.

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


    @property
    def d1_adaptive(self) -> float:
        return self.d1(self.b)


    @property
    def d2_adaptive(self) -> float:
        return self.d2(self.b)


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

    @property
    def black_scholes_adaptive(self) -> float:
        """
        Return the theoretical value of a european option using the Black-Scholes formula.
        Assumes constant volatility, risk-free rate, and no dividends.
        This function automatically uses the cost of carry rate associated with the underlying.

        Future -> 0

        FX -> r - rf

        Equity, Index -> r - q (r if q is 0)

        Returns
        -----------
        float
            The theoretical option value
        """
        self._validate_model("black_scholes_adaptive")
        return self._cost_of_carry_black_scholes(self.b)

    @property
    def black_scholes(self) -> float:
        """
        Return the theoretical value of a european option using the Black-Scholes formula.
        Assumes constant volatility, risk-free rate, and no dividends.

        Returns
        -----------
        float
            The theoretical option value
        """
        self._validate_model("black_scholes")
        _b = self.r
        return self._cost_of_carry_black_scholes(_b)

    @property
    def black_scholes_merton(self) -> float:
        """
        Return the theoretical value of a european option using the Black-Scholes-Merton formula.
        Assumes constant volatility, risk-free rate and a continous dividend yield.
        Its main applications include the pricing of index options and dividend paying stocks.

        Returns
        -----------
        float
            The theoretical option value
        """
        self._validate_model("black_scholes_merton")
        _b = self.r - self.q
        return self._cost_of_carry_black_scholes(_b)

    @property
    def black_76(self) -> float:
        """
        Return the theoretical value of a european option using the Black formula (Sometimes known as the Black-76 Model).
        Assumes constant volatility, risk-free rate, and no dividends.
        Its main application includes the pricing of options on futures, bonds and swaptions, where the underlying has no cost-of-carry.

        Returns
        -----------
        float
            The theoretical option value
        """
        self._validate_model("black_76")
        _b = 0
        return self._cost_of_carry_black_scholes(_b)

    @property
    def garman_kohlhagen(self) -> float:
        """
        Return the theoretical value of a european option using the Garman-Kohlhagen formula, which differentiates itself by including two interest rates.
        Assumes constant volatility, domestic & foreign risk-free rates and no dividends.
        Its main application includes pricing FX Options.

        Raises
        -----------
        ValueError
            Raised when the foreign interest rate isn't defined

        Returns
        -----------
        float
            The theoretical option value
        """
        self._validate_model("garman_kohlhagen")
        if self.rf is not None:
            _b = self.r - self.rf
            return self._cost_of_carry_black_scholes(_b)
        else:
            raise NameError("The foreign interest rate (rf) must be defined.")

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
                * (BjerksundStenslandFormulas.std_bivariate_normal_cdf(-e1, -f1, rho)
                - (i2 / s) ** kappa * BjerksundStenslandFormulas.std_bivariate_normal_cdf(-e2, -f2, rho)
                - (i1 / s) ** kappa * BjerksundStenslandFormulas.std_bivariate_normal_cdf(-e3, -f3, -rho)
                + (i1 / i2) ** kappa * BjerksundStenslandFormulas.std_bivariate_normal_cdf(-e4, -f4, -rho)))

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
                return self.intrinsic_value
            else:
                alpha = (i - k) * i ** (-beta)

                return (alpha * s ** beta
                    - alpha * self._phi(b = b, gamma = beta, h = i, i = i, s = s, r = r, t = self.t)
                    + self._phi(b = b, gamma = 1, h = i, i = i, s = s, r = r, t = self.t)
                    - self._phi(b = b, gamma = 1, h = k, i = i, s = s, r = r, t = self.t)
                    - k * self._phi(b = b, gamma = 0, h = i, i = i, s = s, r = r, t = self.t)
                    + k * self._phi(b = b, gamma = 0, h = k, i = i, s = s, r = r, t = self.t))


    @property
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
        self._validate_model("bjerksund_stensland_1993")

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

    @property
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
        self._validate_model("bjerksund_stensland_2002")

        if self.option_type == OptionType.Call:
            s = self.s
            k = self.k
            r = self.r
            b = self.b
            return self._bjerksund_stensland_call_2002(s, k , r, b)

        elif self.option_type == OptionType.Put:
            # Bjerksund-Stendland put-call transformation
            s = self.k
            k = self.s
            r = self.r - self.b
            b = -self.b
            return self._bjerksund_stensland_call_2002(s, k , r, b)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")

    @property
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
        self._validate_model("bjerksund_stensland_combined")

        if self.option_type == OptionType.Call:
            s = self.s
            k = self.k
            r = self.r
            b = self.b
            flat_boundary = self._bjerksund_stensland_call_1993(s, k , r, b)
            two_step_boundary = self._bjerksund_stensland_call_2002(s, k , r, b)

            return 2 * two_step_boundary - flat_boundary

        elif self.option_type == OptionType.Put:
            # Bjerksund-Stendland put-call transformation
            s = self.k
            k = self.s
            r = self.r - self.b
            b = -self.b
            flat_boundary = self._bjerksund_stensland_call_1993(s, k , r, b)
            two_step_boundary = self._bjerksund_stensland_call_2002(s, k , r, b)

            return 2 * two_step_boundary - flat_boundary
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    def universal_binomial_tree(self, up_factor: float, down_factor: float, p: float, n: int) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, where inputs such as the up- and down-factors can be changed.
        This binomial tree serves as the template for other binomial pricing models like Cox-Ross-Rubenstein, Leisen-Reimer, Jarrow-Rudd and Rendleman-Bartter.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        up_factor : float
            Move up multiplier.

        down_factor : float
            Move down multiplier.

        p : float
            Move up probability, (Move down probability is 1 - p)

        Raises
        -----------
        ValueError
            Raised if the number of steps is not positive or if the risk-neutral probability is invalid.

        Returns
        -----------
        float
            The theoretical option value.

        Notes
        -----------
        Cox-Ross-Rubenstein:
            p = (np.exp(b * dt) - down_factor) / (up_factor - down_factor)
            up_factor = np.exp(self.sigma * np.sqrt(dt))
            down_factor = 1 / up_factor

        Rendleman-Bartter:

            p = (np.exp(b * dt) - down_factor) / (up_factor - down_factor)
            up_factor = np.exp((b - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt))
            down_factor = np.exp((b - 0.5 * self.sigma**2) * dt - self.sigma * np.sqrt(dt))

        Leisen-Reimer:

            p = self._h(self.d2, n)
            up_factor = np.exp(b * dt) * (h(self.d1) / h(self.d2))
            down_factor = ( (np.exp(b * dt) - p * up_factor) / (1 - p))

        Jarrow-Rudd:

            p = 0.5
            up_factor = np.exp(b - self.sigma ** 2 / 2 ) * dt + self.sigma * np.sqrt(dt)
            down_factor = np.exp( (b - self.sigma ** 2 / 2)* dt - self.sigma * np.sqrt(dt)
        """

        if n <= 0:
            raise ValueError("The number of steps must be positive")

        if p < 0 or p > 1:
            raise ValueError(f"Invalid risk-neutral probability: {p:.4f}. Check input parameters.")

        dt = self.t / n

        underlying_price = np.zeros(n + 1)
        for i in range(n + 1):
            underlying_price[i] = self.s * (up_factor ** (n - i)) * (down_factor ** i)


        option_values = self.intrinsic_value_variable(underlying_price)

        for step in range(n - 1, -1, -1): # backward induction
            for i in range(step + 1):

                # Calculate continuation value (expected discounted future value)
                continuation_value = cast(float, np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1]))

                if self.exercise_style == OptionExerciseStyle.European:
                    option_values[i] = continuation_value # European option, can only be exercised at maturity
                else:  # American option
                    current_price = self.s * (up_factor ** (step - i)) * (down_factor ** i)

                    intrinsic_value = self.intrinsic_value_variable(current_price)

                    option_values[i] = np.maximum(continuation_value, intrinsic_value)

        return cast(float, option_values[0])

    def binomial_cox_ross_rubinstein(self, n: int) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, based on the Cox-Ross-Rubinstein model.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        Returns
        -----------
        float
            The theoretical value of the option.

        Notes
        -----------
        The paper by John C. Cox, Stephen A. Ross and Mark Rubinstein (Options Pricing: A simplified approach, 1979) outlining the model:
                https://doi.org/10.1016/0304-405X(79)90015-1
                https://www.unisalento.it/documents/20152/615419/Option+Pricing+-+A+Simplified+Approach.pdf/b473132a-94d9-7615-3feb-5d458c0d0331?version=1.0&download=true
        """

        self._validate_model("binomial_cox_ross_rubinstein")


        up_factor = np.exp(self.sigma * np.sqrt(self.t / n))
        down_factor = 1 / up_factor
        p = (np.exp(self.b * (self.t / n)) - down_factor) / (up_factor - down_factor)

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

    def binomial_cox_ross_rubinstein_drift(self, n: int, drift: float) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, based on a
        modified version of the Cox-Ross-Rubinstein model, which accounts for drift.

        By changing the drift parameter, one can skew the tree into resulting in more nodes
        of the tree upwards or downwards.
        Setting drift = 0 results in the same values given by the regular CRR-Model.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        drift : float
            The drift

        Returns
        -----------
        float
            The theoretical value of the option.
        """

        self._validate_model("binomial_cox_ross_rubinstein_drift")


        up_factor = np.exp(drift * (self.t / n)  + self.sigma * np.sqrt(self.t / n))
        down_factor = np.exp(drift * (self.t / n)  - self.sigma * np.sqrt(self.t / n))
        p = (np.exp(self.b * (self.t / n)) - down_factor) / (up_factor - down_factor)

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

    def binomial_rendleman_bartter(self, n: int) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, based on the Rendleman-Bartter model.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        Returns
        -----------
        float
            The theoretical value of the option.

        Notes
        -----------
        The paper by Richard J. Rendleman, Jr and Brit J. Bartter (Two-State Option Pricing, 1979) outlining the model:
            https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1979.tb00058.x
            http://efinance.org.cn/cn/fm/19791201Two-State%20Option%20Pricing,%20pp.%201093-1110.pdf

        """
        self._validate_model("binomial_rendleman_bartter")

        up_factor = np.exp((self.b - 0.5 * self.sigma**2) * (self.t / n) + self.sigma * np.sqrt(self.t / n))
        down_factor = np.exp((self.b - 0.5 * self.sigma**2) * (self.t / n) - self.sigma * np.sqrt(self.t / n))
        p = (np.exp(self.b * (self.t / n)) - down_factor) / (up_factor - down_factor)

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

    def binomial_leisen_reimer(self, n: int) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, based on the Leisen-Reimer model.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        Returns
        -----------
        float
            The theoretical value of the option.

        Notes
        -----------
        The paper by Dietmar Leisen and Matthias Reimer (Binomial Models for Option Valuation-Examining and Improving Convergence, 1996) outlining the model:
            https://doi.org/10.1080/13504869600000015
            https://downloads.dxfeed.com/specifications/dxLibOptions/Leisen+Reimer+Binomial+tree.pdf
        """
        self._validate_model("binomial_leisen_reimer")

        p = BinomialFormulas.peizer_pratt_inversion_1(self.d2_adaptive, n)
        up_factor = np.exp(self.b * (self.t / n)) * BinomialFormulas.peizer_pratt_inversion_1(self.d1_adaptive, n) / BinomialFormulas.peizer_pratt_inversion_1(self.d2_adaptive, n)
        down_factor = (np.exp(self.b * (self.t / n)) - p * up_factor) / (1 - p)

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

    def binomial_jarrow_rudd(self, n: int) -> float:
        """
        Return the theoretical value of an option using a binomial tree based on the Jarrow-Rudd model.
        The Jarrow-Rudd model is also known as the equal-probability model, due to its value for p being 0.5.
        The Jarrow-Rudd binomial tree is, contrary to CRR or Leisen-Reimer, not risk-neutral.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        Returns
        -----------
        float
            The theoretical value of the option.

        Notes
        -----------
        The book by Robert Jarrow and Andrew Rudd (Option Pricing, 1983) outlining the model:
            https://books.google.com/books/about/Option_Pricing.html?id=bFrQAAAAIAAJ
            https://doi.org/10.1016/0378-4266(86)90028-2
            (Didn't find a direct source)

        """
        self._validate_model("binomial_jarrow_rudd")

        p = 0.5
        up_factor = np.exp((self.b - 0.5 * self.sigma ** 2) * (self.t / n) + self.sigma * np.sqrt((self.t / n)))
        down_factor = np.exp((self.b - 0.5 * self.sigma ** 2) * (self.t / n) - self.sigma * np.sqrt((self.t / n)))

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

    def binomial_jarrow_rudd_risk_neutral(self, n: int) -> float:
        """
        Return the theoretical value of an option using a binomial tree based on a modified version of the Jarrow-Rudd model.
        The Jarrow-Rudd model is also known as the equal-probability model, due to its value for p being 0.5.

        By changing the value of p to a risk neutral value, the model adjusted becomes risk neutral, contrary to its predecesor.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        Returns
        -----------
        float
            The theoretical value of the option.
        """
        self._validate_model("binomial_jarrow_rudd_risk_neutral")

        up_factor = np.exp((self.b - 0.5 * self.sigma ** 2) * (self.t / n) + self.sigma * np.sqrt((self.t / n)))
        down_factor = np.exp((self.b - 0.5 * self.sigma ** 2) * (self.t / n) - self.sigma * np.sqrt((self.t / n)))
        p = (np.exp(self.b * (self.t / n)) - down_factor) / (up_factor - down_factor)

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

    def binomial_tian(self, n: int) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree  based on the Tian (1993) model.
        Some evidence points to the model having a smoother convergence than other binomial trees.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        Returns
        -----------
        float
            The theoretical value of the option.

        Notes
        -----------
        The paper by Yisong Tian (A modified lattice approach to option pricing, 1993) outlining the model:
            https://onlinelibrary.wiley.com/doi/10.1002/fut.3990130509

        """
        self._validate_model("binomial_tian")

        nu = np.exp(self.sigma ** 2 * (self.t / n))
        up_factor =  0.5 * np.exp(self.b *(self.t / n)) * nu * (nu + 1 + np.sqrt(nu ** 2 + 2* nu - 3))
        down_factor =  0.5 * np.exp(self.b *(self.t / n)) * nu * (nu + 1 - np.sqrt(nu ** 2 + 2* nu - 3))
        p = (np.exp(self.b * (self.t / n)) - down_factor) / (up_factor - down_factor)

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

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
            return self.black_scholes_adaptive

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
                return self.intrinsic_value

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

            return self.intrinsic_value

    @property
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

        self._validate_model("barone_adesi_whaley")

        if self.option_type == OptionType.Call:
            return self._barone_adesi_whaley_call()

        elif self.option_type == OptionType.Put:
            return self._barone_adesi_whaley_put()
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def bachelier(self) -> float:
        """
        Return the theoretical value of a european option using the Bachelier model.

        Bachelier assumed a normal distribution for the underlying prices, which allowed the model to account for negative
        prices.

        Raises
        -----------
        InvalidOptionTypeException
            Raised when the option's exercise style is not recognized or supported.

        Returns
        -----------
        float
            The theoretical value of the option.
        """
        self._validate_model("bachelier")

        d1 = (self.s - self.k) / self.sigma * np.sqrt(self.t)

        if self.option_type == OptionType.Call:
            return (self.s - self.k) * norm.cdf(d1) + self.sigma * np.sqrt(self.t) * norm.pdf(d1)
        elif self.option_type == OptionType.Put:
            return (self.k - self.s) * norm.cdf(-d1) + self.sigma * np.sqrt(self.t) * norm.pdf(d1)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


    @property
    def bachelier_modified(self) -> float:
        """
        Return the theoretical value of a european option using the modified Bachelier model.
        The modified Bachelier model takes the time value of money in a risk-neutral world into account.

        Bachelier assumed a normal distribution for the underlying prices, which allowed the model to account for negative
        prices.

        Raises
        -----------
        InvalidOptionTypeException
            Raised when the option's exercise style is not recognized or supported.

        Returns
        -----------
        float
            The theoretical value of the option.
        """
        self._validate_model("bachelier")

        d1 = (self.s - self.k) / self.sigma * np.sqrt(self.t)

        if self.option_type == OptionType.Call:
            return self.s * norm.cdf(d1) - self.k * np.exp(-self.r * self.t) * norm.cdf(d1) + self.sigma * np.sqrt(self.t) * norm.pdf(d1)
        elif self.option_type == OptionType.Put:
            return self.k * np.exp(-self.r * self.t) * norm.cdf(-d1) - self.s * norm.cdf(-d1) + self.sigma * np.sqrt(self.t) * norm.pdf(d1)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")


class EuropeanCall(Option):
    def __init__(
        self,
        s: float,
        k: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
        premium: float | None = None,
        rf: float | None = None,
        transaction_costs: float | None = 0
    ) -> None:

        super().__init__(
            s=s,
            k=k,
            t=t,
            r=r,
            q=q,
            sigma=sigma,
            option_type="call",
            exercise_style="european",
            premium=premium,
            rf = rf,
            transaction_costs = transaction_costs
        )

class EuropeanPut(Option):
    def __init__(
        self, 
        s: float,
        k: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
        premium: float | None = None,
        rf: float | None = None,
        transaction_costs: float | None = 0
    ) -> None:
        
        super().__init__(
            s=s,
            k=k,
            t=t,
            r=r,
            q=q,
            sigma=sigma,
            option_type="put",
            exercise_style="european",
            premium=premium,
            rf = rf,
            transaction_costs = transaction_costs
        )

class AmericanCall(Option):
    def __init__(
        self, 
        s: float,
        k: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
        premium: float | None = None,
        rf: float | None = None,
        transaction_costs: float | None = 0
    ) -> None:
        
        super().__init__(
            s=s,
            k=k,
            t=t,
            r=r,
            q=q,
            sigma=sigma,
            option_type="call",
            exercise_style="american",
            premium=premium,
            rf = rf,
            transaction_costs = transaction_costs
        )

class AmericanPut(Option):
    def __init__(
        self,
        s: float,
        k: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
        premium: float | None = None,
        rf: float | None = None,
        transaction_costs: float | None = 0
    ) -> None:
        super().__init__(
            s=s,
            k=k,
            t=t,
            r=r,
            q=q,
            sigma=sigma,
            option_type="put",
            exercise_style="american",
            premium=premium,
            rf = rf,
            transaction_costs = transaction_costs
        )

class Strategy:
    def __init__(self, *options: Option) -> None:
        self.options = options
        raise NotImplementedError()



if __name__ == "__main__":

    new = Option(100,95,0.25,0.08,0,0.12, OptionType.Put, "european", b =0)

    print(new.strike_delta)
    print(new.zeta)
