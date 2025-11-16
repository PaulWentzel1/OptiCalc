from opticalc.core.vanilla_base import VanillaOptionBase
from opticalc.core.enums import Direction, ExerciseStyle, OptionType, Underlying

from opticalc.core.american_option import AmericanOption
from opticalc.core.european_option import EuropeanOption

from opticalc.utils.exceptions import InvalidExerciseException


class Option(VanillaOptionBase):
    """
    A generalized option. The class will change type into a EuropeanOption, AmericanOption etc. based on the input given for
    exercise_style
    The class represents a financial option (Gives the holder the right, but not the obligation, to buy/sell the specific
    underlying asset).

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

    exercise_style : ExerciseStyle or str
        The exercise style of the option. Valid: european, american, bermuda, asian,
        ExerciseStyle.European, ExerciseStyle.American etc.

    b : float or None, default None
        The cost of carry rate.

    rf : float or None, default None
        The foreign interest rate. Used for FX Options, in which case r is the domestic interest rate.

    premium : float or None, default None
        The current price of the option. Used to derive implied volatility and calculate P&L.

    transaction_costs : float or None, default to None
        The transaction costs associated with trading the option.

    underlying_type: OptionUnderlying, str or None, default None.
        The type of underlying asset the option tracks.

    direction: OptionDirection, str or None, default None.
        The direction of the option, if it is sold or bought.
    """
    def __new__(
            cls,
            s: float,
            k: float,
            t: float,
            r: float,
            q: float,
            sigma: float,
            option_type: OptionType | str,
            exercise_style: ExerciseStyle | str,
            b: float | None = None,
            rf: float | None = None,
            premium: float | None = None,
            transaction_costs: float | None = None,
            underlying_type: Underlying | str | None = None,
            direction: Direction | str | None = None,
            underlying_contracts: int | None = None,
            ) -> AmericanOption | EuropeanOption:
        """
        Return the option subclass (EuropeanOption, AmericanOption etc) based on exercise_style.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised if the option's exercise is invalid.

        Returns
        -----------
        AmericanOption or EuropeanOption
            The appropriate option subclass.
        """
        if isinstance(exercise_style, ExerciseStyle):
            style_str = exercise_style.value.lower()
        else:
            style_str = str(exercise_style).lower()

        if style_str == "american":
            return AmericanOption(
                s=s,
                k=k,
                t=t,
                r=r,
                q=q,
                sigma=sigma,
                option_type=option_type,
                b=b,
                rf=rf,
                premium=premium,
                transaction_costs=transaction_costs,
                underlying_type=underlying_type,
                direction=direction,
                underlying_contracts=underlying_contracts,
            )
        elif style_str == "european":
            return EuropeanOption(
                s=s,
                k=k,
                t=t,
                r=r,
                q=q,
                sigma=sigma,
                option_type=option_type,
                b=b,
                rf=rf,
                premium=premium,
                transaction_costs=transaction_costs,
                underlying_type=underlying_type,
                direction=direction,
                underlying_contracts=underlying_contracts,
            )
        else:
            raise InvalidExerciseException(f"Invalid input '{exercise_style}'. Valid inputs for exercise_style"
                                                 f" are: {[element.value for element in ExerciseStyle]}")

    @property
    def intrinsic_value(self) -> float: ...

    def intrinsic_value_variable(self, s: float | None = None, k: float | None = None) -> float: ...

    @property
    def extrinsic_value(self) -> float: ...

    def profit_at_expiry_variable(self, s: float | None = None,
                                  premium: float | None = None,
                                  transaction_costs: float | None = None) -> float: ...

    @property
    def moneyness(self) -> str: ...

    @property
    def at_the_forward(self) -> bool: ...

    @property
    def at_the_forward_underlying(self) -> float: ...

    @property
    def at_the_forward_strike(self) -> float: ...
