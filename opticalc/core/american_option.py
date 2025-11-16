from opticalc.core.vanilla_base import VanillaOptionBase
from opticalc.core.enums import Direction, ExerciseStyle, OptionType, Underlying

from opticalc.pricing.bjerksund_stensland_pricing import BjerksundStenslandPricing
from opticalc.pricing.binomial_pricing import BinomialPricing
from opticalc.pricing.barone_adesi_whaley_pricing import BaroneAdesiWhaleyPricing


class AmericanOption(VanillaOptionBase, BjerksundStenslandPricing, BinomialPricing, BaroneAdesiWhaleyPricing):
    """
    A American-exercise style option. American options can be exercised at any point until the end of their maturity,
    contrary to european or bermuda options.
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
    def __init__(
        self,
        s: float,
        k: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
        option_type: OptionType | str,
        b: float | None = None,
        rf: float | None = None,
        premium: float | None = None,
        transaction_costs: float | None = None,
        underlying_type: Underlying | str | None = None,
        direction: Direction | str | None = None,
        underlying_contracts: int | None = None,
    ) -> None:
        super().__init__(
            s=s,
            k=k,
            t=t,
            r=r,
            q=q,
            sigma=sigma,
            option_type=option_type,
            exercise_style=ExerciseStyle.American,
            b=b,
            rf=rf,
            premium=premium,
            transaction_costs=transaction_costs,
            underlying_type=underlying_type,
            direction=direction,
            underlying_contracts=underlying_contracts,
            )

    @property
    def intrinsic_value(self) -> float:
        return super().intrinsic_value

    def intrinsic_value_variable(self, s: float | None = None, k: float | None = None) -> float:
        return super().intrinsic_value_variable(s, k)

    @property
    def extrinsic_value(self) -> float:
        return super().extrinsic_value

    def profit_at_expiry_variable(self, s: float | None = None,
                                  premium: float | None = None,
                                  transaction_costs: float | None = None) -> float:
        return super().profit_at_expiry_variable(s, premium, transaction_costs)

    @property
    def moneyness(self) -> str:
        return super().moneyness

    @property
    def at_the_forward(self) -> bool:
        return super().at_the_forward

    @property
    def at_the_forward_underlying(self) -> float:
        return super().at_the_forward_underlying

    @property
    def at_the_forward_strike(self) -> float:
        return super().at_the_forward_strike
