import numpy as np
from scipy.stats import norm  # type: ignore

from opticalc.core.vanilla_base import VanillaOptionBase
from opticalc.core.enums import Direction, ExerciseStyle, OptionType, Underlying

from opticalc.greeks.black_scholes_greeks import BlackScholesGreeks

from opticalc.pricing.bachelier_pricing import BachelierPricing
from opticalc.pricing.binomial_pricing import BinomialPricing
from opticalc.pricing.black_scholes_pricing import BlackScholesPricing

from opticalc.utils.constants import CALL_PUT_PARITY_THRESHOLD


class EuropeanOption(VanillaOptionBase, BlackScholesPricing, BinomialPricing, BachelierPricing, BlackScholesGreeks):
    """
    A European-exercise style option. European options can only be exercised at the end of their maturity, contrary to
    american or bermuda options.
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

    underlying_type: Underlying, str or None, default None.
        The type of underlying asset the option tracks.

    direction: Direction, str or None, default None.
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
            exercise_style=ExerciseStyle.European,
            b=b,
            rf=rf,
            premium=premium,
            transaction_costs=transaction_costs,
            underlying_type=underlying_type,
            direction=direction,
            underlying_contracts=underlying_contracts,
            )

    @property
    def call_put_parity(self) -> float:
        """
        Return value of the opposite option (Inverse option type) needed for the Call-Put parity to be true.
        Current option: Call -> Put value
        Current option: Put  -> Call value

        Returns
        -----------
        float
            The value of the identical option with opposite option type.
        """
        if self.option_type == OptionType.Call:
            call_value = self.black_scholes_adaptive()
            put_value = (self.k * np.exp(-self.r * self.t) * norm.cdf(-self.d2) - self.s
                         * np.exp((self.b - self.r) * self.t) * norm.cdf(-self.d1))
            parity_value = put_value + self.s * np.exp((self.b - self.r) * self.t) - self.k * np.exp(-self.r * self.t)

            if round(call_value, CALL_PUT_PARITY_THRESHOLD) == round(parity_value, CALL_PUT_PARITY_THRESHOLD):
                return put_value
            else:
                raise ValueError("The Call-Put-Parity is not valid. Check parameters for invalid inputs.")

        else:
            call_value = (self.s * norm.cdf(self.d1) * np.exp((self.b - self.r) * self.t)
                          - self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2))
            put_value = self.black_scholes_adaptive()
            parity_value = call_value - self.s * np.exp((self.b - self.r) * self.t) + self.k * np.exp(-self.r * self.t)

            if round(put_value, CALL_PUT_PARITY_THRESHOLD) == round(parity_value, CALL_PUT_PARITY_THRESHOLD):
                return call_value
            else:
                raise ValueError("The Call-Put-Parity is not valid. Check parameters for invalid inputs.")
