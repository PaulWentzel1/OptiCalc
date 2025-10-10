from opticalc.core.base import OptionBase
from opticalc.core.enums import Direction, OptionExerciseStyle, OptionType, Underlying
from opticalc.pricing.black_scholes_pricing import BlackScholesPricing


class Option(OptionBase):
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

    underlying_type: OptionUnderlying, str or None, default None.
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
        underlying_type: Underlying | str | None = None,
        direction: Direction | str | None = None,
        underlying_contracts: int | None = None,
        experimental: bool = False

    ) -> None:
        super().__init__(
        s = s,
        k = k,
        t = t,
        r = r,
        q = q,
        sigma = sigma,
        option_type = option_type,
        exercise_style = exercise_style,
        b = b,
        rf = rf,
        premium = premium,
        transaction_costs = transaction_costs,
        underlying_type = underlying_type,
        direction = direction,
        underlying_contracts = underlying_contracts,
        experimental = experimental
    )

    @property
    def black_scholes(self) -> float:
        engine = BlackScholesPricing(self)
        return engine.black_scholes_merton()



if __name__ == "__main__":
    new = Option(1,2,3,4,5,6,OptionType.Call,OptionExerciseStyle.American,underlying_type= Underlying.Equity)

from typing import Any

from opticalc.core.params import OptionParams


def cache_option_params(option: OptionParams) -> dict[str, Any]:
    """Helper to extract commonly used parameters for caching"""
    return {
        's': option.s,
        'k': option.k,
        't': option.t,
        'r': option.r,
        'q': option.q,
        'sigma': option.sigma,
        'option_type': option.option_type
    }

# class BlackScholesPricing:
#     """Black-Scholes pricing model with cached parameters for performance"""
#     def __init__(self, option: OptionParams):
#         self.option = option
#         # Cache common parameters in one line
#         self.__dict__.update(cache_option_params(option))

#     def __getattr__(self, name: str):
#         """Fallback for uncommon parameters"""
#         if hasattr(self.option, name):
#             return getattr(self.option, name)
#         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

#     @property
#     def black_scholes_merton(self):
#         # Fast access to cached parameters!
#         return self.s * self.k  # Invalid, just for testing


#     def delta(self):
#         return 0.5 * self.sigma  # Fast access


#     def gamma(self):
#         return self.k / self.s  # Fast access



# class ImpliedVolatility:
#     """Implied volatility calculator with cached parameters for performance"""
#     def __init__(self, option: OptionParams):
#         self.option = option
#         # Cache common parameters in one line
#         self.__dict__.update(cache_option_params(option))


#     def __getattr__(self, name: str):
#         """Fallback for uncommon parameters"""
#         if hasattr(self.option, name):
#             return getattr(self.option, name)
#         raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")


#     def implied_vol(self, market_price: float):
#         # Fast parameter access
#         base_vol = 0.20 if self.option_type == OptionType.Call else 0.25
#         return base_vol * (market_price / self.s)




# class EuropeanOption(OptionBase):
#     """A European-exercise style option with lazy-loaded pricing models"""
#     def __init__(
#         self,
#         s: float,
#         k: float,
#         t: float,
#         r: float,
#         q: float,
#         sigma: float,
#         option_type: OptionType | str,
#         b: float | None = None,
#         rf: float | None = None,
#         premium: float | None = None,
#         transaction_costs: float | None = None,
#         underlying_type: Underlying | str | None = None,
#         direction: Direction | str | None = None,
#     ) -> None:

#         super().__init__(
#             s=s,
#             k=k,
#             t=t,
#             r=r,
#             q=q,
#             sigma=sigma,
#             option_type=option_type,
#             exercise_style=OptionExerciseStyle.European,
#             b=b,
#             rf=rf,
#             premium=premium,
#             transaction_costs=transaction_costs,
#             underlying_type=underlying_type,
#             direction=direction,
#             experimental=False
#         )

#         # Lazy-loaded pricing models (created only when accessed)
#         self._bs_model = None
#         self._binomial_model = None
#         self._iv_model = None

#     @property
#     def bs(self) -> BlackScholesPricing:
#         """Lazy-loaded Black-Scholes model"""
#         if self._bs_model is None:
#             self._bs_model = BlackScholesPricing(self)
#         return self._bs_model

#     @property
#     def binomial(self) -> "BinomialPricing":
#         """Lazy-loaded Binomial model"""
#         if self._binomial_model is None:
#             self._binomial_model = BinomialPricing(self)
#         return self._binomial_model

#     @property
#     def iv(self) -> ImpliedVolatility:
#         """Lazy-loaded Implied Volatility model"""
#         if self._iv_model is None:
#             self._iv_model = ImpliedVolatility(self)
#         return self._iv_model

#     # Convenient direct access to commonly used methods
#     @property
#     def black_scholes_merton(self):
#         """Direct access to Black-Scholes price"""
#         return self.bs.black_scholes_merton

#     @property
#     def delta(self):
#         """Direct access to delta"""
#         return self.bs.delta()

#     def binomial_price(self, steps: int = 100):
#         """Direct access to binomial price"""
#         return self.binomial.binomial_price(steps)

#     def implied_vol(self, market_price: float):
#         """Direct access to implied volatility"""
#         return self.iv.implied_vol(market_price)


# class AmericanOption(OptionBase):
#     """American option with binomial and implied volatility models"""

#     def __init__(
#         self,
#         s: float,
#         k: float,
#         t: float,
#         r: float,
#         q: float,
#         sigma: float,
#         option_type: OptionType | str,
#         b: float | None = None,
#         rf: float | None = None,
#         premium: float | None = None,
#         transaction_costs: float | None = None,
#         underlying_type: Underlying | str | None = None,
#         direction: Direction | str | None = None,
#     ) -> None:
#         super().__init__(
#             s=s,
#             k=k,
#             t=t,
#             r=r,
#             q=q,
#             sigma=sigma,
#             option_type=option_type,
#             exercise_style=OptionExerciseStyle.American,
#             b=b,
#             rf=rf,
#             premium=premium,
#             transaction_costs=transaction_costs,
#             underlying_type=underlying_type,
#             direction=direction,
#             experimental=False
#         )

#         self._binomial_model = None
#         self._iv_model = None

#     @property
#     def binomial(self) -> BinomialPricing:
#         if self._binomial_model is None:
#             self._binomial_model = BinomialPricing(self)
#         return self._binomial_model

#     @property
#     def iv(self) -> ImpliedVolatility:
#         if self._iv_model is None:
#             self._iv_model = ImpliedVolatility(self)
#         return self._iv_model

#     def price(self, steps: int = 100):
#         """American options typically use binomial pricing"""
#         return self.binomial.binomial_price(steps)