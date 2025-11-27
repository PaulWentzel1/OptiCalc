from typing import cast

import numpy as np
from scipy.stats import norm  # type: ignore

from opticalc.core.enums import OptionType, Underlying
from opticalc.pricing.base import PricingBase
from opticalc.utils.exceptions import InvalidOptionTypeException


class BlackScholesGreeks(PricingBase):
    """
    Calculate the greeks of european exercise style options.
    """

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return np.exp((self.b - self.r) * self.t) * norm.cdf(self.d1_cost_of_carry(self.b))
        elif self.option_type == OptionType.Put:
            return -np.exp((self.b - self.r) * self.t) * norm.cdf(- self.d1_cost_of_carry(self.b))
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")

    @property
    def delta_mirror(self) -> float:
        """
        Return the Delta Mirror strikes of the option.


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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
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
        """
        return self.k * np.exp((- self.b - self.sigma ** 2 / 2) * self.t)

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return self.s * np.exp(-norm.ppf(delta * np.exp((self.r - self.b) * self.t)) * self.sigma * np.sqrt(self.t) + (self.b + self.sigma ** 2 / 2) * self.t)
        elif self.option_type == OptionType.Put:
            return self.s * np.exp(norm.ppf(-delta * np.exp((self.r - self.b) * self.t)) * self.sigma * np.sqrt(self.t) + (self.b + self.sigma ** 2 / 2) * self.t)
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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        delta = delta * np.exp(-self.b * self.t)
        if self.option_type == OptionType.Call:
            return self.s * np.exp(-norm.ppf(delta * np.exp((self.r - self.b) * self.t)) * self.sigma * np.sqrt(self.t) + (self.b + self.sigma ** 2 / 2) * self.t)
        elif self.option_type == OptionType.Put:
            return self.s * np.exp(norm.ppf(-delta * np.exp((self.r - self.b) * self.t)) * self.sigma * np.sqrt(self.t) + (self.b + self.sigma ** 2 / 2) * self.t)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")

    @property
    def dual_delta(self) -> float:
        """
        Return the Dual Delta of the option.

        Dual Delta is the first derivative of the option's value with respect to the strike price.

        Returns
        -----------
        float
            The Dual Delta of the option.

        Raises
        -----------
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return np.exp(-self.r * self.t) * norm.cdf(self.d2_cost_of_carry(self.b))
        elif self.option_type == OptionType.Put:
            return np.exp(-self.r * self.t) * norm.cdf(self.d2_cost_of_carry(self.b))
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")

    @property
    def dual_gamma(self) -> float:
        """
        Return the Dual Gamma of the option.

        Dual Gamma describes the change in Dual Delta with respect to the option's strike price.

        Returns
        -----------
        float
            The Dual Gamma of the option.

        Raises
        -----------
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return np.exp(-self.r * self.t) * (norm.pdf(self.d2_cost_of_carry(self.b)) / (self.k * self.sigma * np.sqrt(self.t)))

    @property
    def dual_theta(self) -> float:
        """
        Return the Dual Theta of the option.

        Dual Theta describes the change in Theta with respect to time.

        Returns
        -----------
        float
            The Dual Gamma of the option.

        Raises
        -----------
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return - self.black_scholes_cost_of_carry(self.b)

    @property
    def alpha(self) -> float:
        """
        Return the Alpha the option.

        Alpha is also known as Gamma Rent and describes the Theta per Gamma ratio for an options position.

        A high Alpha means that the owner of the option's premium is not adequately compensated for its time decay.

        Returns
        -----------
        float
            The Alpha of the option.

        Raises
        -----------
        InvalidOptionExerciseException
            Raised when the option's exercise typoe is not recognized or supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return np.abs(self.theta / self.gamma)

    @property
    def vanna(self) -> float:
        """
        Return the greek Vanna of an Option.

        Vanna is also known as Ddeltavol or DvegaDspot and is a second-order
        partial derivative. Vanna measures how much Delta will change due
        to changes in the volatility and how much Vega will change due to
        changes in the underlying's spot price. In other words, Vanna represents
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
        InvalidOptionTypeException
            Raised when the option's type is not recognized or supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return ((-np.exp((self.b - self.r) * self.t) * self.d2_cost_of_carry(self.b)) / (self.sigma)) * norm.pdf(self.d1_cost_of_carry(self.b))

    @property
    def max_vanna(self) -> float:
        """
        Return the maximal value of the option's Vanna / DdeltaDvol / DvegaDspot.

        Returns
        -----------
        float
            The maximal value of the option's Vanna.
        """
        return self.k * np.exp(-self.b * self.t - self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2)

    @property
    def min_vanna(self) -> float:
        """
        Return the minimal value of the option's Vanna / DdeltaDvol / DvegaDspot.

        Returns
        -----------
        float
            The minimal value of the option's Vanna.
        """
        return self.k * np.exp(-self.b * self.t + self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2)

    @property
    def vanna_min_strike(self) -> float:
        """
        Return the strike price where the Vanna attains its minimum.

        Returns
        -----------
        float
            The strike price of the option where Vanna is the lowest.
        """
        return self.s * np.exp(self.b * self.t - self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2)

    @property
    def vanna_max_strike(self) -> float:
        """
        Return the strike price where the Vanna attains its maximum.

        Returns
        -----------
        float
            The strike price of the option where Vanna is the highest.
        """
        return self.s * np.exp(self.b * self.t + self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2)

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.vanna * (1 / self.sigma) * (self.d1_cost_of_carry(self.b) * self.d2_cost_of_carry(self.b) - (self.d1_cost_of_carry(self.b) / self.d2_cost_of_carry(self.b)) - 1)

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return -np.exp((self.b - self.r) * self.t) * (norm.pdf(self.d1_cost_of_carry(self.b)) * ((self.b) / (self.sigma * np.sqrt(self.t)) - (self.d2_cost_of_carry(self.b)) / (2 * self.t)) + (self.b - self.r) * norm.cdf(self.d1_cost_of_carry(self.b)))
        elif self.option_type == OptionType.Put:
            return -np.exp((self.b - self.r) * self.t) * (norm.pdf(self.d1_cost_of_carry(self.b)) * ((self.b) / (self.sigma * np.sqrt(self.t)) - (self.d2_cost_of_carry(self.b)) / (2 * self.t)) - (self.b - self.r) * norm.cdf(-self.d1_cost_of_carry(self.b)))
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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return np.exp((self.b - self.r) * self.t) * norm.cdf(self.d1_cost_of_carry(self.b)) * (self.s / self.black_scholes_cost_of_carry(self.b))
        elif self.option_type == OptionType.Put:
            return np.exp((self.b - self.r) * self.t) * (norm.cdf(self.d1_cost_of_carry(self.b)) - 1) * (self.s / self.black_scholes_cost_of_carry(self.b))
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
        """
        return self.sigma * np.abs(self.elasticity)

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return (norm.pdf(self.d1_cost_of_carry(self.b)) * np.exp((self.b - self.r) * self.t)) / (self.s * self.sigma * np.sqrt(self.t))

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return (self.s * (norm.pdf(self.d1_cost_of_carry(self.b)) * np.exp((self.b - self.r) * self.t)) / (self.s * self.sigma * np.sqrt(self.t))) / 100

    @property
    def gamma_percent_max_underlying(self) -> float:
        """
        Return the underlying price where the Gamma Percent attains its maximum.

        Returns
        -----------
        float
            The underlying price of the option where Gamma Percent is greatest.
        """
        return self.k * np.exp((- self.b - self.sigma ** 2 / 2) * self.t)

    @property
    def gamma_percent_max_strike(self) -> float:
        """
        Return the strike where the Gamma Percent attains its maximum.

        Returns
        -----------
        float
            The strike of the option where Gamma Percent is greatest.
        """
        return self.s * np.exp((self.b + self.sigma ** 2 / 2) * self.t)

    @property
    def gamma_max_underlying(self) -> float:
        """
        Return the underlying price where the Gamma attains its maximum.

        Returns
        -----------
        float
            The underlying price of the option where Gamma is greatest.
        """
        return self.k * np.exp((- self.b - 3 * self.sigma ** 2 / 2) * self.t)

    @property
    def gamma_max_strike(self) -> float:
        """
        Return the strike where the Gamma attains its maximum.

        Returns
        -----------
        float
            The strike of the option where Gamma is greatest.
        """
        return self.s * np.exp((self.b + self.sigma ** 2 / 2) * self.t)

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
            Invalid option parameters. r < sigma ** 2 + 2b must be true.
        """
        if not self.r < self.sigma ** 2 + 2 * self.b:
            raise ValueError("The returned value must be greater than 0 for the saddle point to exist. Therefore r < sigma ** 2 + 2b")

        return 1 / (2 * (self.sigma ** 2 + 2 * self.b - self.r))

    @property
    def gamma_saddle_underlying(self) -> float:
        """
        Return the underlying price where the Gamma attains its saddle point.

        Returns
        -----------
        float
           The underlying price of the option where Gamma has a saddle point.
        """
        return self.k * np.exp((-self.b - 3 * self.sigma ** 2 / 2) * self.gamma_saddle_time)

    @property
    def gamma_saddle(self) -> float:
        """
        Return the Gamma value found at the Option's Gamma saddle point.

        Returns
        -----------
        float
            The Gamma found at the saddle point.
        """
        return (np.sqrt(np.exp(1) / np.pi) * np.sqrt((2 * self.b - self.r) / self.sigma ** 2 + 1)) / self.k

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.gamma * ((self.d1_cost_of_carry(self.b) * self.d2_cost_of_carry(self.b) - 1) / self.sigma)

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
        """
        lower_boundary = cast(float, self.k * np.exp(- self.b * self.t - self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2))
        upper_boundary = cast(float, self.k * np.exp(- self.b * self.t + self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2))
        text = f"Zomma and Zomma Percent are negative for the underlying prices between {lower_boundary:.4f} and {upper_boundary:.4f}"
        text_alternative = f"Zomma and Zomma Percent are positive for the underlying prices outside of {lower_boundary:.4f} and {upper_boundary:.4f}"
        return lower_boundary, upper_boundary, text, text_alternative

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
        """
        lower_boundary = cast(float, self.s * np.exp(self.b * self.t) - self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2)
        upper_boundary = cast(float, self.s * np.exp(self.b * self.t) + self.sigma * np.sqrt(self.t) * np.sqrt(4 + self.t * self.sigma ** 2) / 2)
        text = f"Zomma and Zomma Percent are negative for the strike prices between {lower_boundary:.4f} and {upper_boundary:.4f}"
        text_alternative = f"Zomma and Zomma Percent are positive for the strike prices outside of {lower_boundary:.4f} and {upper_boundary:.4f}"
        return lower_boundary, upper_boundary, text, text_alternative

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.gamma_percent * ((self.d1_cost_of_carry(self.b) * self.d2_cost_of_carry(self.b) - 1) / self.sigma)

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return -((self.gamma * (1 + self.d1_cost_of_carry(self.b) / (self.sigma * np.sqrt(self.t)))) / self.s)

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return - self.gamma * (self.d1_cost_of_carry(self.b) / (100 * self.sigma * np.sqrt(self.t)))

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.gamma * (self.r - self.b + (self.b * self.d1_cost_of_carry(self.b) / self.sigma * np.sqrt(self.t)) + ((1 - self.d1_cost_of_carry(self.b) * self.d2_cost_of_carry(self.b)) / 2 * self.t))

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.gamma_percent * (self.r - self.b + (self.b * self.d1_cost_of_carry(self.b) / self.sigma * np.sqrt(self.t)) + ((1 - self.d1_cost_of_carry(self.b) * self.d2_cost_of_carry(self.b)) / 2 * self.t))

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.s * np.exp((self.b - self.r) * self.t) * norm.pdf(self.d1_cost_of_carry(self.b)) * np.sqrt(self.t)

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return (self.sigma / 10) * self.s * np.exp((self.b - self.r) * self.t) * norm.pdf(self.d1_cost_of_carry(self.b)) * np.sqrt(self.t)

    @property
    def vega_max_underlying_local(self) -> float:
        """
        Return the underlying price where Vega attains its local maximum.

        Returns
        -----------
        float
            The underlying price of the option where Vega is greatest.
        """
        return self.k * np.exp((- self.b + self.sigma ** 2 / 2) * self.t)

    @property
    def vega_max_strike_local(self) -> float:
        """
        Return the strike price where Vega attains its maximum local.

        Returns
        -----------
        float
            The strike price of the option where Vega is greatest.
        """
        return self.s * np.exp((self.b + self.sigma ** 2 / 2) * self.t)

    @property
    def vega_black_76_max_time(self) -> float:
        """
        Return the time to maturity where Vega attains its maximum for the Black-76 pricing model.

        Returns
        -----------
        float
            The time to maturity of the option where Vega is greatest.
        """
        return (2 * (1 + np.sqrt(1 + (8 * self.r * (1/self.sigma ** 2) + 1) * np.log(self.s / self.k) ** 2))) / (8 * self.r + self.sigma ** 2)

    @property
    def vega_max_time_global(self) -> float:
        """
        Return the time to maturity where Vega attains its global maximum.

        Returns
        -----------
        float
            The time to maturity of the option where Vega is greatest.
        """
        return 1 / (2 * self.r)

    @property
    def vega_max_underlying_global(self) -> float:
        """
        Return the underlying price where Vega attains its global maximum.

        Returns
        -----------
        float
            The underlying price of the option where Vega is greatest.
        """
        return self.k * np.exp((-self.b + self.sigma ** 2 / 2) / (2 * self.r))

    @property
    def vega_max(self) -> float:
        """
        Return the value Vega attains at its global maximum.

        Returns
        -----------
        float
            The value of Vega at its global maximum.
        """
        return self.k / (2 * np.sqrt(self.r * np.e * np.pi))

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
        """
        return self.vega * self.sigma / self.black_scholes_cost_of_carry(self.b)

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.vega * ((self.d1_cost_of_carry(self.b) * self.d2_cost_of_carry(self.b)) / self.sigma)

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
        """
        lower_boundary = self.k * np.exp((- self.b - self.sigma ** 2 / 2) * self.t)
        upper_boundary = self.k * np.exp((- self.b + self.sigma ** 2 / 2) * self.t)
        text = f"Vomma and Vomma Percent are negative for the underlying prices between {lower_boundary:.4f} and {upper_boundary:.4f}"
        text_alternative = f"Vomma and Vomma Percent are positive for the underlying prices outside of {lower_boundary:.4f} and {upper_boundary:.4f}"
        return lower_boundary, upper_boundary, text, text_alternative

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
        """

        lower_boundary = self.s * np.exp((self.b - self.sigma ** 2 / 2) * self.t)
        upper_boundary = self.s * np.exp((self.b + self.sigma ** 2 / 2) * self.t)
        text = f"Vomma and Vomma Percent are negative for the underlying prices between {lower_boundary:.4f} and {upper_boundary:.4f}"
        text_alternative = f"Vomma and Vomma Percent are positive for the underlying prices outside of {lower_boundary:.4f} and {upper_boundary:.4f}"
        return lower_boundary, upper_boundary, text, text_alternative

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.vega_percent * ((self.d1_cost_of_carry(self.b) * self.d2_cost_of_carry(self.b)) / self.sigma)

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.vomma * (1 / self.sigma) * (self.d1_cost_of_carry(self.b) * self.d2_cost_of_carry(self.b) - (self.d1_cost_of_carry(self.b) / self.d2_cost_of_carry(self.b)) - (self.d2_cost_of_carry(self.b) / self.d1_cost_of_carry(self.b)) - 1)

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return self.vega * (self.r - self.b + self.b * self.d1_cost_of_carry(self.b) / self.sigma * np.sqrt(self.t) - ((1 + self.d1_cost_of_carry(self.b) * self.d2_cost_of_carry(self.b)) / 2 * self.t))

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
        """
        return self.s * np.exp((self.b - self.r) * self.t) * norm.pdf(self.d1_cost_of_carry(self.b)) * (np.sqrt(self.t) / (2 * self.sigma))

    @property
    def ddeltadvar(self) -> float:
        """
        Return the DdeltaDvar of the option.

        DdeltaDvar is the change in Delta for a change in the variance (Variance Vanna).

        Returns
        -----------
        float
            The Variance Vega of the option.
        """
        return - self.s * np.exp((self.b - self.r) * self.t) * norm.pdf(self.d1_cost_of_carry(self.b)) * (self.d2_cost_of_carry(self.b) / (2 * self.sigma ** 2))

    @property
    def variance_vomma(self) -> float:
        """
        Return the Variance Vomma of the option.

        Variance Vomma describes the Variance Vega's sensitivity to a small change in the variance.

        Returns
        -----------
        float
            The Variance Vomma of the option.
        """
        return ((self.s * np.exp((self.b - self.r) * self.t) * np.sqrt(self.t)) / (4 * self.sigma ** 3)) * norm.pdf(self.d1_cost_of_carry(self.b)) * (self.d1_cost_of_carry(self.b) * self.d2_cost_of_carry(self.b) - 1)

    @property
    def variance_ultima(self) -> float:
        """
        Return the Variance Ultima of the option.

        The Variance Ultima is the Black-Scholes-Merton formulas third derivative with respect to variance.

        Returns
        -----------
        float
            The Variance Ultima of the option.
        """
        return ((self.s * np.exp((self.b - self.r) * self.t) * np.sqrt(self.t)) / (8 * self.sigma ** 5)) * norm.pdf(self.d1_cost_of_carry(self.b)) * ((self.d1_cost_of_carry(self.b) * self.d2_cost_of_carry(self.b) - 1) * (self.d1_cost_of_carry(self.b) * self.d2_cost_of_carry(self.b) - 3) - (self.d1_cost_of_carry(self.b) ** 2 + self.d2_cost_of_carry(self.b) ** 2))

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return -((self.s * np.exp((self.b - self.r) * self.t) * norm.pdf(self.d1_cost_of_carry(self.b)) * self.sigma) / (2 * np.sqrt(self.t))) - (self.b - self.r) * self.s * np.exp((self.b - self.r) * self.t) * norm.cdf(self.d1_cost_of_carry(self.b)) - self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2_cost_of_carry(self.b))
        elif self.option_type == OptionType.Put:
            return -((self.s * np.exp((self.b - self.r) * self.t) * norm.pdf(self.d1_cost_of_carry(self.b)) * self.sigma) / (2 * np.sqrt(self.t))) + (self.b - self.r) * self.s * np.exp((self.b - self.r) * self.t) * norm.cdf(-self.d1_cost_of_carry(self.b)) + self.r * self.k * np.exp(-self.r * self.t) * norm.cdf(-self.d2_cost_of_carry(self.b))
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
            Raised when the trading days are negative.
        """
        if trading_days < 0:
            raise ValueError(f"The option's trading days '{trading_days}' is not valid, it has to be positive.")
        return self.theta / trading_days

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return -(self.s * norm.pdf(self.d1_cost_of_carry(self.b)) * self.sigma) / (2 * np.sqrt(self.t))

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
        ValueError
            Raised if the number of trading days is invalid
        """
        if trading_days < 0:
            raise ValueError(f"The option's trading days '{trading_days}' is not valid, it has to be positive.")

        return self.driftless_theta / trading_days

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
        """

        if trading_days < 0:
            raise ValueError(f"The option's trading days '{trading_days}' is not valid, it has to be positive.")

        return (self.theta / trading_days) / self.vega

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.underlying_type and self.underlying_type == Underlying.Future:
            return -self.t * self.black_scholes_cost_of_carry(self.b)
        elif self.option_type == OptionType.Call:
            return self.t * self.k * np.exp(-self.r * self.t) * norm.cdf(self.d2_cost_of_carry(self.b))
        elif self.option_type == OptionType.Put:
            return -self.t * self.k * np.exp(-self.r * self.t) * norm.cdf(-self.d2_cost_of_carry(self.b))
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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return - self.t * self.s * np.exp((self.b - self.r) * self.t) * norm.cdf(self.d1_cost_of_carry(self.b))
        elif self.option_type == OptionType.Put:
            return self.t * self.s * np.exp((self.b - self.r) * self.t) * norm.cdf(-self.d1_cost_of_carry(self.b))
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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return self.t * self.s * np.exp((self.b - self.r) * self.t) * norm.cdf(self.d1_cost_of_carry(self.b))
        elif self.option_type == OptionType.Put:
            return -self.t * self.s * np.exp((self.b - self.r) * self.t) * norm.cdf(-self.d1_cost_of_carry(self.b))
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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return cast(float, norm.cdf(self.d2_cost_of_carry(self.b)))
        elif self.option_type == OptionType.Put:
            return cast(float, norm.cdf(-self.d2_cost_of_carry(self.b)))
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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return - np.exp(- self.r * self.t) * norm.cdf(self.d2_cost_of_carry(self.b))
        elif self.option_type == OptionType.Put:
            return np.exp(-self.r * self.t) * norm.cdf(-self.d2_cost_of_carry(self.b))
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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
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

        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """

        if p > 1 or p < 0:
            raise ValueError(f"The Risk-neutral probability p must be between 0 and 1. Input was: {p}")

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return cast(float, -norm.pdf(self.d2_cost_of_carry(self.b)) * (self.d1_cost_of_carry(self.b) / self.sigma))
        elif self.option_type == OptionType.Put:
            return cast(float, norm.pdf(self.d2_cost_of_carry(self.b)) * (self.d1_cost_of_carry(self.b) / self.sigma))
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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return norm.pdf(self.d2_cost_of_carry(self.b)) * (self.b / (self.sigma * np.sqrt(self.t)) - self.d1_cost_of_carry(self.b) / 2 * self.t)
        elif self.option_type == OptionType.Put:
            return - norm.pdf(self.d2_cost_of_carry(self.b)) * (self.b / (self.sigma * np.sqrt(self.t)) - self.d1_cost_of_carry(self.b) / 2 * self.t)
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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type not in (OptionType.Call, OptionType.Put):
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
        else:
            return (norm.pdf(self.d2_cost_of_carry(self.b)) * np.exp(-self.r * self.t)) / (self.k * self.sigma * np.sqrt(self.t))

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

        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """

        if p > 1 or p < 0:
            raise ValueError(f"The Risk-neutral probability p must be between 0 and 1. Input was: {p}")

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
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        z = np.log(self.k / self.s) / self.sigma * np.sqrt(self.t)
        mu = (self.b - self.sigma ** 2 / 2) / self.sigma ** 2
        _lambda = np.sqrt(mu ** 2 + 2 * self.r / self.sigma ** 2)
        if self.option_type == OptionType.Call:
            return (self.k / self.s) ** (mu + _lambda) * norm.cdf(-z) + (self.k / self.s) ** (mu - _lambda) * norm.cdf(-z + 2 * _lambda * self.sigma * np.sqrt(self.t))
        elif self.option_type == OptionType.Put:
            return - norm.pdf(self.d2_cost_of_carry(self.b)) * (self.b / (self.sigma * np.sqrt(self.t)) - self.d1_cost_of_carry(self.b) / 2 * self.t)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")

    @property
    def epsilon(self) -> float:
        """
        Return the greek Epsilon.
        Epsilon represents dividend risk, e.g. the percentage change in option value per percentage change in the underlying's dividend yield.

        Returns
        -----------
        The Epsilon of the option.

        Raises
        -----------
        InvalidOptionTypeException
            Raised if the option's type isn't supported.
        """
        if self.option_type == OptionType.Call:
            return -self.s * self.t * np.exp((self.b - self.r) * self.t) * norm.cdf(self.d1_cost_of_carry(self.b))
        elif self.option_type == OptionType.Put:
            return self.s * self.t * np.exp((self.b - self.r) * self.t) * norm.cdf(-self.d1_cost_of_carry(self.b))
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
