import numpy as np
from scipy.stats import norm  # type: ignore

from opticalc.core.enums import OptionType
from opticalc.pricing.base import PricingBase
from opticalc.utils.exceptions import InvalidOptionTypeException


class BachelierPricing(PricingBase):
    """
    Calculate the value of european-exercise style options using the Bachelier model .
    """

    @PricingBase.european_only
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
        d1 = (self.s - self.k) / self.sigma * np.sqrt(self.t)

        if self.option_type == OptionType.Call:
            return (self.s - self.k) * norm.cdf(d1) + self.sigma * np.sqrt(self.t) * norm.pdf(d1)
        elif self.option_type == OptionType.Put:
            return (self.k - self.s) * norm.cdf(-d1) + self.sigma * np.sqrt(self.t) * norm.pdf(d1)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")

    @PricingBase.european_only
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

        d1 = (self.s - self.k) / self.sigma * np.sqrt(self.t)

        if self.option_type == OptionType.Call:
            return (self.s * norm.cdf(d1) - self.k * np.exp(-self.r * self.t)
                    * norm.cdf(d1) + self.sigma * np.sqrt(self.t) * norm.pdf(d1))

        elif self.option_type == OptionType.Put:
            return (self.k * np.exp(-self.r * self.t) * norm.cdf(-d1) - self.s
                    * norm.cdf(-d1) + self.sigma * np.sqrt(self.t) * norm.pdf(d1))

        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid.")
