from options_calculator.pricing.base import PricingBase
from options_calculator.utils.exceptions import MissingParameterException


class BlackScholesPricing(PricingBase):
    """
    Calculate the value of european-exercise style options using various implementations of the Black-Scholes model.
    """

    @PricingBase._european_only
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
        return self.black_scholes_cost_of_carry(self.b)

    @PricingBase._european_only
    def black_scholes(self) -> float:
        """
        Return the theoretical value of a european option using the Black-Scholes formula.
        Assumes constant volatility, risk-free rate, and no dividends.

        Returns
        -----------
        float
            The theoretical option value
        """
        _b = self.r
        return self.black_scholes_cost_of_carry(_b)

    @PricingBase._european_only
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
        _b = self.r - self.q
        return self.black_scholes_cost_of_carry(_b)

    @PricingBase._european_only
    def black_76(self) -> float:
        """
        Return the theoretical value of a european option using the Black formula (Sometimes known as the Black-76 Model).
        Assumes constant volatility, risk-free rate, and no dividends.
        Its main application includes the pricing of options on futures, bonds and swaptions, where the underlying has no
        cost-of-carry.

        Returns
        -----------
        float
            The theoretical option value
        """
        _b = 0
        return self.black_scholes_cost_of_carry(_b)

    @PricingBase._european_only
    def garman_kohlhagen(self) -> float:
        """
        Return the theoretical value of a european option using the Garman-Kohlhagen formula, which differentiates itself by
        including two interest rates. Assumes constant volatility, domestic & foreign risk-free rates and no dividends.
        Its main application includes pricing FX Options.

        Raises
        -----------
        MissingParameterException
            Raised when the foreign interest rate isn't defined

        Returns
        -----------
        float
            The theoretical option value
        """
        if self.rf is not None:
            _b = self.r - self.rf
            return self.black_scholes_cost_of_carry(_b)
        else:
            raise MissingParameterException("The foreign interest rate (rf) must be defined.")
