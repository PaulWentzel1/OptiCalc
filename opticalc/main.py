import numpy as np
from scipy.stats import norm, multivariate_normal
from math import comb
from enum import Enum

from numpy.typing import NDArray
from typing import Any, cast

from utils import InvalidOptionTypeException, UnsupportedModelException, InvalidOptionExerciseException


class OptionExerciseStyle(Enum):
    """Option Exercise Style"""
    European = "European"
    American = "American"
    Bermuda = "Bermuda"

class OptionType(Enum):
    """Option Type"""
    Call = "Call"
    Put = "Put"

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

    option_type : str
        The Option type. Valid: Call, Put.

    exercise_type : str,
        The exercise style of the option. Valid: European, American, Bermuda etc

    premium : float or None, default None
        The current price of the option. Used to derive implied volatilty.

    rf : float or None, default None
        The foreign interest rate. Used for FX Options, in which case r is the domestic interest rate.

    transaction_costs : float or None, default to None
        The transaction costs associated with trading the option.
    """

    def __init__(
        self, 
        s: float,
        k: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
        option_type: str, # consider replacing type with enum class
        exercise_type: str, # consider replacing type with enum class
        premium: float | None = None,
        rf: float | None = None,
        transaction_costs: float | None = None
    ) -> None:
        # Input validation
        self.s = s
        self.k = k
        self.t = t
        self.r = r
        self.q = q
        self.sigma = sigma
        self.option_type = option_type.lower()
        self.exercise_type = exercise_type.lower()
        self.premium = premium
        self.rf = rf
        self.transaction_costs = transaction_costs
    
    def __repr__(self) -> str:
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
            f"This is a {self.option_type} option with a {self.exercise_type}-style exercise.\n"
            f"Option-specific details:\n"
            f"Strike price: {self.k}\n"
            f"Time to expiry: {self.t}\n"
            f"Volatility of the underlying: {self.sigma}\n"
            f"Dividend yield of the underlying: {self.q}\n"
            f"Current spot price of the underlying: {self.s}\n"
            f"{'='*line_length}"
        )

    def _check_european(self, function_name: str) -> None:
        """
        Verifies whether or not an option has a european-style exercise.
        If no issue is found, the method will return None.

        Parameters
        -----------
        function_name : str
            The name of the method, for clarity.

        Raises
        -----------
        UnsupportedModelException
            If the option doesn't have a european-style exercise.

        Returns
        -----------
        None
            If the option has a european-style exercise.
        """
        if self.exercise_type != "european":
            raise UnsupportedModelException(
                f"{function_name} is only usable for European-style options. "
                f"The current option has a {self.exercise_type}-style exercise")

    def _check_american(self, function_name: str) -> None:
        """
        Verifies whether or not an option has a american-style exercise.
        If no issue is found, the method will return None.

        Parameters
        -----------
        function_name : str
            The name of the method, for clarity.

        Raises
        -----------
        UnsupportedModelException
            If the option doesn't have a american-style exercise.

        Returns
        -----------
        None
            If the option has a american-style exercise.
        """
        if self.exercise_type != "american":
            raise UnsupportedModelException(
                f"{function_name} is only usable for American-style options. "
                f"The current option has a {self.exercise_type}-style exercise")


    @property
    def available_models(self) -> list[str]:
        """
        Returns the available pricing models for a specific option.

        Returns
        -----------
        list[str]
            A list of all valid pricing models.
        """
        if self.exercise_type == "european":
            valid = ["black_scholes", "black_scholes_merton",
                     "black_76", "binomial_leisen_reimer",
                     "binomial_jarrow_rudd", "binomial_rendleman_bartter",
                    "binomial_cox_ross_rubinstein"]

            if self.rf is not None:
                valid.append("garman_kohlhagen")
            return valid

        elif self.exercise_type == "american":
            valid = ["bjerksund_stensland_1993","bjerksund_stensland_2002",
                     "bjerksund_stensland_combined", "binomial_leisen_reimer",
                     "binomial_jarrow_rudd", "binomial_rendleman_bartter",
                     "binomial_cox_ross_rubinstein"]

            return valid

        elif self.exercise_type == "bermuda":
            valid = [""]    # Placeholder
            return valid
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_type}' is not valid.")


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
            Raised if the intended model doesn't support the opt32ion's exercise.

        InvalidOptionExerciseException
            Raised if the option's exercise type isn't supported.
        """

        valid_european = ["black_scholes", "black_scholes_merton", "black_76", "binomial_leisen_reimer", "binomial_jarrow_rudd", "binomial_rendleman_bartter", "binomial_cox_ross_rubinstein"]
        if self.rf is not None:
            valid_european.append("garman_kohlhagen")
        valid_american = ["bjerksund_stensland_1993","bjerksund_stensland_2002", "bjerksund_stensland_combined", "binomial_leisen_reimer", "binomial_jarrow_rudd", "binomial_rendleman_bartter", "binomial_cox_ross_rubinstein"]
        #valid_bermuda = []

        if self.exercise_type == "european":
            if function_name not in valid_european:
                raise UnsupportedModelException(
                f"{function_name} is not usable for European-style options."
                f"The current option has a {self.exercise_type}-style exercise")

        elif self.exercise_type == "american":
            if function_name not in valid_american:
                raise UnsupportedModelException(
                f"{function_name} is not usable for American-style options."
                f"The current option has a {self.exercise_type}-style exercise")
        # elif self.exercise_type == "bermuda":
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_type}' is not valid.")

    @property
    def intrinsic_value(self) -> float:
        """
        Return the intrinsic value of the option, given the current underlying price.

        Returns
        -----------
        float
            The option's intrinsic value given its strike and the underlying's current price.
        """
        return cast(float, self.variable_intrinsic_value())


    def variable_intrinsic_value(self, underlying_price: float | NDArray[Any] | None = None) -> float | NDArray[Any]:
        """
        Return the intrinsic of the option, given a specific underlying price.

        Raises
        -----------
        InvalidOptionTypeException
            Raised when the the option type is something else than "call" and "put".

        Returns
        -----------
        float
            The option's intrinsic value, given its strike and the underlying's price.
        """

        if underlying_price is None:
            underlying_price = self.s

        if self.option_type == "call":
            return np.maximum(underlying_price - self.k, 0)
        elif self.option_type == "put":
            return np.maximum(self.k - underlying_price, 0)
        else:
            raise InvalidOptionTypeException(f"The Option type {self.option_type} is not valid")

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

        Returns:
            The option's moneyness.
        """
        if np.absolute(self.s - self.k) < 0.05:
            return "At the money."
        elif self.intrinsic_value > 0:
            return "In the money."
        else:
            return "Out of the money."





    @property
    def plot(self):
        ...

    @property
    def delta(self):
        ...

    @property
    def vega(self):
        ...

    @property
    def theta(self):
        ...
    
    @property
    def rho(self):
        ...
    
    @property
    def epsilon(self):
        ...
    
    @property
    def gamma(self):
        ...
    
    @property
    def vanna(self):
        ...
    
    @property
    def charm(self):
        ...

    @property
    def vomma(self):
        ...

    @property
    def vera(self):
        ...   
    
    @property
    def veta(self):
        ...
    
    @property
    def speed(self):
        ...

    @property
    def zomma(self):
        ...
    
    @property
    def color(self):
        ...

    @property
    def ultima(self):
        ...
    
    @property
    def first_order_greels(self):
        ...

    @property
    def second_order_greeks(self):
        ...

    @property
    def third_order_greeks(self):
        ...

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
            Raised when the the option type is something else than "call" and "put".
        """
        d1 = self.d1(b)
        d2 = self.d2(b)

        if self.option_type == "call":
            return self.s * norm.cdf(d1) * np.exp((b - self.r) * self.t) - self.k * np.exp(-self.r * self.t) * norm.cdf(d2)
        elif self.option_type == "put":     
            return self.k * np.exp(-self.r * self.t) * norm.cdf(-d2) - self.s * np.exp((b - self.r) * self.t) * norm.cdf(-d1)
        else:
            raise InvalidOptionTypeException("The Option type given is not valid.")

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
        b = self.r
        return self._cost_of_carry_black_scholes(b)
    
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
        b = self.r - self.q
        return self._cost_of_carry_black_scholes(b)
    
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
        b = 0
        return self._cost_of_carry_black_scholes(b)
    
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
            b = self.r - self.rf
            return self._cost_of_carry_black_scholes(b)
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

    @staticmethod
    def _std_bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
        """
        Return the values of the Cumulative Bivariate normal distribution.

        Computes P(x <= a, y <= b) where x & y follows a standardized bivariate 
        normal distribution with the correlation coefficient rho.

        
        Parameters
        -----------
        a : float
            Upper limit for first variable.

        b : float
            Upper limit for second variable.

        rho : float 
            The correlation between a and b.
    
        Returns
        -----------
        float
            The cumulative probability P(x <= a, y <= b).
        """


        mean: list[float] = [0, 0]
        cov: list[list[float]] = [[1, rho], [rho, 1]]

        return multivariate_normal.cdf([a, b], mean=mean, cov=cov,allow_singular=True)

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
                * (self._std_bivariate_normal_cdf(-e1, -f1, rho)
                - (i2 / s) ** kappa * self._std_bivariate_normal_cdf(-e2, -f2, rho)
                - (i1 / s) ** kappa * self._std_bivariate_normal_cdf(-e3, -f3, -rho)
                + (i1 / i2) ** kappa * self._std_bivariate_normal_cdf(-e4, -f4, -rho)))
   
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
            Raised when the the option type is something else than "call" and "put".
        
        Returns
        -----------
        float
            The theoretical option value
        """
        self._validate_model("bjerksund_stensland_1993")
        b = self.r - self.q
        if self.option_type == "call":
            s = self.s
            k = self.k
            r = self.r
            b = b
            return self._bjerksund_stensland_call_1993(s, k , r, b)
            
        elif self.option_type == "put":
            # Bjerksund-Stendland put-call transformation
            s = self.k
            k = self.s
            r = self.r - b
            b = -b
            return self._bjerksund_stensland_call_1993(s, k , r, b) 
        else:
            raise InvalidOptionTypeException("The Option type given is not valid.")

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
            Raised when the the option type is something else than "call" and "put".
        
        Returns
        -----------
        float
            The theoretical option value
        """
        self._validate_model("bjerksund_stensland_2002")
        b = self.r - self.q
        if self.option_type == "call":
            s = self.s
            k = self.k
            r = self.r
            b = b
            return self._bjerksund_stensland_call_2002(s, k , r, b)
            
        elif self.option_type == "put":
            # Bjerksund-Stendland put-call transformation
            s = self.k
            k = self.s
            r = self.r - b
            b = -b
            return self._bjerksund_stensland_call_2002(s, k , r, b) 
        else:
            raise InvalidOptionTypeException("The Option type given is not valid.")

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
            Raised when the the option type is something else than "call" and "put".

        Returns
        -----------
        float
            The theoretical value of the option.
        """
        self._validate_model("bjerksund_stensland_combined")
        b = self.r - self.q

        if self.option_type == "call":
            s = self.s
            k = self.k
            r = self.r
            b = b
            flat_boundary = self._bjerksund_stensland_call_1993(s, k , r, b)
            two_step_boundary = self._bjerksund_stensland_call_2002(s, k , r, b)

            return 2 * two_step_boundary - flat_boundary

        elif self.option_type == "put":
            # Bjerksund-Stendland put-call transformation
            s = self.k
            k = self.s
            r = self.r - b
            b = -b
            flat_boundary = self._bjerksund_stensland_call_1993(s, k , r, b)
            two_step_boundary = self._bjerksund_stensland_call_2002(s, k , r, b)

            return 2 * two_step_boundary - flat_boundary

        else:
            raise InvalidOptionTypeException("The Option type given is not valid.")

    @staticmethod
    def _h_1(x: float, n: int) -> float:
        """
        Return the Preizer-Pratt inversion method 1 on the variable x.
        Required for Leisen-Reimer binomial tree.
        """

        if x == 0: # Extra check since np.sign returns 0 if x is 0
            sign = 1
        else:
            sign = np.sign(x)

        return 0.5 + 0.5 * sign * np.sqrt(1 - np.exp(-((x / (n + 1/3))**2) * (n + 1/6)))

    @staticmethod
    def _h_2(x: float, n: int) -> float:
        """
        Return the Preizer-Pratt inversion method 2 on the variable x.
        Required for Leisen-Reimer binomial tree.
        """

        if x == 0: # Extra check since np.sign returns 0 if x is 0
            sign = 1
        else:
            sign = np.sign(x)

        return 0.5+sign*0.5*np.sqrt(1 - np.exp(-((x / (n + 1/3 + 0.1/(n + 1)))**2) * (n + 1/6)))




    def universal_binomial_tree(self, up_factor: float, down_factor: float, p: float, n: int, b: float | None = None) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, where inputs such as the up- and down-factors can be changed.
        This binomial tree serves as the template for other binomial pricing models like Cox-Ross-Rubenstein, Leisen-Reimer, Jarrow-Rudd and Rendleman-Bartter.


        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        up_factor : float
            Move up multiplier


        down_factor : float
            Move down multiplier

        p : float
            Move up probability, (Move down probability is 1 - p)

        b : float or None, default None
           The cost of carry rate. If None, defaults to r - q.

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

        if b is None:
            b = self.r - self.q

        if p < 0 or p > 1:
            raise ValueError(f"Invalid risk-neutral probability: {p:.4f}. Check input parameters.")

        dt = self.t / n

        underlying_price = np.zeros(n + 1)
        for i in range(n + 1):
            underlying_price[i] = self.s * (up_factor ** (n - i)) * (down_factor ** i)


        option_values = self.variable_intrinsic_value(underlying_price)

        for step in range(n - 1, -1, -1): # backward induction
            for i in range(step + 1):

                # Calculate continuation value (expected discounted future value)
                continuation_value = cast(float, np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1]))

                if self.exercise_type == 'european':
                    option_values[i] = continuation_value # European option, can only be exercised at maturity
                else:  # American option
                    current_price = self.s * (up_factor ** (step - i)) * (down_factor ** i)

                    intrinsic_value = self.variable_intrinsic_value(current_price)

                    option_values[i] = np.maximum(continuation_value, intrinsic_value)

        return cast(float, option_values[0])


    def binomial_cox_ross_rubinstein(self, n: int, b: float | None = None) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, based on the Cox-Ross-Rubinstein model.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        b : float or None, default None
           The cost of carry rate. If None, defaults to r - q.

        Returns
        -----------
        float
            The theoretical value of the option.
        """

        self._validate_model("binomial_cox_ross_rubinstein")

        if b is None:
            b = self.r - self.q

        up_factor = np.exp(self.sigma * np.sqrt(self.t / n))
        down_factor = 1 / up_factor
        p = (np.exp(b * (self.t / n)) - down_factor) / (up_factor - down_factor)

        return self.universal_binomial_tree(up_factor, down_factor, p, n, b)

    #Return the theoretical value of an option using a risk-neutral binomial tree, based on the Leisen-Reimer model.
    def binomial_rendleman_bartter(self, n: int, b: float | None = None) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, based on the Rendleman-Bartter model.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        b : float or None, default None
           The cost of carry rate. If None, defaults to r - q.

        Returns
        -----------
        float
            The theoretical value of the option.
        """

        self._validate_model("binomial_rendleman_bartter")

        if b is None:
            b = self.r - self.q

        up_factor = np.exp((b - 0.5 * self.sigma**2) * (self.t / n) + self.sigma * np.sqrt(self.t / n))
        down_factor = np.exp((b - 0.5 * self.sigma**2) * (self.t / n) - self.sigma * np.sqrt(self.t / n))
        p = (np.exp(b * (self.t / n)) - down_factor) / (up_factor - down_factor)

        return self.universal_binomial_tree(up_factor, down_factor, p, n, b)

#Return the theoretical value of an option using a risk-neutral binomial tree, based on the Leisen-Reimer model.
    def binomial_leisen_reimer(self, n: int, b: float | None = None) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, based on the Leisen-Reimer model.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        b : float or None, default None
           The cost of carry rate. If None, defaults to r - q.

        Returns
        -----------
        float
            The theoretical value of the option.
        """
        self._validate_model("binomial_leisen_reimer")

        if b is None:
            b = self.r - self.q

        p = self._h_1(self.d2(b), n)
        up_factor = np.exp(b * (self.t / n)) * self._h_1(self.d1(b), n) / self._h_1(self.d2(b), n)
        down_factor = (np.exp(b * (self.t / n)) - p * up_factor) / (1 - p)

        return self.universal_binomial_tree(up_factor, down_factor, p, n, b)


    def binomial_jarrow_rudd(self, n: int, b: float | None = None) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, based on the Jarrow-Rudd model.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        b : float or None, default None
           The cost of carry rate. If None, defaults to r - q.

        Returns
        -----------
        float
            The theoretical value of the option.
        """
        self._validate_model("binomial_jarrow_rudd")

        if b is None:
            b = self.r - self.q

        p = 0.5
        up_factor = np.exp((b - 0.5 * self.sigma ** 2) * (self.t / n) + self.sigma * np.sqrt((self.t / n)))
        down_factor = np.exp((b - 0.5 * self.sigma ** 2) * (self.t / n) - self.sigma * np.sqrt((self.t / n)))

        return self.universal_binomial_tree(up_factor, down_factor, p, n, b)

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
            exercise_type="european",
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
            exercise_type="european",
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
            exercise_type="american",
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
            exercise_type="american",
            premium=premium,
            rf = rf,
            transaction_costs = transaction_costs
        )

class Strategy:
    def __init__(self, *options: Option) -> None:
        self.options = options
        ...



if __name__ == "__main__":
    new = Option(100,75,12/350,np.exp(0.05/(12/350)) -1 ,0,0.324,"call", "european")
    new_call = Option(100,80,0.5,0.07,0,0.3,"call", "american")
    new_put = Option(100,80,0.5,0.07,0,0.3,"put", "american")
    b = 0.07


    print(f"{new_put.binomial_cox_ross_rubinstein(n = 25)}")

    print(f"{new_put.binomial_jarrow_rudd(n = 25)}")

    print(f"{new_put.binomial_leisen_reimer(n = 25)}")

    print(f"{new_put.binomial_rendleman_bartter(n = 25)}")

