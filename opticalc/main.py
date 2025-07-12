import numpy as np
from scipy.stats import norm, multivariate_normal
from enum import Enum
from typing import cast

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

    market_price : float or None, default None
        The current price of the option. Used to derive implied volatilty.

    rf : float or None, default None
        The foreign interest rate. Used for FX Options, in which case r is the domestic interest rate.
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
        market_price: float | None = None,
        rf: float | None = None

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
        self.market_price = market_price
        self.rf = rf
    
    def __repr__(self) -> str:
        """
        Return a string containing general information about the option.

        Returns
        -----------
        str
            Basic information about the option and its given parameters.
        """

        line_length = 63
        return f"""    {"-"*line_length}
    This is a {self.option_type} option with a {self.exercise_type}-style exercise.

    Option-specific details:
    {"-"*line_length}
    Strike price: {self.k}
    Time to expiry: {self.t}
    Volatility of the underlying: {self.sigma}
    Dividend yield of the underlying: {self.q}
    Current spot price of the underlying: {self.s}  
    {"-"*line_length}"""


    def _check_european(self, function_name: str) -> None:
        """
        Verifies whether or not an option has a european-style exercise.
        If no issue is found, the method will return None.
        
        Parameters
        -----------
        function_name : str
            The name of the method, for clarity.

        Returns
        -----------
        None
            If the option has a european-style exercise.

        Raises
        -----------
        UnsupportedModelException
            If the option doesn't have a european-style exercise.
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

        Returns
        -----------
        None
            If the option has a american-style exercise.

        Raises
        -----------
        UnsupportedModelException
            If the option doesn't have a american-style exercise.
        """
        if self.exercise_type != "american":
            raise UnsupportedModelException(
                f"{function_name} is only usable for American-style options. "
                f"The current option has a {self.exercise_type}-style exercise")

    @property
    def payoff(self) -> float:
        if self.option_type == "call":
            return max(self.s - self.k, 0)
        elif self.option_type == "put":
            return max(self.k - self.s, 0)

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
            valid = ["black_scholes", "black_scholes_merton", "black_76", "binomial"]
            if self.rf is not None:
                valid.append("garman_kohlhagen")
            return valid
        
        elif self.exercise_type == "american":
            valid = ["bjerksund_stensland_1993", "binomial"]
            return valid
    
        elif self.exercise_type == "bermuda":
            valid = [""]    # Placeholder
            return valid
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_type}' is not valid.")

    @property
    def plot(self):
        ...

    @property
    def binomial(self):
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
    #  ^ move down later

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
        Return the theoretical price of a european option using the Black-Scholes formula.
        Assumes constant volatility, risk-free rate, and no dividends.
        
        Returns
        -----------
        float
            The theoretical price of the option.
        """
        self._check_european("Black-Scholes")
        b = self.r
        return self._cost_of_carry_black_scholes(b)
    
    @property
    def black_scholes_merton(self) -> float:
        """
        Return the theoretical price of a european option using the Black-Scholes-Merton formula.
        Assumes constant volatility, risk-free rate and a continous dividend yield. 
        Its main applications include the pricing of index options and dividend paying stocks.

        Returns
        -----------
        float
            The theoretical price of the option.
        """
        self._check_european("Black-Scholes-Merton")
        b = self.r - self.q
        return self._cost_of_carry_black_scholes(b)
    
    @property   
    def black_76(self) -> float:
        """
        Return the theoretical price of a european option using the Black formula (Sometimes known as the Black-76 Model).
        Assumes constant volatility, risk-free rate, and no dividends.
        Its main application includes the pricing of options on futures, bonds and swaptions, where the underlying has no cost-of-carry.
        
        Returns
        -----------
        float
            The theoretical price of the option.
        """
        self._check_european("Black 76")
        b = 0
        return self._cost_of_carry_black_scholes(b)
    
    @property
    def garman_kohlhagen(self) -> float:
        """
        Return the theoretical price of a european option using the Garman-Kohlhagen formula, which differentiates itself by including two interest rates.
        Assumes constant volatility, domestic & foreign risk-free rates and no dividends.
        Its main application includes pricing FX Options.
        
        Returns
        -----------
        float
            The theoretical price of the option.
        """
        self._check_european("Garman-Kohlhagen")
        if self.rf is not None:
            b = self.r - self.rf
            return self._cost_of_carry_black_scholes(b)
        else:
            raise ValueError("The foreign interest rate (rf) must be defined.")


    def _phi(self, s: float, t: float, gamma: float, h: float, i: float, r: float, b: float) -> float:
        """
        Calculate the value of Phi function, an important component of the Bjerksund-Stensland model(s)

        Parameters
        -----------
        b : float
            The cost of carry rate, which is determined by the given pricing model.
        
        gamma : float
            ...
         (s, t, gamma, h, i, r, b, sigma)
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
    def std_bivariate_normal_cdf(a: float, b: float, rho: float) -> float:
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
                * (self.std_bivariate_normal_cdf(-e1, -f1, rho)
                - (i2 / s) ** kappa * self.std_bivariate_normal_cdf(-e2, -f2, rho)
                - (i1 / s) ** kappa * self.std_bivariate_normal_cdf(-e3, -f3, -rho)
                + (i1 / i2) ** kappa * self.std_bivariate_normal_cdf(-e4, -f4, -rho)))
   

    def _bjerksund_stensland_call_1993(self, s: float, k: float, r: float, b: float) -> float:
        """
        Return the theoretical price of an american call option using the Bjerksund-Stensland approximation model (1993).
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
            The theoretical price of the option.
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
                return self.payoff
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
        Return the theoretical price of an american option using the Bjerksund-Stensland approximation model (1993).
        Assumes constant volatility, risk-free rate and allows for a continous dividend yield. 
        While not as accurate as numerical methods like the Binomial pricing model, it is a faster altrnative.
        
        Raises
        -----------
        InvalidOptionTypeException
            Raised when the the option type is something else than "call" and "put".
        
        Returns
        -----------
        float
            The theoretical price of the option.
        """
        b = self.r - self.q
        self._check_american("bjerksund_stensland_1993")
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
        Return the theoretical price of an american call option using the Bjerksund-Stensland approximation model (2002).
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
            The theoretical price of the option.
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
        Return the theoretical price of an american option using the Bjerksund-Stensland approximation model (2002).
        Assumes constant volatility, risk-free rate and allows for a continous dividend yield. 
        While not as accurate as numerical methods like the Binomial pricing model, it is a faster altrnative.
        
        Raises
        -----------
        InvalidOptionTypeException
            Raised when the the option type is something else than "call" and "put".
        
        Returns
        -----------
        float
            The theoretical price of the option.
        """
        
        b = self.r - self.q
        self._check_american("bjerksund_stensland_1993")
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



class EuropeanCall(Option):
    def __init__(
        self, 
        s: float,
        k: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
        market_price: float | None = None,
        rf: float | None = None
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
            market_price=market_price,
            rf = rf
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
        market_price: float | None = None,
        rf: float | None = None
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
            market_price=market_price,
            rf = rf
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
        market_price: float | None = None,
        rf: float | None = None
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
            market_price=market_price,
            rf = rf
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
        market_price: float | None = None,
        rf: float | None = None
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
            market_price=market_price,
            rf = rf
        )

class Strategy:
    def __init__(self, *options: Option) -> None:
        self.options = options
        ...


if __name__ == "__main__":
    stock_price = 300
    strike_price = 280
    time_to_maturity = 0.25
    risk_free_rate = 0.03
    volatility = 0.2

    new_option = Option(stock_price, strike_price, time_to_maturity, risk_free_rate,0, volatility, "call", "american")
    european_call = EuropeanCall(stock_price, strike_price, time_to_maturity, risk_free_rate,0,volatility)
    american_put = AmericanPut(stock_price, strike_price, time_to_maturity, risk_free_rate,0,volatility)



    print(american_put.bjerksund_stensland_2002)




