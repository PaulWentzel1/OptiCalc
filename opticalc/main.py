import numpy as np
from scipy.stats import norm
from enum import Enum
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
        The current spot price of the underlying

    k : float
        The strike of the option

    t : float
        The time left until the option expires

    r : float
        The risk-free rate

    q : float
        A continuous dividend yield

    sigma : float
        The volatility of the underlying

    option_type : str
        The Option type. Valid: Call, Put,

    exercise_type : str,
        The exercise style of the option. Valid: European, American, Bermuda etc

    market_price : float or None, default None
        The current price of the option. Used to derive implied volatilty

    rf : float or None, default None
        The foreign interest rate. Used for FX Options, in which case r is the domestic interest rate 
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
        self.exercise_type = exercise_type.lower() # american/european, consider replacing with enum class or something 
        self.market_price = market_price
        self.rf = rf

    def _check_european(self, function_name: str) -> None:
        if self.exercise_type != "european":
            raise UnsupportedModelException(
                f"{function_name} is only usable for European-style options. "
                f"The current option has a {self.exercise_type}-style exercise")


    def d1(self, b: float) -> float:
        return (np.log(self.s / self.k) + (b + self.sigma ** 2 / 2) * self.t) / (self.sigma * np.sqrt(self.t))
    

    def d2(self, b: float) -> float:
        return self.d1(b) - self.sigma * np.sqrt(self.t)

    def _cost_of_carry_black_scholes(self, b: float) -> float:
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
        self._check_european("Black-Scholes")
        b = self.r
        return self._cost_of_carry_black_scholes(b)
    
    @property
    def black_scholes_merton(self) -> float:
        self._check_european("Black-Scholes-Merton")
        b = self.r - self.q
        return self._cost_of_carry_black_scholes(b)
    
    @property
    def black_76(self) -> float:
        self._check_european("Black 76")
        b = 0
        return self._cost_of_carry_black_scholes(b)
    
    @property
    def garman_kohlhagen(self) -> float:
        self._check_european("Garman-Kohlhagen")
        if self.rf is not None:
            b = self.r - self.rf
            return self._cost_of_carry_black_scholes(b)
        else:
            raise ValueError("The foreign interest rate (rf) must be defined.")



    def __repr__(self) -> str:
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
    {"-"*line_length}
    Black-Scholes price: {self.black_scholes}
    Black-Scholes-Merton price: {self.black_scholes_merton}
    Black-76 price: {self.black_76}
    Garman Kohlhagen: {self.garman_kohlhagen if self.rf is not None else "No value for rf is given, can't be calculated"}
    {"-"*line_length}"""

    @property
    def available_models(self) -> list[str]:
        if self.exercise_type == "european":
            valid = ["black_scholes", "black_scholes_merton", "black_76"]
            if self.rf is not None:
                valid.append("garman_kohlhagen")
            return valid
        elif self.exercise_type == "american":
            valid = [""]    # Placeholder
            return valid
    
        elif self.exercise_type == "bermuda":
            valid = [""]    # Placeholder
            return valid
        else:
            raise InvalidOptionExerciseException(f"The option's exercise type '{self.exercise_type}' is not valid.")

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

if __name__ == "__main__":
    stock_price = 300
    strike_price = 250
    time_to_maturity = 3
    risk_free_rate = 0.03
    volatility = 0.15

    new_option = Option(stock_price, strike_price, time_to_maturity, risk_free_rate,0, volatility, "call", "european")

    european_call = EuropeanCall(stock_price, strike_price, time_to_maturity, risk_free_rate,0,volatility)
    european_put = EuropeanPut(stock_price, strike_price, time_to_maturity, risk_free_rate,0,volatility)

    print(f"Available pricing models: {european_call.available_models}")

    print(european_put.black_scholes_merton)

    american_call = AmericanCall(stock_price, strike_price, time_to_maturity, risk_free_rate, 0, volatility)
