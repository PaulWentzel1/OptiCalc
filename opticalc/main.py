import numpy as np
from scipy.stats import norm
from enum import Enum


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
    
    d1 : float or None, default None
        Currently little to no use as a parameter

    d2 : float or None, default None
        Currently little to no use as a parameter
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
        d1: float | None = None,
        d2: float | None = None
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
        self.d1_val = d1
        self.d2_val = d2


    @property
    def d1(self) -> float:
        return (np.log(self.s / self.k) + (self.r + (self.sigma ** 2 / 2 )) * self.t) / (self.sigma * np.sqrt(self.t))
    
    @property
    def d2(self) -> float:
        return self.d1 - self.sigma * np.sqrt(self.t)
    
    def __repr__(self) -> str:
        return """
            
        """

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
        d1: float | None = None,
        d2: float | None = None
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
            d1=d1,
            d2=d2
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
        d1: float | None = None,
        d2: float | None = None
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
            d1=d1,
            d2=d2
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
        d1: float | None = None,
        d2: float | None = None
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
            d1=d1,
            d2=d2
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
        d1: float | None = None,
        d2: float | None = None
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
            d1=d1,
            d2=d2
        )

if __name__ == "__main__":
    stock_price = 5
    strike_price = 10
    time_to_maturity = 3.5
    risk_free_rate = 0.05
    volatility = 0.20

    new_option = Option(stock_price, strike_price, time_to_maturity, risk_free_rate,0,volatility, "call", "european")

    european_call = EuropeanCall(stock_price, strike_price, time_to_maturity, risk_free_rate,0,volatility)

    print(new_option.d1)

    print(european_call.d1)

    print(new_option)