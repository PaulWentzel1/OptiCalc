# 01. Getting started

## Installing OptiCalc
OptiCalc can be installed via PyPI
```bash
pip install options_calculator
```

## Creating a basic Option object
The option type object is the centerpiece of OptiCalc. In this section we will introduce the basics of creating a minimal setup for a standard european call option.

```python
import options_calculator as op

underlying_price = 100
strike_price = 110
time_to_expiry = 1/12
interest_rate = 0.03
dividends = 0.00
volatility = 0.20
option_type = "call"

#  Initializing a European Option
european_option = op.EuropeanOption(s = underlying_price,
                                    k = strike_price,
                                    t = time_to_expiry,
                                    r = interest_rate,
                                    q = 0.00,
                                    sigma = volatility,
                                    option_type = option_type)

print(f"Option price: {european_option.black_scholes():.4f}")
```
## Creating an advanced Option object
The next example showcases all possible assignable parameters to an option in OptiCalc.
These parameters allow for more complex features such as multiple exercises which are used by Bermuda options or reversing the implied volatility which requires the ```premium``` parameter.
```python
import options_calculator as op

underlying_price = 210
strike_price = 204
exercise_dates = [0.1, 0.2, 0.3, 0.4]
interest_rate = 0.03
dividends = 0.00
volatility = 0.20
option_type = "put"
cost_of_carry = 0.24
foreign_interest_rate = 0.04
option_premium = 2.15
transaction_costs = 0.01
underlying = "FX"
trade_direction = "long"
num_contracts = 100

bermuda_option = op.BermudaOption(s=underlying_price,
                                  k=strike_price,
                                  t=exercise_dates,
                                  r=interest_rate,
                                  q=dividends,
                                  sigma=volatility,
                                  option_type=option_type,
                                  b=cost_of_carry,
                                  rf=foreign_interest_rate,
                                  premium=option_premium,
                                  transaction_costs=transaction_costs,
                                  underlying_type=underlying,
                                  direction=trade_direction,
                                  underlying_contracts=num_contracts)

print(f"Bermuda Price (Binomial): {bermuda_option.binomial_cox_ross_rubinstein():.4f}")
```
This concludes this introduction to creating option type objects. These serve as the foundation for pricing, greeks, plotting and most other features found in OptiCalc.