# OptiCalc: An intuitive and versatile options library.
## Overview
```OptiCalc``` is a Python library designed to provide the user with a high-level and intuitive API for option pricing, greeks calculation, implied volatility calculation and more.

The core idea behind OptiCalc is to allow for complex calculations with a simple interface, filling the gap between simple End-user developed applications (EUDAs) and more advanced libraries/modules such as QuantLib. OptiCalc is not necessarily intended for speed or low-level complexity. For that, QuantLib is much more suitable.
## Table of Contents
- [Getting started with OptiCalc](#getting-started-with-opticalc)
- [Features](#features)
- [License](#license)
- [Dependencies](#dependencies)
- [Roadmap](#roadmap)


<!--## Getting started with OptiCalc
OptiCalc is available on PyPI and can be installed via
```sh
pip install opticalc
```
-->
## Features
### High-level Option Classes
Currently 4 option classes (```EuropeanOption```, ```AmericanOption```, ```BermudaOption``` and ```Option```) are available for use.

These option classes offer flexibility with their attributes, from simple, minimal inputs for quick calculations to more complex ones:
```py
import opticalc as op

# Minimal inputs
new_european_option = op.EuropeanOption(s = 100,
                                        k = 110,
                                        t = 0.10,
                                        r = 0.04,
                                        q = 0.00,
                                        sigma = 0.20,
                                        option_type = op.OptionType.Call)

# All possible inputs used
new_american_option = op.Option(s = 200,
                                k = 150,
                                t = 0.25,
                                r = 0.035,
                                q = 0.05,
                                sigma = 0.15,
                                option_type = "put",
                                exercise_style = op.ExerciseStyle.American,
                                b = None,
                                rf = 0.02,
                                premium = 5.5,
                                transaction_costs = 0.233,
                                underlying_type = op.Underlying.FX,
                                direction = "long",
                                underlying_contracts = 100)
```
### Dynamic input
Assigning values to an option's attributes is also handled internally in a dynamic way, which allows for ease of use and personal preference when changing or assigning values.
```py
import opticalc as op

option = op.Option(...)

option.option_type = op.OptionType.Call
print(option.option_type)  # Output: op.OptionType.Call

option.option_type = "put"
print(option.option_type)  # Output: op.OptionType.Put

option.option_type = "Call"
print(option.option_type)  # Output: op.OptionType.Call

option.exercise_style = "european"
print(option.exercise_style)  # Output: op.ExerciseStyle.European

option.exercise_style = "BERMUDA"
print(option.exercise_style)  # Output: op.ExerciseStyle.Bermuda

option.underlying_type = op.Underlying.FX
print(option.underlying_type)  # Output: op.Underlying.FX

option.direction = "lOnG"
print(option.direction)  # Output: op.Direction.Long
```

### Extensive methods and formulas
Option-type objects like ```EuropeanOption```, ```AmericanOption``` and ```Option``` contain methods
#### EuropeanOption
| | | |
|---|---|---|
| alpha | at_the_forward | at_the_forward_strike |
| at_the_forward_underlying | b | bachelier |
| bachelier_modified | binomial_cox_ross_rubinstein | binomial_cox_ross_rubinstein_drift |
| binomial_jarrow_rudd | binomial_jarrow_rudd_risk_neutral | binomial_leisen_reimer |
| binomial_rendleman_bartter | binomial_tian | black_76 |
| black_scholes | black_scholes_adaptive | black_scholes_cost_of_carry |
| black_scholes_merton | bleed_offset_volatility | carry_rho |
| charm | colour | colour_percent |
| d1 | d1_cost_of_carry | d2 |
| d2_cost_of_carry | ddeltadvar | delta |
| delta_mirror | direction | driftless_theta |
| driftless_theta_daily | dual_delta | dual_gamma |
| dual_theta | dzetadtime | dzetadvol |
| elasticity | epsilon | exercise_style |
| extrinsic_value | gamma | gamma_max_strike |
| gamma_max_underlying | gamma_percent | gamma_percent_max_strike |
| gamma_percent_max_underlying | gamma_saddle | gamma_saddle_time |
| gamma_saddle_underlying | garman_kohlhagen | intrinsic_value |
| intrinsic_value_variable | max_vanna | method_class |
| min_vanna | modify_cost_of_carry | moneyness |
| option_type | otm_to_itm_probability | phi |
| premium | probability_mirror_strike | profit_at_expiry_variable |
| rho | short_term_option_volatility | sigma |
| speed | speed_percent | straddle_symmetric_underlying |
| strike_delta | strike_from_futures_delta | strike_from_spot_delta |
| strike_gamma | strike_gamma_probability | strike_probability |
| theta | theta_daily | transaction_costs |
| ultima | underlying_contracts | underlying_type |
| universal_binomial_tree | validate_pricing_model | vanna |
| vanna_max_strike | vanna_min_strike | variance_ultima |
| variance_vega | variance_vomma | vega |
| vega_black_76_max_time | vega_elasticity | vega_max |
| vega_max_strike_local | vega_max_time_global | vega_max_underlying_global |
| vega_max_underlying_local | vega_percent | vera |
| veta | vomma | vomma_percent |
| vomma_range_strike | vomma_range_underlying | yanna |
| zeta | zomma | zomma_percent |
| zomma_range_strikes | zomma_range_underlying | |

#### AmericanOption
| | | |
|---|---|---|
| at_the_forward | at_the_forward_strike | at_the_forward_underlying |
| b | binomial_cox_ross_rubinstein | binomial_cox_ross_rubinstein_drift |
| binomial_jarrow_rudd | binomial_jarrow_rudd_risk_neutral | binomial_leisen_reimer |
| binomial_rendleman_bartter | binomial_tian | bjerksund_stensland_1993 |
| bjerksund_stensland_2002 | bjerksund_stensland_call_1993 | bjerksund_stensland_call_2002 |
| bjerksund_stensland_combined | black_scholes_cost_of_carry | d1 |
| d1_cost_of_carry | d2 | d2_cost_of_carry |
| direction | exercise_style | extrinsic_value |
| intrinsic_value | intrinsic_value_variable | method_class |
| modify_cost_of_carry | moneyness | option_type |
| premium | profit_at_expiry_variable | transaction_costs |
| underlying_contracts | underlying_type | universal_binomial_tree |
| validate_pricing_model | | |

### The cost-of-carry logic
OptiCalc includes flexible handling of the option's cost of carry factor ```b```. The cost of carry can either be defined when initializing a Option-type object or left to default to ```None```. In that case, OptiCalc will try to infer ```b``` by looking at the option's underlying, where ```b``` will be set in the following manner:

    Future:         b = 0
    FX:             b = r - rf
    Stock Index:    b = r - q
    Bond:           b = r - q
    Commodity:      b = r - q
    Swap:           b = r - q
    Not defined:    b = r - q

```py
import opticalc as op

underlying_price = 80
strike = 65
time_to_expiry = 0.333
risk_free_rate = 0.04
dividends = 0.01
volatility = 0.20
option_type = op.OptionType.Put
# b not defined!

new_option = op.EuropeanOption(underlying_price, strike, time_to_expiry,
                               risk_free_rate, dividends, volatility, option_type)
```

Since ```b``` is not explicitly defined when initializing the object, currently ```b = None```. Given that, OptiCalc will autocalculate ```b```. Since the underlying type is not defined, OptiCalc defaults to ```b = r - q```.

```py
print(new_option.b)  # Output: 0.03
```
Should the user want to, one can also modify ```b``` by simply assigning a new value to ```b```. This will assign ```b``` a custom value which will means in future calls, no autocalculation based on the underlying will take place.
This new value can simply be removed by calling the property ```reset_b``` which will result in ```b``` simply falling back to appropriate value based on the underlying or ```r - q``` in the default case.

```py
new_option.b = 0.05
print(new_option.b)  # Output after modification: 0.05

new_option.reset_b
print(new_option.b) # Output: 0.03

new_option.underlying_type = op.Underlying.Future
new_option.reset_b
print(new_option.b) # Output: 0.00
```

### Pricers
As of now, OptiCalc offers 5 different pricing classes, with 15 pricing methods available for european-style options and 12 for american-style options.

- **Black Scholes** (European options)
    - Black-Scholes
    - Black-Scholes-Merton
    - Black-76
    - Garman Kohlhagen
    - Adaptive Black-Scholes

- **Binomial Trees** (European and American options)
    - Cox-Ross-Rubinstein
    - Cox-Ross-Rubinstein with drift
    - Rendleman-Bartter
    - Leisen-Reimer
    - Jarrow-Rudd
    - Jarrow-Rudd with risk neutrality
    - Tian
    - Universal binomial tree

- **Bjerksund-Stensland** (American options)
    - Bjerksund-Stensland 1993 model
    - Bjerksund-Stensland 2002 model
    - Combined Bjerksund-Stensland 1993 and 2002 model

- **Barone-Adesi and Whaley** (American options)
    - Barone-Adesi and Whaley model
- **Bachelier** (European options)
    - Classic Bachelier model
    - Modified Bachelier model

###

## License
OptiCalc is released under a **[MIT License](LICENSE)**

## Dependencies
- [SciPy](https://scipy.org/): Adds statistical and probability functions like distributions.
- [NumPy](https://www.numpy.org/): Adds support for vectorized operations and the functions to process these.

## Roadmap
As OptiCalc is still under development, many features are still work-in progress and will be implemented in a similar order following the outline below:
- New pricing modules
    - Longstaff-Schwartz 
    - Heston
    - Merton Jump Diffusion
    - Variance Gamma
    - Trinomial Trees
    - Monte Carlo
- Greeks support for american options
- Implied volatility solver
- Testing
- Proper documentation
- Bermuda exercise for options
- Extend support for exotic options (Asian, Barrier, Basket...)
- Global settings class for constants etc.
- Plotting (Payoff graphs, Greeks)
- OptionStrategy class (Allows for the combination of several options)
    - Aggregate greeks
- Fully vectorized operations (Allowing for multiple options to be processed at the same time)
- Option chains object








