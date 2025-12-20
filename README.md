# OptiCalc: An intuitive and versatile options library
## Overview
```OptiCalc``` or ```options_calculator``` is a Python library designed to provide the user with a high-level and intuitive API for option pricing, greeks calculation, implied volatility calculation and more.

The core idea behind OptiCalc is to allow for complex calculations with a simple interface, filling the gap between simple End-user developed applications (EUDAs) and more advanced libraries/modules such as QuantLib. OptiCalc is not necessarily intended for speed or low-level complexity. For that, QuantLib is much more suitable.
## Table of Contents
- [Getting started with OptiCalc](#getting-started-with-opticalc)
- [Documentation](#documentation)
- [Features](#features)
- [License](#license)
- [Dependencies](#dependencies)
- [Roadmap](#roadmap)

## Getting started with OptiCalc
OptiCalc is available on PyPI and can be installed via
```shell
# Via PyPI
pip install options_calculator
```
```sh
# Via uv
uv pip install options_calculator
```
## Documentation
For detailed guides and explainations of OptiCalc's logic, please refer to the documentation in the [docs folder](./docs/):

- [01. Getting Started](./docs/01_getting_started.md): Installing the library and initializing your first Option objects.

## Features
### High-level Option Classes
Currently 4 option classes (```EuropeanOption```, ```AmericanOption```, ```BermudaOption``` and ```Option```) are available for use.

These option classes offer flexibility with their attributes, from simple, minimal inputs for quick calculations to more complex ones:
```py
import options_calculator as op

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
import options_calculator as op

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
import options_calculator as op

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
As of now, OptiCalc offers 5 different pricing classes, with 15 pricing methods available for european-style options, 12 for american-style options and 8 for bermuda-style options.

### Pricing Model Compatibility

| Model | European exercise | American exercise | Bermuda exercise | Implementation Type | Notes |
| :--- | :---: | :---: | :---: | :--- |:--- |
| **Black-Scholes** | ✅ | ❌ | ❌ | Analytical ||
| **Black-Scholes-Merton** | ✅ | ❌ | ❌ | Analytical ||
| **Black-76** | ✅ | ❌ | ❌ | Analytical ||
| **Garman-Kohlhagen** | ✅ | ❌ | ❌ | Analytical ||
| **Adaptive Black-Scholes** | ✅ | ❌ | ❌ | Analytical ||
| **Bachelier** | ✅ | ❌ | ❌ | Analytical ||
| ... | ... | ... | ... | ... | ... |
| **Cox-Ross-Rubinstein** | ✅ | ✅ | ✅ | Numerical (Tree) |
| **Bjerksund-Stensland** | ❌ | ✅ | ❌ | Analytical (Approx) |
| **Barone-Adesi & Whaley** | ❌ | ✅ | ❌ | Analytical (Approx) |

Todo remove
- **Black Scholes** (European options)
    - Black-Scholes
    - Black-Scholes-Merton
    - Black-76
    - Garman Kohlhagen
    - Adaptive Black-Scholes

- **Binomial Trees** (European, Bermuda and American options)
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
    - Modified Bachelier model ##
## License
OptiCalc is released under a **[MIT License](LICENSE)**

## Dependencies
- [SciPy](https://scipy.org/): Adds statistical and probability functions like distributions.
- [NumPy](https://www.numpy.org/): Adds support for vectorized operations and the functions to process these.
- [Matplotlib](https://matplotlib.org/): Allows for visualization of greeks and option payoffs.

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
- Implied volatility
- Testing
- Expanded Documentation
- Extend support for exotic options (Asian, Barrier, Basket...)
- Global settings class for constants etc.
- Expand Plotting (Payoff graphs, Greeks)
- OptionStrategy class (Allows for the combination of several options)
    - Aggregate greeks
- Fully vectorized option classes (Option chains)







