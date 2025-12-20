# 02. Pricing Vanilla European Exercise Style Options

## Introduction to European Options
Options with a European exercise differentiate themselves from other exercises like American and Bermuda by the fact that the option can only be exercised at maturity.
This means that the option only possesses a single exercise date.

## Available pricing models


### Pricing with Black-Scholes
In its current implementation, OptiCalc includes five different variations of the Black-Scholes options pricing framework, with their main differences stemming from the handling of the cost-of-carry factor b.
These pricing formulas are only available for vanilla European-exercise style options.

```python
import options_calculator as op

# Initialize the option
european_option = op.EuropeanOption(s = 100,
                                    k = 110,
                                    t = 0.10,
                                    r = 0.04,
                                    q = 0.00,
                                    sigma = 0.20,
                                    option_type = op.OptionType.Call)


print(f"Price using Black-Scholes {}")
print(f"Price using Black-Scholes-Merton {}")
print(f"Price using Black-76 {}")

print(f"Price using Garman-Kohlhagen {}")
print(f"Price using Adaptive Black-Scholkes {}")







```



### Pricing with Bachelier

### Pricing with Binomial trees
