# 02. Pricing European Exercise Style Options

## Introduction to European Options

## Available pricing models
### Pricing with Black-Scholes
In its current implementation, OptiCalc includes five different variations of the Black-Scholes options pricing framework, with their main differences stemming from the handling of the cost-of-carry factor b.
These pricing formulas are only available for vanilla European-exercise style options.
```python
import options_calculator as op

# Initialize the option
european_option = op.EuropeanOption(s = 100
									


					)









```



### Pricing with Bachelier

### Pricing with Binomial trees


Black-Scholes	✅	❌	❌	Analytical	
Black-Scholes-Merton	✅	❌	❌	Analytical	
Black-76	✅	❌	❌	Analytical	
Garman-Kohlhagen	✅	❌	❌	Analytical	
Adaptive Black-Scholes	✅	❌	❌	Analytical	
Classic Bachelier	✅	❌	❌	Analytical	
Modified Bachelier
## Example