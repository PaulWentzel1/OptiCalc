

__doc__ = """
OptiCalc is a Python library designed to provide the user with a high-level and intuitive API for option pricing,
greeks calculation, implied volatility calculation and more.

The core idea behind OptiCalc is to allow for complex calculations with a simple interface, filling the gap between simple
End-user developed applications (EUDAs) and more advanced libraries/modules such as QuantLib.
OptiCalc is not necessarily intended for speed or low-level complexity. For that, QuantLib is much more suitable.
"""
from opticalc.core.option import (
    Option
)

from opticalc.core.american_option import (
    AmericanOption
)

from opticalc.core.bermuda_option import (
    BermudaOption
)

from opticalc.core.european_option import (
    EuropeanOption
)

from opticalc.core.enums import (
    OptionType,
    ExerciseStyle,
    Underlying,
    Direction,
)

__all__ = [
    "Option",
    "EuropeanOption",
    "AmericanOption",
    "BermudaOption",
    "OptionType",
    "ExerciseStyle",
    "Underlying",
    "Direction"
]
