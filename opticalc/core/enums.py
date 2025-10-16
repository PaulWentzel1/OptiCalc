from enum import Enum


class OptionType(Enum):
    """Option type."""
    Call = "call"
    Put = "put"
    Straddle = "straddle"  # Maybe not as optiontypes but rather a subclass like EuropeanOption
    Butterfly = "butterfly"  # ^


class OptionExerciseStyle(Enum):
    """Option exercise style."""
    European = "european"
    American = "american"
    Bermuda = "bermuda"
    Asian = "asian"


class Direction(Enum):
    """Direction of the option."""
    Long = "long"
    Short = "short"


class Underlying(Enum):
    """Type of underlying."""
    Equity = "equity"
    Stock_index = "index"
    Future = "future"
    FX = "fx"
    Interest_rate = "interest_rate"  # Bonds
    Commodity = "commodity"  # Only for spot commodity
    Swap = "swap"
