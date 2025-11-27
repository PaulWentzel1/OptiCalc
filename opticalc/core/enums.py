from enum import Enum


class OptionType(Enum):
    """Option type."""
    Call = "call"
    Put = "put"


class ExerciseStyle(Enum):
    """Option exercise style."""
    European = "european"
    American = "american"
    Bermuda = "bermuda"


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


class Moneyness(Enum):
    """Stage of moneyness."""
    OTM = "out of the money"
    ATM = "at the money"
    ATF = "at the forward"
    ITM = "in the money"


class PricingClass(Enum):
    BachelierPricing = "bachelier"
    BaroneAdesiWhaleyPricing = "barone_adesi_whaley"
    BinomialPricing = "binomial"
    BjerksundStenslandPricing = "bjerksund_stensland"
    BlackScholesPricing = "black_scholes"


class Model(Enum):
    Bachelier = "bachelier"
    BachelierModified = "bachelier_modified"

    BaroneAdesiWhaley = "barone_adesi_whaley"

    BlackScholesAdaptive = "black_scholes_adaptive"
    BlackScholes = "black_scholes"
    BlackScholesMertom = "black_scholes_merton"
    Black76 = "black_76"
    GarmanKohlhagen = "garman_kohlhagen"

    UniversalBinomialTree = "universal_binomial_tree"
    BinomialCoxRossRubinstein = "binomial_cox_ross_rubinstein"
    BinomialCoxRossRubinsteinDrift = "binomial_cox_ross_rubinstein_drift"
    BinomialRendlemanBartter = "binomial_rendleman_bartter"
    BinomialLeisenReimer = "binomial_leisen_reimer"
    BinomialJarrowRudd = "binomial_jarrow_rudd"
    BinomialJarrowRuddRiskNeutral = "binomial_jarrow_rudd_risk_neutral"
    BinomialTian = "binomial_tian"

    BjerksundStenslandCall1993 = "bjerksund_stensland_call_1993"
    BjerksundStensland1993 = "bjerksund_stensland_1993"
    BjerksundStenslandCall2002 = "bjerksund_stensland_call_2002"
    BjerksundStensland2002 = "bjerksund_stensland_2002"
    BjerksundStenslandCombined = "bjerksund_stensland_combined"


class BarrierTypes(Enum):
    UpAndOut = "up_and_out"
    DownAndOut = "down_and_out"
    UpAndIn = "up_and_in"
    DownAndIn = "down_and_in"


class AsianAverageTypes(Enum):
    ArithmeticAverage = "arithmetic_average"
    GeometricAverage = "geometric_average"
