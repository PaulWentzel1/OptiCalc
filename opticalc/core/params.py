from dataclasses import dataclass, field

from opticalc.core.enums import Direction, OptionExerciseStyle, OptionType, Underlying


@dataclass
class OptionParams:
    """Contains the parameters for a Option-type Object"""
    s: float
    k: float
    t: float
    r: float
    q: float
    sigma: float
    option_type: OptionType | str
    exercise_style: OptionExerciseStyle | str
    _override_b: float | None = field(repr= False) # | None = None
    rf: float | None = None
    premium: float | None = None
    transaction_costs: float | None = None
    underlying_type: Underlying | str | None = None
    direction: Direction | str | None = None
    underlying_contracts: int | None = None
    experimental: bool = False

    def __init__(
        self,
        s: float,
        k: float,
        t: float,
        r: float,
        q: float,
        sigma: float,
        option_type: OptionType | str,
        exercise_style: OptionExerciseStyle | str,
        b: float | None = None,
        rf: float | None = None,
        premium: float | None = None,
        transaction_costs: float | None = None,
        underlying_type: Underlying | str | None = None,
        direction: Direction | str | None = None,
        underlying_contracts: int | None = None,
        experimental: bool = False
    ) -> None:
        self.s = s
        self.k = k
        self.t = t
        self.r = r
        self.q = q
        self.sigma = sigma
        self.option_type = option_type
        self.exercise_style = exercise_style
        self._override_b = b
        self.rf = rf
        self.premium = premium
        self.transaction_costs = transaction_costs
        self.underlying_type = underlying_type
        self.direction = direction
        self.underlying_contracts = underlying_contracts
        self.experimental = experimental

    def get_b(self) -> float | None:
        return self._override_b

    @property
    def b(self) -> float:
        """
        Autocalculates the cost of carry rate (b) based on the option's underlying type or uses overrides provided by the user.
        """
        if self._override_b is not None:
            return self._override_b
        else:
            if self.underlying_type == Underlying.Future:
                return 0

            elif self.underlying_type == Underlying.FX:
                if self.rf is None:
                    raise NameError("The foreign interest rate (rf) must be defined.")
                return self.r - self.rf # fx options
            else:
                return self.r - self.q # equity, index, commodity (spot) options


    def modify_cost_of_carry(self, b: float | None = None) -> None:
        """
        Modify the cost of carry with a custom value. Useful for exotic options or plotting changes in cost of carry.
        """
        if b is not None:
            self._override_b = b
        else:
            self._override_b = None
