from dataclasses import dataclass, field

import numpy as np

from opticalc.core.enums import Direction, ExerciseStyle, OptionType, Underlying
from opticalc.utils.exceptions import MissingParameterException


@dataclass
class OptionParams:
    """
    OptionParams contains the parameters for a Option-type object and the cost-of-carry logic.
    This logic is used to determine b based on user input and underlying type.
    """
    s: float
    k: float
    _t: float = field(init=False, repr=False)
    _exercise_dates: list[float] | None = field(init=False, repr=False)
    r: float
    q: float
    sigma: float
    option_type: OptionType | str
    exercise_style: ExerciseStyle | str
    _override_b: float | None = field(repr=False)
    rf: float | None = None
    premium: float | None = None
    transaction_costs: float | None = None
    underlying_type: Underlying | str | None = None
    direction: Direction | str | None = None
    underlying_contracts: int | None = None

    def __init__(
        self,
        s: float,
        k: float,
        t: float | list[float],
        r: float,
        q: float,
        sigma: float,
        option_type: OptionType | str,
        exercise_style: ExerciseStyle | str,
        b: float | None = None,
        rf: float | None = None,
        premium: float | None = None,
        transaction_costs: float | None = None,
        underlying_type: Underlying | str | None = None,
        direction: Direction | str | None = None,
        underlying_contracts: int | None = None,
    ) -> None:
        self.s = s
        self.k = k
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

        if isinstance(t, (list, tuple)):
            sorted_dates = sorted(t)
            self._t = sorted_dates[-1]
            self._exercise_dates = sorted_dates
        else:

            self._t = t
            self._exercise_dates = None

    @property
    def t(self) -> float:
        """
        Returns the time to final maturity. This maintains compatibility with European/American pricers.
        """
        return self._t

    @t.setter
    def t(self, value: float | list[float]) -> None:
        """
        Sets the expiry/maturity/exercise dates of the option to a customized value.
        """
        exercises = np.asarray(value)

        if exercises.ndim == 0:
            # Handle Scalar case
            self._t = float(exercises)
            self._exercise_dates = None
        else:
            exercises.sort()
            self._exercise_dates = exercises  # type: ignore
            self._t = float(exercises[-1])

    @property
    def exercise_dates(self) -> list[float]:
        """
        Returns the list of exercise dates. If t was initialized as a scalar, returns a single-item list [t].
        """
        if self._exercise_dates is not None:
            return self._exercise_dates
        return [self._t]

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
                    raise MissingParameterException("The foreign interest rate (rf) must be defined.")
                return self.r - self.rf  # fx options
            else:
                return self.r - self.q  # equity, index, commodity (spot) options

    @b.setter
    def b(self, value: float | None) -> None:
        """
        Sets the cost-of-carry rate b to a customized value.
        """
        self._override_b = value

    @property
    def reset_b(self) -> None:
        """Reset the cost-of-carry rate b."""
        self._override_b = None
