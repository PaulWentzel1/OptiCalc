
from options_calculator.core.option import Option, EuropeanOption, AmericanOption, BermudaOption


class Strategy:
    def __init__(self, *options: Option | EuropeanOption | AmericanOption | BermudaOption) -> None:
        self.options = options
        raise NotImplementedError()
