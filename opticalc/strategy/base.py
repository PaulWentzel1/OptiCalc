
from opticalc.core.option import Option, EuropeanOption, AmericanOption


class Strategy:
    def __init__(self, *options: Option | EuropeanOption | AmericanOption | BermudaOption) -> None:
        self.options = options
        raise NotImplementedError()
