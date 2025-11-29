from scipy.optimize import newton # type: ignore

from options_calculator.pricing.black_scholes_pricing import BlackScholesPricing


class ImpliedVolatility(BlackScholesPricing):

    def newton_raphson(self) -> float:
        ...
        # if not self.premium:
        #     raise ValueError("Premium must be defined!")

        # root = float(self.black_scholes_cost_of_carry(self.b) - self.premium) ** 2

        # newton(root, )

if __name__ == "__main__":
    ...
