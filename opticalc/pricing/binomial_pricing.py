from typing import cast, Any

import numpy as np
from numpy.typing import NDArray

from opticalc.pricing.base import PricingBase
from opticalc.core.enums import OptionExerciseStyle


class BinomialPricing(PricingBase):
    @staticmethod
    def peizer_pratt_inversion_1(x: float, n: int) -> float:
        """
        Return the Preizer-Pratt inversion method 1 on the variable x.
        Required for Leisen-Reimer binomial tree.
        """
        if x == 0:  # Extra check since np.sign returns 0 if x is 0
            sign = 1
        else:
            sign = np.sign(x)

        return 0.5 + 0.5 * sign * np.sqrt(1 - np.exp(-((x / (n + 1/3)) ** 2) * (n + 1/6)))

    @staticmethod
    def peizer_pratt_inversion_2(x: float, n: int) -> float:
        """
        Return the Preizer-Pratt inversion method 2 on the variable x.
        Required for Leisen-Reimer binomial tree.
        """
        if x == 0:  # Extra check since np.sign returns 0 if x is 0
            sign = 1
        else:
            sign = np.sign(x)

        return 0.5 + sign * 0.5 * np.sqrt(1 - np.exp(-((x / (n + 1/3 + 0.1 / (n + 1))) ** 2) * (n + 1/6)))

    def universal_binomial_tree(self, up_factor: float, down_factor: float, p: float, n: int) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, where inputs such as the up- and
        down-factors can be changed. This binomial tree serves as the template for other binomial pricing models like
        Cox-Ross-Rubenstein, Cox-Ross-Rubenstein with drift, Rendleman-Bartter, Leisen-Reimer, Jarrow-Rudd,
        Jarrow-Rudd risk neutral and Tian.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        up_factor : float
            Move up multiplier.

        down_factor : float
            Move down multiplier.

        p : float
            Move up probability, (Move down probability is 1 - p)

        Raises
        -----------
        ValueError
            Raised if the number of steps is not positive or if the risk-neutral probability is invalid.

        Returns
        -----------
        float
            The theoretical option value.

        Notes
        -----------
        Cox-Ross-Rubenstein:
            p = (np.exp(b * dt) - down_factor) / (up_factor - down_factor)
            up_factor = np.exp(self.sigma * np.sqrt(dt))
            down_factor = 1 / up_factor

        Rendleman-Bartter:

            p = (np.exp(b * dt) - down_factor) / (up_factor - down_factor)
            up_factor = np.exp((b - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt))
            down_factor = np.exp((b - 0.5 * self.sigma**2) * dt - self.sigma * np.sqrt(dt))

        Leisen-Reimer:

            p = self._h(self.d2, n)
            up_factor = np.exp(b * dt) * (h(self.d1) / h(self.d2))
            down_factor = ( (np.exp(b * dt) - p * up_factor) / (1 - p))

        Jarrow-Rudd:

            p = 0.5
            up_factor = np.exp(b - self.sigma ** 2 / 2 ) * dt + self.sigma * np.sqrt(dt)
            down_factor = np.exp( (b - self.sigma ** 2 / 2)* dt - self.sigma * np.sqrt(dt)
        """

        if n <= 0:
            raise ValueError("The number of steps must be positive")

        if p < 0 or p > 1:
            raise ValueError(f"Invalid risk-neutral probability: {p:.4f}. Check input parameters.")

        dt = self.t / n

        underlying_price = np.zeros(n + 1)
        for i in range(n + 1):
            underlying_price[i] = self.s * (up_factor ** (n - i)) * (down_factor ** i)

        option_values = cast(NDArray[Any], self.vanilla_intrinsic_value_variable(underlying_price))

        for step in range(n - 1, -1, -1):  # backward induction
            for i in range(step + 1):

                # Calculate continuation value (expected discounted future value)
                continuation_value = cast(float, np.exp(-self.r * dt) * (p * option_values[i] + (1 - p) * option_values[i + 1]))

                if self.exercise_style == OptionExerciseStyle.European:
                    option_values[i] = continuation_value  # European option, can only be exercised at maturity
                else:  # American option
                    current_price = self.s * (up_factor ** (step - i)) * (down_factor ** i)

                    intrinsic_value = self.vanilla_intrinsic_value_variable(current_price)

                    option_values[i] = np.maximum(continuation_value, intrinsic_value)

        return cast(float, option_values[0])

    def binomial_cox_ross_rubinstein(self, n: int) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, based on the Cox-Ross-Rubinstein model.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        Returns
        -----------
        float
            The theoretical value of the option.

        Notes
        -----------
        The paper by John C. Cox, Stephen A. Ross and Mark Rubinstein (Options Pricing: A simplified approach, 1979) outlining the model:
                https://doi.org/10.1016/0304-405X(79)90015-1
                https://www.unisalento.it/documents/20152/615419/Option+Pricing+-+A+Simplified+Approach.pdf/b473132a-94d9-7615-3feb-5d458c0d0331?version=1.0&download=true
        """
        up_factor = np.exp(self.sigma * np.sqrt(self.t / n))
        down_factor = 1 / up_factor
        p = (np.exp(self.b * (self.t / n)) - down_factor) / (up_factor - down_factor)

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

    def binomial_cox_ross_rubinstein_drift(self, n: int, drift: float) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, based on a
        modified version of the Cox-Ross-Rubinstein model, which accounts for drift.

        By changing the drift parameter, one can skew the tree into resulting in more nodes
        of the tree upwards or downwards.
        Setting drift = 0 results in the same values given by the regular CRR-Model.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        drift : float
            The drift

        Returns
        -----------
        float
            The theoretical value of the option.
        """

        up_factor = np.exp(drift * (self.t / n) + self.sigma * np.sqrt(self.t / n))
        down_factor = np.exp(drift * (self.t / n) - self.sigma * np.sqrt(self.t / n))
        p = (np.exp(self.b * (self.t / n)) - down_factor) / (up_factor - down_factor)

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

    def binomial_rendleman_bartter(self, n: int) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, based on the Rendleman-Bartter model.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        Returns
        -----------
        float
            The theoretical value of the option.

        Notes
        -----------
        The paper by Richard J. Rendleman, Jr and Brit J. Bartter (Two-State Option Pricing, 1979) outlining the model:
            https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1979.tb00058.x
            http://efinance.org.cn/cn/fm/19791201Two-State%20Option%20Pricing,%20pp.%201093-1110.pdf

        """

        up_factor = np.exp((self.b - 0.5 * self.sigma**2) * (self.t / n) + self.sigma * np.sqrt(self.t / n))
        down_factor = np.exp((self.b - 0.5 * self.sigma**2) * (self.t / n) - self.sigma * np.sqrt(self.t / n))
        p = (np.exp(self.b * (self.t / n)) - down_factor) / (up_factor - down_factor)

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

    def binomial_leisen_reimer(self, n: int) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree, based on the Leisen-Reimer model.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        Returns
        -----------
        float
            The theoretical value of the option.

        Notes
        -----------
        The paper by Dietmar Leisen and Matthias Reimer (Binomial Models for Option Valuation-Examining and Improving Convergence, 1996) outlining the model:
            https://doi.org/10.1080/13504869600000015
            https://downloads.dxfeed.com/specifications/dxLibOptions/Leisen+Reimer+Binomial+tree.pdf
        """

        p = self.peizer_pratt_inversion_1(self.d2(self.b), n)
        up_factor = np.exp(self.b * (self.t / n)) * self.peizer_pratt_inversion_1(self.d1(self.b), n) / self.peizer_pratt_inversion_1(self.d2(self.b), n)
        down_factor = (np.exp(self.b * (self.t / n)) - p * up_factor) / (1 - p)

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

    def binomial_jarrow_rudd(self, n: int) -> float:
        """
        Return the theoretical value of an option using a binomial tree based on the Jarrow-Rudd model.
        The Jarrow-Rudd model is also known as the equal-probability model, due to its value for p being 0.5.
        The Jarrow-Rudd binomial tree is, contrary to CRR or Leisen-Reimer, not risk-neutral.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        Returns
        -----------
        float
            The theoretical value of the option.

        Notes
        -----------
        The book by Robert Jarrow and Andrew Rudd (Option Pricing, 1983) outlining the model:
            https://books.google.com/books/about/Option_Pricing.html?id=bFrQAAAAIAAJ
            https://doi.org/10.1016/0378-4266(86)90028-2
            (Didn't find a direct source)

        """

        p = 0.5
        up_factor = np.exp((self.b - 0.5 * self.sigma ** 2) * (self.t / n) + self.sigma * np.sqrt((self.t / n)))
        down_factor = np.exp((self.b - 0.5 * self.sigma ** 2) * (self.t / n) - self.sigma * np.sqrt((self.t / n)))

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

    def binomial_jarrow_rudd_risk_neutral(self, n: int) -> float:
        """
        Return the theoretical value of an option using a binomial tree based on a modified version of the Jarrow-Rudd model.
        The Jarrow-Rudd model is also known as the equal-probability model, due to its value for p being 0.5.

        By changing the value of p to a risk neutral value, the model adjusted becomes risk neutral, contrary to its predecesor.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        Returns
        -----------
        float
            The theoretical value of the option.
        """

        up_factor = np.exp((self.b - 0.5 * self.sigma ** 2) * (self.t / n) + self.sigma * np.sqrt((self.t / n)))
        down_factor = np.exp((self.b - 0.5 * self.sigma ** 2) * (self.t / n) - self.sigma * np.sqrt((self.t / n)))
        p = (np.exp(self.b * (self.t / n)) - down_factor) / (up_factor - down_factor)

        return self.universal_binomial_tree(up_factor, down_factor, p, n)

    def binomial_tian(self, n: int) -> float:
        """
        Return the theoretical value of an option using a risk-neutral binomial tree  based on the Tian (1993) model.
        Some evidence points to the model having a smoother convergence than other binomial trees.

        Parameters
        -----------
        n : int
            The amount of steps in the binomial tree.

        Returns
        -----------
        float
            The theoretical value of the option.

        Notes
        -----------
        The paper by Yisong Tian (A modified lattice approach to option pricing, 1993) outlining the model:
            https://onlinelibrary.wiley.com/doi/10.1002/fut.3990130509

        """
        nu = np.exp(self.sigma ** 2 * (self.t / n))
        up_factor = 0.5 * np.exp(self.b * (self.t / n)) * nu * (nu + 1 + np.sqrt(nu ** 2 + 2 * nu - 3))
        down_factor = 0.5 * np.exp(self.b * (self.t / n)) * nu * (nu + 1 - np.sqrt(nu ** 2 + 2 * nu - 3))
        p = (np.exp(self.b * (self.t / n)) - down_factor) / (up_factor - down_factor)

        return self.universal_binomial_tree(up_factor, down_factor, p, n)
