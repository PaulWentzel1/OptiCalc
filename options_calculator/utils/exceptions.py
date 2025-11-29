class InvalidOptionTypeException(Exception):
    """Invalid option type."""
    ...


class InvalidExerciseException(Exception):
    """Invalid opiton exercise."""
    ...


class UnsupportedModelException(Exception):
    """Pricing model is not supported."""
    ...


class InvalidUnderlyingException(Exception):
    """Invalid underlying type."""
    ...


class InvalidDirectionException(Exception):
    """Invalid direction."""
    ...


class MissingParameterException(ValueError):
    """Missing a needed parameter."""
    ...
