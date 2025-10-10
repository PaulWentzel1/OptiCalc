class InvalidOptionTypeException(Exception):
    """Invalid option type."""
    ...

class InvalidOptionExerciseException(Exception):
    """Invalid opiton exercise."""
    ...

class UnsupportedModelException(Exception):
    """Priicing model is not supported."""
    ...

class InvalidUnderlyingException(Exception):
    """Invalid underlying type."""
    ...

class InvalidDirectionException(Exception):
    """Invalid direction."""
    ...