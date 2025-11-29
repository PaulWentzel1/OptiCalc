from options_calculator.core.enums import Model
from options_calculator.utils.exceptions import MissingParameterException


def convert_model(model: Model | str) -> Model:
    if isinstance(model, str):
        try:
            model = Model(model.lower())
        except ValueError as e:
            raise MissingParameterException(f"Invalid input '{model}'. Valid inputs for underlying_type"
                                            f" are: {[element.value for element in Model]}") from e
    return model


def get_method_class(obj: object, method_name: str) -> str | None:
    """
    Return the class in which a specific method was defined. Used to verify pricing models on a group/class-basis.

    Parameters
    -----------
    method_name : str
        Name of the method to verify.

    Returns:
        The class where the method is defined.
    """

    cls = obj if isinstance(obj, type) else obj.__class__

    for base_class in cls.__mro__:
        if method_name in base_class.__dict__:
            return base_class.__name__
    return None
