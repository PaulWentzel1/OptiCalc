from opticalc.core.enums import Model
from opticalc.utils.exceptions import MissingParameterException

def convert_model(model: Model | str) -> Model:
        if isinstance(model, str):
            try:
                model = Model(model.lower())
            except ValueError as e:
                raise MissingParameterException(f"Invalid input '{model}'. Valid inputs for underlying_type"
                                                        f" are: {[element.value for element in Model]}") from e
        return model