from typing import Optional

import attr
from paragraph2actions.actions import Action, Chemical

_ALLOWED_POST_TREATMENTS = {
    "annealing",
    "calcination",
    "carbonization",
    "heat",
    "oxidation",
    "pyrolysis",
    "reduction",
}
_ALLOWED_SYNTHESIS_METHODS = {
    "ald",
    "impregnation",
    "precipitation",
    "pyrolysis",
}


@attr.s(auto_attribs=True)
class ALD(Action):
    metal: str
    precursor: str
    support: str
    deposition_temperature: Optional[str] = None
    flux_pressure: Optional[str] = None
    flux_temperature: Optional[str] = None


@attr.s(auto_attribs=True)
class Centrifugate(Action):
    """
    Centrifugate action.

    Note: we store the rpm as a string to avoid numerical changes when converting.
    But we check that the given value is a number.
    """

    rpm: Optional[str] = None
    duration: Optional[str] = None
    temperature: Optional[str] = None

    def __attrs_post_init__(self):
        # Validate the given RPM value
        if self.rpm is not None:
            try:
                _ = float(self.rpm)
            except ValueError as e:
                raise ValueError(
                    f'The rpm value, "{self.rpm}", must be a number.'
                ) from e


@attr.s(auto_attribs=True)
class Concentrate(Action):
    duration: Optional[str] = None
    temperature: Optional[str] = None


@attr.s(auto_attribs=True)
class Copolymerize(Action):
    material: Optional[Chemical] = None
    precursor: Optional[Chemical] = None
    gas: Optional[str] = None
    duration: Optional[str] = None
    temperature: Optional[str] = None
    ramp: Optional[str] = None


@attr.s(auto_attribs=True)
class DrySolution(Action):
    material: Optional[str] = None
    duration: Optional[str] = None
    temperature: Optional[str] = None


@attr.s(auto_attribs=True)
class Grind(Action):
    material: Optional[str] = None
    duration: Optional[str] = None
    temperature: Optional[str] = None


@attr.s(auto_attribs=True)
class Impregnate(Action):
    material: Chemical
    support_material: Chemical
    solvent: Chemical
    target_loading: Optional[str] = None


@attr.s(auto_attribs=True)
class Leach(Action):
    material: Optional[Chemical] = None
    duration: Optional[str] = None
    temperature: Optional[str] = None


@attr.s(auto_attribs=True)
class PostTreatment(Action):
    treatment_type: str
    gas: str
    duration: Optional[str] = None
    temperature: Optional[str] = None
    ramp: Optional[str] = None

    def __attrs_post_init__(self):
        if self.treatment_type not in _ALLOWED_POST_TREATMENTS:
            raise ValueError(
                f'Invalid treatment type "{self.treatment_type}", '
                f"must be one of {_ALLOWED_POST_TREATMENTS}."
            )


@attr.s(auto_attribs=True)
class Precipitate(Action):
    material: Optional[Chemical] = None
    support_material: Optional[Chemical] = None
    solvent: Optional[Chemical] = None
    target_concentration: Optional[str] = None
    ph: Optional[str] = None
    target_loading: Optional[str] = None


@attr.s(auto_attribs=True)
class Pyrolize(Action):
    material: Optional[Chemical] = None
    gas: Optional[str] = None
    duration: Optional[str] = None
    temperature: Optional[str] = None
    ramp: Optional[str] = None


@attr.s(auto_attribs=True)
class Reflux(Action):
    duration: Optional[str] = None
    temperature: Optional[str] = None
    dean_stark: bool = False
    atmosphere: Optional[str] = None


@attr.s(auto_attribs=True)
class Stir(Action):
    duration: Optional[str] = None
    temperature: Optional[str] = None
    atmosphere: Optional[str] = None
    rpm: Optional[str] = None

    def __attrs_post_init__(self):
        # Validate the given RPM value
        if self.rpm is not None:
            try:
                _ = float(self.rpm)
            except ValueError as e:
                raise ValueError(
                    f'The rpm value, "{self.rpm}", must be a number.'
                ) from e


@attr.s(auto_attribs=True)
class SynthesisMethod(Action):
    method: str

    def __attrs_post_init__(self):
        if self.method not in _ALLOWED_SYNTHESIS_METHODS:
            raise ValueError(
                f'Invalid synthesis method "{self.method}", '
                f"must be one of {_ALLOWED_SYNTHESIS_METHODS}."
            )


@attr.s(auto_attribs=True)
class SynthesisProduct(Action):
    catalyst_name: str
    target_loading: Optional[str] = None


@attr.s(auto_attribs=True)
class SynthesisVariant(Action):
    """
    Denotes a variation to a synthesis, where a related target is created
    from another precursor or set of precursors.

    Note (July 2022): in this initial implementation, we limit the precursors to one only.
    """

    target_metal: Chemical
    precursor: Chemical
    temperature: Optional[str] = None


@attr.s(auto_attribs=True)
class Transfer(Action):
    vessel: str


@attr.s(auto_attribs=True)
class Wait(Action):
    duration: str
    temperature: Optional[str] = None
    atmosphere: Optional[str] = None


@attr.s(auto_attribs=True)
class Wash(Action):
    material: Chemical
    repetitions: int = 1
    duration: Optional[str] = None
    temperature: Optional[str] = None
