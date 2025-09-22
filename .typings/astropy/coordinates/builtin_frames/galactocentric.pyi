from _typeshed import Incomplete
from astropy.coordinates import representation as r
from astropy.coordinates.baseframe import BaseCoordinateFrame
from astropy.utils.state import ScienceState
from collections.abc import MappingView

__all__ = ['Galactocentric']

class _StateProxy(MappingView):
    """
    `~collections.abc.MappingView` with a read-only ``getitem`` through
    `~types.MappingProxyType`.

    """
    _mappingproxy: Incomplete
    def __init__(self, mapping) -> None: ...
    def __getitem__(self, key):
        """Read-only ``getitem``."""
    def __deepcopy__(self, memo): ...

class galactocentric_frame_defaults(ScienceState):
    '''Global setting of default values for the frame attributes in the `~astropy.coordinates.Galactocentric` frame.

    These constancts may be updated in future versions of ``astropy``. Note
    that when using `~astropy.coordinates.Galactocentric`, changing values
    here will not affect any attributes that are set explicitly by passing
    values in to the `~astropy.coordinates.Galactocentric`
    initializer. Modifying these defaults will only affect the frame attribute
    values when using the frame as, e.g., ``Galactocentric`` or
    ``Galactocentric()`` with no explicit arguments.

    This class controls the parameter settings by specifying a string name,
    with the following pre-specified options:

    - \'pre-v4.0\': The current default value, which sets the default frame
      attribute values to their original (pre-astropy-v4.0) values.
    - \'v4.0\': The attribute values as updated in Astropy version 4.0.
    - \'latest\': An alias of the most recent parameter set (currently: \'v4.0\')

    Alternatively, user-defined parameter settings may be registered, with
    :meth:`~astropy.coordinates.galactocentric_frame_defaults.register`,
    and used identically as pre-specified parameter sets. At minimum,
    registrations must have unique names and a dictionary of parameters
    with keys "galcen_coord", "galcen_distance", "galcen_v_sun", "z_sun",
    "roll". See examples below.

    This class also tracks the references for all parameter values in the
    attribute ``references``, as well as any further information the registry.
    The pre-specified options can be extended to include similar
    state information as user-defined parameter settings -- for example, to add
    parameter uncertainties.

    The preferred method for getting a parameter set and metadata, by name, is
    :meth:`~astropy.coordinates.galactocentric_frame_defaults.get_from_registry`
    since it ensures the immutability of the registry.

    See :ref:`astropy:astropy-coordinates-galactocentric-defaults` for more
    information.

    Examples
    --------
    The default `~astropy.coordinates.Galactocentric` frame parameters can be
    modified globally::

        >>> from astropy.coordinates import galactocentric_frame_defaults
        >>> _ = galactocentric_frame_defaults.set(\'v4.0\') # doctest: +SKIP
        >>> Galactocentric() # doctest: +SKIP
        <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
            (266.4051, -28.936175)>, galcen_distance=8.122 kpc, galcen_v_sun=(12.9, 245.6, 7.78) km / s, z_sun=20.8 pc, roll=0.0 deg)>
        >>> _ = galactocentric_frame_defaults.set(\'pre-v4.0\') # doctest: +SKIP
        >>> Galactocentric() # doctest: +SKIP
        <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
            (266.4051, -28.936175)>, galcen_distance=8.3 kpc, galcen_v_sun=(11.1, 232.24, 7.25) km / s, z_sun=27.0 pc, roll=0.0 deg)>

    The default parameters can also be updated by using this class as a context
    manager::

        >>> with galactocentric_frame_defaults.set(\'pre-v4.0\'):
        ...     print(Galactocentric()) # doctest: +FLOAT_CMP
        <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
            (266.4051, -28.936175)>, galcen_distance=8.3 kpc, galcen_v_sun=(11.1, 232.24, 7.25) km / s, z_sun=27.0 pc, roll=0.0 deg)>

    Again, changing the default parameter values will not affect frame
    attributes that are explicitly specified::

        >>> import astropy.units as u
        >>> with galactocentric_frame_defaults.set(\'pre-v4.0\'):
        ...     print(Galactocentric(galcen_distance=8.0*u.kpc)) # doctest: +FLOAT_CMP
        <Galactocentric Frame (galcen_coord=<ICRS Coordinate: (ra, dec) in deg
            (266.4051, -28.936175)>, galcen_distance=8.0 kpc, galcen_v_sun=(11.1, 232.24, 7.25) km / s, z_sun=27.0 pc, roll=0.0 deg)>

    Additional parameter sets may be registered, for instance to use the
    Dehnen & Binney (1998) measurements of the solar motion. We can also
    add metadata, such as the 1-sigma errors. In this example we will modify
    the required key "parameters", change the recommended key "references" to
    match "parameters", and add the extra key "error" (any key can be added)::

        >>> state = galactocentric_frame_defaults.get_from_registry("v4.0")
        >>> state["parameters"]["galcen_v_sun"] = (10.00, 225.25, 7.17) * (u.km / u.s)
        >>> state["references"]["galcen_v_sun"] = "https://ui.adsabs.harvard.edu/full/1998MNRAS.298..387D"
        >>> state["error"] = {"galcen_v_sun": (0.36, 0.62, 0.38) * (u.km / u.s)}
        >>> galactocentric_frame_defaults.register(name="DB1998", **state)

    Just as in the previous examples, the new parameter set can be retrieved with::

        >>> state = galactocentric_frame_defaults.get_from_registry("DB1998")
        >>> print(state["error"]["galcen_v_sun"])  # doctest: +FLOAT_CMP
        [0.36 0.62 0.38] km / s

    '''
    _latest_value: str
    _value: Incomplete
    _references: Incomplete
    _state: Incomplete
    _registry: Incomplete
    def parameters(cls): ...
    def references(cls): ...
    @classmethod
    def get_from_registry(cls, name: str) -> dict[str, dict]:
        '''
        Return Galactocentric solar parameters and metadata given string names
        for the parameter sets. This method ensures the returned state is a
        mutable copy, so any changes made do not affect the registry state.

        Returns
        -------
        state : dict
            Copy of the registry for the string name.
            Should contain, at minimum:

            - "parameters": dict
                Galactocentric solar parameters
            - "references" : Dict[str, Union[str, Sequence[str]]]
                References for "parameters".
                Fields are str or sequence of str.

        Raises
        ------
        KeyError
            If invalid string input to registry
            to retrieve solar parameters for Galactocentric frame.

        '''
    @classmethod
    def validate(cls, value): ...
    @classmethod
    def register(cls, name: str, parameters: dict, references: Incomplete | None = None, **meta: dict) -> None:
        """Register a set of parameters.

        Parameters
        ----------
        name : str
            The registration name for the parameter and metadata set.
        parameters : dict
            The solar parameters for Galactocentric frame.
        references : dict or None, optional
            References for contents of `parameters`.
            None becomes empty dict.
        **meta : dict, optional
            Any other properties to register.

        """

class Galactocentric(BaseCoordinateFrame):
    '''
    A coordinate or frame in the Galactocentric system.

    This frame allows specifying the Sun-Galactic center distance, the height of
    the Sun above the Galactic midplane, and the solar motion relative to the
    Galactic center. However, as there is no modern standard definition of a
    Galactocentric reference frame, it is important to pay attention to the
    default values used in this class if precision is important in your code.
    The default values of the parameters of this frame are taken from the
    original definition of the frame in 2014. As such, the defaults are somewhat
    out of date relative to recent measurements made possible by, e.g., Gaia.
    The defaults can, however, be changed at runtime by setting the parameter
    set name in `~astropy.coordinates.galactocentric_frame_defaults`.

    The current default parameter set is ``"pre-v4.0"``, indicating that the
    parameters were adopted before ``astropy`` version 4.0. A regularly-updated
    parameter set can instead be used by setting
    ``galactocentric_frame_defaults.set (\'latest\')``, and other parameter set
    names may be added in future versions. To find out the scientific papers
    that the current default parameters are derived from, use
    ``galcen.frame_attribute_references`` (where ``galcen`` is an instance of
    this frame), which will update even if the default parameter set is changed.

    The position of the Sun is assumed to be on the x axis of the final,
    right-handed system. That is, the x axis points from the position of
    the Sun projected to the Galactic midplane to the Galactic center --
    roughly towards :math:`(l,b) = (0^\\circ,0^\\circ)`. For the default
    transformation (:math:`{\\rm roll}=0^\\circ`), the y axis points roughly
    towards Galactic longitude :math:`l=90^\\circ`, and the z axis points
    roughly towards the North Galactic Pole (:math:`b=90^\\circ`).

    For a more detailed look at the math behind this transformation, see
    the document :ref:`astropy:coordinates-galactocentric`.

    The frame attributes are listed under **Other Parameters**.
    '''
    default_representation = r.CartesianRepresentation
    default_differential = r.CartesianDifferential
    galcen_coord: Incomplete
    galcen_distance: Incomplete
    galcen_v_sun: Incomplete
    z_sun: Incomplete
    roll: Incomplete
    frame_attribute_references: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @classmethod
    def get_roll0(cls):
        """The additional roll angle (about the final x axis) necessary to align the
        final z axis to match the Galactic yz-plane.  Setting the ``roll``
        frame attribute to -this method's return value removes this rotation,
        allowing the use of the `~astropy.coordinates.Galactocentric` frame
        in more general contexts.

        """
