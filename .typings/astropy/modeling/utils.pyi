from _typeshed import Incomplete
from collections import UserDict

__all__ = ['poly_map_domain', 'ellipse_extent']

def poly_map_domain(oldx, domain, window):
    """
    Map domain into window by shifting and scaling.

    Parameters
    ----------
    oldx : array
          original coordinates
    domain : list or tuple of length 2
          function domain
    window : list or tuple of length 2
          range into which to map the domain
    """
def ellipse_extent(a, b, theta):
    """
    Calculates the half size of a box encapsulating a rotated 2D
    ellipse.

    Parameters
    ----------
    a : float or `~astropy.units.Quantity`
        The ellipse semimajor axis.
    b : float or `~astropy.units.Quantity`
        The ellipse semiminor axis.
    theta : float or `~astropy.units.Quantity` ['angle']
        The rotation angle as an angular quantity
        (`~astropy.units.Quantity` or `~astropy.coordinates.Angle`) or
        a value in radians (as a float). The rotation angle increases
        counterclockwise.

    Returns
    -------
    offsets : tuple
        The absolute value of the offset distances from the ellipse center that
        define its bounding box region, ``(dx, dy)``.

    Examples
    --------
    .. plot::
        :include-source:

        import numpy as np
        import matplotlib.pyplot as plt
        from astropy.modeling.models import Ellipse2D
        from astropy.modeling.utils import ellipse_extent, render_model

        amplitude = 1
        x0 = 50
        y0 = 50
        a = 30
        b = 10
        theta = np.pi / 4

        model = Ellipse2D(amplitude, x0, y0, a, b, theta)
        dx, dy = ellipse_extent(a, b, theta)
        limits = [x0 - dx, x0 + dx, y0 - dy, y0 + dy]
        model.bounding_box = limits

        image = render_model(model)

        plt.imshow(image, cmap='binary', interpolation='nearest', alpha=.5,
                  extent = limits)
        plt.show()
    """

class _ConstraintsDict(UserDict):
    """
    Wrapper around UserDict to allow updating the constraints
    on a Parameter when the dictionary is updated.
    """
    _model: Incomplete
    constraint_type: Incomplete
    def __init__(self, model, constraint_type) -> None: ...
    def __setitem__(self, key, val) -> None: ...

class _SpecialOperatorsDict(UserDict):
    """
    Wrapper around UserDict to allow for better tracking of the Special
    Operators for CompoundModels. This dictionary is structured so that
    one cannot inadvertently overwrite an existing special operator.

    Parameters
    ----------
    unique_id: int
        the last used unique_id for a SPECIAL OPERATOR
    special_operators: dict
        a dictionary containing the special_operators

    Notes
    -----
    Direct setting of operators (`dict[key] = value`) into the
    dictionary has been deprecated in favor of the `.add(name, value)`
    method, so that unique dictionary keys can be generated and tracked
    consistently.
    """
    _unique_id: Incomplete
    def __init__(self, unique_id: int = 0, special_operators={}) -> None: ...
    def _set_value(self, key, val) -> None: ...
    def __setitem__(self, key, val) -> None: ...
    def _get_unique_id(self): ...
    def add(self, operator_name, operator):
        """
        Adds a special operator to the dictionary, and then returns the
        unique key that the operator is stored under for later reference.

        Parameters
        ----------
        operator_name: str
            the name for the operator
        operator: function
            the actual operator function which will be used

        Returns
        -------
        the unique operator key for the dictionary
            `(operator_name, unique_id)`
        """
