import abc
from .core import Model
from .parameters import Parameter
from _typeshed import Incomplete

__all__ = ['Projection', 'Pix2SkyProjection', 'Sky2PixProjection', 'Zenithal', 'Cylindrical', 'PseudoCylindrical', 'Conic', 'PseudoConic', 'QuadCube', 'HEALPix', 'AffineTransformation2D', 'projcodes', 'Pix2Sky_ZenithalPerspective', 'Pix2Sky_AZP', 'Pix2Sky_SlantZenithalPerspective', 'Pix2Sky_SZP', 'Pix2Sky_Gnomonic', 'Pix2Sky_TAN', 'Pix2Sky_Stereographic', 'Pix2Sky_STG', 'Pix2Sky_SlantOrthographic', 'Pix2Sky_SIN', 'Pix2Sky_ZenithalEquidistant', 'Pix2Sky_ARC', 'Pix2Sky_ZenithalEqualArea', 'Pix2Sky_ZEA', 'Pix2Sky_Airy', 'Pix2Sky_AIR', 'Pix2Sky_CylindricalPerspective', 'Pix2Sky_CYP', 'Pix2Sky_CylindricalEqualArea', 'Pix2Sky_CEA', 'Pix2Sky_PlateCarree', 'Pix2Sky_CAR', 'Pix2Sky_Mercator', 'Pix2Sky_MER', 'Pix2Sky_SansonFlamsteed', 'Pix2Sky_SFL', 'Pix2Sky_Parabolic', 'Pix2Sky_PAR', 'Pix2Sky_Molleweide', 'Pix2Sky_MOL', 'Pix2Sky_HammerAitoff', 'Pix2Sky_AIT', 'Pix2Sky_ConicPerspective', 'Pix2Sky_COP', 'Pix2Sky_ConicEqualArea', 'Pix2Sky_COE', 'Pix2Sky_ConicEquidistant', 'Pix2Sky_COD', 'Pix2Sky_ConicOrthomorphic', 'Pix2Sky_COO', 'Pix2Sky_BonneEqualArea', 'Pix2Sky_BON', 'Pix2Sky_Polyconic', 'Pix2Sky_PCO', 'Pix2Sky_TangentialSphericalCube', 'Pix2Sky_TSC', 'Pix2Sky_COBEQuadSphericalCube', 'Pix2Sky_CSC', 'Pix2Sky_QuadSphericalCube', 'Pix2Sky_QSC', 'Pix2Sky_HEALPix', 'Pix2Sky_HPX', 'Pix2Sky_HEALPixPolar', 'Pix2Sky_XPH', 'Sky2Pix_ZenithalPerspective', 'Sky2Pix_AZP', 'Sky2Pix_SlantZenithalPerspective', 'Sky2Pix_SZP', 'Sky2Pix_Gnomonic', 'Sky2Pix_TAN', 'Sky2Pix_Stereographic', 'Sky2Pix_STG', 'Sky2Pix_SlantOrthographic', 'Sky2Pix_SIN', 'Sky2Pix_ZenithalEquidistant', 'Sky2Pix_ARC', 'Sky2Pix_ZenithalEqualArea', 'Sky2Pix_ZEA', 'Sky2Pix_Airy', 'Sky2Pix_AIR', 'Sky2Pix_CylindricalPerspective', 'Sky2Pix_CYP', 'Sky2Pix_CylindricalEqualArea', 'Sky2Pix_CEA', 'Sky2Pix_PlateCarree', 'Sky2Pix_CAR', 'Sky2Pix_Mercator', 'Sky2Pix_MER', 'Sky2Pix_SansonFlamsteed', 'Sky2Pix_SFL', 'Sky2Pix_Parabolic', 'Sky2Pix_PAR', 'Sky2Pix_Molleweide', 'Sky2Pix_MOL', 'Sky2Pix_HammerAitoff', 'Sky2Pix_AIT', 'Sky2Pix_ConicPerspective', 'Sky2Pix_COP', 'Sky2Pix_ConicEqualArea', 'Sky2Pix_COE', 'Sky2Pix_ConicEquidistant', 'Sky2Pix_COD', 'Sky2Pix_ConicOrthomorphic', 'Sky2Pix_COO', 'Sky2Pix_BonneEqualArea', 'Sky2Pix_BON', 'Sky2Pix_Polyconic', 'Sky2Pix_PCO', 'Sky2Pix_TangentialSphericalCube', 'Sky2Pix_TSC', 'Sky2Pix_COBEQuadSphericalCube', 'Sky2Pix_CSC', 'Sky2Pix_QuadSphericalCube', 'Sky2Pix_QSC', 'Sky2Pix_HEALPix', 'Sky2Pix_HPX', 'Sky2Pix_HEALPixPolar', 'Sky2Pix_XPH']

projcodes: Incomplete

class _ParameterDS(Parameter):
    """
    Same as `Parameter` but can indicate its modified status via the ``dirty``
    property. This flag also gets set automatically when a parameter is
    modified.

    This ability to track parameter's modified status is needed for automatic
    update of WCSLIB's prjprm structure (which may be a more-time intensive
    operation) *only as required*.

    """
    dirty: bool
    def __init__(self, *args, **kwargs) -> None: ...
    def validate(self, value) -> None: ...

class Projection(Model, metaclass=abc.ABCMeta):
    """Base class for all sky projections."""
    r0: Incomplete
    _separable: bool
    _prj: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    @abc.abstractmethod
    def inverse(self):
        """
        Inverse projection--all projection models must provide an inverse.
        """
    @property
    def prjprm(self):
        """WCSLIB ``prjprm`` structure."""
    def _update_prj(self) -> None:
        """
        A default updater for projection's pv.

        .. warning::
            This method assumes that PV0 is never modified. If a projection
            that uses PV0 is ever implemented in this module, that projection
            class should override this method.

        .. warning::
            This method assumes that the order in which PVi values (i>0)
            are to be assigned is identical to the order of model parameters
            in ``param_names``. That is, pv[1] = model.parameters[0], ...

        """
    def __getstate__(self): ...
    def __setstate__(self, state): ...

class Pix2SkyProjection(Projection):
    """Base class for all Pix2Sky projections."""
    n_inputs: int
    n_outputs: int
    _input_units_strict: bool
    _input_units_allow_dimensionless: bool
    def __new__(cls, *args, **kwargs): ...
    inputs: Incomplete
    outputs: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def input_units(self): ...
    @property
    def return_units(self): ...
    def evaluate(self, x, y, *args, **kwargs): ...
    @property
    def inverse(self): ...

class Sky2PixProjection(Projection):
    """Base class for all Sky2Pix projections."""
    n_inputs: int
    n_outputs: int
    _input_units_strict: bool
    _input_units_allow_dimensionless: bool
    def __new__(cls, *args, **kwargs): ...
    inputs: Incomplete
    outputs: Incomplete
    def __init__(self, *args, **kwargs) -> None: ...
    @property
    def input_units(self): ...
    @property
    def return_units(self): ...
    def evaluate(self, phi, theta, *args, **kwargs): ...
    @property
    def inverse(self): ...

class Zenithal(Projection, metaclass=abc.ABCMeta):
    """Base class for all Zenithal projections.

    Zenithal (or azimuthal) projections map the sphere directly onto a
    plane.  All zenithal projections are specified by defining the
    radius as a function of native latitude, :math:`R_\\theta`.

    The pixel-to-sky transformation is defined as:

    .. math::
        \\phi &= \\arg(-y, x) \\\\\n        R_\\theta &= \\sqrt{x^2 + y^2}

    and the inverse (sky-to-pixel) is defined as:

    .. math::
        x &= R_\\theta \\sin \\phi \\\\\n        y &= R_\\theta \\cos \\phi
    """

class Pix2Sky_ZenithalPerspective(Pix2SkyProjection, Zenithal):
    """
    Zenithal perspective projection - pixel to sky.

    Corresponds to the ``AZP`` projection in FITS WCS.

    .. math::
        \\phi &= \\arg(-y \\cos \\gamma, x) \\\\\n        \\theta &= \\left\\{\\genfrac{}{}{0pt}{}{\\psi - \\omega}{\\psi + \\omega + 180^{\\circ}}\\right.

    where:

    .. math::
        \\psi &= \\arg(\\rho, 1) \\\\\n        \\omega &= \\sin^{-1}\\left(\\frac{\\rho \\mu}{\\sqrt{\\rho^2 + 1}}\\right) \\\\\n        \\rho &= \\frac{R}{\\frac{180^{\\circ}}{\\pi}(\\mu + 1) + y \\sin \\gamma} \\\\\n        R &= \\sqrt{x^2 + y^2 \\cos^2 \\gamma}

    Parameters
    ----------
    mu : float
        Distance from point of projection to center of sphere
        in spherical radii, μ.  Default is 0.

    gamma : float
        Look angle γ in degrees.  Default is 0°.

    """
    mu: Incomplete
    gamma: Incomplete
    def _mu_validator(self, value) -> None: ...

class Sky2Pix_ZenithalPerspective(Sky2PixProjection, Zenithal):
    """
    Zenithal perspective projection - sky to pixel.

    Corresponds to the ``AZP`` projection in FITS WCS.

    .. math::
        x &= R \\sin \\phi \\\\\n        y &= -R \\sec \\gamma \\cos \\theta

    where:

    .. math::
        R = \\frac{180^{\\circ}}{\\pi} \\frac{(\\mu + 1) \\cos \\theta}
            {(\\mu + \\sin \\theta) + \\cos \\theta \\cos \\phi \\tan \\gamma}

    Parameters
    ----------
    mu : float
        Distance from point of projection to center of sphere
        in spherical radii, μ. Default is 0.

    gamma : float
        Look angle γ in degrees. Default is 0°.

    """
    mu: Incomplete
    gamma: Incomplete
    def _mu_validator(self, value) -> None: ...

class Pix2Sky_SlantZenithalPerspective(Pix2SkyProjection, Zenithal):
    """
    Slant zenithal perspective projection - pixel to sky.

    Corresponds to the ``SZP`` projection in FITS WCS.

    Parameters
    ----------
    mu : float
        Distance from point of projection to center of sphere
        in spherical radii, μ.  Default is 0.

    phi0 : float
        The longitude φ₀ of the reference point, in degrees.  Default
        is 0°.

    theta0 : float
        The latitude θ₀ of the reference point, in degrees.  Default
        is 90°.

    """
    mu: Incomplete
    phi0: Incomplete
    theta0: Incomplete
    def _mu_validator(self, value) -> None: ...

class Sky2Pix_SlantZenithalPerspective(Sky2PixProjection, Zenithal):
    """
    Zenithal perspective projection - sky to pixel.

    Corresponds to the ``SZP`` projection in FITS WCS.

    Parameters
    ----------
    mu : float
        distance from point of projection to center of sphere
        in spherical radii, μ.  Default is 0.

    phi0 : float
        The longitude φ₀ of the reference point, in degrees.  Default
        is 0°.

    theta0 : float
        The latitude θ₀ of the reference point, in degrees.  Default
        is 90°.

    """
    mu: Incomplete
    phi0: Incomplete
    theta0: Incomplete
    def _mu_validator(self, value) -> None: ...

class Pix2Sky_Gnomonic(Pix2SkyProjection, Zenithal):
    """
    Gnomonic projection - pixel to sky.

    Corresponds to the ``TAN`` projection in FITS WCS.

    See `Zenithal` for a definition of the full transformation.

    .. math::
        \\theta = \\tan^{-1}\\left(\\frac{180^{\\circ}}{\\pi R_\\theta}\\right)
    """
class Sky2Pix_Gnomonic(Sky2PixProjection, Zenithal):
    """
    Gnomonic Projection - sky to pixel.

    Corresponds to the ``TAN`` projection in FITS WCS.

    See `Zenithal` for a definition of the full transformation.

    .. math::
        R_\\theta = \\frac{180^{\\circ}}{\\pi}\\cot \\theta
    """
class Pix2Sky_Stereographic(Pix2SkyProjection, Zenithal):
    """
    Stereographic Projection - pixel to sky.

    Corresponds to the ``STG`` projection in FITS WCS.

    See `Zenithal` for a definition of the full transformation.

    .. math::
        \\theta = 90^{\\circ} - 2 \\tan^{-1}\\left(\\frac{\\pi R_\\theta}{360^{\\circ}}\\right)
    """
class Sky2Pix_Stereographic(Sky2PixProjection, Zenithal):
    """
    Stereographic Projection - sky to pixel.

    Corresponds to the ``STG`` projection in FITS WCS.

    See `Zenithal` for a definition of the full transformation.

    .. math::
        R_\\theta = \\frac{180^{\\circ}}{\\pi}\\frac{2 \\cos \\theta}{1 + \\sin \\theta}
    """

class Pix2Sky_SlantOrthographic(Pix2SkyProjection, Zenithal):
    """
    Slant orthographic projection - pixel to sky.

    Corresponds to the ``SIN`` projection in FITS WCS.

    See `Zenithal` for a definition of the full transformation.

    The following transformation applies when :math:`\\xi` and
    :math:`\\eta` are both zero.

    .. math::
        \\theta = \\cos^{-1}\\left(\\frac{\\pi}{180^{\\circ}}R_\\theta\\right)

    The parameters :math:`\\xi` and :math:`\\eta` are defined from the
    reference point :math:`(\\phi_c, \\theta_c)` as:

    .. math::
        \\xi &= \\cot \\theta_c \\sin \\phi_c \\\\\n        \\eta &= - \\cot \\theta_c \\cos \\phi_c

    Parameters
    ----------
    xi : float
        Obliqueness parameter, ξ.  Default is 0.0.

    eta : float
        Obliqueness parameter, η.  Default is 0.0.

    """
    xi: Incomplete
    eta: Incomplete

class Sky2Pix_SlantOrthographic(Sky2PixProjection, Zenithal):
    """
    Slant orthographic projection - sky to pixel.

    Corresponds to the ``SIN`` projection in FITS WCS.

    See `Zenithal` for a definition of the full transformation.

    The following transformation applies when :math:`\\xi` and
    :math:`\\eta` are both zero.

    .. math::
        R_\\theta = \\frac{180^{\\circ}}{\\pi}\\cos \\theta

    But more specifically are:

    .. math::
        x &= \\frac{180^\\circ}{\\pi}[\\cos \\theta \\sin \\phi + \\xi(1 - \\sin \\theta)] \\\\\n        y &= \\frac{180^\\circ}{\\pi}[\\cos \\theta \\cos \\phi + \\eta(1 - \\sin \\theta)]

    """
    xi: Incomplete
    eta: Incomplete

class Pix2Sky_ZenithalEquidistant(Pix2SkyProjection, Zenithal):
    """
    Zenithal equidistant projection - pixel to sky.

    Corresponds to the ``ARC`` projection in FITS WCS.

    See `Zenithal` for a definition of the full transformation.

    .. math::
        \\theta = 90^\\circ - R_\\theta
    """
class Sky2Pix_ZenithalEquidistant(Sky2PixProjection, Zenithal):
    """
    Zenithal equidistant projection - sky to pixel.

    Corresponds to the ``ARC`` projection in FITS WCS.

    See `Zenithal` for a definition of the full transformation.

    .. math::
        R_\\theta = 90^\\circ - \\theta
    """
class Pix2Sky_ZenithalEqualArea(Pix2SkyProjection, Zenithal):
    """
    Zenithal equidistant projection - pixel to sky.

    Corresponds to the ``ZEA`` projection in FITS WCS.

    See `Zenithal` for a definition of the full transformation.

    .. math::
        \\theta = 90^\\circ - 2 \\sin^{-1} \\left(\\frac{\\pi R_\\theta}{360^\\circ}\\right)
    """
class Sky2Pix_ZenithalEqualArea(Sky2PixProjection, Zenithal):
    """
    Zenithal equidistant projection - sky to pixel.

    Corresponds to the ``ZEA`` projection in FITS WCS.

    See `Zenithal` for a definition of the full transformation.

    .. math::
        R_\\theta &= \\frac{180^\\circ}{\\pi} \\sqrt{2(1 - \\sin\\theta)} \\\\\n                 &= \\frac{360^\\circ}{\\pi} \\sin\\left(\\frac{90^\\circ - \\theta}{2}\\right)
    """

class Pix2Sky_Airy(Pix2SkyProjection, Zenithal):
    """
    Airy projection - pixel to sky.

    Corresponds to the ``AIR`` projection in FITS WCS.

    See `Zenithal` for a definition of the full transformation.

    Parameters
    ----------
    theta_b : float
        The latitude :math:`\\theta_b` at which to minimize the error,
        in degrees.  Default is 90°.
    """
    theta_b: Incomplete

class Sky2Pix_Airy(Sky2PixProjection, Zenithal):
    """
    Airy - sky to pixel.

    Corresponds to the ``AIR`` projection in FITS WCS.

    See `Zenithal` for a definition of the full transformation.

    .. math::
        R_\\theta = -2 \\frac{180^\\circ}{\\pi}\\left(\\frac{\\ln(\\cos \\xi)}{\\tan \\xi} +
            \\frac{\\ln(\\cos \\xi_b)}{\\tan^2 \\xi_b} \\tan \\xi \\right)

    where:

    .. math::
        \\xi &= \\frac{90^\\circ - \\theta}{2} \\\\\n        \\xi_b &= \\frac{90^\\circ - \\theta_b}{2}

    Parameters
    ----------
    theta_b : float
        The latitude :math:`\\theta_b` at which to minimize the error,
        in degrees.  Default is 90°.

    """
    theta_b: Incomplete

class Cylindrical(Projection, metaclass=abc.ABCMeta):
    """Base class for Cylindrical projections.

    Cylindrical projections are so-named because the surface of
    projection is a cylinder.
    """
    _separable: bool

class Pix2Sky_CylindricalPerspective(Pix2SkyProjection, Cylindrical):
    """
    Cylindrical perspective - pixel to sky.

    Corresponds to the ``CYP`` projection in FITS WCS.

    .. math::
        \\phi &= \\frac{x}{\\lambda} \\\\\n        \\theta &= \\arg(1, \\eta) + \\sin{-1}\\left(\\frac{\\eta \\mu}{\\sqrt{\\eta^2 + 1}}\\right)

    where:

    .. math::
        \\eta = \\frac{\\pi}{180^{\\circ}}\\frac{y}{\\mu + \\lambda}

    Parameters
    ----------
    mu : float
        Distance from center of sphere in the direction opposite the
        projected surface, in spherical radii, μ. Default is 1.

    lam : float
        Radius of the cylinder in spherical radii, λ. Default is 1.

    """
    mu: Incomplete
    lam: Incomplete
    def _mu_validator(self, value) -> None: ...
    def _lam_validator(self, value) -> None: ...

class Sky2Pix_CylindricalPerspective(Sky2PixProjection, Cylindrical):
    """
    Cylindrical Perspective - sky to pixel.

    Corresponds to the ``CYP`` projection in FITS WCS.

    .. math::
        x &= \\lambda \\phi \\\\\n        y &= \\frac{180^{\\circ}}{\\pi}\\left(\\frac{\\mu + \\lambda}{\\mu + \\cos \\theta}\\right)\\sin \\theta

    Parameters
    ----------
    mu : float
        Distance from center of sphere in the direction opposite the
        projected surface, in spherical radii, μ.  Default is 0.

    lam : float
        Radius of the cylinder in spherical radii, λ.  Default is 0.

    """
    mu: Incomplete
    lam: Incomplete
    def _mu_validator(self, value) -> None: ...
    def _lam_validator(self, value) -> None: ...

class Pix2Sky_CylindricalEqualArea(Pix2SkyProjection, Cylindrical):
    """
    Cylindrical equal area projection - pixel to sky.

    Corresponds to the ``CEA`` projection in FITS WCS.

    .. math::
        \\phi &= x \\\\\n        \\theta &= \\sin^{-1}\\left(\\frac{\\pi}{180^{\\circ}}\\lambda y\\right)

    Parameters
    ----------
    lam : float
        Radius of the cylinder in spherical radii, λ.  Default is 1.
    """
    lam: Incomplete

class Sky2Pix_CylindricalEqualArea(Sky2PixProjection, Cylindrical):
    """
    Cylindrical equal area projection - sky to pixel.

    Corresponds to the ``CEA`` projection in FITS WCS.

    .. math::
        x &= \\phi \\\\\n        y &= \\frac{180^{\\circ}}{\\pi}\\frac{\\sin \\theta}{\\lambda}

    Parameters
    ----------
    lam : float
        Radius of the cylinder in spherical radii, λ.  Default is 0.
    """
    lam: Incomplete

class Pix2Sky_PlateCarree(Pix2SkyProjection, Cylindrical):
    """
    Plate carrée projection - pixel to sky.

    Corresponds to the ``CAR`` projection in FITS WCS.

    .. math::
        \\phi &= x \\\\\n        \\theta &= y
    """
    @staticmethod
    def evaluate(x, y): ...

class Sky2Pix_PlateCarree(Sky2PixProjection, Cylindrical):
    """
    Plate carrée projection - sky to pixel.

    Corresponds to the ``CAR`` projection in FITS WCS.

    .. math::
        x &= \\phi \\\\\n        y &= \\theta
    """
    @staticmethod
    def evaluate(phi, theta): ...

class Pix2Sky_Mercator(Pix2SkyProjection, Cylindrical):
    """
    Mercator - pixel to sky.

    Corresponds to the ``MER`` projection in FITS WCS.

    .. math::
        \\phi &= x \\\\\n        \\theta &= 2 \\tan^{-1}\\left(e^{y \\pi / 180^{\\circ}}\\right)-90^{\\circ}
    """
class Sky2Pix_Mercator(Sky2PixProjection, Cylindrical):
    """
    Mercator - sky to pixel.

    Corresponds to the ``MER`` projection in FITS WCS.

    .. math::
        x &= \\phi \\\\\n        y &= \\frac{180^{\\circ}}{\\pi}\\ln \\tan \\left(\\frac{90^{\\circ} + \\theta}{2}\\right)
    """

class PseudoCylindrical(Projection, metaclass=abc.ABCMeta):
    """Base class for pseudocylindrical projections.

    Pseudocylindrical projections are like cylindrical projections
    except the parallels of latitude are projected at diminishing
    lengths toward the polar regions in order to reduce lateral
    distortion there.  Consequently, the meridians are curved.
    """
    _separable: bool

class Pix2Sky_SansonFlamsteed(Pix2SkyProjection, PseudoCylindrical):
    """
    Sanson-Flamsteed projection - pixel to sky.

    Corresponds to the ``SFL`` projection in FITS WCS.

    .. math::
        \\phi &= \\frac{x}{\\cos y} \\\\\n        \\theta &= y
    """
class Sky2Pix_SansonFlamsteed(Sky2PixProjection, PseudoCylindrical):
    """
    Sanson-Flamsteed projection - sky to pixel.

    Corresponds to the ``SFL`` projection in FITS WCS.

    .. math::
        x &= \\phi \\cos \\theta \\\\\n        y &= \\theta
    """
class Pix2Sky_Parabolic(Pix2SkyProjection, PseudoCylindrical):
    """
    Parabolic projection - pixel to sky.

    Corresponds to the ``PAR`` projection in FITS WCS.

    .. math::
        \\phi &= \\frac{180^\\circ}{\\pi} \\frac{x}{1 - 4(y / 180^\\circ)^2} \\\\\n        \\theta &= 3 \\sin^{-1}\\left(\\frac{y}{180^\\circ}\\right)
    """
class Sky2Pix_Parabolic(Sky2PixProjection, PseudoCylindrical):
    """
    Parabolic projection - sky to pixel.

    Corresponds to the ``PAR`` projection in FITS WCS.

    .. math::
        x &= \\phi \\left(2\\cos\\frac{2\\theta}{3} - 1\\right) \\\\\n        y &= 180^\\circ \\sin \\frac{\\theta}{3}
    """
class Pix2Sky_Molleweide(Pix2SkyProjection, PseudoCylindrical):
    """
    Molleweide's projection - pixel to sky.

    Corresponds to the ``MOL`` projection in FITS WCS.

    .. math::
        \\phi &= \\frac{\\pi x}{2 \\sqrt{2 - \\left(\\frac{\\pi}{180^\\circ}y\\right)^2}} \\\\\n        \\theta &= \\sin^{-1}\\left(
                \\frac{1}{90^\\circ}\\sin^{-1}\\left(\\frac{\\pi}{180^\\circ}\\frac{y}{\\sqrt{2}}\\right)
                + \\frac{y}{180^\\circ}\\sqrt{2 - \\left(\\frac{\\pi}{180^\\circ}y\\right)^2}
            \\right)
    """
class Sky2Pix_Molleweide(Sky2PixProjection, PseudoCylindrical):
    """
    Molleweide's projection - sky to pixel.

    Corresponds to the ``MOL`` projection in FITS WCS.

    .. math::
        x &= \\frac{2 \\sqrt{2}}{\\pi} \\phi \\cos \\gamma \\\\\n        y &= \\sqrt{2} \\frac{180^\\circ}{\\pi} \\sin \\gamma

    where :math:`\\gamma` is defined as the solution of the
    transcendental equation:

    .. math::

        \\sin \\theta = \\frac{\\gamma}{90^\\circ} + \\frac{\\sin 2 \\gamma}{\\pi}
    """
class Pix2Sky_HammerAitoff(Pix2SkyProjection, PseudoCylindrical):
    """
    Hammer-Aitoff projection - pixel to sky.

    Corresponds to the ``AIT`` projection in FITS WCS.

    .. math::
        \\phi &= 2 \\arg \\left(2Z^2 - 1, \\frac{\\pi}{180^\\circ} \\frac{Z}{2}x\\right) \\\\\n        \\theta &= \\sin^{-1}\\left(\\frac{\\pi}{180^\\circ}yZ\\right)
    """
class Sky2Pix_HammerAitoff(Sky2PixProjection, PseudoCylindrical):
    """
    Hammer-Aitoff projection - sky to pixel.

    Corresponds to the ``AIT`` projection in FITS WCS.

    .. math::
        x &= 2 \\gamma \\cos \\theta \\sin \\frac{\\phi}{2} \\\\\n        y &= \\gamma \\sin \\theta

    where:

    .. math::
        \\gamma = \\frac{180^\\circ}{\\pi} \\sqrt{\\frac{2}{1 + \\cos \\theta \\cos(\\phi / 2)}}
    """

class Conic(Projection, metaclass=abc.ABCMeta):
    '''Base class for conic projections.

    In conic projections, the sphere is thought to be projected onto
    the surface of a cone which is then opened out.

    In a general sense, the pixel-to-sky transformation is defined as:

    .. math::

        \\phi &= \\arg\\left(\\frac{Y_0 - y}{R_\\theta}, \\frac{x}{R_\\theta}\\right) / C \\\\\n        R_\\theta &= \\mathrm{sign} \\theta_a \\sqrt{x^2 + (Y_0 - y)^2}

    and the inverse (sky-to-pixel) is defined as:

    .. math::
        x &= R_\\theta \\sin (C \\phi) \\\\\n        y &= R_\\theta \\cos (C \\phi) + Y_0

    where :math:`C` is the "constant of the cone":

    .. math::
        C = \\frac{180^\\circ \\cos \\theta}{\\pi R_\\theta}
    '''
    sigma: Incomplete
    delta: Incomplete

class Pix2Sky_ConicPerspective(Pix2SkyProjection, Conic):
    """
    Colles' conic perspective projection - pixel to sky.

    Corresponds to the ``COP`` projection in FITS WCS.

    See `Conic` for a description of the entire equation.

    The projection formulae are:

    .. math::
        C &= \\sin \\theta_a \\\\\n        R_\\theta &= \\frac{180^\\circ}{\\pi} \\cos \\eta [ \\cot \\theta_a - \\tan(\\theta - \\theta_a)] \\\\\n        Y_0 &= \\frac{180^\\circ}{\\pi} \\cos \\eta \\cot \\theta_a

    Parameters
    ----------
    sigma : float
        :math:`(\\theta_1 + \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 90.

    delta : float
        :math:`(\\theta_1 - \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 0.
    """
class Sky2Pix_ConicPerspective(Sky2PixProjection, Conic):
    """
    Colles' conic perspective projection - sky to pixel.

    Corresponds to the ``COP`` projection in FITS WCS.

    See `Conic` for a description of the entire equation.

    The projection formulae are:

    .. math::
        C &= \\sin \\theta_a \\\\\n        R_\\theta &= \\frac{180^\\circ}{\\pi} \\cos \\eta [ \\cot \\theta_a - \\tan(\\theta - \\theta_a)] \\\\\n        Y_0 &= \\frac{180^\\circ}{\\pi} \\cos \\eta \\cot \\theta_a

    Parameters
    ----------
    sigma : float
        :math:`(\\theta_1 + \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 90.

    delta : float
        :math:`(\\theta_1 - \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 0.
    """
class Pix2Sky_ConicEqualArea(Pix2SkyProjection, Conic):
    """
    Alber's conic equal area projection - pixel to sky.

    Corresponds to the ``COE`` projection in FITS WCS.

    See `Conic` for a description of the entire equation.

    The projection formulae are:

    .. math::
        C &= \\gamma / 2 \\\\\n        R_\\theta &= \\frac{180^\\circ}{\\pi} \\frac{2}{\\gamma}
            \\sqrt{1 + \\sin \\theta_1 \\sin \\theta_2 - \\gamma \\sin \\theta} \\\\\n        Y_0 &= \\frac{180^\\circ}{\\pi} \\frac{2}{\\gamma}
            \\sqrt{1 + \\sin \\theta_1 \\sin \\theta_2 - \\gamma \\sin((\\theta_1 + \\theta_2)/2)}

    where:

    .. math::
        \\gamma = \\sin \\theta_1 + \\sin \\theta_2

    Parameters
    ----------
    sigma : float
        :math:`(\\theta_1 + \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 90.

    delta : float
        :math:`(\\theta_1 - \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 0.
    """
class Sky2Pix_ConicEqualArea(Sky2PixProjection, Conic):
    """
    Alber's conic equal area projection - sky to pixel.

    Corresponds to the ``COE`` projection in FITS WCS.

    See `Conic` for a description of the entire equation.

    The projection formulae are:

    .. math::
        C &= \\gamma / 2 \\\\\n        R_\\theta &= \\frac{180^\\circ}{\\pi} \\frac{2}{\\gamma}
            \\sqrt{1 + \\sin \\theta_1 \\sin \\theta_2 - \\gamma \\sin \\theta} \\\\\n        Y_0 &= \\frac{180^\\circ}{\\pi} \\frac{2}{\\gamma}
            \\sqrt{1 + \\sin \\theta_1 \\sin \\theta_2 - \\gamma \\sin((\\theta_1 + \\theta_2)/2)}

    where:

    .. math::
        \\gamma = \\sin \\theta_1 + \\sin \\theta_2

    Parameters
    ----------
    sigma : float
        :math:`(\\theta_1 + \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 90.

    delta : float
        :math:`(\\theta_1 - \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 0.
    """
class Pix2Sky_ConicEquidistant(Pix2SkyProjection, Conic):
    """
    Conic equidistant projection - pixel to sky.

    Corresponds to the ``COD`` projection in FITS WCS.

    See `Conic` for a description of the entire equation.

    The projection formulae are:

    .. math::

        C &= \\frac{180^\\circ}{\\pi} \\frac{\\sin\\theta_a\\sin\\eta}{\\eta} \\\\\n        R_\\theta &= \\theta_a - \\theta + \\eta\\cot\\eta\\cot\\theta_a \\\\\n        Y_0 = \\eta\\cot\\eta\\cot\\theta_a

    Parameters
    ----------
    sigma : float
        :math:`(\\theta_1 + \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 90.

    delta : float
        :math:`(\\theta_1 - \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 0.
    """
class Sky2Pix_ConicEquidistant(Sky2PixProjection, Conic):
    """
    Conic equidistant projection - sky to pixel.

    Corresponds to the ``COD`` projection in FITS WCS.

    See `Conic` for a description of the entire equation.

    The projection formulae are:

    .. math::

        C &= \\frac{180^\\circ}{\\pi} \\frac{\\sin\\theta_a\\sin\\eta}{\\eta} \\\\\n        R_\\theta &= \\theta_a - \\theta + \\eta\\cot\\eta\\cot\\theta_a \\\\\n        Y_0 = \\eta\\cot\\eta\\cot\\theta_a

    Parameters
    ----------
    sigma : float
        :math:`(\\theta_1 + \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 90.

    delta : float
        :math:`(\\theta_1 - \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 0.
    """
class Pix2Sky_ConicOrthomorphic(Pix2SkyProjection, Conic):
    """
    Conic orthomorphic projection - pixel to sky.

    Corresponds to the ``COO`` projection in FITS WCS.

    See `Conic` for a description of the entire equation.

    The projection formulae are:

    .. math::

        C &= \\frac{\\ln \\left( \\frac{\\cos\\theta_2}{\\cos\\theta_1} \\right)}
                  {\\ln \\left[ \\frac{\\tan\\left(\\frac{90^\\circ-\\theta_2}{2}\\right)}
                                   {\\tan\\left(\\frac{90^\\circ-\\theta_1}{2}\\right)} \\right] } \\\\\n        R_\\theta &= \\psi \\left[ \\tan \\left( \\frac{90^\\circ - \\theta}{2} \\right) \\right]^C \\\\\n        Y_0 &= \\psi \\left[ \\tan \\left( \\frac{90^\\circ - \\theta_a}{2} \\right) \\right]^C

    where:

    .. math::

        \\psi = \\frac{180^\\circ}{\\pi} \\frac{\\cos \\theta}
               {C\\left[\\tan\\left(\\frac{90^\\circ-\\theta}{2}\\right)\\right]^C}

    Parameters
    ----------
    sigma : float
        :math:`(\\theta_1 + \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 90.

    delta : float
        :math:`(\\theta_1 - \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 0.
    """
class Sky2Pix_ConicOrthomorphic(Sky2PixProjection, Conic):
    """
    Conic orthomorphic projection - sky to pixel.

    Corresponds to the ``COO`` projection in FITS WCS.

    See `Conic` for a description of the entire equation.

    The projection formulae are:

    .. math::

        C &= \\frac{\\ln \\left( \\frac{\\cos\\theta_2}{\\cos\\theta_1} \\right)}
                  {\\ln \\left[ \\frac{\\tan\\left(\\frac{90^\\circ-\\theta_2}{2}\\right)}
                                   {\\tan\\left(\\frac{90^\\circ-\\theta_1}{2}\\right)} \\right] } \\\\\n        R_\\theta &= \\psi \\left[ \\tan \\left( \\frac{90^\\circ - \\theta}{2} \\right) \\right]^C \\\\\n        Y_0 &= \\psi \\left[ \\tan \\left( \\frac{90^\\circ - \\theta_a}{2} \\right) \\right]^C

    where:

    .. math::

        \\psi = \\frac{180^\\circ}{\\pi} \\frac{\\cos \\theta}
               {C\\left[\\tan\\left(\\frac{90^\\circ-\\theta}{2}\\right)\\right]^C}

    Parameters
    ----------
    sigma : float
        :math:`(\\theta_1 + \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 90.

    delta : float
        :math:`(\\theta_1 - \\theta_2) / 2`, where :math:`\\theta_1` and
        :math:`\\theta_2` are the latitudes of the standard parallels,
        in degrees.  Default is 0.
    """
class PseudoConic(Projection, metaclass=abc.ABCMeta):
    """Base class for pseudoconic projections.

    Pseudoconics are a subclass of conics with concentric parallels.
    """

class Pix2Sky_BonneEqualArea(Pix2SkyProjection, PseudoConic):
    """
    Bonne's equal area pseudoconic projection - pixel to sky.

    Corresponds to the ``BON`` projection in FITS WCS.

    .. math::

        \\phi &= \\frac{\\pi}{180^\\circ} A_\\phi R_\\theta / \\cos \\theta \\\\\n        \\theta &= Y_0 - R_\\theta

    where:

    .. math::

        R_\\theta &= \\mathrm{sign} \\theta_1 \\sqrt{x^2 + (Y_0 - y)^2} \\\\\n        A_\\phi &= \\arg\\left(\\frac{Y_0 - y}{R_\\theta}, \\frac{x}{R_\\theta}\\right)

    Parameters
    ----------
    theta1 : float
        Bonne conformal latitude, in degrees.
    """
    _separable: bool
    theta1: Incomplete

class Sky2Pix_BonneEqualArea(Sky2PixProjection, PseudoConic):
    """
    Bonne's equal area pseudoconic projection - sky to pixel.

    Corresponds to the ``BON`` projection in FITS WCS.

    .. math::
        x &= R_\\theta \\sin A_\\phi \\\\\n        y &= -R_\\theta \\cos A_\\phi + Y_0

    where:

    .. math::
        A_\\phi &= \\frac{180^\\circ}{\\pi R_\\theta} \\phi \\cos \\theta \\\\\n        R_\\theta &= Y_0 - \\theta \\\\\n        Y_0 &= \\frac{180^\\circ}{\\pi} \\cot \\theta_1 + \\theta_1

    Parameters
    ----------
    theta1 : float
        Bonne conformal latitude, in degrees.
    """
    _separable: bool
    theta1: Incomplete

class Pix2Sky_Polyconic(Pix2SkyProjection, PseudoConic):
    """
    Polyconic projection - pixel to sky.

    Corresponds to the ``PCO`` projection in FITS WCS.
    """
class Sky2Pix_Polyconic(Sky2PixProjection, PseudoConic):
    """
    Polyconic projection - sky to pixel.

    Corresponds to the ``PCO`` projection in FITS WCS.
    """
class QuadCube(Projection, metaclass=abc.ABCMeta):
    """Base class for quad cube projections.

    Quadrilateralized spherical cube (quad-cube) projections belong to
    the class of polyhedral projections in which the sphere is
    projected onto the surface of an enclosing polyhedron.

    The six faces of the quad-cube projections are numbered and laid
    out as::

              0
        4 3 2 1 4 3 2
              5

    """
class Pix2Sky_TangentialSphericalCube(Pix2SkyProjection, QuadCube):
    """
    Tangential spherical cube projection - pixel to sky.

    Corresponds to the ``TSC`` projection in FITS WCS.
    """
class Sky2Pix_TangentialSphericalCube(Sky2PixProjection, QuadCube):
    """
    Tangential spherical cube projection - sky to pixel.

    Corresponds to the ``TSC`` projection in FITS WCS.
    """
class Pix2Sky_COBEQuadSphericalCube(Pix2SkyProjection, QuadCube):
    """
    COBE quadrilateralized spherical cube projection - pixel to sky.

    Corresponds to the ``CSC`` projection in FITS WCS.
    """
class Sky2Pix_COBEQuadSphericalCube(Sky2PixProjection, QuadCube):
    """
    COBE quadrilateralized spherical cube projection - sky to pixel.

    Corresponds to the ``CSC`` projection in FITS WCS.
    """
class Pix2Sky_QuadSphericalCube(Pix2SkyProjection, QuadCube):
    """
    Quadrilateralized spherical cube projection - pixel to sky.

    Corresponds to the ``QSC`` projection in FITS WCS.
    """
class Sky2Pix_QuadSphericalCube(Sky2PixProjection, QuadCube):
    """
    Quadrilateralized spherical cube projection - sky to pixel.

    Corresponds to the ``QSC`` projection in FITS WCS.
    """
class HEALPix(Projection, metaclass=abc.ABCMeta):
    """Base class for HEALPix projections."""

class Pix2Sky_HEALPix(Pix2SkyProjection, HEALPix):
    """
    HEALPix - pixel to sky.

    Corresponds to the ``HPX`` projection in FITS WCS.

    Parameters
    ----------
    H : float
        The number of facets in longitude direction.

    X : float
        The number of facets in latitude direction.

    """
    _separable: bool
    H: Incomplete
    X: Incomplete

class Sky2Pix_HEALPix(Sky2PixProjection, HEALPix):
    """
    HEALPix projection - sky to pixel.

    Corresponds to the ``HPX`` projection in FITS WCS.

    Parameters
    ----------
    H : float
        The number of facets in longitude direction.

    X : float
        The number of facets in latitude direction.

    """
    _separable: bool
    H: Incomplete
    X: Incomplete

class Pix2Sky_HEALPixPolar(Pix2SkyProjection, HEALPix):
    '''
    HEALPix polar, aka "butterfly" projection - pixel to sky.

    Corresponds to the ``XPH`` projection in FITS WCS.
    '''
class Sky2Pix_HEALPixPolar(Sky2PixProjection, HEALPix):
    '''
    HEALPix polar, aka "butterfly" projection - pixel to sky.

    Corresponds to the ``XPH`` projection in FITS WCS.
    '''

class AffineTransformation2D(Model):
    """
    Perform an affine transformation in 2 dimensions.

    Parameters
    ----------
    matrix : array
        A 2x2 matrix specifying the linear transformation to apply to the
        inputs

    translation : array
        A 2D vector (given as either a 2x1 or 1x2 array) specifying a
        translation to apply to the inputs

    """
    n_inputs: int
    n_outputs: int
    standard_broadcasting: bool
    _separable: bool
    matrix: Incomplete
    translation: Incomplete
    def _matrix_validator(self, value) -> None:
        """Validates that the input matrix is a 2x2 2D array."""
    def _translation_validator(self, value) -> None:
        '''
        Validates that the translation vector is a 2D vector.  This allows
        either a "row" vector or a "column" vector where in the latter case the
        resultant Numpy array has ``ndim=2`` but the shape is ``(1, 2)``.
        '''
    inputs: Incomplete
    outputs: Incomplete
    def __init__(self, matrix=..., translation=..., **kwargs) -> None: ...
    @property
    def inverse(self):
        """
        Inverse transformation.

        Raises `~astropy.modeling.InputParameterError` if the transformation cannot be inverted.
        """
    @classmethod
    def evaluate(cls, x, y, matrix, translation):
        """
        Apply the transformation to a set of 2D Cartesian coordinates given as
        two lists--one for the x coordinates and one for a y coordinates--or a
        single coordinate pair.

        Parameters
        ----------
        x, y : array, float
              x and y coordinates
        """
    @staticmethod
    def _create_augmented_matrix(matrix, translation): ...
    @property
    def input_units(self): ...

# Names in __all__ with no definition:
#   Pix2Sky_AIR
#   Pix2Sky_AIT
#   Pix2Sky_ARC
#   Pix2Sky_AZP
#   Pix2Sky_BON
#   Pix2Sky_CAR
#   Pix2Sky_CEA
#   Pix2Sky_COD
#   Pix2Sky_COE
#   Pix2Sky_COO
#   Pix2Sky_COP
#   Pix2Sky_CSC
#   Pix2Sky_CYP
#   Pix2Sky_HPX
#   Pix2Sky_MER
#   Pix2Sky_MOL
#   Pix2Sky_PAR
#   Pix2Sky_PCO
#   Pix2Sky_QSC
#   Pix2Sky_SFL
#   Pix2Sky_SIN
#   Pix2Sky_STG
#   Pix2Sky_SZP
#   Pix2Sky_TAN
#   Pix2Sky_TSC
#   Pix2Sky_XPH
#   Pix2Sky_ZEA
#   Sky2Pix_AIR
#   Sky2Pix_AIT
#   Sky2Pix_ARC
#   Sky2Pix_AZP
#   Sky2Pix_BON
#   Sky2Pix_CAR
#   Sky2Pix_CEA
#   Sky2Pix_COD
#   Sky2Pix_COE
#   Sky2Pix_COO
#   Sky2Pix_COP
#   Sky2Pix_CSC
#   Sky2Pix_CYP
#   Sky2Pix_HPX
#   Sky2Pix_MER
#   Sky2Pix_MOL
#   Sky2Pix_PAR
#   Sky2Pix_PCO
#   Sky2Pix_QSC
#   Sky2Pix_SFL
#   Sky2Pix_SIN
#   Sky2Pix_STG
#   Sky2Pix_SZP
#   Sky2Pix_TAN
#   Sky2Pix_TSC
#   Sky2Pix_XPH
#   Sky2Pix_ZEA
