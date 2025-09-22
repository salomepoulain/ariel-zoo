import abc
from .utils import convert_parameter_to_model_parameter as convert_parameter_to_model_parameter
from astropy.cosmology._typing import _CosmoT as _CosmoT
from astropy.cosmology.connect import convert_registry as convert_registry
from astropy.cosmology.core import Cosmology as Cosmology
from astropy.modeling import FittableModel as FittableModel, Model as Model
from astropy.utils.decorators import classproperty as classproperty
from typing import Generic

__all__: list[str]

class _CosmologyModel(FittableModel, Generic[_CosmoT], metaclass=abc.ABCMeta):
    '''Base class for Cosmology redshift-method Models.

    .. note::

        This class is not publicly scoped so should not be used directly.
        Instead, from a Cosmology instance use ``.to_format("astropy.model")``
        to create an instance of a subclass of this class.

    `_CosmologyModel` (subclasses) wrap a redshift-method of a
    :class:`~astropy.cosmology.Cosmology` class, converting each non-`None`
    |Cosmology| :class:`~astropy.cosmology.Parameter` to a
    :class:`astropy.modeling.Model` :class:`~astropy.modeling.Parameter`
    and the redshift-method to the model\'s ``__call__ / evaluate``.

    See Also
    --------
    astropy.cosmology.Cosmology.to_format
    '''
    @abc.abstractmethod
    def _cosmology_class(self) -> type[_CosmoT]:
        """Cosmology class as a private attribute.

        Set in subclasses.
        """
    @abc.abstractmethod
    def _method_name(self) -> str:
        """Cosmology method name as a private attribute.

        Set in subclasses.
        """
    def cosmology_class(cls) -> type[_CosmoT]:
        """|Cosmology| class."""
    def _cosmology_class_sig(cls):
        """Signature of |Cosmology| class."""
    @property
    def cosmology(self) -> _CosmoT:
        """Return |Cosmology| using `~astropy.modeling.Parameter` values."""
    def method_name(self) -> str:
        """Redshift-method name on |Cosmology| instance."""
    def evaluate(self, *args, **kwargs):
        """Evaluate method {method!r} of {cosmo_cls!r} Cosmology.

        The Model wraps the :class:`~astropy.cosmology.Cosmology` method,
        converting each |Cosmology| :class:`~astropy.cosmology.Parameter` to a
        :class:`astropy.modeling.Model` :class:`~astropy.modeling.Parameter`
        (unless the Parameter is None, in which case it is skipped).
        Here an instance of the cosmology is created using the current
        Parameter values and the method is evaluated given the input.

        Parameters
        ----------
        *args, **kwargs
            The first ``n_inputs`` of ``*args`` are for evaluating the method
            of the cosmology. The remaining args and kwargs are passed to the
            cosmology class constructor.
            Any unspecified Cosmology Parameter use the current value of the
            corresponding Model Parameter.

        Returns
        -------
        Any
            Results of evaluating the Cosmology method.
        """

def from_model(model: _CosmologyModel[_CosmoT]) -> _CosmoT:
    '''Load |Cosmology| from `~astropy.modeling.Model` object.

    Parameters
    ----------
    model : `_CosmologyModel` subclass instance
        See ``Cosmology.to_format.help("astropy.model") for details.

    Returns
    -------
    `~astropy.cosmology.Cosmology` subclass instance

    Examples
    --------
    >>> from astropy.cosmology import Cosmology, Planck18
    >>> model = Planck18.to_format("astropy.model", method="lookback_time")
    >>> print(Cosmology.from_format(model))
    FlatLambdaCDM(name="Planck18", H0=67.66 km / (Mpc s), Om0=0.30966,
                  Tcmb0=2.7255 K, Neff=3.046, m_nu=[0. 0. 0.06] eV, Ob0=0.04897)
    '''
def to_model(cosmology: _CosmoT, *_: object, method: str) -> _CosmologyModel[_CosmoT]:
    '''Convert a `~astropy.cosmology.Cosmology` to a `~astropy.modeling.Model`.

    Parameters
    ----------
    cosmology : `~astropy.cosmology.Cosmology` subclass instance
    method : str, keyword-only
        The name of the method on the ``cosmology``.

    Returns
    -------
    `_CosmologyModel` subclass instance
        The Model wraps the |Cosmology| method, converting each non-`None`
        :class:`~astropy.cosmology.Parameter` to a
        :class:`astropy.modeling.Model` :class:`~astropy.modeling.Parameter`
        and the method to the model\'s ``__call__ / evaluate``.

    Examples
    --------
    >>> from astropy.cosmology import Planck18
    >>> model = Planck18.to_format("astropy.model", method="lookback_time")
    >>> model
    <FlatLambdaCDMCosmologyLookbackTimeModel(H0=67.66 km / (Mpc s), Om0=0.30966,
        Tcmb0=2.7255 K, Neff=3.046, m_nu=[0.  , 0.  , 0.06] eV, Ob0=0.04897,
        name=\'Planck18\')>
    '''
def model_identify(origin: str, format: str | None, *args: object, **kwargs: object) -> bool:
    """Identify if object uses the :class:`~astropy.modeling.Model` format.

    Returns
    -------
    bool
    """
