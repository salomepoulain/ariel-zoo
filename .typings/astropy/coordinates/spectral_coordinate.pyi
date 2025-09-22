from _typeshed import Incomplete
from astropy.coordinates.spectral_quantity import SpectralQuantity
from astropy.utils.exceptions import AstropyUserWarning

__all__ = ['SpectralCoord']

class NoVelocityWarning(AstropyUserWarning): ...
class NoDistanceWarning(AstropyUserWarning): ...

class SpectralCoord(SpectralQuantity):
    """
    A spectral coordinate with its corresponding unit.

    .. note:: The |SpectralCoord| class is new in Astropy v4.1 and should be
              considered experimental at this time. Note that we do not fully
              support cases where the observer and target are moving
              relativistically relative to each other, so care should be taken
              in those cases. It is possible that there will be API changes in
              future versions of Astropy based on user feedback. If you have
              specific ideas for how it might be improved, please  let us know
              on the |astropy-dev mailing list| or at
              http://feedback.astropy.org.

    Parameters
    ----------
    value : ndarray or `~astropy.units.Quantity` or `SpectralCoord`
        Spectral values, which should be either wavelength, frequency,
        energy, wavenumber, or velocity values.
    unit : unit-like
        Unit for the given spectral values.
    observer : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`, optional
        The coordinate (position and velocity) of observer. If no velocities
        are present on this object, the observer is assumed to be stationary
        relative to the frame origin.
    target : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`, optional
        The coordinate (position and velocity) of target. If no velocities
        are present on this object, the target is assumed to be stationary
        relative to the frame origin.
    radial_velocity : `~astropy.units.Quantity` ['speed'], optional
        The radial velocity of the target with respect to the observer. This
        can only be specified if ``redshift`` is not specified.
    redshift : float, optional
        The relativistic redshift of the target with respect to the observer.
        This can only be specified if ``radial_velocity`` cannot be specified.
    doppler_rest : `~astropy.units.Quantity`, optional
        The rest value to use when expressing the spectral value as a velocity.
    doppler_convention : str, optional
        The Doppler convention to use when expressing the spectral value as a velocity.
    """
    def __new__(cls, value, unit: Incomplete | None = None, observer: Incomplete | None = None, target: Incomplete | None = None, radial_velocity: Incomplete | None = None, redshift: Incomplete | None = None, **kwargs): ...
    _radial_velocity: Incomplete
    _observer: Incomplete
    _target: Incomplete
    def __array_finalize__(self, obj) -> None: ...
    @staticmethod
    def _validate_coordinate(coord, label: str = ''):
        '''
        Checks the type of the frame and whether a velocity differential and a
        distance has been defined on the frame object.

        If no distance is defined, the target is assumed to be "really far
        away", and the observer is assumed to be "in the solar system".

        Parameters
        ----------
        coord : `~astropy.coordinates.BaseCoordinateFrame`
            The new frame to be used for target or observer.
        label : str, optional
            The name of the object being validated (e.g. \'target\' or \'observer\'),
            which is then used in error messages.
        '''
    def replicate(self, value: Incomplete | None = None, unit: Incomplete | None = None, observer: Incomplete | None = None, target: Incomplete | None = None, radial_velocity: Incomplete | None = None, redshift: Incomplete | None = None, doppler_convention: Incomplete | None = None, doppler_rest: Incomplete | None = None, copy: bool = False):
        """
        Return a replica of the `SpectralCoord`, optionally changing the
        values or attributes.

        Note that no conversion is carried out by this method - this keeps
        all the values and attributes the same, except for the ones explicitly
        passed to this method which are changed.

        If ``copy`` is set to `True` then a full copy of the internal arrays
        will be made.  By default the replica will use a reference to the
        original arrays when possible to save memory.

        Parameters
        ----------
        value : ndarray or `~astropy.units.Quantity` or `SpectralCoord`, optional
            Spectral values, which should be either wavelength, frequency,
            energy, wavenumber, or velocity values.
        unit : unit-like
            Unit for the given spectral values.
        observer : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`, optional
            The coordinate (position and velocity) of observer.
        target : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`, optional
            The coordinate (position and velocity) of target.
        radial_velocity : `~astropy.units.Quantity` ['speed'], optional
            The radial velocity of the target with respect to the observer.
        redshift : float, optional
            The relativistic redshift of the target with respect to the observer.
        doppler_rest : `~astropy.units.Quantity`, optional
            The rest value to use when expressing the spectral value as a velocity.
        doppler_convention : str, optional
            The Doppler convention to use when expressing the spectral value as a velocity.
        copy : bool, optional
            If `True`, and ``value`` is not specified, the values are copied to
            the new `SkyCoord` - otherwise a reference to the same values is used.

        Returns
        -------
        sc : `SpectralCoord` object
            Replica of this object
        """
    @property
    def quantity(self):
        """
        Convert the ``SpectralCoord`` to a `~astropy.units.Quantity`.
        Equivalent to ``self.view(u.Quantity)``.

        Returns
        -------
        `~astropy.units.Quantity`
            This object viewed as a `~astropy.units.Quantity`.

        """
    @property
    def observer(self):
        """
        The coordinates of the observer.

        If set, and a target is set as well, this will override any explicit
        radial velocity passed in.

        Returns
        -------
        `~astropy.coordinates.BaseCoordinateFrame`
            The astropy coordinate frame representing the observation.
        """
    @observer.setter
    def observer(self, value) -> None: ...
    @property
    def target(self):
        """
        The coordinates of the target being observed.

        If set, and an observer is set as well, this will override any explicit
        radial velocity passed in.

        Returns
        -------
        `~astropy.coordinates.BaseCoordinateFrame`
            The astropy coordinate frame representing the target.
        """
    @target.setter
    def target(self, value) -> None: ...
    @property
    def radial_velocity(self):
        """
        Radial velocity of target relative to the observer.

        Returns
        -------
        `~astropy.units.Quantity` ['speed']
            Radial velocity of target.

        Notes
        -----
        This is different from the ``.radial_velocity`` property of a
        coordinate frame in that this calculates the radial velocity with
        respect to the *observer*, not the origin of the frame.
        """
    @property
    def redshift(self):
        """
        Redshift of target relative to observer. Calculated from the radial
        velocity.

        Returns
        -------
        `astropy.units.Quantity`
            Redshift of target.
        """
    @staticmethod
    def _calculate_radial_velocity(observer, target, as_scalar: bool = False):
        """
        Compute the line-of-sight velocity from the observer to the target.

        Parameters
        ----------
        observer : `~astropy.coordinates.BaseCoordinateFrame`
            The frame of the observer.
        target : `~astropy.coordinates.BaseCoordinateFrame`
            The frame of the target.
        as_scalar : bool
            If `True`, the magnitude of the velocity vector will be returned,
            otherwise the full vector will be returned.

        Returns
        -------
        `~astropy.units.Quantity` ['speed']
            The radial velocity of the target with respect to the observer.
        """
    @staticmethod
    def _normalized_position_vector(observer, target):
        """
        Calculate the normalized position vector between two frames.

        Parameters
        ----------
        observer : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`
            The observation frame or coordinate.
        target : `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`
            The target frame or coordinate.

        Returns
        -------
        pos_hat : `BaseRepresentation`
            Position representation.
        """
    def with_observer_stationary_relative_to(self, frame, velocity: Incomplete | None = None, preserve_observer_frame: bool = False):
        """
        A new  `SpectralCoord` with the velocity of the observer altered,
        but not the position.

        If a coordinate frame is specified, the observer velocities will be
        modified to be stationary in the specified frame. If a coordinate
        instance is specified, optionally with non-zero velocities, the
        observer velocities will be updated so that the observer is co-moving
        with the specified coordinates.

        Parameters
        ----------
        frame : str, `~astropy.coordinates.BaseCoordinateFrame` or `~astropy.coordinates.SkyCoord`
            The observation frame in which the observer will be stationary. This
            can be the name of a frame (e.g. 'icrs'), a frame class, frame instance
            with no data, or instance with data. This can optionally include
            velocities.
        velocity : `~astropy.units.Quantity` or `~astropy.coordinates.CartesianDifferential`, optional
            If ``frame`` does not contain velocities, these can be specified as
            a 3-element `~astropy.units.Quantity`. In the case where this is
            also not specified, the velocities default to zero.
        preserve_observer_frame : bool
            If `True`, the final observer frame class will be the same as the
            original one, and if `False` it will be the frame of the velocity
            reference class.

        Returns
        -------
        new_coord : `SpectralCoord`
            The new coordinate object representing the spectral data
            transformed based on the observer's new velocity frame.
        """
    def with_radial_velocity_shift(self, target_shift: Incomplete | None = None, observer_shift: Incomplete | None = None):
        """
        Apply a velocity shift to this spectral coordinate.

        The shift can be provided as a redshift (float value) or radial
        velocity (`~astropy.units.Quantity` with physical type of 'speed').

        Parameters
        ----------
        target_shift : float or `~astropy.units.Quantity` ['speed']
            Shift value to apply to current target.
        observer_shift : float or `~astropy.units.Quantity` ['speed']
            Shift value to apply to current observer.

        Returns
        -------
        `SpectralCoord`
            New spectral coordinate with the target/observer velocity changed
            to incorporate the shift. This is always a new object even if
            ``target_shift`` and ``observer_shift`` are both `None`.
        """
    def to_rest(self):
        """
        Transforms the spectral axis to the rest frame.
        """
    def __repr__(self) -> str: ...
