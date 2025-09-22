__all__ = ['kepler_fits_reader']

def kepler_fits_reader(filename, unit_parse_strict: str = 'warn'):
    '''
    This serves as the FITS reader for KEPLER or TESS files within
    astropy-timeseries.

    This function should generally not be called directly, and instead this
    time series reader should be accessed with the
    :meth:`~astropy.timeseries.TimeSeries.read` method::

        >>> from astropy.timeseries import TimeSeries
        >>> ts = TimeSeries.read(\'kplr33122.fits\', format=\'kepler.fits\')  # doctest: +SKIP

    Parameters
    ----------
    filename : `str` or `pathlib.Path`
        File to load.
    unit_parse_strict : str, optional
        Behaviour when encountering invalid column units in the FITS header.
        Default is "warn", which will emit a ``UnitsWarning`` and create a
        :class:`~astropy.units.core.UnrecognizedUnit`.
        Values are the ones allowed by the ``parse_strict`` argument of
        :class:`~astropy.units.core.Unit`: ``raise``, ``warn`` and ``silent``.

    Returns
    -------
    ts : `~astropy.timeseries.TimeSeries`
        Data converted into a TimeSeries.
    '''
