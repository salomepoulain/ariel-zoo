from _typeshed import Incomplete

__all__ = ['time_support']

def time_support(*, scale: Incomplete | None = None, format: Incomplete | None = None, simplify: bool = True):
    """
    Enable support for plotting `astropy.time.Time` instances in
    matplotlib.

    May be (optionally) used with a ``with`` statement.

      >>> import matplotlib.pyplot as plt
      >>> from astropy import units as u
      >>> from astropy import visualization
      >>> with visualization.time_support():  # doctest: +IGNORE_OUTPUT
      ...     plt.figure()
      ...     plt.plot(Time(['2016-03-22T12:30:31', '2016-03-22T12:30:38', '2016-03-22T12:34:40']))
      ...     plt.draw()

    Parameters
    ----------
    scale : str, optional
        The time scale to use for the times on the axis. If not specified,
        the scale of the first Time object passed to Matplotlib is used.
    format : str, optional
        The time format to use for the times on the axis. If not specified,
        the format of the first Time object passed to Matplotlib is used.
    simplify : bool, optional
        If possible, simplify labels, e.g. by removing 00:00:00.000 times from
        ISO strings if all labels fall on that time.
    """
