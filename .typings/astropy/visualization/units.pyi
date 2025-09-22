__all__ = ['quantity_support']

def quantity_support(format: str = 'latex_inline'):
    """
    Enable support for plotting `astropy.units.Quantity` instances in
    matplotlib.

    May be (optionally) used with a ``with`` statement.

      >>> import matplotlib.pyplot as plt
      >>> from astropy import units as u
      >>> from astropy import visualization
      >>> with visualization.quantity_support():
      ...     plt.figure()
      ...     plt.plot([1, 2, 3] * u.m)
      [...]
      ...     plt.plot([101, 125, 150] * u.cm)
      [...]
      ...     plt.draw()

    Parameters
    ----------
    format : `astropy.units.format.Base` subclass or str
        The name of a format or a formatter class.  If not
        provided, defaults to ``latex_inline``.

    """
