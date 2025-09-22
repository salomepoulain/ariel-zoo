__all__ = ['NDSlicingMixin']

class NDSlicingMixin:
    """Mixin to provide slicing on objects using the `NDData`
    interface.

    The ``data``, ``mask``, ``uncertainty`` and ``wcs`` will be sliced, if
    set and sliceable. The ``unit`` and ``meta`` will be untouched. The return
    will be a reference and not a copy, if possible.

    Examples
    --------
    Using this Mixin with `~astropy.nddata.NDData`:

        >>> from astropy.nddata import NDData, NDSlicingMixin
        >>> class NDDataSliceable(NDSlicingMixin, NDData):
        ...     pass

    Slicing an instance containing data::

        >>> nd = NDDataSliceable([1,2,3,4,5])
        >>> nd[1:3]
        NDDataSliceable([2, 3])

    Also the other attributes are sliced for example the ``mask``::

        >>> import numpy as np
        >>> mask = np.array([True, False, True, True, False])
        >>> nd2 = NDDataSliceable(nd, mask=mask)
        >>> nd2slc = nd2[1:3]
        >>> nd2slc[nd2slc.mask]
        NDDataSliceable([—])

    Be aware that changing values of the sliced instance will change the values
    of the original::

        >>> nd3 = nd2[1:3]
        >>> nd3.data[0] = 100
        >>> nd2
        NDDataSliceable([———, 100, ———, ———,   5])

    See Also
    --------
    NDDataRef
    NDDataArray
    """
    def __getitem__(self, item): ...
    def _slice(self, item):
        """Collects the sliced attributes and passes them back as `dict`.

        It passes uncertainty, mask and wcs to their appropriate ``_slice_*``
        method, while ``meta`` and ``unit`` are simply taken from the original.
        The data is assumed to be sliceable and is sliced directly.

        When possible the return should *not* be a copy of the data but a
        reference.

        Parameters
        ----------
        item : slice
            The slice passed to ``__getitem__``.

        Returns
        -------
        dict :
            Containing all the attributes after slicing - ready to
            use them to create ``self.__class__.__init__(**kwargs)`` in
            ``__getitem__``.
        """
    def _slice_uncertainty(self, item): ...
    def _slice_mask(self, item): ...
    def _slice_wcs(self, item): ...
    def _handle_wcs_slicing_error(self, err, item) -> None: ...
