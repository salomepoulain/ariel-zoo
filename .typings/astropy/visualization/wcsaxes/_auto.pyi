__all__ = ['auto_assign_coord_positions']

def auto_assign_coord_positions(ax) -> None:
    """
    Given a ``WCSAxes`` instance, automatically update any dynamic tick, tick
    label and axis label positions.

    This function operates in-place on the axes and assumes that
    ``_update_ticks`` has already been called on all the ``CoordinateHelper``
    instances.
    """
