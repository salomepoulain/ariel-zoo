from _typeshed import Incomplete
from astropy.io.fits.header import Header

__all__ = ['CompImageHeader', '_bintable_header_to_image_header', '_image_header_to_empty_bintable']

class CompImageHeader(Header):
    def __init__(self, *args, **kwargs) -> None: ...

def _bintable_header_to_image_header(bintable_header): ...
def _image_header_to_empty_bintable(image_header, name: Incomplete | None = None, huge_hdu: bool = False, compression_type: Incomplete | None = None, tile_shape: Incomplete | None = None, hcomp_scale: Incomplete | None = None, hcomp_smooth: Incomplete | None = None, quantize_level: Incomplete | None = None, quantize_method: Incomplete | None = None, dither_seed: Incomplete | None = None, axes: Incomplete | None = None, generate_dither_seed: Incomplete | None = None): ...
