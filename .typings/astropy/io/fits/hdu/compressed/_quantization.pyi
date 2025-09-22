from _typeshed import Incomplete

__all__ = ['Quantize']

class QuantizationFailedException(Exception): ...

class Quantize:
    """
    Quantization of floating-point data following the FITS standard.
    """
    row: Incomplete
    quantize_level: Incomplete
    dither_method: Incomplete
    bitpix: Incomplete
    def __init__(self, *, row: int, dither_method: int, quantize_level: int, bitpix: int) -> None: ...
    def decode_quantized(self, buf, scale, zero):
        """
        Unquantize data.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to unquantize.

        Returns
        -------
        np.ndarray
            The unquantized buffer.
        """
    def encode_quantized(self, buf):
        """
        Quantize data.

        Parameters
        ----------
        buf : bytes or array_like
            The buffer to quantize.

        Returns
        -------
        np.ndarray
            A buffer with quantized data.
        """
