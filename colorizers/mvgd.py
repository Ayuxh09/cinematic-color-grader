import numpy as np
from color_matcher import ColorMatcher


class TransferMVGD:
    def __init__(self):
        self.matcher = ColorMatcher()

    def transform(self, src, ref):
        """
        Apply MVGD color transfer from reference image to source image.

        Args:
            src (np.ndarray): Source image as float32, [0..1]
            ref (np.ndarray): Reference image as float32, [0..1]

        Returns:
            np.ndarray: Color transferred image [0..1]
        """
        # ColorMatcher expects 8-bit images
        src_uint8 = (src * 255).astype(np.uint8)
        ref_uint8 = (ref * 255).astype(np.uint8)

        result_uint8 = self.matcher.transfer(src_uint8, ref_uint8, method="mvgd")

        # Back to float32 [0..1]
        return result_uint8.astype(np.float32) / 255.0
