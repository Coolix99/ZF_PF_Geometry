import logging
import tifffile

logger = logging.getLogger(__name__)

def get_Image(file):
    with tifffile.TiffFile(file) as tif:
        try:
            image = tif.asarray()
        except Exception as e:
            logger.error(f"Error reading image from {file}: {e}")
            return None
        return image
