import tifffile

def get_Image(file):
    with tifffile.TiffFile(file) as tif:
        try:
            image=tif.asarray()
        except:
            return None
        return image
