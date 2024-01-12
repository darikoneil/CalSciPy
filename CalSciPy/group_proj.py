
@validate_longest_numpy_dimension(axis=0, pos=0)
def grouped_z_project(images: np.ndarray, bin_size: Union[Tuple[int, int, int], int],
                      function: Callable = np.mean) -> np.ndarray:
    """
    Utilize grouped z-project to downsample data

    Downsample example function -> np.mean

    :param images: A numpy array containing a tiff stack (frames, y pixels, x pixels)
    :type images: numpy.ndarray
    :param bin_size:  size of each bin passed to downsampling function
    :type bin_size: tuple[int, int, int] or tuple[int]
    :param function: group-z projecting function
    :type function: Callable = np.mean
    :return: downsampled image (frames, y pixels, x pixels)
    :rtype: numpy.ndarray
    """
    return skimage.measure.block_reduce(images, block_size=bin_size, func=function).astype(images.dtype)