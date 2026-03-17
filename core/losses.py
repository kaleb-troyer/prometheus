# Exposes a set of functions which accept parameters such as solar field
# layout, heliostat flux images, and ideal flux profile, and return modified
# copies of these, based on factors such as soiling or shading.
# 2026-03-13
# Kaleb Troyer

import numpy as np

def hel_soiling(
    hel_imgs: dict,
    fluxgrid: np.ndarray,
    hel_soil: dict,
) -> tuple[dict, np.ndarray]:
    """
    Returns modified copies of each heliostat image and the fluxgrid according
    to the provided heliostat soiling factors.

    Parameters
    ---------------
    hel_imgs : dict
        A dictionary of heliostats (keys) and arrays (values) of incident flux
        on the receiver.
    fluxgrid : np.ndarray
        An array of the ideal flux values incident on the receiver before
        soiling.
    hel_soil : dict
        A dictionary of heliostats (keys) and their soiling factor (values)
        from 0 to 1.

    Returns
    ---------------
    hel_imgs_soiled : dict
        A dictionary of modified heliostat flux images.
    fluxgrid_soiled : np.ndarray
        An array of the new ideal flux values after soiling.
    """

    # placeholder function, theoretical usage is as follows:
    # 1) copy and scale hel_imgs using factors from hel_soil
    # 2) calculate fractional loss of power due to soiling using
    #    copy of hel_imgs and original
    # 3) copy and scale fluxgrid based on fractional loss of power
    # 4) return new hel_imgs and fluxgrid as tuple

    pass

def hel_shading(
    hel_imgs: dict,
    fluxgrid: np.ndarray,
    fld_layout: dict,
    occluder: np.ndarray,
    dimensions: tuple,
    location: tuple,
) -> tuple[dict, np.ndarray]:
    """
    Returns modified copies of each heliostat image and the fluxgrid according
    to the provided occluder and occlusion factor.

    Parameters
    ---------------
    hel_imgs : dict
        A dictionary of heliostats (keys) and arrays (values) of incident flux
        on the receiver.
    fluxgrid : np.ndarray
        An array of the ideal flux values incident on the receiver before
        soiling.
    fld_layout : dict
        A dictionary of heliostats (keys) and their (x, y) coordinates (values)
        relative to the solar tower.
    occluder : np.ndarray
        An array describing the shape of the occluder's shadow on the solar
        field. The array is scaled and translated to match the dimensions and
        location variables. Values may range from 0 to 1, where 0 indicates no
        shading and 1 indicates complete shading.
    dimensions : np.ndarray
        A tuple of the x and y dimensions of the occluder, in meters.
    location : np.ndarray
        A tuple of the x and y coordinates of the occluder, in meters.

    Returns
    ---------------
    hel_imgs_soiled : dict
        A dictionary of modified heliostat flux images.
    fluxgrid_soiled : np.ndarray
        An array of the new ideal flux values after soiling.
    """

    # placeholder function, theoretical usage is as follows:
    # 1) create a copy of hel_imgs
    # 2) scale, translate, and apply the occluder image
    # 3) identify which heliostats are effected by the occluder and scale their
    #    flux image values according to the occlusion factor
    # 4) calculate fractional loss of power due to shading
    # 3) copy and scale ideal fluxgrid based on fractional loss of power
    # 4) return new hel_imgs and fluxgrid as tuple

    pass

def img_to_occl(
    file: str,
) -> np.ndarray:
    """
    Converts a grayscale image into an array of occlusion values from 0 to 1.

    Parameters
    ---------------
    file : str
        Complete path to the grayscale occlusion image.

    Returns
    ---------------
    occluder : np.ndarray
        An array describing the shape of the occluder's shadow on the solar
        field.
    """

    # placeholder function, theoretical usage is as follows:
    # 1) load a grayscale image file
    # 2) convert the image into an NxM array, where N and M are the dimensions
    #    of the image and each entry corresponds to a pixel and value.

    pass

# EOF
