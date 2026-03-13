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

    # placeholder function, theoretical operation is as follows:
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
        An array describing the shape, size, and location of the occluder's
        shadow on the solar field. The array is scaled and translated to match
        the total size of the solar field. Values may range from 0 to 1, where
        0 indicates no shading and 1 indicates complete shading.

    Returns
    ---------------
    hel_imgs_soiled : dict
        A dictionary of modified heliostat flux images.
    fluxgrid_soiled : np.ndarray
        An array of the new ideal flux values after soiling.
    """

    # placeholder function, theoretical operation is as follows:
    # 1) create a copy of hel_imgs
    # 2) scale and translate the occluder image so it matches the proportions
    #    of the solar field
    # 3) identify which heliostats are effected by the occluder and scale their
    #    flux image according to the occlusion factor
    # 4) calculate fractional loss of power due to shading
    # 3) copy and scale fluxgrid based on fractional loss of power
    # 4) return new hel_imgs and fluxgrid as tuple
    #
    # NOTE, as written the function scales the image to match the total size of
    # the solar field. It may be easier to specify the dimensions and location
    # of the occluder instead.

    pass

# EOF
