import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from healpy.newvisufunc import projview, newprojplot
from matplotlib.ticker import (MultipleLocator, AutoLocator)
from scipy.interpolate import interp1d
import pickle

#Define the function that loads all spherical harmonic coefficients given the path of the folder.
def load_all_dict(path):

    '''Be sure to dowload from the git page the spha_coeff folder: Then use the path of the dowloaded folder as an input for this function'''

    with open(path + 'alm_dict_N_ne.pkl', 'rb') as f:
        alm_dict_N_ne = pickle.load(f)

    with open(path + 'alm_dict_S_ne.pkl', 'rb') as f:
        alm_dict_S_ne = pickle.load(f)

    with open(path + 'alm_dict_N_gamma2.pkl', 'rb') as f:
        alm_dict_N_gamma2 = pickle.load(f)

    with open(path + 'alm_dict_S_gamma2.pkl', 'rb') as f:
        alm_dict_S_gamma2 = pickle.load(f)

    with open(path + 'alm_dict_N_rodi.pkl', 'rb') as f:
        alm_dict_N_rodi = pickle.load(f)

    with open(path + 'alm_dict_S_rodi.pkl', 'rb') as f:
        alm_dict_S_rodi = pickle.load(f)

    return alm_dict_N_ne, alm_dict_S_ne, alm_dict_N_gamma2, alm_dict_S_gamma2, alm_dict_N_rodi, alm_dict_S_rodi


# Define the function to interpolate spherical harmonic coefficients based on input conditions
def spha_coefficients_from_conditions(alm_dict, solar_activity_input, clock_angle_input, season_input):

    """
    Retrieves or interpolates spherical harmonic coefficients based on given solar activity, clock angle, and season.

    This function determines the appropriate spherical harmonic coefficients from a preloaded dictionary. If an exact 
    match for solar activity and clock angle is found, it returns the corresponding coefficients. Otherwise, it performs 
    circular interpolation for the clock angle and linear interpolation for solar activity to estimate the values.

    Parameters:
    alm_dict (dict): A dictionary containing spherical harmonic coefficients for different conditions.
    solar_activity_input (float): The solar activity level, mapped to predefined categories (e.g., LSA, MSA, HSA).
    clock_angle_input (float): The clock angle in radians, mapped to predefined categories.
    season_input (str): The season key used to access the appropriate coefficient subset.

    Returns:
    numpy.ndarray: The interpolated or directly retrieved spherical harmonic coefficients.

    Raises:
    KeyError: If the provided season_input is not found in the dictionary.
    ValueError: If the solar activity or clock angle is outside the expected range.

    Example:
    coefficients = spha_coefficients_from_conditions(alm_dict, 84, np.radians(90), "winter")

    PSA: This docstring has been written with the assistance of AI.
    """

    # Map of solar activity levels and corresponding numerical values
    solar_activity_levels = {'LSA': 70, 'MSA': 84, 'HSA': 133}
    # Map of clock angle tags and corresponding radian values
    clock_angle_values = {'UR': np.radians(30), 'R': np.radians(90), 'BR': np.radians(150),
                          'B': np.radians(210), 'BL': np.radians(270), 'L': np.radians(330)}
    
    # Get lists of solar activity values and clock angles
    solar_activities = list(solar_activity_levels.values())
    clock_angles = list(clock_angle_values.values())

    # Handle case when exact matches for solar activity and clock angle are found
    if solar_activity_input in solar_activities and clock_angle_input in clock_angles:
        # Get corresponding tags for exact matches
        solar_activity_tag = next(key for key, value in solar_activity_levels.items() if value == solar_activity_input)
        clock_angle_tag = next(key for key, value in clock_angle_values.items() if value == clock_angle_input)
        # Return the corresponding coefficients
        return alm_dict[solar_activity_tag][season_input][clock_angle_tag]

    # Handle case for circular interpolation of clock angles only
    if solar_activity_input in solar_activities and clock_angle_input not in clock_angles:
        solar_activity_tag = next(key for key, value in solar_activity_levels.items() if value == solar_activity_input)

        # Sort clock angles for circular interpolation and add a wrap-around value
        sorted_angles = np.array(clock_angles + [clock_angles[0] + 2 * np.pi])  # Wrap-around for circular interpolation
        sorted_keys = list(clock_angle_values.keys()) + [list(clock_angle_values.keys())[0]]  # Wrap-around tags

        # Find the closest surrounding angles for interpolation
        idx_upper = np.searchsorted(sorted_angles, clock_angle_input)
        clock_angle_lower = sorted_angles[idx_upper - 1]
        clock_angle_upper = sorted_angles[idx_upper]

        clock_angle_lower_tag = sorted_keys[idx_upper - 1]
        clock_angle_upper_tag = sorted_keys[idx_upper]

        # Retrieve corresponding coefficients for interpolation
        coef_upper = alm_dict[solar_activity_tag][season_input][clock_angle_upper_tag]
        coef_lower = alm_dict[solar_activity_tag][season_input][clock_angle_lower_tag]

        # Interpolate coefficients for the given clock angle
        CFI_clock_angle = np.zeros_like(coef_upper)
        for i in range(coef_upper.shape[0]):
            interp_func = interp1d([clock_angle_lower, clock_angle_upper], 
                                   [coef_lower[i], coef_upper[i]], 
                                   kind='linear', fill_value='extrapolate')
            CFI_clock_angle[i] = interp_func(clock_angle_input)

        return CFI_clock_angle
    
    # Handle case for interpolation of solar activity level only
    if solar_activity_input not in solar_activities and clock_angle_input in clock_angles:
        # Find the exact clock angle tag
        clock_angle_tag = next(key for key, value in clock_angle_values.items() if np.isclose(clock_angle_input, value))
        
        # Find the closest two solar activity levels for interpolation
        if solar_activity_input > max(solar_activities):  # Extrapolate above max value
            solar_activity_upper = sorted(solar_activities)[-1]
            solar_activity_lower = sorted(solar_activities)[-2]
        elif solar_activity_input < min(solar_activities):  # Extrapolate below min value
            solar_activity_upper = sorted(solar_activities)[1]
            solar_activity_lower = sorted(solar_activities)[0]
        else:  # Interpolate between closest solar activity levels
            solar_activity_upper = max([sa for sa in solar_activities if sa <= solar_activity_input])
            solar_activity_lower = min([sa for sa in solar_activities if sa >= solar_activity_input])

        # Get the solar activity tags
        solar_activity_upper_tag = next(key for key, value in solar_activity_levels.items() if value == solar_activity_upper)
        solar_activity_lower_tag = next(key for key, value in solar_activity_levels.items() if value == solar_activity_lower)

        # Retrieve coefficients for both solar activity levels
        coef_upper = alm_dict[solar_activity_upper_tag][season_input][clock_angle_tag]
        coef_lower = alm_dict[solar_activity_lower_tag][season_input][clock_angle_tag]

        # Interpolate over solar activity levels
        CFI = np.zeros_like(coef_upper)
        for i in range(coef_upper.shape[0]):
            interp_func = interp1d([solar_activity_lower, solar_activity_upper], 
                                [coef_lower[i], coef_upper[i]], 
                                kind='linear', fill_value='extrapolate')
            CFI[i] = interp_func(solar_activity_input)

        return CFI


    # Handle circular interpolation for both solar activity and clock angle
    if solar_activity_input not in solar_activities and clock_angle_input not in clock_angles:
        # Circular interpolation for clock angle
        sorted_angles = np.array(clock_angles + [clock_angles[0] + 2 * np.pi])
        sorted_keys = list(clock_angle_values.keys()) + [list(clock_angle_values.keys())[0]]

        idx_upper = np.searchsorted(sorted_angles, clock_angle_input)
        clock_angle_lower = sorted_angles[idx_upper - 1]
        clock_angle_upper = sorted_angles[idx_upper]

        clock_angle_lower_tag = sorted_keys[idx_upper - 1]
        clock_angle_upper_tag = sorted_keys[idx_upper]

        # Determine solar activity interpolation range
        if solar_activity_input > max(solar_activities):  # Extrapolate above max value
            solar_activity_upper = sorted(solar_activities)[-1]
            solar_activity_lower = sorted(solar_activities)[-2]
            extrapolate = True

        elif solar_activity_input < min(solar_activities):  # Extrapolate below min value
            solar_activity_upper = sorted(solar_activities)[1]
            solar_activity_lower = sorted(solar_activities)[0]
            extrapolate = True

        else:  # Interpolate between closest solar activity levels
            solar_activity_upper = max([sa for sa in solar_activities if sa <= solar_activity_input])
            solar_activity_lower = min([sa for sa in solar_activities if sa >= solar_activity_input])
            extrapolate = False

        solar_activity_upper_tag = next(key for key, value in solar_activity_levels.items() if value == solar_activity_upper)
        solar_activity_lower_tag = next(key for key, value in solar_activity_levels.items() if value == solar_activity_lower)

        # Retrieve coefficients for interpolation
        coef_upper_upper = alm_dict[solar_activity_upper_tag][season_input][clock_angle_upper_tag]
        coef_upper_lower = alm_dict[solar_activity_upper_tag][season_input][clock_angle_lower_tag]
        coef_lower_upper = alm_dict[solar_activity_lower_tag][season_input][clock_angle_upper_tag]
        coef_lower_lower = alm_dict[solar_activity_lower_tag][season_input][clock_angle_lower_tag]

        # Initialize arrays for interpolated coefficients
        CFI_solar_activity_lower = np.zeros_like(coef_upper_upper)
        CFI_solar_activity_upper = np.zeros_like(coef_upper_upper)
        CFI = np.zeros_like(coef_upper_upper)

        # Perform 2D interpolation: first over clock angles, then over solar activity levels
        for i in range(coef_lower_lower.shape[0]):
            # Interpolate over clock angles for lower solar activity level
            interp_func_I = interp1d([clock_angle_lower, clock_angle_upper], 
                                     [coef_lower_lower[i], coef_lower_upper[i]], 
                                     kind='linear', fill_value='extrapolate')
            CFI_solar_activity_lower[i] = interp_func_I(clock_angle_input)

            # Interpolate over clock angles for upper solar activity level
            interp_func_II = interp1d([clock_angle_lower, clock_angle_upper], 
                                      [coef_upper_lower[i], coef_upper_upper[i]], 
                                      kind='linear', fill_value='extrapolate')
            CFI_solar_activity_upper[i] = interp_func_II(clock_angle_input)

            # Interpolate over solar activity levels
            interp_func_III = interp1d([solar_activity_lower, solar_activity_upper], 
                                       [CFI_solar_activity_lower[i], CFI_solar_activity_upper[i]], 
                                       kind='linear', fill_value='extrapolate')
            CFI[i] = interp_func_III(solar_activity_input)

        return CFI
    
