"""IEA Task 37 Combined Case Study AEP Calculation Code

Written by Nicholas F. Baker, PJ Stanley, and Jared Thomas (BYU FLOW lab)
Created 10 June 2018
Updated 11 Jul 2018 to include read-in of .yaml turb locs and wind freq dist.
Completed 26 Jul 2018 for commenting and release
Modified 22 Aug 2018 implementing multiple suggestions from Erik Quaeghebeur.
Modified 25 Sep 2025 implementing vectorization for GaussianWake function.
"""

from __future__ import print_function   # For Python 3 compatibility
import numpy as np
import sys
import yaml                             # For reading .yaml files
from math import radians as DegToRad    # For converting degrees to radians


# Structured datatype for holding coordinate pair
coordinate = np.dtype([('x', 'f8'), ('y', 'f8')])


def WindFrame(turb_coords, wind_dir_deg):
    """Convert map coordinates to downwind/crosswind coordinates."""
    
    # Convert from meteorological polar system (CW, 0 deg.=N) to standard polar system (CCW, 0 deg.=W)
    wind_dir_deg = 270. - wind_dir_deg
    # Convert inflow wind direction from degrees to radians
    wind_dir_rad = np.radians(wind_dir_deg)
    
    # Constants to use below
    cos_dir = np.cos(-wind_dir_rad)
    sin_dir = np.sin(-wind_dir_rad)
    
    # Ensure turb_coords is a numpy array
    turb_coords = np.array(turb_coords)
    
    # Create an empty array with the same shape as turb_coords but with the dtype for coordinate
    frame_coords = np.empty(turb_coords.shape)
    
    # Convert to downwind(x) & crosswind(y) coordinates
    frame_coords[:, 0] = (turb_coords[:, 0] * cos_dir) - (turb_coords[:, 1] * sin_dir)
    frame_coords[:, 1] = (turb_coords[:, 0] * sin_dir) + (turb_coords[:, 1] * cos_dir)
    
    return frame_coords


def GaussianWake_vetorizado_optimizado(frame_coords, turb_diam):
    """
    Return the total wake loss for each turbine due to upstream wakes,
    using vectorized operations for high performance and precision.
    """
    # Ensure double precision
    frame_coords = frame_coords.astype(np.float64)
    num_turb = len(frame_coords)

    # Constants
    CT = 4.0 * (1. / 3.) * (1.0 - 1. / 3.)
    k = 0.0324555

    # Extract coordinates and reshape to column vectors
    x_coords = frame_coords[:, 0].reshape(-1, 1)
    y_coords = frame_coords[:, 1].reshape(-1, 1)

    # Calculate pairwise differences between all turbines
    x_diff = x_coords.T - x_coords   # X-coordinate difference matrix
    y_diff = y_coords.T - y_coords   # Y-coordinate difference matrix

    # Mask for turbines where the primary turbine is downwind of the target (x_diff > 0)
    mask = x_diff > 0

    # Initialize sigma with zeros and calculate only for valid downwind cases
    sigma = np.zeros_like(x_diff)
    sigma[mask] = k * x_diff[mask] + turb_diam / np.sqrt(8.)

    # Calculate exponent only where mask is True
    exponent = np.zeros_like(sigma)
    exponent[mask] = -0.5 * (y_diff[mask] / sigma[mask])**2

    # Calculate radical (Bastankhah model factor) only for valid downwind cases
    # For turbines not downwind, we set a neutral value (1.0)
    radical = np.ones_like(sigma)
    radical[mask] = 1. - CT / (8. * sigma[mask]**2 / turb_diam**2)

    # To prevent negative values inside the square root, apply np.maximum and compute sqrt
    radical_val = np.ones_like(sigma)
    radical_val[mask] = np.sqrt(np.maximum(radical[mask], 0))

    # Calculate loss matrix using the wake model equation
    loss_matrix = np.zeros_like(sigma)
    loss_matrix[mask] = (1. - radical_val[mask]) * np.exp(exponent[mask])

    # Aggregate losses for each turbine using the root-sum-of-squares (RSS) of deficits
    loss = np.sqrt(np.sum(loss_matrix**2, axis=1))

    return loss


def calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
            turb_diam, turb_ci, turb_co, rated_ws, rated_pwr):
    """Calculate the wind farm AEP."""
    num_bins = len(wind_freq)  # Number of bins used for our windrose

    #  Power produced by the wind farm from each wind direction
    pwr_produced = np.zeros(num_bins)
    # For each wind bin
    for i in range(num_bins):
        # Find the farm's power for the current direction
        pwr_produced[i] = DirPower(turb_coords, wind_dir[i], wind_speed[i],
                                   turb_diam, turb_ci, turb_co,
                                   rated_ws, rated_pwr)

    #  Convert power to AEP
    hrs_per_year = 365.*24.
    AEP = hrs_per_year * (wind_freq * pwr_produced)
    AEP /= 1.E6  # Convert to MWh

    return AEP



################################################################## DIR POWER ####################################################################
def DirPower(turb_coords, wind_dir_deg, wind_speed,
             turb_diam, turb_ci, turb_co, rated_ws, rated_pwr):
    # Return the power produced by each turbine.
    num_turb = len(turb_coords)

    # Shift coordinate frame of reference to downwind/crosswind
    frame_coords = WindFrame(turb_coords, wind_dir_deg)
    
    # Use the Simplified Bastankhah Gaussian wake model for wake deficits (vectorized version)
    loss = GaussianWake_vetorizado_optimizado(frame_coords, turb_diam)
    
    # Effective windspeed is freestream multiplied by wake deficits
    wind_speed_eff = wind_speed*(1.-loss)
    # By default, the turbine's power output is zero
    turb_pwr = np.zeros(num_turb)

    # Check to see if turbine produces power for experienced wind speed
    for n in range(num_turb):
        # If we're between the cut-in and rated wind speeds
        if ((turb_ci <= wind_speed_eff[n])
                and (wind_speed_eff[n] < rated_ws)):
            # Calculate the curve's power
            turb_pwr[n] = rated_pwr * ((wind_speed_eff[n]-turb_ci)
                                       / (rated_ws-turb_ci))**3
        # If we're between the rated and cut-out wind speeds
        elif ((rated_ws <= wind_speed_eff[n])
                and (wind_speed_eff[n] < turb_co)):
            # Produce the rated power
            turb_pwr[n] = rated_pwr

    # Sum the power from all turbines for this direction
    pwrDir = np.sum(turb_pwr)

    return pwrDir

def getTurbLocYAML(file_name):
    """Retrieve turbine locations from a simplified <.yaml> layout file."""
    with open(file_name, 'r') as f:
        data = yaml.safe_load(f)
    
    turb_xc = np.asarray(data['xc'])
    turb_yc = np.asarray(data['yc'])
    return np.column_stack((turb_xc, turb_yc))


def getWindRoseYAML(file_name):
    """Retrieve wind rose data (bins, freqs, speeds) from <.yaml> file."""
    # Read in the .yaml file
    with open(file_name, 'r') as f:
        props = yaml.safe_load(f)['definitions']['wind_inflow']['properties']

    # Rip wind directional bins, their frequency, and the farm windspeed
    # (Convert from <list> to <ndarray>)
    wind_dir = np.asarray(props['direction']['bins'])
    wind_freq = np.asarray(props['probability']['default'])
    # (Convert from <list> to <ndarray>)
    wind_speed = np.asarray(props['speed']['default'])

    return wind_dir, wind_freq, wind_speed


def getTurbAtrbtYAML(file_name):
    '''Retreive turbine attributes from the <.yaml> file'''
    # Read in the .yaml file
    with open(file_name, 'r') as f:
        defs = yaml.safe_load(f)['definitions']
        op_props = defs['operating_mode']['properties']
        turb_props = defs['wind_turbine_lookup']['properties']
        rotor_props = defs['rotor']['properties']

    # Rip the turbine attributes
    # (Convert from <list> to <float>)
    turb_ci = float(op_props['cut_in_wind_speed']['default'])
    turb_co = float(op_props['cut_out_wind_speed']['default'])
    rated_ws = float(op_props['rated_wind_speed']['default'])
    rated_pwr = float(turb_props['power']['maximum'])
    turb_diam = float(rotor_props['radius']['default']) * 2.

    return turb_ci, turb_co, rated_ws, rated_pwr, turb_diam


if __name__ == "__main__":
    """Used for demonstration.

    An example command line syntax to run this file is:

        python iea37-aepcalc.py iea37-ex16.yaml

    For Python .yaml capability, in the terminal type "pip install pyyaml".
    """
    # Read necessary values from .yaml files
    # Get turbine locations
    turb_coords = getTurbLocYAML(sys.argv[1])
    # Use default/standard filenames for standalone evaluation
    import os
    base_dir = os.path.dirname(os.path.abspath(__file__))
    fname_turb = os.path.join(base_dir, "iea37-335mw.yaml")
    fname_wr = os.path.join(base_dir, "iea37-windrose.yaml")
    # Get the array wind sampling bins, frequency at each bin, and wind speed
    wind_dir, wind_freq, wind_speed = getWindRoseYAML(fname_wr)
    # Pull the needed turbine attributes from file
    turb_ci, turb_co, rated_ws, rated_pwr, turb_diam = getTurbAtrbtYAML(
        fname_turb)

    # Calculate the AEP from ripped values
    AEP = calcAEP(turb_coords, wind_freq, wind_speed, wind_dir,
                  turb_diam, turb_ci, turb_co, rated_ws, rated_pwr)
    # Print AEP for each binned direction, with 5 digits behind the decimal.
    print(np.array2string(AEP, precision=5, floatmode='fixed',
                          separator=', ', max_line_width=62))
    # Print AEP summed for all directions
    print(np.around(np.sum(AEP), decimals=5))