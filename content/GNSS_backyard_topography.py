"""
Dea Dressel
AA 272
Final Project
Fall 2023
"""

# Imports
import sys

import numpy as np
import pandas as pd

import gnss_lib_py as glp
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib import ticker
import plotly.graph_objects as go

import requests
import urllib3
import urllib

# From Derek's android_tutorial_272.ipynb.
import site
site.main()

# Constants
NPTS = 200
NGRIDX = 200
NGRIDY = 200
NPLOTS = 3

def unpack_and_format_fix_data(filename):
    """
    Unpacks the location fix measurements from raw Android dataset.
    Renames the latitude, longitude, and altitude measurements to
    differentiate between different fix types.

    Adapted from Derek's android_tutorial_272.ipynb.

    Inputs:
        filename: Path to GNSS Logger .txt data file.

    Output:
        fixes: List containing various fix types and the corresponding
               lat_rx_deg, lon_rx_deg, and alt_rx_m data.
        fix_types: List of types of fix measurements in dataset.
    """
    # Unpack location fix measurements
    fix_data = glp.AndroidRawFixes(filename)

    # Identify the types of fix measurements in dataset
    fix_types = np.unique(fix_data["fix_provider"]).tolist()

    # Rename data headers
    fixes = []
    for provider in fix_types:
        fix_provider = fix_data.where("fix_provider",provider)
        fix_provider.rename({"lat_rx_deg":"lat_rx_" + provider + "_deg",
                            "lon_rx_deg":"lon_rx_" + provider + "_deg",
                            "alt_rx_m":"alt_rx_" + provider + "_m",
                            "VerticalAccuracyMeters" : "VerticalAccuracyMeters" + provider,
                            "NumberOfUsedSignals" : "NumberOfUsedSignals" + provider,
                            "AccuracyMeters" : "AccuracyMeters" + provider
                            }, inplace=True)
        fixes.append(fix_provider)

    return fix_data, fixes, fix_types

def plot_trajectory_fixes(fixes):
    """
    Leverage gnss_lib_py to plot trajectory fixes.

    Input:
        fixes: list of fix provider data.
    """
    trajectory_fix_fig = glp.plot_map(*fixes)
    trajectory_fix_fig.update_layout(title_text="Trajectory Taken while Collecting Data",
                                     title_font_size=20,
                                     font=go.layout.Font(family="Times New Roman"),
                                     margin={"r": 100, "t": 100, "l": 100, "b": 100},
                                    xaxis_title=dict(text='Date', font=dict(size=16, color='#FFFFFF')),
                                    yaxis_title="Date")

    trajectory_fix_fig.show()


def interpolate_and_plot_topography_fixes(fixes, fix_types):
    """
    """
    
    # Create Fused, GNSS, and Vertical Accuracy figures
    fig_fused, axs_fused = plt.subplots(1,3)
    plt.subplots_adjust(wspace=0.5)
    fig_gnss, axs_gnss = plt.subplots(1,3)
    plt.subplots_adjust(wspace=0.5)
    fig_vert_accuracy, ax_vert_accuracy = plt.subplots(2,1)
    plt.subplots_adjust(hspace=0.5)
    
    # 
    z_gnss, z_fused = None, None
    x_plot, y_plot = None, None

    # Loop over fixes
    for i, fix in enumerate(fixes):
        
        # Identify fix_type
        fix_type = fix_types[i]
        
        # Get latitude, longitude, and altitude data for specific fix type
        lat = fix[f'lat_rx_{fix_type}_deg']
        lon = fix[f'lon_rx_{fix_type}_deg']
        alt = fix[f'alt_rx_{fix_type}_m']
        
        # Identify (lon, lat) grid space
        min_lat, max_lat = min(lat), max(lat)
        min_lon, max_lon = min(lon), max(lon)

        # Create dense grid values
        x = np.linspace(min_lon, max_lon, NGRIDX)
        y = np.linspace(min_lat, max_lat, NGRIDY)
        x_plot, y_plot = x, y
        
        # Create unstructured triangular grid using Delaunay triangulation
        triangulation = tri.Triangulation(lon, lat)

        # Linearly interpolate (lon, lat) data on (x, y) grid
        interpolator = tri.LinearTriInterpolator(triangulation, alt)

        # Create meshgrid and interpolate altitude values
        X, Y = np.meshgrid(x, y)
        z = interpolator(X, Y)

        # Switch based on fix type
        if "fused" == fix_type:
            # Fused "Fix"
            axs = axs_fused
            fig = fig_fused
            color = "-r"
            fig_tit = "Fused"
            z_fused = z
        else:
            # GNSS "Fix"
            axs = axs_gnss
            fig = fig_gnss
            color = "-b"
            fig_tit = "GNSS"
            z_gnss = z


        # Create plots
        plot_vertical_accuracy(fig_vert_accuracy, ax_vert_accuracy[i], fix, fix_type, color, fig_tit)

        plot_trajectory(axs[0], lon, lat, min_lon, max_lon, min_lat, max_lat, title=f"Trajectory", marker="-k")

        plot_interpolation(axs[1], triangulation, min_lon, max_lon, min_lat, max_lat, title=f"Triangular Interpolation")

        plot_contour(fig, axs[2], x, y, z, min_lon, max_lon, min_lat, max_lat, title=f"Topography")

        # Set figure title and axis labels
        fig_title = "GNSS Measurements" if fix_type == "gnss" else "Fused Measurements"
        fig.suptitle(fig_title, fontsize=20)
        fig.supxlabel('Longitude (deg)')
        fig.supylabel('Latitude (deg)')
        
    plt.show()
    return z_gnss, z_fused, x_plot, y_plot

### PLOTTING HELPER FUNCTIONS ###
    
def plot_vertical_accuracy(fig, ax, fix, fix_type, color, fig_tit):
    # Plot vertical accuracy data
    vertical_accuracy = fix[f'VerticalAccuracyMeters{fix_type}']
    ax.plot(vertical_accuracy, color)
    ax.set_title(f"Vertical Accuracy of {fig_tit} Measurements")
    fig.supxlabel('Measurement Index')
    fig.supylabel('Vertical Accuracy (m)')

def plot_trajectory(ax, lon, lat, min_lon, max_lon, min_lat, max_lat, title, marker):
    # Plot trajectory
    ax.plot(lon, lat, marker, ms=3)
    ax.set(xlim=(min_lon, max_lon), ylim=(min_lat, max_lat))
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}"))
    ax.grid(visible=True, which='both', axis='both')
    ax.set_aspect('equal')

def plot_interpolation(ax, triangulation, min_lon, max_lon, min_lat, max_lat, title):
    # Plot interpolation
    ax.triplot(triangulation, '-k')
    ax.set(xlim=(min_lon, max_lon), ylim=(min_lat, max_lat))
    ax.set_title(title)
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}"))
    ax.grid(visible=True, which='both', axis='both')
    ax.set_aspect('equal')

def plot_contour(fig, ax, x, y, z, min_lon, max_lon, min_lat, max_lat, title):
    # Plot contour 
    ax.contour(x, y, z, levels=14, linewidths=0.5, colors='k')
    cntr = ax.contourf(x, y, z, levels=14, cmap="Greens")
    ax.set(xlim=(min_lon, max_lon), ylim=(min_lat, max_lat))
    ax.set_title(title)
    fig.colorbar(cntr, ax=ax, label="Altitude (m)")
    ax.xaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}"))
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("{x:,.4f}"))
    ax.grid(visible=True, which='both', axis='both')
    ax.set_aspect('equal')

def plot_ground_truth(z_measured):
    # Create figures
    fig, axs = plt.subplots(1,3)
    plt.subplots_adjust(wspace=0.5)

    # Read in ground truth data
    df = pd.read_csv("data/ground_truth/xyz_usgs.txt", delimiter="\t")
    lon = df["lon"]
    lat = df["lat"]
    alt = df["alt"] - 30

    # Convert altitude from MSL to WGS84

    # Identify (lon, lat) grid space
    min_lat, max_lat = min(lat), max(lat)
    min_lon, max_lon = min(lon), max(lon)
    x = np.linspace(min_lon, max_lon, NGRIDX)
    y = np.linspace(min_lat, max_lat, NGRIDY)

    # Perform interpolation
    triangulation = tri.Triangulation(lon, lat)
    interpolator = tri.LinearTriInterpolator(triangulation, alt)
    X, Y = np.meshgrid(x, y)
    z = interpolator(X, Y)

    # Remove (lat,lon) data points from ground truth z that do not exist in measured z
    z_gt = z - z_measured
    z_gt += z_measured

    # Create plots
    plot_trajectory(axs[0], lon, lat, min_lon, max_lon, min_lat, max_lat, title=f"Sparse Data", marker="ko")

    plot_interpolation(axs[1], triangulation, min_lon, max_lon, min_lat, max_lat, title=f"Triangular Interpolation")

    plot_contour(fig, axs[2], x, y, z_gt, min_lon, max_lon, min_lat, max_lat, title=f"Topography")
    
    # Edit figure
    fig.suptitle("USGS Measurements", fontsize=20)
    fig.supxlabel('Longitude (deg)')
    fig.supylabel('Latitude (deg)')
    plt.show()

    return z



### USGS QUERY HELPER FUNCTIONS ### 

def make_usgs_request(usgs_params):
    """
    Continuously makes a USGS altitude query until a response is received.
    """
    usgs_url = r'https://epqs.nationalmap.gov/v1/json?'
    count = 0
    response = None
    while True:
        try:
            response = requests.get((usgs_url + urllib.parse.urlencode(usgs_params)))
        except (OSError, urllib3.exceptions.ProtocolError) as error:
            print('\n')
            print(f'Number of tries: {count}')
            print(error)
            print('\n')
            count += 1
            continue
        break
    return response
            
def get_z_usgs_ground_truth():
    """
    Queries the USGS service for measured altitudes over a sparse
    (lat, lon) dataset.
    """
    # Read in sparse (lat,lon) data
    df_xy = pd.read_csv("data/ground_truth/xy_sparse_data.txt", delimiter="\t")
    df_alt = pd.DataFrame()

    # Query USGS for altitude at each lat/lon combination
    elevations = []
    for lat in df_xy["lat"]:
        for lon in df_xy["lon"]:
            usgs_params = {'x': lon, 'y': lat, 'units': "Meters", 'output': 'json'}
            response = make_usgs_request(usgs_params)
            el = response.json()['value']
            elevations.append(el)

    # Add to DF and write altitude data to csv
    df_alt['elevations'] = elevations
    df_alt.to_csv('ground_truth/z_data.txt', sep='\t', index=False)

def format_usgs_data():
    """
    Reads in xy sparse data and queried USGS z data. Reformats to
    capture each measurement.
    """
    # Read in xyz values
    df_xy = pd.read_csv("data/ground_truth/xy_sparse_data.txt", delimiter="\t")
    df_alt = pd.read_csv("data/ground_truth/z_data.txt", delimiter="\t")

    # Create empty dataframe
    df_ll = pd.DataFrame()
    lons = []
    lats = []
    for lat in df_xy["lat"]:
        for lon in df_xy["lon"]:
            lats.append(lat)
            lons.append(lon)

    # write lat, lon, lat measurements to csv
    df_ll["lat"] = lats
    df_ll['lon'] = lons
    df_ll['alt'] = df_alt['elevations']
    df_ll.to_csv('data/ground_truth/xyz_usgs.txt', sep='\t', index=False)

### EXPERIMENTAL RESULTS CALCULATION ### 

def calculate_RMSE(z_gt, z_gnss, z_fused):
    """
    Calcualte the Root Mean Squared Error for the altitude measurements.
    """

    # GNSS compared to Ground Truth
    z_gt_subset_gnss = z_gt[z_gnss.nonzero()]
    z_gnss_nonzero = z_gnss[z_gnss.nonzero()]
    N_gnss = len(z_gt_subset_gnss)

    # Fused compared to Ground Truth
    z_gt_subset_fused = z_gt[z_fused.nonzero()]
    z_fused_nonzero = z_fused[z_fused.nonzero()]
    N_fused = len(z_gt_subset_fused)

    rmse_gnss = np.sqrt( np.sum( (z_gt_subset_gnss - z_gnss_nonzero)**2 ) / N_gnss)
    rmse_fused = np.sqrt( np.sum( (z_gt_subset_fused - z_fused_nonzero)**2 )/ N_fused)

    return rmse_gnss, rmse_fused



if __name__ == "__main__":
    """
    Welcome to my project! With this script, you can quickly create a low-cost
    topographic map using GNSS "Fix" and Fused "Fix" data collected with GNSS Logger.

    Possible Arugment:
        -usgs: query USGS database. Only need to use this argument if
               ground_truth/xyz_usgs.txt doesn't exist yet.
    """
    args = sys.argv[1:]

    # File containing GNSS Logger data
    filename = "data/GNSS_Logger/gnss_log_2023_11_30_13_55_16.txt"
    
    # Unpack data and plot trajectories
    fix_data, fixes, fix_types = unpack_and_format_fix_data(filename)

    # Plot GNSS and Fused Fixes (aka remove Network)
    plot_trajectory_fixes(fixes[:2])

    # Interpolate data and plot topographies
    z_gnss, z_fused, x_plot, y_plot = interpolate_and_plot_topography_fixes(fixes[:2], fix_types)

    # USE ONLY IF NEED TO QUERY USGS FOR GROUND TRUTH DATA
    # THIS PROCESS TAKES A LONG TIME
    if 1 == len(args) and "-usgs" == args[0]:
        # Only use this 
        get_z_usgs_ground_truth()
        format_usgs_data()

    # Plot ground truth
    z_gt = plot_ground_truth(z_gnss)

    # Calculate experimental results
    rmse_gnss, rmse_fused = calculate_RMSE(z_gt, z_gnss, z_fused)
    print("\n~~ Root Mean Square Error Results ~~")
    print("RMSE GNSS: ", rmse_gnss)
    print("RMSE FUSED: ", rmse_fused)

    