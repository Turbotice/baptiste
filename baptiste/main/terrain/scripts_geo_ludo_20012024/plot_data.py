#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:38:04 2024

@author: moreaul
"""

%matplotlib qt5
import os
import shutil
from obspy import read
from obspy.core import UTCDateTime
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import mplcursors
from datetime import datetime


# path2data = '/Users/moreaul/Desktop/Geophones/Toit_ISTerre_17012024'
# acqu_numb = '0001'

path2data = 'W:\\Banquise\\Data_test_terrain\\Geophones\\Grenoble_20240118_dehors'
acqu_numb = '0001'

def read_data(path2data):
    miniseed_files = []
    
    # Iterate over files in the specified directory
    for filename in os.listdir(path2data):
        if filename.endswith(".miniseed"):
            file_path = os.path.join(path2data, filename)
            miniseed_files.append(file_path)
    
    # Read MiniSEED files
    streams = []
    for file_path in miniseed_files:
        stream = read(file_path)
        streams.append(stream)
    
    return streams, miniseed_files

class ZoomHandler:
    def __init__(self, ax, time_vector, data_vector):
        self.ax = ax
        self.time_vector = time_vector
        self.data_vector = data_vector
        self.original_xlim = ax.get_xlim()
        self.original_ylim = ax.get_ylim()

        # Initialize rectangle selector
        self.rs = RectangleSelector(ax, self.on_rectangle_select, drawtype='box', useblit=True, button=[1],
                                    minspanx=5, minspany=5, spancoords='pixels', interactive=True)

    def on_click(self, event):
        if event.dblclick:
            self.reset_zoom()

    def on_rectangle_select(self, eclick, erelease):
        # Extract rectangle coordinates
        x1, y1 = eclick.xdata, eclick.ydata
        x2, y2 = erelease.xdata, erelease.ydata

        # Apply zoom to the selected area
        self.ax.set_xlim(min(x1, x2), max(x1, x2))
        self.ax.set_ylim(min(y1, y2), max(y1, y2))
        plt.draw()

    def reset_zoom(self):
        self.ax.set_xlim(self.original_xlim)
        self.ax.set_ylim(self.original_ylim)
        plt.draw()

def convert_to_utc_times(trace, time_vector):
    start_time_utc = UTCDateTime(trace.stats.starttime)
    return [start_time_utc + t for t in time_vector]





# Read MiniSEED file directly
seismic_data_streams, miniseed_files = read_data(path2data +'/' +acqu_numb)

# Extract a time and data vectors for plotting
stream2plot = seismic_data_streams[1]
first_trace = stream2plot[0]
time_vector = convert_to_utc_times(first_trace, first_trace.times())
data_vector = first_trace.data

# Extract the start time of the first trace
start_time_utc = UTCDateTime(first_trace.stats.starttime)

# Create a simple scatter plot
fig, ax = plt.subplots()
ax.plot([datetime.utcfromtimestamp(t.timestamp) for t in time_vector], data_vector)

# Set the title with the start time
fig.suptitle(f"Seismic Data - {start_time_utc.date}", fontsize=16)

# Create an instance of ZoomHandler
zoom_handler = ZoomHandler(ax, time_vector, data_vector)
# Connect the on_click method to the figure
fig.canvas.mpl_connect('button_press_event', zoom_handler.on_click)
# Enable interactive labels using mplcursors
mplcursors.cursor(hover=True)
# Adjust the backend to make it work better in Spyder
plt.ion()
plt.show(block=True)  # Use block=True to make it work better in Spyder

plt.show()




