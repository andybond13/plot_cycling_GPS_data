#!python2

datashader_plot = False

import os
import math
import pandas as pd
import sys
from numba import jit
import warnings
warnings.filterwarnings('ignore')
from vincenty import vincenty_inverse
from mpl_toolkits import mplot3d
import matplotlib.colors as colors
import matplotlib.colorbar as colorbar
import matplotlib.cm as cm
from pyproj import Proj, transform
import ggps
from fitparse import FitFile
#https://towardsdatascience.com/how-tracking-apps-analyse-your-gps-data-a-hands-on-tutorial-in-python-756d4db6715d
import gpxpy
import matplotlib.pyplot as plt
import datetime
from geopy import distance
from math import sqrt, floor
import numpy as np
import pandas as pd
#import plotly.plotly as py
import chart_studio.plotly as py
import plotly.graph_objs as go
import haversine
if datashader_plot:
    #http://datashader.org/topics/nyc_taxi.html
    from bokeh.models import BoxZoomTool
    from bokeh.plotting import figure, output_notebook, show
    from bokeh.tile_providers import STAMEN_TERRAIN
    import datashader as ds
    from datashader import transfer_functions as tf
    from datashader.bokeh_ext import InteractiveImage
    from functools import partial
    from datashader.utils import export_image
    from datashader.colors import colormap_select, Greys9, Hot, inferno, viridis

figsize = (6,6)
dpi = 300
color_by = 'velo'
plot_width  = int(750)
plot_height = int(plot_width//1.2)
options = dict(line_color=None, fill_color='blue', size=5)
background = 'black'

msToMPH = 2.23694
semicircleToDegree = 180.0 / math.pow(2.0, 31)

def calc_direction(lon, lat):
    nData = len(lon)
    heading = np.zeros( (nData,) )

    for i in range(0, nData):
        if (i == 0):
            dx = lon[i+1] - lon[i]
            dy = lat[i+1] - lat[i]
        elif (i == nData-1):
            dx = lon[i] - lon[i-1]
            dy = lat[i] - lat[i-1]
        else:
            dx = 0.5 * (lon[i+1] - lon[i-1])
            dy = 0.5 * (lat[i+1] - lat[i-1])

        heading[i] = 90.0 - math.degrees(math.atan2(dy, dx))

    return heading

def clean_file(f):
    with open(f, 'r') as fopen:
        f_edit = f+'.edit'
        with open(f_edit, 'w') as fnew:
            i = 0
            for line in fopen:
                i = i+1
                if i == 1:
                    lf = line.lstrip()
                else:
                    lf = line
                fnew.write(lf)
    return f_edit

def base_plot(tools='pan,wheel_zoom,reset',plot_width=plot_width, plot_height=plot_height, **plot_args):
    p = figure(tools=tools, plot_width=plot_width, plot_height=plot_height,
        x_range=x_range, y_range=y_range, outline_line_color=None,
        min_border=0, min_border_left=0, min_border_right=0,
        min_border_top=0, min_border_bottom=0, **plot_args)

    p.axis.visible = False
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    p.add_tools(BoxZoomTool(match_aspect=True))

    return p

def web_mercator(lat, long):
    return transform(Proj(init='epsg:4326'), Proj(init='epsg:3857'), long, lat)

def update_bounds(data, min_lat, max_lat, min_long, max_long):

    this_min_lat = min(data['lat'].values)
    this_max_lat = max(data['lat'].values)
    this_min_long = min(data['lon'].values)
    this_max_long = max(data['lon'].values)

    if (min_lat == None or this_min_lat < min_lat):
        min_lat = this_min_lat
    if (max_lat == None or this_max_lat > max_lat):
        max_lat = this_max_lat
    if (min_long == None or this_min_long < min_long):
        min_long = this_min_long
    if (max_long == None or this_max_long > max_long):
        max_long = this_max_long

    return min_lat, max_lat, min_long, max_long

def smooth(x,window_len=11,window='hanning'):
#https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError, "Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'"


    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y[(window_len/2-1):-(window_len/2)]   #corrects length

def calc_time_min(times):
    nTimes = len(times)
    dTime = np.diff(times)
    try:
        for i in range(0, nTimes-1):
            dTime[i] = dTime[i].total_seconds()
    except:
        dTime = np.divide(dTime, np.timedelta64(1, 's'))
    time_min = np.zeros( (nTimes,) ) 
    for i in range(1, nTimes):
        time_min[i] = time_min[i-1] + dTime[i-1] / 60.0
    return time_min

@jit(forceobj=True)
def calc_velo_and_slope(df):

    lats = np.array(df['lat'].values)  
    longs = np.array(df['lon'].values)
    #times = np.array( [x.to_datetime64() for x in df['time'].values] )
    times = np.array( [x.to_pydatetime() for x in df['time'].values] )
    alts = np.array(df['alt'].values)

    dAlt = np.diff(alts)
    dTime = np.diff(times)
    for i in range(0, len(lats)-1):
        dTime[i] = dTime[i].total_seconds()

    dists = np.zeros( (len(lats)-1,) )
    dists3d = np.zeros( (len(lats)-1,) )
    for i in range(1, len(lats)):
        start = (lats[i-1], longs[i-1])
        end = (lats[i], longs[i])
#        dist = haversine.haversine(start, end, unit='ft')
        dist = vincenty_inverse(start, end).ft
        dz = dAlt[i-1]
        dist3d = math.sqrt(dist*dist + dz*dz)
        dists[i-1] = dist 
        dists3d[i-1] = dist3d 

    velo = np.zeros( (len(lats),) ) 
    slope = np.zeros( (len(lats),) )

    time_min = calc_time_min(times) 
    for i in range(1, len(lats)):
        if (i == 0):
            velo[i] = dists3d[0]/dTime[0]
            slope[i] = dAlt[0]/dists[0]
        elif (i == len(lats)-1):
            velo[i] = dists3d[-1]/dTime[-1]
            slope[i] = dAlt[-1]/dists[-1]
        else:
            velo[i] = (dists3d[i] + dists3d[i-1]) / (dTime[i] + dTime[i-1]) 
            slope[i] = (dAlt[i] + dAlt[i-1]) / (dists[i] + dists[i-1])

    #convert velo from ft/s to mph
    velo = velo * 3600.0 / 5280.0

    #filter velo & slope
    velo = smooth(velo, window_len=8) 
    slope = smooth(slope, window_len=8) 

    wm_lat = np.zeros( (len(lats),) ) 
    wm_lon = np.zeros( (len(longs),) )
    if datashader_plot: 
        for i in range(0, len(lats)):
            wm_lat[i],wm_lon[i] = web_mercator(lats[i], longs[i])

    return velo,slope,time_min,wm_lat,wm_lon

def read_file(f):

    filebase, file_extension = os.path.splitext(f)
    data = None
    if ('.gpx' in file_extension):
        gpx_file = open(f, 'r')
        gpx = gpxpy.parse(gpx_file)
        data = gpx.tracks[0].segments[0].points

        df = pd.DataFrame(columns=['lon','lat', 'alt', 'time'])
        for point in data:
            df = df.append({'lon': point.longitude, 'lat':point.latitude, 'alt':point.elevation, 'time':point.time}, ignore_index=True)

        velo,slope,time_min,wm_lat,wm_lon = calc_velo_and_slope(df)
        df['velo'] = velo
        df['slope'] = slope


    if ('.tcx' in file_extension):
        fnew = clean_file(f)
        handler = ggps.TcxHandler()
        handler.parse(fnew)
        trackpoints = handler.trackpoints

        df = pd.DataFrame(columns=['lon','lat', 'alt', 'time'])
        for i in range(0, len(trackpoints)):
            point = trackpoints[i].values
            df = df.append({'lon': float(point['longitudedegrees']), 'lat':float(point['latitudedegrees']), 'alt':float(point['altitudefeet']), 'time':pd.Timestamp(point['time'])}, ignore_index=True)

        velo,slope,time_min,wm_lat,wm_lon = calc_velo_and_slope(df)
        df['velo'] = velo
        df['slope'] = slope

    if ('.fit' in file_extension):
        fitfile = FitFile(f)
        df = pd.DataFrame(columns=['lon','lat', 'alt', 'time', 'velo'])

        for record in fitfile.get_messages('record'):
            rData = {}
            for record_data in record:
                rData[record_data.name] = record_data.value
            df = df.append({'lon': rData['position_long']*semicircleToDegree, 'lat':rData['position_lat']*semicircleToDegree, 'alt':0.0, 'time':pd.Timestamp(rData['timestamp']), 'velo':rData['enhanced_speed']*msToMPH, 'slope':0.0}, ignore_index=True)

   
        time_min = calc_time_min( df['time'].values ) 
        wm_lat = np.zeros( (len(df['lat']),) ) 
        wm_lon = np.zeros( (len(df['lon']),) )
        if datashader_plot: 
            for i in range(0, len(lats)):
                wm_lat[i],wm_lon[i] = web_mercator( df['lat'].values[i], df['long'].values[i])

    heading = calc_direction(df['lon'].values, df['lat'].values)
    df['time_min'] = time_min
    df['wm_lat'] = wm_lat
    df['wm_lon'] = wm_lon
    df['heading'] = heading

    return df,filebase


files = sys.argv[1:]
min_lat = None
max_lat = None
min_long = None
max_long = None
data = None

for f in files:
    print f
    data,filebase = read_file(f)
    min_lat, max_lat, min_long, max_long = update_bounds(data, min_lat, max_lat, min_long, max_long)

    velo = data['velo'].values
    cmap = cm.viridis
    norm = colors.Normalize(vmin=np.min(velo), vmax=np.max(velo))
    slope = data['slope'].values
    cmap_slope = cm.coolwarm
    max_slope = max(abs(np.min(slope)), abs(np.max(slope)) )
    norm_slope = colors.Normalize(vmin=-max_slope, vmax=max_slope)
   
    fig1 = plt.figure(figsize=figsize) 
#    plt.plot(data['lon'], data['lat'])
    for i in xrange(len(data['lon'])-1):
        if (color_by == 'velo'):
            plt.plot( data['lon'].values[i:i+2], data['lat'].values[i:i+2], color=cmap(norm(velo[i])) )
        elif (color_by == 'slope'):
            plt.plot( data['lon'].values[i:i+2], data['lat'].values[i:i+2], color=cmap_slope(norm_slope(slope[i])) )
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.savefig(filebase+'_map.png', dpi=dpi)

    fig2 = plt.figure(figsize=figsize) 
    for i in xrange(len(data['lon'])-1):
        if (color_by == 'velo'):
            plt.plot( data['time_min'].values[i:i+2], data['alt'].values[i:i+2], color=cmap(norm(velo[i])) )
        elif (color_by == 'slope'):
            plt.plot( data['time_min'].values[i:i+2], data['alt'].values[i:i+2], color=cmap_slope(norm_slope(slope[i])) )
    plt.xlabel('Time (min)')
    plt.ylabel('Elevation (ft)')
    plt.savefig(filebase+'_elev.png', dpi=dpi)

    fig3 = plt.figure(figsize=figsize) 
    for i in xrange(len(data['lon'])-1):
        if (color_by == 'velo'):
            plt.plot( data['time_min'].values[i:i+2], data['slope'].values[i:i+2], color=cmap(norm(velo[i])) )
        elif (color_by == 'slope'):
            plt.plot( data['time_min'].values[i:i+2], data['slope'].values[i:i+2], color=cmap_slope(norm_slope(slope[i])) )
    plt.xlabel('Time (min)')
    plt.ylabel('Slope (ft/ft)')
    plt.savefig(filebase+'_slope.png', dpi=dpi)

    fig4 = plt.figure(figsize=figsize) 
    plt.plot(data['time_min'], data['velo'])
    plt.xlabel('Time (min)')
    plt.ylabel('Speed (mph)')
    plt.savefig(filebase+'_speed.png', dpi=dpi)

#    _data = [go.Scatter3d(x=data['lon'], y=data['lat'], z=data['alt'], mode='lines')]
#    py.sign_in('andybond13', 'pNIE4xlJTh4Fp08mlGO5')
#    py.plot(_data)

    fig5 = plt.figure()
    ax = plt.axes(projection='3d')
    for i in xrange(len(data['lon'])-1):
        if (color_by == 'velo'):
            ax.plot( data['lon'].values[i:i+2], data['lat'].values[i:i+2], data['alt'].values[i:i+2], color=cmap(norm(velo[i])) )
        elif (color_by == 'slope'):
            ax.plot( data['lon'].values[i:i+2], data['lat'].values[i:i+2], data['alt'].values[i:i+2], color=cmap_slope(norm_slope(slope[i])) )
    plt.savefig(filebase+'_map3d.png', dpi=dpi)

    fig6 = plt.figure(figsize=figsize) 
    plt.scatter(data['slope'], data['velo'], alpha=0.5, s=1, c=data['heading'], cmap=cm.hsv) 
    plt.xlabel('Slope (ft/ft)')
    plt.ylabel('Speed (mph)')
    plt.savefig(filebase+'_slopeSpeed.png', dpi=dpi)

plt.show() 


if datashader_plot:
    output_notebook()

    wm_minlong,wm_minlat = web_mercator(min_long, min_lat)
    wm_maxlong,wm_maxlat = web_mercator(max_long, max_lat)
    FRAME = x_range, y_range = ((wm_minlong, wm_maxlong), (wm_minlat, wm_maxlat))

    def create_image(x_range, y_range, w=plot_width, h=plot_height):
        cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(data, 'wm_lon', 'wm_lat',  ds.count('velo'))
        img = tf.shade(agg, cmap=viridis, how='eq_hist')
        return tf.dynspread(img, threshold=0.5, max_px=4)

    def create_image90(x_range, y_range, w=plot_width, h=plot_height):
        cvs = ds.Canvas(plot_width=w, plot_height=h, x_range=x_range, y_range=y_range)
        agg = cvs.points(data, 'wm_lon', 'wm_lat',  ds.count('velo'))
        img = tf.shade(agg.where(agg>np.percentile(agg,90)), cmap=viridis, how='eq_hist')
        return tf.dynspread(img, threshold=0.3, max_px=4)

    p = base_plot()
    export = partial(export_image, export_path="export", background=background)
    p.add_tile(STAMEN_TERRAIN)
    export(create_image90(*FRAME),"NYCT_90th")
    InteractiveImage(p, create_image90)
