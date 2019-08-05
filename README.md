# plot_cycling_GPS_data
Plotting cycling (& other) GPS data from Garmin/Strava

The purpose of this script is to create custom plots of cycling GPS data. These can be obtained from a Strava website (see "Bulk Download", https://support.strava.com/hc/en-us/articles/216918437-Exporting-your-Data-and-Bulk-Export#Bulk) or directly from GPS fitness tracking devices.

Currently recognized file types are:
-*.gpx
-*.tcx
-*.fit

Given the data available, the script currently processes and plots:
-2D map, colored by speed
-3D map, colored by speed,
-speed
-elevation, colored by speed
-slope, colored by speed
-speed vs. slope, colored by compass heading

The script can be modified to create custom plots beyond these.
