#!bash

source ~/.bash_profile

python2 plot_GPS_cycling_data.py *.fit *.gpx *.tcx
rm *.edit

#currently plots .gpx, *.fit, and .tcx files
#-makes 6 plots for each file (does not combine) --> should combine plot 1 at least
#-datashader capability doesn't work
