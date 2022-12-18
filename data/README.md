# Datasets

The data was taken from [UCR Time Series Classification Archive](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/). We used all datasets with equal time series length and without missing values. The UCR Archive already handles these cases (in a special folder), replacing missing values using linear interpolation and adding low-amplitude random numbers to the end of the time series with different lengths.

This folder originally contained a subfolder `UCRArchive_2018`, which has all `128` datasets without missing values and with equal time series lengths. If you want to test the `Jupyter` notebooks locally, you will probably need to load them manually.