# sunpos.py
import math
import pandas as pd

def sunpos(when, location, refraction = 0):

    # Falls 'when' ein String ist, in ein pandas Timestamp umwandeln
    if isinstance(when, str):
        when = pd.to_datetime(when)

    # Extract Time    
    dt = when.to_pydatetime()  # in ein datetime-Objekt umwandeln
    utc_offset = dt.utcoffset().total_seconds() / 3600 if dt.utcoffset() else 0
    timestamp_list = [dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, int(utc_offset)]
    year, month, day, hour, minute, second, timezone = timestamp_list
    
    # Extract Location
    latitude, longitude = location
    
# Math typing shortcuts
    rad, deg = math.radians, math.degrees
    sin, cos, tan = math.sin, math.cos, math.tan
    asin, atan2 = math.asin, math.atan2
    
# Convert latitude and longitude to radians
    rlat = rad(latitude)
    rlon = rad(longitude)
    
# Decimal hour of the day at Greenwich
    greenwichtime = hour - timezone + minute / 60 + second / 3600 # Days from J2000, accurate from 1901 to 2099
    daynum = (
        367 * year
        - 7 * (year + (month + 9) // 12) // 4
        + 275 * month // 9
        + day
        - 730531.5
        + greenwichtime / 24
    )
    
# Mean longitude of the sun
    mean_long = daynum * 0.01720279239 + 4.894967873
    
# Mean anomaly of the Sun
    mean_anom = daynum * 0.01720197034 + 6.240040768
    
# Ecliptic longitude of the sun
    eclip_long = (
        mean_long
        + 0.03342305518 * sin(mean_anom)
        + 0.0003490658504 * sin(2 * mean_anom)
    )
    
# Obliquity of the ecliptic
    obliquity = 0.4090877234 - 0.000000006981317008 * daynum
    
# Right ascension of the sun
    rasc = atan2(cos(obliquity) * sin(eclip_long), cos(eclip_long))
    
# Declination of the sun
    decl = asin(sin(obliquity) * sin(eclip_long))
    
# Local sidereal time
    sidereal = 4.894961213 + 6.300388099 * daynum + rlon
    
# Hour angle of the sun
    hour_ang = sidereal - rasc
    
# Local elevation of the sun
    elevation = asin(sin(decl) * sin(rlat) + cos(decl) * cos(rlat) * cos(hour_ang))# Local azimuth of the sun
    azimuth = atan2(
        -cos(decl) * cos(rlat) * sin(hour_ang),
        sin(decl) - sin(rlat) * sin(elevation),
    )
    
# Convert azimuth and elevation to degrees
    azimuth = into_range(deg(azimuth), 0, 360)
    elevation = into_range(deg(elevation), -180, 180)
    
# Refraction correction (optional)
    if refraction:
        targ = rad((elevation + (10.3 / (elevation + 5.11))))
        elevation += (1.02 / tan(targ)) / 60
        
# Return azimuth and elevation in degrees
    return (round(azimuth, 2), round(elevation, 2))
    
def into_range(x, range_min, range_max):
    shiftedx = x - range_min
    delta = range_max - range_min
    return (((shiftedx % delta) + delta) % delta) + range_min
    
if __name__ == "__main__":
    
# Close Encounters latitude, longitude
    location = (40.602778, -104.741667)
    
# Fourth of July, 2022 at 11:20 am MDT (-6 hours)
    when = (2022, 7, 4, 11, 20, 0, -6)
    
# Get the Sun's apparent location in the sky
    azimuth, elevation = sunpos(when, location, True)
    
# Output the results
    print("\nWhen: ", when)
    print("Where: ", location)
    print("Azimuth: ", azimuth)
    print("Elevation: ", elevation)
    
# When:  (2022, 7, 4, 11, 20, 0, -6)
# Where:  (40.602778, -104.741667)
# Azimuth:  121.38
# Elevation:  61.91