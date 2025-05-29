import numpy as np
import os
import math
import pandas as pd


rtk = False

kNbTrials = 100

def main():
    ppk_root = "/home/samqiao/ASRL/vtr3_testing/localization_data/ppk/"
    output_root = '/home/samqiao/ASRL/vtr3_testing/localization_data/ppk/'
    # Get the list of folders in the output_root directory
    sequences = [ f for f in os.listdir(ppk_root) if os.path.isdir(os.path.join(ppk_root, f)) ]

    print(sequences)

    for sequence in sequences:

        

        seq_path = os.path.join(output_root, sequence)
        
        # Get the file that finises with BESTPOS.ASCII
        files = [f for f in os.listdir(seq_path) if f.endswith('BESTPOS.ASCII')]
        if len(files) == 0:
            print('No BESTPOS.ASCII file found in ' + seq_path)
            continue
        elif len(files) > 1:
            print('Multiple BESTPOS.ASCII files found in ' + seq_path)
            continue
        bestpos_file = files[0]

        # Get the file that finises with _gps_fix.csv
        files = [f for f in os.listdir(seq_path) if f.endswith('_gps_fix.csv')]
        if len(files) == 0:
            print('No _gps_fix.csv file found in ' + seq_path)
            continue
        elif len(files) > 1:
            print('Multiple _gps_fix.csv files found in ' + seq_path)
            continue
        gps_fix_file = files[0]

        # Read the BESTPOS.ASCII and _gps_fix.csv files
        best_pos = pd.read_csv(os.path.join(seq_path, bestpos_file), delimiter=',')
        gps_fix = np.loadtxt(os.path.join(seq_path, gps_fix_file), delimiter=',', skiprows=1)   

        gps_fix_lat = np.round(gps_fix[:, 1], 11)

        counter = 0
        time_correspondence = None
        while counter < kNbTrials:
            # Get a random index in best_pos
            index = np.random.randint(0, len(best_pos))
            # Get the latitude of the best_pos at that index
            lat = best_pos.iloc[index,11]
            # Get the index of the exact same latitude in gps_fix
            index_gps_fix = np.where(gps_fix_lat == lat)[0]
            # If the index is not empty, we have a match
            if len(index_gps_fix) > 0:
                time_correspondence = np.array([best_pos.iloc[index, 6], gps_fix[index_gps_fix[0], 0]])
                print('Found time correspondence: ' + str(time_correspondence))
                break
            else:
                counter += 1
        if time_correspondence is None:
            print('ERROR: No time correspondence found for sequence ' + sequence)
            continue
        
        
        output_path = os.path.join(output_root, sequence, 'gps_cartesian.txt')
        with open(output_path, 'w') as output_file:
            output_file.write('timestamp,latitude,longitude,altitude,x,y,z\n')
            if rtk:
                with open(os.path.join(seq_path,bestpos_file), 'r') as f:
                    for i, line in enumerate(f):
                        parts = line.split(',')
                        gps_time = float(parts[6])
                        ros_time = (gps_time - time_correspondence[0] + time_correspondence[1])*1e9
                        lat = float(parts[11])
                        lon = float(parts[12])
                        alt = float(parts[13])
                        x, y, z = gnss_to_cartesian(lat, lon, alt)
                        output_file.write(str(ros_time) + ',' + str(lat) + ',' + str(lon) + ',' + str(alt) + ',' + str(x) + ',' + str(y) + ',' + str(z) + '\n')
                        #print('Processing line: ' + str(gps_time) + ' lat: ' + str(lat) + ' lon: ' + str(lon) + ' alt: ' + str(alt))
                        #print('Next')
            else:
                nb_skip = 14
                with open(os.path.join(seq_path, sequence + '.txt'), 'r') as f:
                    for i, line in enumerate(f):
                        if 'Master ' in line:
                            nb_skip += 3
                        if i < nb_skip:
                            continue
                        # Split line into with space as delimiter
                        parts = line.split(' ')
                        # Remove empty strings
                        parts = list(filter(None, parts))
                        # Get GPS time
                        gps_time = float(parts[11])
                        ros_time = (gps_time - time_correspondence[0] + time_correspondence[1])*1e9
                        # Get latitude and longitude
                        lat_deg = float(parts[0])
                        sign_lat = math.copysign(1, lat_deg)
                        lat = float(parts[0]) + sign_lat*float(parts[1])/60.0 + sign_lat*float(parts[2])/3600.0
                        lon_deg = float(parts[3])
                        sign_lon = math.copysign(1, lon_deg)
                        lon = float(parts[3]) + sign_lon*float(parts[4])/60.0 + sign_lon*float(parts[5])/3600.0

                        # Convert to Cartesian coordinates
                        x = float(parts[6])
                        y = float(parts[7])
                        z = float(parts[8])

                        # Write to file
                        output_file.write(str(ros_time) + ',' + str(lat) + ',' + str(lon) + ',' + str(z) + ',' + str(x) + ',' + str(y) + ',' + str(z) + '\n')


                        #print('Processing line: ' + str(gps_time) + ' lat: ' + str(lat) + ' lon: ' + str(lon))
        print('Finished processing sequence: ' + sequence)


                
# Copied from Leonardo's script for converting warthog data
def gnss_to_cartesian(lat, lon, alt=0):
    """
    Convert geodetic coordinates (latitude, longitude, altitude) 
    to ECEF (Earth-Centered, Earth-Fixed) Cartesian coordinates.
    
    :param lat: Latitude in decimal degrees
    :param lon: Longitude in decimal degrees
    :param alt: Altitude in meters (default 0)
    :return: x, y, z coordinates in meters
    """
    # WGS84 ellipsoid constants
    a = 6378137.0  # semi-major axis in meters
    f = 1 / 298.257223563  # flattening
    e2 = 2 * f - f * f  # square of first eccentricity
    
    # Convert latitude and longitude to radians
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)
    
    # Calculate prime vertical radius of curvature
    N = a / math.sqrt(1 - e2 * math.sin(lat_rad)**2)
    
    # Calculate ECEF coordinates
    x = (N + alt) * math.cos(lat_rad) * math.cos(lon_rad)
    y = (N + alt) * math.cos(lat_rad) * math.sin(lon_rad)
    z = ((1 - e2) * N + alt) * math.sin(lat_rad)
    
    return x, y, z

if __name__ == '__main__':
    main()