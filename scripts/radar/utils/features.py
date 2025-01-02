import numpy as np
from scipy import ndimage

def cen2018features(fft_data: np.ndarray, min_range=58, zq=4.0, sigma_gauss=17) -> np.ndarray:
    """Extract features from polar radar data using the method described in cen_icra18
    Args:
        fft_data (np.ndarray): Polar radar power readings
        min_range (int): targets with a range bin less than or equal to this value will be ignored.
        zq (float): if y[i] > zq * sigma_q then it is considered a potential target point
        sigma_gauss (int): std dev of the gaussian filter used to smooth the radar signal
        
    Returns:
        np.ndarray: N x 2 array of feature locations (azimuth_bin, range_bin)
    """
    nazimuths = fft_data.shape[0]
    # w_median = 200
    # q = fft_data - ndimage.median_filter(fft_data, size=(1, w_median))  # N x R
    q = fft_data - np.mean(fft_data, axis=1, keepdims=True)
    p = ndimage.gaussian_filter1d(q, sigma=17, truncate=3.0) # N x R
    noise = np.where(q < 0, q, 0) # N x R
    nonzero = np.sum(q < 0, axis=-1, keepdims=True) # N x 1
    sigma_q = np.sqrt(np.sum(noise**2, axis=-1, keepdims=True) / nonzero) # N x 1

    def norm(x, sigma):
        return np.exp(-0.5 * (x / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

    nqp = norm(q - p, sigma_q)
    npp = norm(p, sigma_q)
    nzero = norm(np.zeros((nazimuths, 1)), sigma_q)
    y = q * (1 - nqp / nzero) + p * ((nqp - npp) / nzero)
    t = np.nonzero(y > zq * sigma_q)
    # Extract peak centers
    current_azimuth = t[0][0]
    peak_points = [t[1][0]]
    peak_centers = []

    def mid_point(l):
        return l[len(l) // 2]

    for i in range(1, len(t[0])):
        if t[1][i] - peak_points[-1] > 1 or t[0][i] != current_azimuth:
            m = mid_point(peak_points)
            if m > min_range:
                peak_centers.append((current_azimuth, m))
            peak_points = []
        current_azimuth = t[0][i]
        peak_points.append(t[1][i])
    if len(peak_points) > 0 and mid_point(peak_points) > min_range:
        peak_centers.append((current_azimuth, mid_point(peak_points)))

    return np.asarray(peak_centers)

# modified CACFAR algorithm
def modifiedCACFAR(
    raw_scan: np.ndarray,
    minr=2.0,
    maxr=80.0,
    res=0.040308,
    width=137,
    guard=7,
    threshold=0.50,
    threshold2=0.0,
    threshold3=0.23,
    peak_summary_method='max_intensity'):
    # peak_summary_method: median, geometric_mean, max_intensity, weighted_mean
    rows = raw_scan.shape[0]
    cols = raw_scan.shape[1]
    if width % 2 == 0: width += 1
    w2 = int(np.floor(width / 2))
    mincol = int(minr / res + w2 + guard + 1)
    if mincol > cols or mincol < 0: mincol = 0
    maxcol = int(maxr / res - w2 - guard)
    if maxcol > cols or maxcol < 0: maxcol = cols
    N = maxcol - mincol
    targets_polar_pixels = []
    for i in range(rows):
        mean = np.mean(raw_scan[i])
        peak_points = []
        peak_point_intensities = []
        for j in range(mincol, maxcol):
            left = 0
            right = 0
            for k in range(-w2 - guard, -guard):
                left += raw_scan[i, j + k]
            for k in range(guard + 1, w2 + guard):
                right += raw_scan[i, j + k]
            # (statistic) estimate of clutter power
            stat = max(left, right) / w2  # GO-CFAR
            thres = threshold * stat + threshold2 * mean + threshold3
            if raw_scan[i, j] > thres:
                peak_points.append(j)
                peak_point_intensities.append(raw_scan[i, j])
            elif len(peak_points) > 0:
                if peak_summary_method == 'median':
                    r = peak_points[len(peak_points) // 2]
                elif peak_summary_method == 'geometric_mean':
                    r = np.mean(peak_points)
                elif peak_summary_method == 'max_intensity':
                    r = peak_points[np.argmax(peak_point_intensities)]
                elif peak_summary_method == 'weighted_mean':
                    r = np.sum(np.array(peak_points) * np.array(peak_point_intensities) / np.sum(peak_point_intensities))
                else:
                    raise NotImplementedError("peak summary method: {} not supported".format(peak_summary_method))
                targets_polar_pixels.append((i, r))
                peak_points = []
                peak_point_intensities = []
    return np.asarray(targets_polar_pixels)

import heapq

def KStrong(
    raw_scan: np.ndarray,
    minr=2.0,
    maxr=80.0,
    res=0.040308,
    K=4,
    static_threshold=0.23):
    rows = raw_scan.shape[0]
    cols = raw_scan.shape[1]
    mincol = int(minr / res)
    if mincol > cols or mincol < 0: mincol = 0
    maxcol = int(maxr / res)
    if maxcol > cols or maxcol < 0: maxcol = cols
    
    targets_polar_pixels = []

    for i in range(rows):

        temp_intensities = raw_scan[i]
        max_pairs = []

        for j in range(mincol, maxcol):
            if temp_intensities[j] > static_threshold:
                max_pairs.append((temp_intensities[j], j))

        sorted_pairs = sorted(max_pairs, key=lambda x: x[0], reverse=True)

        for k in range(K):
            if k < len(sorted_pairs):
                value, j = sorted_pairs[k]
                assert(value==raw_scan[i, j])
                targets_polar_pixels.append((i, j))
            else:
                break

    return np.asarray(targets_polar_pixels)

def polar_to_cartesian_points(azimuths: np.ndarray, polar_points: np.ndarray, radar_resolution: float,
    downsample_rate=1) -> np.ndarray:
    """Converts points from polar coordinates to cartesian coordinates
    Args:
        azimuths (np.ndarray): The actual azimuth of reach row in the fft data reported by the Navtech sensor
        polar_points (np.ndarray): N x 2 array of points (azimuth_bin, range_bin)
        radar_resolution (float): Resolution of the polar radar data (metres per pixel)
        downsample_rate (float): fft data may be downsampled along the range dimensions to speed up computation
    Returns:
        np.ndarray: N x 2 array of points (x, y) in metric
    """
    N = polar_points.shape[0]
    cart_points = np.zeros((N, 2))
    for i in range(0, N):
        azimuth = azimuths[polar_points[i, 0]]
        r = polar_points[i, 1] * radar_resolution * downsample_rate + radar_resolution / 2
        cart_points[i, 0] = r * np.cos(azimuth)
        cart_points[i, 1] = r * np.sin(azimuth)
    return cart_points

def convert_to_bev(cart_points: np.ndarray, cart_resolution: float, cart_pixel_width: int) -> np.ndarray:
    """Converts points from metric cartesian coordinates to pixel coordinates in the BEV image
    Args:
        cart_points (np.ndarray): N x 2 array of points (x, y) in metric
        cart_pixel_width (int): width and height of the output BEV image
    Returns:
        np.ndarray: N x 2 array of points (u, v) in pixels which can be plotted on the BEV image
    """
    if (cart_pixel_width % 2) == 0:
        cart_min_range = (cart_pixel_width / 2 - 0.5) * cart_resolution
    else:
        cart_min_range = cart_pixel_width // 2 * cart_resolution
    print(cart_min_range)
    pixels = []
    N = cart_points.shape[0]
    for i in range(0, N):
        u = (cart_min_range + cart_points[i, 1]) / cart_resolution
        v = (cart_min_range - cart_points[i, 0]) / cart_resolution
        if 0 < u and u < cart_pixel_width and 0 < v and v < cart_pixel_width:
            pixels.append((u, v))
    return np.asarray(pixels)