import cv2
import numpy as np
import os

import sys
parent_folder = "/home/samqiao/ASRL/vtr3_testing"

# Insert path at index 0 so it's searched first
sys.path.insert(0, parent_folder)

# Load images
img1 = cv2.imread('scripts/direct/unit_tests/unit_test_data/1738179490.9871147_scan.png')
img2 = cv2.imread('scripts/direct/unit_tests/unit_test_data/1738179491.236846.png')

outpath = "scripts/direct/unit_tests"

# Ensure same size
if img1.shape != img2.shape:
    raise ValueError("Images must be the same size")

# Compute absolute difference
diff = cv2.absdiff(img1, img2)

# Highlight differences (optional)
gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_diff, 30, 255, cv2.THRESH_BINARY)

# Count non-zero pixels
num_different_pixels = np.count_nonzero(thresh)
print(f"Number of different pixels: {num_different_pixels}")

# Save or display the diff
cv2.imwrite(os.path.join(outpath,'difference.png'), diff)
