import numpy as np
import csv
def quaternion_to_rotation_matrix(q):
    qw, qx, qy, qz = q
    return np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
import numpy as np

# Load first 1000 ground truth rotations from MH_01_easy
with open('data/machine_hall/MH_01_easy/mav0/state_groundtruth_estimate0/data.csv', 'r') as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    R_meas = []
    for i, row in enumerate(reader):
        if i >= 1000:
            break
        qw, qx, qy, qz = map(float, row[4:8])
        R_meas.append(quaternion_to_rotation_matrix([qw, qx, qy, qz]))

# Save to NPZ file
np.savez('euroc_mav_subset.npz', *[R for R in R_meas])
print("Saved euroc_mav_subset.npz with {} rotations".format(len(R_meas)))