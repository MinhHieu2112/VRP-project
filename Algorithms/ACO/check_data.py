import pandas as pd
import numpy as np
import os

# Check distance matrix
matrix_file = os.path.join('..', 'Data', 'orsm_matrix.csv')
df = pd.read_csv(matrix_file, header=None)
mat = df.values

print('Matrix shape:', mat.shape)
print('Zero values on diagonal:', np.sum(np.diag(mat) == 0))
print('Zero values off-diagonal:', np.sum((mat == 0) & ~np.eye(mat.shape[0], dtype=bool)))
print('Min non-zero value:', np.min(mat[mat > 0]) if np.any(mat > 0) else 'No positive values')
print('Has negative values:', np.any(mat < 0))
print('Has inf values:', np.any(np.isinf(mat)))
print('Has nan values:', np.any(np.isnan(mat)))

# Check locations
locations_file = os.path.join('..', 'Data', 'locations.csv')
loc_df = pd.read_csv(locations_file)
print('Locations shape:', loc_df.shape)
print('Location columns:', list(loc_df.columns))