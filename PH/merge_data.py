import sys
sys.path.append('/Users/utkarshagrawal/Documents/Postdoc/U_1_haar')
from merge_utils import merge

measurement_type = 'weak'
assert measurement_type in ['weak','proj'], print('measurement_type wrong')

setup = 'purification'
evolution_type = 'matchgate'
scrambling_type = 'matchgate'

merge(measurement_type,evolution_type,scrambling_type,setup)