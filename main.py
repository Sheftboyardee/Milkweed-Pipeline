import pipe

import warnings
warnings.filterwarnings('ignore')


# Resize Plot if needed
PLOT_SIZE = (4,4)

# Species and ID columns
SPECIES = 'species'
ID = 'ID'

my_data = pipe.Pipeline(plot_size = PLOT_SIZE, species = SPECIES, ID = ID)
my_data.choose_file()