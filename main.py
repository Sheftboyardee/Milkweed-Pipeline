import pipe
import warnings
warnings.filterwarnings('ignore')

# Customize File Path to CSV
# Pipeline requires csv with last two columns containing species and ID information
FILE_PATH = '../data/Char_selected_no_speciosa.csv'

# Resize Plot as needed, I like to stick with squares
PLOT_SIZE = (4,4)



my_data = pipe.Pipeline(file_path = FILE_PATH, plot_size = PLOT_SIZE, species = 'species', ID = 'ID')
my_data.choose_file()