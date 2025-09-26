RAW_MDATA_PATH = '../data/stwg-v2.h5mu'
MDATA_PATH = '../data/stwg-v2-filtered.h5mu'
OUTPUT_DIR = '../analysis/'
JASPAR_PATH = '../data/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.txt'
REFTSS_PATH = '../data/reftss.pkl'
HPA_PATH = '../data/hpa_tfs.pkl'
CHIPATLAS_PATH = '../data/chipatlas_kidney_promoters.pkl'

PLATFORM = 'batch'
SAMPLE = 'sample'
CELLTYPE = 'celltype'
GEX = 'rna'
ACC = 'atac'

MIN_CELLS = 30
PROXIMAL_BP = 5000
RANDOM_STATE = 0
NORMALIZE_TOTAL = False

# downsampling
NUM_CELLS = None
READS_PER_CELL = None

NUM_TOP_GENES = 1000

MIN_SAMPLES = -1

NUM_TREES = 20
LEARNING_RATE = 0.5

MAX_DEPTH = None
EARLY_STOPPING = 3

FEATURE_FRACTION = 1
BAGGING_FRACTION = 1

LEAVE_P_OUT = 2

IMPORTANCE_THRESHOLD = 0.95
CORRELATION_THRESHOLD = 0.2

_vars = {k: v for k, v in locals().items() if not k.startswith('_')}
for k, v in _vars.items():
        print(f"{k}={v}")