# config.py
OPENAI_API_KEY = 'your-actual-key'
MAPS_DIR = 'datafiles/inputs/maps'

# config.pys
COLOR_SCHEME = {
    'free_space': [1.0, 1.0, 1.0],
    'obstacle': [0.0, 0.0, 0.0],
    'path_start': [1.0, 0.0, 1.0],
    'path_end': [1.0, 1.0, 0.0],
    'searched': [0.0, 1.0, 0.0]
}
EXCEL_CONFIG = {
    'map_sheets': ['Sheet1', 'GridMap'],
    'data_types': {'map': int, 'reward': float}
}