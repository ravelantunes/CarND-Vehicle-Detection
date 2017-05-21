# color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
# orient = 9  # HOG orientations
# pix_per_cell = 8 # HOG pixels per cell
# cell_per_block = 2 # HOG cells per block
# hog_channel = 2 # Can be 0, 1, 2, or "ALL"
# spatial_size = (8, 8) # Spatial binning dimensions
# hist_bins = 128    # Number of histogram bins
# spatial_feat = True # Spatial features on or off
# hist_feat = True # Histogram features on or off
# hog_feat = True # HOG features on or off
# y_start_stop = [None, None] # Min and max in y to search in slide_window()

def get_parameters():
    parameters = {
        'color_space': 'YCrCb',  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
        'orient': 9,  # HOG orientations
        'pix_per_cell': 8, # HOG pixels per cell
        'cell_per_block': 2,  # HOG cells per block
        'hog_channel': 'ALL', # Can be 0, 1, 2, or "ALL"
        'spatial_size': (8, 8),  # Spatial binning dimensions
        'hist_bins': 128,  # Number of histogram bins
        'spatial_feat': True, # Spatial features on or off
        'hist_feat': True,  # Histogram features on or off
        'hog_feat': True , # HOG features on or off
        'y_start_stop': [None, None]  # Min and max in y to search in slide_window()
    }
    return parameters


# (9 orient, 9pix per cell, 5 cell per block)
# Channel H (HSV) only: 0.9611
# Channel S (HSV) only: 0.9544
# Channel V (HSV) only: 0.9842

# Spat 32x32 only: 0.9837 (3072)
# Spat 16x16 only: 0.9876 (768)
# Spat 8x8 only: 0.9823 (192)
# Spat 4x4 only: 0.962 (48)

# Color hist 256 bin: 0.9718 (768f)
# Color hist 128 bin: 0.9716 (384f)
# Color hist 64 bin: 0.9648 (192f)
# Color hist 32 bin: 0.9451 (96f)
# Color hist 16 bin: 0.9451 (48f)
# Color hist 8 bin: 0.9448 (24f)