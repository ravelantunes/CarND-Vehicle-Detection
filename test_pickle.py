import pickle
pickle_file = 'model-only.pickle'
with open(pickle_file, 'rb') as f:
    clf = pickle.load(f)
    f.close()

import parameters
p = parameters.get_parameters()



print(p['pix_per_cell'])
print(clf)

