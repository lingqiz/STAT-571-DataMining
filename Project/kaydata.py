import os
import numpy as np

class DataLoader():

    @staticmethod
    def split_data(dataset, n_test, n_val, shuffle_idx):        
        return (dataset[shuffle_idx[n_test+n_val:]], \
                dataset[shuffle_idx[0:n_test]], \
                dataset[shuffle_idx[n_test:n_test+n_val]])
    
    # Class Methods
    def __init__(self, combine=True):
        fname = ['kay_labels.npy', 'kay_labels_val.npy', 'kay_images.npz']
        path = ['r638s', 'yqb3e', 'ymnjv']

        for name, url in zip(fname, path):
            if not os.path.exists(name):
                os.system('wget -qO $fname https://osf.io/%s/download' % url)

        with np.load(fname[-1]) as obj:
            dat = dict(**obj)

        labels = np.load('kay_labels.npy')
        val_labels = np.load('kay_labels_val.npy')
        
        if combine:
            self.stimuli = np.concatenate((dat['stimuli'], dat['stimuli_test']))
            self.response = np.concatenate((dat['responses'], dat['responses_test']))
            self.labels = np.concatenate((labels, val_labels), axis=1)
        else:
            self.stimuli = dat['stimuli']
            self.response = dat['responses']
            self.labels = labels

        self.roi = dat['roi']
        self.roi_name = dat['roi_names']

    def select_roi(self, roi_names):
        # ROI index for each name
        roi_idx = [np.where(self.roi_name == name)[0][0] for name in roi_names]

        # Get the voxel responses
        index = [self.roi == idx for idx in roi_idx]
        return self.response[:, np.logical_or.reduce(index)]
        