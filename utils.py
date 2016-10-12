import numpy as np
from copy import deepcopy
from sys import stdout

def compress_values(array):
    """ Given an array with potentially non contiguous numbers, compress them """
    unique = np.unique(array)
    d = dict(zip(unique, np.arange(0, unique.shape[0])))
    
    for k, v in d.iteritems(): array[array==k] = v
    return array

def select_clusters(clustering, mask, method='percentage', amount = .3, compress=True):
    """" From a clustering solution, select clusters with a certain amount of 
    voxels within a given mask. 
    
    Arguments:
    clustering - A nibabel image representing the clustering
    mask - A nibabel image mask
    method - Percentage of number of voxels
    amount - argumenrt for method
    compress - should clustering be reindexed to only include continuous numbers?

    """
    from copy import deepcopy
    
    clustering_copy = deepcopy(clustering)

    clustering = clustering.get_data()
    mask = mask.get_data()

    unique_values = np.unique(clustering)
    unique_values = unique_values[unique_values.nonzero()]

    def amnt_inmask(level, clustering, mask, method='percentage'):
        # Make cluster mask
        cluster_mask = clustering == level

        if method == 'percentage':
        	results = mask[cluster_mask].mean()
        elif method == 'sum':
        	results = mask[cluster_mask].sum()

        return results

    cluster_perc_in = np.array([amnt_inmask(level, clustering, mask, method=method) for level in unique_values])

    values_in = unique_values[cluster_perc_in >= amount]
    
    in_mask = np.in1d(clustering, values_in).reshape(clustering.shape)
    
    clustering_copy.get_data()[in_mask == False] = 0

    if compress is True:
    	_ = compress_values(clustering_copy.get_data())

    return in_mask, clustering_copy

def mask_nifti(nifti, mask):
    """ Mask a nifti image using a nifti mask """
    masked_nib = deepcopy(nifti)
    masked_nib.get_data()[mask.get_data() == 0] = 0
    return masked_nib

class ProgressBar():
    """" Custom Progress Bar used during classification process """
    def __init__(self, total, start=False):
        """
        total - total number of events
        start - automatically start when created? 
        """
        self.total = total
        self.current = 0.0
        self.last_int = 0
        if start:
            self.next()

    def update_progress(self, progress):
        display = '\r[{0}] {1}%'.format('#' * (progress / 10), progress)
        stdout.write(display)
        stdout.flush()

    def next(self):
        """ Increment progress bar"""
        if not self.last_int == int((self.current) / self.total * 100):
            self.update_progress(int((self.current) / self.total * 100))
            self.last_int = int((self.current) / self.total * 100)

        if self.current == self.total:
            self.reset()
        else:
            self.current = self.current + 1

    def reset(self):
        print ""
        self.current = 0.0

def mask_diagonal(masked_array):
    """ Given a masked array, it returns the same array with the diagonals masked"""
    if len(masked_array.shape) == 3:
        i, j, k = np.meshgrid(
            *map(np.arange, masked_array.shape), indexing='ij')
        masked_array.mask = (i == j)
    elif len(masked_array.shape) == 2:
        i, j = np.meshgrid(
            *map(np.arange, masked_array.shape), indexing='ij')
        masked_array.mask = (i == j)

    return masked_array