from tools import compress_values
import seaborn as sns
from copy import deepcopy
import numpy as np
from tempfile import NamedTemporaryFile
import nibabel as nib
from surfer import project_volume_data

def surf_clusters(brain, nifti, colormap=None, level_mask=None, **kwargs):
    """" Display a nifti image of a clustering solution (discrete values) onto a pysurfer brain bilaterally
    Args:
    brain - pysurfer brain
    nifti - nifti image to display
    colormap - colormap to use, if none uses husl palette
    spatial_mask - Optional spatial mask to apply. 
    level_mask - Optionally mask certain clusters (levels) of the image """

    args = {'thresh': 0.001, 'alpha': 0.8,
            'colorbar': False, 'remove_existing': True, 'min': 1}
    if kwargs != {}:
        args.update(kwargs)

    if colormap is None:
        n_clusters = int(nifti.get_data().max())
        colormap = sns.color_palette('husl', n_clusters)
        from random import shuffle
        shuffle(colormap)

    if level_mask is not None:
        nifti = deepcopy(nifti)
        data = nifti.get_data()
        unique = np.unique(data[data.nonzero()])

        for val in unique:
            if not val in level_mask:
                data[data == val] = float(0)

        unique = np.unique(data[data.nonzero()])
        colormap = [v for i, v in enumerate(colormap) if i + 1 in unique]

        compress_values(nifti.get_data())

    with NamedTemporaryFile(suffix='.nii.gz') as f:
        nib.save(nifti, f.name)

        l_roi_surf = project_volume_data(f.name, "lh",
                                         subject_id="fsaverage", projsum='max', smooth_fwhm=0)
        r_roi_surf = project_volume_data(f.name, "rh",
                                         subject_id="fsaverage", projsum='max', smooth_fwhm=0)

        # Remap colors given that file is discrete
        l_cols = [colormap[int(np.round(c)) - 1] for c in np.unique(l_roi_surf)[1:]]
        if len(l_cols) < 2:
            l_cols = l_cols + [(0, 0, 0)]

        r_cols = [colormap[int(np.round(c)) - 1] for c in np.unique(r_roi_surf)[1:]]
        if len(r_cols) < 2:
            r_cols = r_cols + [(0, 0, 0)]

        brain.add_data(l_roi_surf, hemi='lh', colormap=l_cols, **args)
        brain.add_data(r_roi_surf, hemi='rh', colormap=r_cols, **args)


def display_coactivation(brain, niftis, colormap=None, reduce_alpha_step = 0, **kwargs):
    args = {'thresh' : 0.001, 'alpha' : 0.85, 'colorbar' : False, 'min' : 0}
    if kwargs != {}:
        args.update(kwargs)

    if colormap is None:
        colormap = sns.color_palette('Set1', len(niftis))
    
    for i, image in enumerate(niftis):      
        with NamedTemporaryFile(suffix='.nii.gz') as f:
            nib.save(image, f.name)
             
            l_roi_surf = project_volume_data(f.name, "lh",
                                subject_id="fsaverage", smooth_fwhm=2)
            r_roi_surf = project_volume_data(f.name, "rh",
                                subject_id="fsaverage", smooth_fwhm=2)

            args['remove_existing'] = i == 0

            color = sns.light_palette(colormap[i], n_colors=10)[5:]
            if l_roi_surf.sum() > 0:
                brain.add_data(l_roi_surf, hemi='lh', colormap=color, **args)
            if r_roi_surf.sum() > 0:
                brain.add_data(r_roi_surf, hemi='rh', colormap=color, **args)
                
            args['alpha'] -= reduce_alpha_step
