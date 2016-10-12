from tools import compress_values
import seaborn as sns
from copy import deepcopy
import numpy as np
from tempfile import NamedTemporaryFile
import nibabel as nib
from surfer import project_volume_data
import matplotlib.pyplot as plt
import pylab as pl

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


def surf_coactivation(brain, niftis, colormap=None, reduce_alpha_step = 0, **kwargs):
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

def plot_clf_polar(importances, palette=None, mask=None, **kwargs):
    """ Make polar plot for classificaiton results.
    importances - formatted importances
    palette -  Colors to use for each region
    mask - List of which regions to include, by default uses all """
    import pandas as pd
    import seaborn as sns
    
    if mask is not None:
        importances = importances[importances.region.isin(mask)]
    
    pplot = pd.pivot_table(importances, values='importance', index='feature', columns=['region'])

    if palette is None:
        palette = sns.color_palette('Set1', importances.region.unique().shape[0])
    if mask is not None:
        palette = [n[0] for n in sorted(zip(np.array(palette)[np.array(mask)-1], mask), key=lambda tup: tup[1])]

    return plot_polar(pplot, overplot=True, palette=palette, **kwargs)


def plot_polar(data, n_top=3, selection='top', overplot=False, labels=None,
               palette='husl', reorder=False, method='weighted', metric='correlation', 
               label_size=26, threshold=None, max_val=None,
               alpha_level=1, legend=False, error_bars=None, min_val=-0.85):
    """ Make a polar plot
    data - Tabular data of shape features x classes 
    n_top - Number of features to select
    selection - Selection method to use `
                (top = M strongest for each class; std = N with greatest std across all)
    overplot - Overlap plots for each class?
    labels - Subset of features to use (overrides auto selection by n_top)
    palette - Color palette to use (can be label or list of colors from seaborn)
    reorder - If True, uses hierarchical clustering to reorder axis
    method - Method to use for clustering
    metric - Metric to use for clustering
    label_size - X axis label size
    threshold - Value to draw an optional line that denotes significance threshold
    max_val - Maximum value of y axis
    min_val - Minimum value of y axis
    alpha_level - transparency value for lines
    legend - Show legend?
    error_bars - Option bootstrapped data to draw error bars """

    n_panels = data.shape[1]

    if labels is None:

        if selection == 'top':
            labels = []
            for i in range(n_panels):
                labels.extend(data.iloc[:, i].sort_values(ascending=False) \
                    .index[:n_top])
            labels = np.unique(labels)
        elif selection == 'std':
            labels = data.T.std().sort_values(ascending=False).index[:n_top]

        data = data.loc[labels,:]
    
    else:
        data = data.loc[labels,:]

    if error_bars is not None:
        error_bars = error_bars.loc[labels,:]

    if reorder is True:
        # Use hierarchical clustering to order
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage, leaves_list
        dists = pdist(data, metric=metric)
        pairs = linkage(dists, method=method)
        pairs[pairs < 0] = 0
        order = leaves_list(pairs)
        data = data.iloc[order,:]

        if error_bars is not None:
            error_bars = error_bars.iloc[order,:]

        labels = [labels[i] for i in order]


    theta = np.linspace(0.0, 2 * np.pi, len(labels), endpoint=False)
    
    ## Add first
    theta = np.concatenate([theta, [theta[0]]])
    if overplot:
        fig, ax = plt.subplots(1, 1, subplot_kw=dict(polar=True))
        fig.set_size_inches(10, 10)
    else:
        fig, axes = plt.subplots(n_panels, 1, sharex=False, sharey=False,
                             subplot_kw=dict(polar=True))
        fig.set_size_inches((6, 6 * n_panels))
        
    if isinstance(palette, str):
        from seaborn import color_palette
        colors = color_palette(palette, n_panels)
    else:
        colors = palette

    for i in range(n_panels):
        if overplot:
            alpha = 0.025
        else:
            ax = axes[i]
            alpha = 0.8

        if max_val is None:
            if error_bars is not None:
                max_val = data.values.max() + error_bars.values.max() + data.values.max() * .02
            else:
                max_val = data.values.max()
        
        ax.set_ylim(min_val, max_val)
        
        d = data.iloc[:,i].values
        d = np.concatenate([d, [d[0]]])
        name = data.columns[i]

        if error_bars is not None:
            e = error_bars.iloc[:,i].values
            e = np.concatenate([e, [e[0]]])
        else:
            e = None

        if error_bars is not None:
            ax.errorbar(theta, d, yerr=e, capsize=0, color=colors[i], elinewidth = 3, linewidth=0)
        else:
            ax.plot(theta, d, alpha=alpha_level - 0.1, color=colors[i], linewidth=8, label=name)
            ax.fill(theta, d, ec='k', alpha=alpha, color=colors[i], linewidth=8)

        ax.set_xticks(theta)
        ax.set_rlabel_position(11.12)
        ax.set_xticklabels(labels, fontsize=label_size)
        [lab.set_fontsize(22) for lab in ax.get_yticklabels()]

    
    if threshold is not None:
        theta = np.linspace(0.0, 2 * np.pi, 999, endpoint=False)
        theta = np.concatenate([theta, [theta[0]]])
        d = np.array([threshold] * 1000)
        ax.plot(theta, d, alpha=1, color='black', linewidth=2, linestyle='--')

    if legend is True:
        ax.legend(bbox_to_anchor=(1.15, 1.1))

    circle = pl.Circle((0, 0), np.abs(min_val), transform=ax.transData._b, color="grey", alpha=0.22 )
    ax.add_artist(circle)

    plt.tight_layout()

    return labels, data