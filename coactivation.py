from neurosynth.analysis.meta import MetaAnalysis
import nibabel as nib
import numpy as np
from copy import deepcopy

def mask_level(img, level):
    img = deepcopy(img)
    data = img.get_data()
    data[:] = np.round(data)
    data[data != level] = 0 
    data[data == level] = 1
    
    return img

def coactivation_contrast(dataset, infile, regions=None, target_thresh=0.03, 
                          other_thresh=0.01, q=0.01, contrast='others'):
    """ Performs meta-analyses to contrast co-activation in a target region vs
    co-activation of other regions. Contrasts every region in "regions" vs
    the other regions in "regions"
    
    dataset: Neurosynth dataset
    infile: Nifti file with masks as levels
    regions: which regions in image to contrast
    target_thresh: activaton threshold for retrieving ids for target region
    other_thresh: activation threshold for ids in other regions
                  - This should be proportionally lower than target thresh since
                    multiple regions are being contrasted to one, and thus should de-weighed
    stat: which image to return from meta-analyis. Default is usually correct
    
    returns: a list of nifti images for each contrast performed of length = len(regions) """

    if isinstance(infile, str):
        image = nib.load(infile)
    else:
        image = infile

    affine = image.get_affine()

    stat="pFgA_z_FDR_%s" % str(q)
        
    meta_analyses = []
    for reg in regions:
        if contrast == 'others':
            other_ids = [dataset.get_studies(mask=mask_level(image, a), activation_threshold=other_thresh)
                             for a in regions if a != reg]
            joined_ids = set()
            for ids in other_ids:
                joined_ids = joined_ids | set(ids)
            joined_ids = list(joined_ids)
        elif contrast == 'joint': 
            mask = nib.Nifti1Image((image.get_data() != 0).astype('int'), affine=image.get_affine())
            joined_ids = dataset.get_studies(mask = mask, activation_threshold=other_thresh)
        else:
            joined_ids = None

        reg_ids = dataset.get_studies(mask=mask_level(image, reg), activation_threshold=target_thresh)
        meta_analyses.append(MetaAnalysis(dataset, reg_ids, ids2=joined_ids, q=q))
        
    return [nib.nifti1.Nifti1Image(dataset.masker.unmask(
                ma.images[stat]), affine, dataset.masker.get_header()) for ma in meta_analyses]

