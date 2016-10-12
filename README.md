# neurosynth-lfc
Final parcellation images are available under images/

Follow along the Clustering, Coactivation, and Functional preference profiles notebooks to recreate analyses, results and visualizations from the article. These notebooks are intended to allow researchers to easily perform similar analyses on other brain areas of interest.

Requirements

- Python 2.7.x
- neurosynth (github.com/neurosynth/neurosynth)
 Note: Install directly from github: pip install git+https://github.com/neurosynth/neurosynth.git
- Scipy/Numpy
- scikit-learn
- joblib 0.10
- nibabel 1.x
- fastcluster

For visualization:


- pandas
- seaborn
- pysurfer
  - Note, pysurfer can be difficult to install. Feel free to visualize using nilearn or your package of choice instead. I've had success - installing it under canopy.

Unzip pre-generated Neurosynth dataset prior to analysis
