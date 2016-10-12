import numpy as np
from utils import ProgressBar, mask_diagonal
from joblib import Parallel, delayed
import pandas as pd 
from scipy.stats import norm

from neurosynth.analysis.classify import classify

def classify_parallel(classifier, scoring, region_data, importance_function):
    """ Parallel classification function. Used to classify for each region if study 
    was activated or not (typically based on neurosynth features)
    classifier: sklearn classifier
    scoring: sklearn scoring function
    region_data: contains (X, y) data for a given region
    importance function: function to format importance vector (i.e. what to pull out from fitted classifier)

    returns summary dictionary with score, importance, preditions and importance vectors """

    X, y = region_data

    output = classify(
        X, y, classifier=classifier, cross_val='4-Fold', scoring=scoring)
    output['importance'] = importance_function(output['clf'].clf)
    return output

def log_odds_ratio(clf):
    """ Extracts log odds-ratio from naive bayes classifier """
    return np.log(clf.theta_[1] / clf.theta_[0])

class RegionalClassifier(object):

    """" Object used to classify on a region by region basis (from a cluster solution) 
    if studies activated a region using Neurosynth features (e.g. topics) 
    as classification features """

    def __init__(self, dataset, mask_img, classifier=None, cv='4-Fold',
                 thresh=0.05, thresh_low=0):
        """
        dataset - Neurosynth dataset
        mask_img - Path to Nifti image containing discrete regions coded as levels 
        classifier - sklearn classifier
        cv - cross validation strategy
        thresh - Threshold used to determine if a study is considered to have activated a region
        thresh_low - Threshold used to determine if a study is considered to be inactivate in a region

        """
        self.mask_img = mask_img
        self.classifier = classifier
        self.dataset = dataset
        self.thresh = thresh
        self.thresh_low = thresh_low
        self.cv = cv
        self.data = None

    def load_data(self):
        """ Loads data to set up classificaiton problem. Most importantly self.data is filled in, which consists
        of a Numpy array (length = number of regions) with X and y data for each region """
        from neurosynth.analysis.reduce import average_within_regions

        all_ids = self.dataset.image_table.ids

        high_thresh = average_within_regions(
            self.dataset, self.mask_img, threshold=self.thresh)
        low_thresh = average_within_regions(
            self.dataset, self.mask_img, threshold=self.thresh_low)

        self.data = np.empty(high_thresh.shape[0], dtype=np.object)
        for i, on_mask in enumerate(high_thresh):
            on_data = self.dataset.get_feature_data(
                ids=np.array(all_ids)[np.where(on_mask == True)[0]]).dropna()

            off_mask = low_thresh[i]
            off_ids = list(
                set(all_ids) - set(np.array(all_ids)[np.where(off_mask == True)[0]]))
            off_data = self.dataset.feature_table.get_feature_data(
                ids=off_ids).dropna()

            y = np.array([0] * off_data.shape[0] + [1] * on_data.shape[0])
            X = np.vstack((np.array(off_data), np.array(on_data)))

            from sklearn.preprocessing import scale
            X = scale(X, with_mean=False)
            self.data[i] = (X, y)

        self.feature_names = self.dataset.get_feature_data().columns.tolist()
        self.n_regions = self.data.shape[0]

    def initalize_containers(self):
        """ Makes all the containers that will hold results from classificaiton """
        self.class_score = np.zeros(self.n_regions)
        self.predictions = np.empty(self.n_regions, np.object)
        self.importance = mask_diagonal(
            np.ma.masked_array(np.zeros((self.n_regions, len(self.feature_names)))))
        self.fit_clfs = np.empty(self.n_regions, np.object)

    def classify(self, scoring='accuracy', n_jobs=1, importance_function=None):
        """
        scoring -  scoring function or type (str)
        n_jobs - Number of parallel jobs
        importance_function - Function to extract importance vectors from classifiers (differs by algorithm)
        """
        if importance_function is None:
            importance_function = log_odds_ratio

        if self.data is None:
            self.load_data()
            self.initalize_containers()

        print("Classifying...")
        pb = ProgressBar(self.n_regions, start=True)

        for index, output in enumerate(Parallel(n_jobs=n_jobs)(
                delayed(classify_parallel)(
                    self.classifier, scoring, region_data, importance_function) for region_data in self.data)):
            self.class_score[index] = output['score']
            self.importance[index] = output['importance']
            self.predictions[index] = output['predictions']
            pb.next()

    def get_formatted_importances(self, feature_names=None):
        """ Returns a pandas table of importances for each feature for each region. 
        Optionally takes new names for each feature (i.e. nickanames) """
        import pandas as pd
        if feature_names is None:
            feature_names = self.feature_names

        o_fi = pd.DataFrame(self.importance, columns=feature_names)

        # Melt feature importances, and add top_words for each feeature
        o_fi['region'] = range(1, o_fi.shape[0] + 1)
        return pd.melt(o_fi, var_name='feature', value_name='importance', id_vars=['region'])

def permutation_parallel(X, y, cla, feat_names, region, i):   
    newY = np.random.permutation(y)
    cla_fits = cla.fit(X, newY)
    fit_w = np.log(cla_fits.theta_[1] / cla_fits.theta_[0])
    
    results = []
    for n, lo in enumerate(fit_w):
        results.append([region + 1, i, feat_names[n], lo])
        
    return results

def permute_log_odds(clf, boot_n, feature_names=None, region_names = None, n_jobs=1):
    """ Given a fitted RegionalClassifier object, permute the column "importances" (i.e. log odds ratios)
    by resampling across studies. The function returns a pandas dataframe with z-score and p-values for each
    combination between a region and a topic in the Dataset """

    def z_score_array(arr, dist):
        return np.array([(v - dist[dist.region == i + 1].lor.mean()) / dist[dist.region == i + 1].lor.std() 
                         for i, v in enumerate(arr.tolist())])
                                           
    pb = ProgressBar(len(clf.data), start=True)
    overall_results = []
    
    if feature_names is None:
        feature_names = clf.feature_names

    if region_names is None:
        region_names = range(1, len(clf.data) + 1)

    # For each region, run boot_n number of permutations in parallel, and save to a list
    for reg, (X, y) in enumerate(clf.data):
        results = Parallel(n_jobs = n_jobs)(delayed(permutation_parallel)(
            X, y, clf.classifier, feature_names, reg, i) for i in range(boot_n))
        for result in results:
            for res in result:
                overall_results.append(res)
        pb.next()
                                             
    # Combine permuted data to a dataframe                                           
    perm_results = pd.DataFrame(overall_results, columns=['region', 'perm_n', 'topic_name', 'lor'])

    # Reshape observed log odds ratios with real data, z-score observed value on permuted null distribution
    lor = pd.DataFrame(clf.importance, index=range(1, clf.importance.shape[0] + 1), columns=feature_names)
    lor_z = lor.apply(lambda x: z_score_array(x, perm_results[perm_results.topic_name == x.name]))
    lor_z.index = region_names
                             
    # Transform to long format and add p-values
    all_roi_z = pd.melt(pd.concat([lor_z]).reset_index(),value_name='lor_z', id_vars='index')
    all_roi_z = all_roi_z.rename(columns={'index' : 'ROI'})
    all_roi_z['p'] = (1 - norm.cdf(all_roi_z.lor_z.abs())) * 2

    return all_roi_z

def bootstrap_parallel(X, y, cla, feat_names, region, i):
    ## Split into classes
    X0 = X[y == 0]
    X1 = X[y == 1]

    ## Sample with replacement from each class
    X0_boot = X0[np.random.choice(X0.shape[0], X0.shape[0])]
    X1_boot = X1[np.random.choice(X1.shape[0], X1.shape[0])]

    # Recombine
    X_boot = np.vstack([X0_boot, X1_boot])
    
    cla_fits = cla.fit(X_boot, y)
    fit_w = np.log(cla_fits.theta_[1] / cla_fits.theta_[0])
    
    results = []
    for n, lo in enumerate(fit_w):
        results.append([region, i, feat_names[n], lo])
        
    return results
def bootstrap_log_odds(clf, boot_n, feature_names=None, region_names = None, n_jobs=1):

    def percentile(n):
        def percentile_(x):
            return np.percentile(x, n)
        percentile_.__name__ = 'percentile_%s' % n
        return percentile_

    pb = ProgressBar(len(clf.data), start=True)

    if feature_names is None:
        feature_names = clf.feature_names

    if region_names is None:
        region_names = range(1, len(clf.data) + 1)

    # For each region, calculate in parallel bootstrapped lor estimates
    overall_boot = []
    for reg, (X, y) in enumerate(clf.data):
        results = Parallel(n_jobs = n_jobs)(delayed(bootstrap_parallel)(
            X, y, clf.classifier, feature_names, region_names[reg], i) for i in range(boot_n))
        for result in results:
            for res in result:
                overall_boot.append(res)
        pb.next()
            
    overall_boot = pd.DataFrame(overall_boot, columns=['region', 'perm_n', 'topic_name', 'fi'])

    # Calculate the 95% confidence intervals from the bootstrapped samples
    return overall_boot.groupby(['region', 'topic_name'])['fi'].agg({'mean' : np.mean, 'low_ci' : percentile(2.5), 'hi_ci' : percentile(97.5)}).reset_index()