#from __future__ import division # division should return floating points, see 054, weird result, not changed after the feature mapper

theLabel = "205_check_variability"

locationFolder = "C:/Kaggle/KaggleCauseEffect/PYTHON/205_software_package_check_variability/"
filenameJSON = "SETTINGS.json"

enableOrderCat = True
enableLogarithm = False
enableSymmetryAB = True

enablePrintNewCalculations = False

onlyBinaryBinary = False
onlyNumericalNumerical = False
onlyCatCat = False
onlyCatbinCatbin = False

trainOnlyCatbinToNumerical = False
trainOnlyNumericalToCatbin = False

# Reverse A or B when making the prediction to check if it matters. 
# If it does not we might make use of this to enhance our predictions.
# Result: don't do this, because it makes the predictions much worse!
reverseA = False
reverseB = False

enableLabelInPickle = False

# How many standard deviations do we need to exceed to be accounted as outlier?
thesholdForOutlier = 5

nJobs = 1
nEstimators = 500
writeImportances = False
printImportances = False

recalculateAllFeatures = True ######### <-------------

enableScoreSeparate = True

########################################################################################
#-# CE.PY
# Cause Effect from Kaggle
########################################################################################

import numpy as np
from scipy import stats
np.random.seed(12345678) # fix random seed to get same numbers
import pickle
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline


########################################################################################
#-# TRAIN.PY
########################################################################################

def feature_extractor():
    features = [
                #(['precision1', 'recall1', 'accuracy1', 'afstandROC1', 'afstandROC_accent1', 'precision2', 'recall2', 'afstandROC2', 'afstandROC_accent2', 'dif precision', 'dif recall', 'dif afstandROC', 'dif afstandROC_accent'], ['A','B'], MultiColumnTransform(statsBinaryBinary_dif)),
                ('A: Number of Samples', 'A', SimpleTransform(transformer=len)), # len(A) = len(B)
                ('A: Number of Unique Samples', 'A', SimpleTransform(transformer=count_unique)),
                ('B: Number of Unique Samples', 'B', SimpleTransform(transformer=count_unique)),
                (['kmeans 1', 'kmeans 2', 'kmeans 3', 'kmeans 4', 'kmeans 5', 'kmeans 6'], ['A','B'], MultiColumnTransform(find_kmeans)),
                (['polyfit4_remainder_1', 'polyfit4_remainder_2', 'polyfit4_remainder_3', 'polyfit4_remainder_4', 'polyfit4_remainder_5', 'polyfit4_remainder_6'], ['A','B'], MultiColumnTransform(polyfit4_remainder)),
                (['information_gain_YX', 'information_gain_XY', 'cumProbX', 'cumProbY'], ['A','B', 'AT', 'BT'], MultiColumnTransform(to_bin_bin_feats)),
                (['information_gain_YX max', 'information_gain_XY max', 'cumProbX max', 'cumProbY max', 'information_gain_YX min', 'information_gain_XY min', 'cumProbX min', 'cumProbY min'], ['A','B', 'AT', 'BT'], MultiColumnTransform(to_bin_bin_feats_minmax)),
                ('A: Fraction of Unique Samples', 'A', SimpleTransform(transformer=fraction_unique)),
                ('B: Fraction of Unique Samples', 'B', SimpleTransform(transformer=fraction_unique)),
                #('A: Normalized Entropy', 'A', SimpleTransform(transformer=normalized_entropy)),
                #('B: Normalized Entropy', 'B', SimpleTransform(transformer=normalized_entropy)),
                ('A: Normal Test Z', 'A', SimpleTransform(transformer=normaltest_z)),
                ('B: Normal Test Z', 'B', SimpleTransform(transformer=normaltest_z)),
                ('A: Normal Test P', 'A', SimpleTransform(transformer=normaltest_p)),
                ('B: Normal Test P', 'B', SimpleTransform(transformer=normaltest_p)),
                ('Normal test Z dif', ['A','B'], MultiColumnTransform(normaltest_z_dif)),
                ('Normal test P dif', ['A','B'], MultiColumnTransform(normaltest_p_dif)),
                ('A: Uniform Test Chi', 'A', SimpleTransform(transformer=uniform_test_chi)),
                ('B: Uniform Test Chi', 'B', SimpleTransform(transformer=uniform_test_chi)),
                ('A: Uniform Test P', 'A', SimpleTransform(transformer=uniform_test_p)),
                ('B: Uniform Test P', 'B', SimpleTransform(transformer=uniform_test_p)),
                ('Uniform test Z dif', ['A','B'], MultiColumnTransform(uniform_test_chi_dif)),
                ('Uniform test P dif', ['A','B'], MultiColumnTransform(uniform_test_p_dif)),
                ('A: Mean', 'A', SimpleTransform(transformer=my_mean)),
                ('B: Mean', 'B', SimpleTransform(transformer=my_mean)),
                ('A: Median', 'A', SimpleTransform(transformer=my_median)),
                ('B: Median', 'B', SimpleTransform(transformer=my_median)),
                ('A: Range', 'A', SimpleTransform(transformer=my_range)),
                ('B: Range', 'B', SimpleTransform(transformer=my_range)),
                ('A: Min', 'A', SimpleTransform(transformer=my_min)),
                ('B: Min', 'B', SimpleTransform(transformer=my_min)),
                ('A: Max', 'A', SimpleTransform(transformer=my_max)),
                ('B: Max', 'B', SimpleTransform(transformer=my_max)),
                ('A: Std', 'A', SimpleTransform(transformer=my_std)),
                ('B: Std', 'B', SimpleTransform(transformer=my_std)),
                (['shannonentropy_x', 'shannonentropy_y', 'shannonentropy_dif', 'condentropy_xy', 'condentropy_yx', 'condentropy_dif', 'mutualinfo_xy', 'corrent_xy', 'covent_xy'], ['A','B', 'AT', 'BT'], MultiColumnTransform(information_theory_other)),
                (['mutual_info_score', 'homogeneity_xy', 'homogeneity_yx', 'homogeneity_dif', 'completeness_dif', 'vmeasure'], ['A','B', 'AT', 'BT'], MultiColumnTransform(information_theory_sklearn_metrics_ew)),
                (['mutual_info_score ef', 'homogeneity_xy ef', 'homogeneity_yx ef', 'homogeneity_dif ef', 'completeness_dif ef', 'vmeasure ef'], ['A','B', 'AT', 'BT'], MultiColumnTransform(information_theory_sklearn_metrics_ef)),
                ('A: Percentile 01', 'A', SimpleTransform(transformer=percentile01)),
                ('B: Percentile 01', 'B', SimpleTransform(transformer=percentile01)),
                ('A: Percentile 02', 'A', SimpleTransform(transformer=percentile02)),
                ('B: Percentile 02', 'B', SimpleTransform(transformer=percentile02)),
                ('A: Percentile 05', 'A', SimpleTransform(transformer=percentile05)),
                ('B: Percentile 05', 'B', SimpleTransform(transformer=percentile05)),
                ('A: Percentile 10', 'A', SimpleTransform(transformer=percentile10)),
                ('B: Percentile 10', 'B', SimpleTransform(transformer=percentile10)),
                ('A: Percentile 25', 'A', SimpleTransform(transformer=percentile25)),
                ('B: Percentile 25', 'B', SimpleTransform(transformer=percentile25)),
                ('A: Percentile 75', 'A', SimpleTransform(transformer=percentile75)),
                ('B: Percentile 75', 'B', SimpleTransform(transformer=percentile75)),
                ('A: Percentile 90', 'A', SimpleTransform(transformer=percentile90)),
                ('B: Percentile 90', 'B', SimpleTransform(transformer=percentile90)),
                ('A: Percentile 95', 'A', SimpleTransform(transformer=percentile95)),
                ('B: Percentile 95', 'B', SimpleTransform(transformer=percentile95)),
                ('A: Percentile 010205', 'A', SimpleTransform(transformer=percentile010205)),
                ('B: Percentile 010205', 'B', SimpleTransform(transformer=percentile010205)),
                ('A: max in histogram', 'A', SimpleTransform(transformer=maxValueInHistogram)),
                ('B: max in histogram', 'B', SimpleTransform(transformer=maxValueInHistogram)),
                ('A: max in histogram eq', 'A', SimpleTransform(transformer=maxValueInHistogramEq)),
                ('B: max in histogram eq', 'B', SimpleTransform(transformer=maxValueInHistogramEq)),
                ('A: type', 'AT', SimpleTransform(transformer=my_type)),
                ('B: type', 'BT', SimpleTransform(transformer=my_type)),
                ('A: Normalized median', 'A', SimpleTransform(transformer=normalized_median)),
                ('B: Normalized median', 'B', SimpleTransform(transformer=normalized_median)),
                ('A: Normalized range', 'A', SimpleTransform(transformer=normalized_range)),
                ('B: Normalized range', 'B', SimpleTransform(transformer=normalized_range)),
                ('A: Skewness', 'A', SimpleTransform(transformer=skewness)),
                ('B: Skewness', 'B', SimpleTransform(transformer=skewness)),
                ('A: Kurtosis', 'A', SimpleTransform(transformer=kurtosis)),
                ('B: Kurtosis', 'B', SimpleTransform(transformer=kurtosis)),
                #('A: Skewtest Z', 'A', SimpleTransform(transformer=skewtest_z)),
                #('B: Skewtest Z', 'B', SimpleTransform(transformer=skewtest_z)),
                #('A: Skewtest P', 'A', SimpleTransform(transformer=skewtest_p)),
                #('B: Skewtest P', 'B', SimpleTransform(transformer=skewtest_p)),skewtest_both
                (['A: Skewtest Z', 'A: Skewtest P'], 'A', SimpleTransform(transformer=skewtest_both)),
                (['B: Skewtest Z', 'B: Skewtest P'], 'B', SimpleTransform(transformer=skewtest_both)),
                ('FractionNearMin A', 'A', SimpleTransform(transformer=fractionNearMin)),
                ('FractionNearMin B', 'B', SimpleTransform(transformer=fractionNearMin)),
                #('A: Fraction max', 'A', SimpleTransform(transformer=fractionNearMax)),
                #('B: Fraction max', 'B', SimpleTransform(transformer=fractionNearMax)),
                ('FractionNearMinMax A', 'A', SimpleTransform(transformer=fractionNearMinMax)),
                ('FractionNearMinMax B', 'B', SimpleTransform(transformer=fractionNearMinMax)),
                ('FractionNearMinMax dif', ['A','B'], MultiColumnTransform(fractionNearMinMax_dif)),
                ('Pearson R', ['A','B'], MultiColumnTransform(correlation)),
                ('Pearson R Magnitude', ['A','B'], MultiColumnTransform(correlation_magnitude)),
                ('Pearson R Magnitude Splitted 1std A B', ['A','B'], MultiColumnTransform(correlation_magnitude_splitted_1)),
                ('Pearson R Magnitude Splitted 2min A B', ['A','B'], MultiColumnTransform(correlation_magnitude_splitted_2)),
                ('Pearson R Magnitude Splitted 3max A B', ['A','B'], MultiColumnTransform(correlation_magnitude_splitted_3)),
                ('Pearson R Magnitude Splitted 1std r B A', ['B','A'], MultiColumnTransform(correlation_magnitude_splitted_1)),
                ('Pearson R Magnitude Splitted 2min r B A', ['B','A'], MultiColumnTransform(correlation_magnitude_splitted_2)),
                ('Pearson R Magnitude Splitted 3max r B A', ['B','A'], MultiColumnTransform(correlation_magnitude_splitted_3)),
                ('Pearson R Magnitude Splitted 1std d', ['A','B'], MultiColumnTransform(correlation_magnitude_splitted_1_dif)),
                ('Pearson R Magnitude Splitted 2min d', ['A','B'], MultiColumnTransform(correlation_magnitude_splitted_2_dif)),
                ('Pearson R Magnitude Splitted 3max d', ['A','B'], MultiColumnTransform(correlation_magnitude_splitted_3_dif)),
                ('Number of Unique Samples Difference', ['A','B'], MultiColumnTransform(count_unique_difference)),
                ('Number of Unique Samples Fraction', ['A','B'], MultiColumnTransform(count_unique_fraction)),
                ('Linear regression P value A B', ['A','B'], MultiColumnTransform(linear_regression_p)),
                ('Linear regression P value B A', ['B','A'], MultiColumnTransform(linear_regression_p)),
                ('Linear regression Std err A B', ['A','B'], MultiColumnTransform(linear_regression_stderr)),
                ('Linear regression Std err B A', ['B','A'], MultiColumnTransform(linear_regression_stderr)),
               #('Linear regression rsquared A B', ['A','B'], MultiColumnTransform(linear_regression_rsquared)),
               #('Linear regression rsquared B A', ['B','A'], MultiColumnTransform(linear_regression_rsquared)),
                ('Linear regression Slope A B',   ['A','B'], MultiColumnTransform(linear_regression_slope)),
                ('Linear regression Slope B A',   ['B','A'], MultiColumnTransform(linear_regression_slope)),
                ('Linear regression Interct A B', ['A','B'], MultiColumnTransform(linear_regression_intercept)),
                ('Linear regression Interct B A', ['B','A'], MultiColumnTransform(linear_regression_intercept)),
                ('Normal Normalized median difference', ['A','B'], MultiColumnTransform(normalized_median_difference)),
                ('Normal Substract Poly A B z', ['A','B'], MultiColumnTransform(normal_substract_poly_z)),
                ('Normal Substract Poly B A z', ['B','A'], MultiColumnTransform(normal_substract_poly_z)),
               #('Normal Substract Poly A B p', ['A','B'], MultiColumnTransform(normal_substract_poly_p)),
               #('Normal Substract Poly B A p', ['B','A'], MultiColumnTransform(normal_substract_poly_p)),
                ('Normal Substract Poly dif z', ['A','B'], MultiColumnTransform(normal_substract_poly_z_dif)),
                ('Normal Substract Poly dif p', ['A','B'], MultiColumnTransform(normal_substract_poly_p_dif)),
                #('Quadratic regression A B rsquared', ['A','B'], MultiColumnTransform(quadratic_regression_rsquared)),
                #('Quadratic regression B A rsquared', ['B','A'], MultiColumnTransform(quadratic_regression_rsquared)),
                #('Quadratic regression A B a', ['A','B'], MultiColumnTransform(quadratic_regression_a)),
                #('Quadratic regression B A a', ['B','A'], MultiColumnTransform(quadratic_regression_a)),
                #('Quadratic regression A B b', ['A','B'], MultiColumnTransform(quadratic_regression_b)),
                #('Quadratic regression B A b', ['B','A'], MultiColumnTransform(quadratic_regression_b)),
                #('Quadratic regression A B c', ['A','B'], MultiColumnTransform(quadratic_regression_c)),
                #('Quadratic regression B A c', ['B','A'], MultiColumnTransform(quadratic_regression_c)),
                #('Quadratic regression dif rsquared', ['A','B'], MultiColumnTransform(quadratic_regression_rsquared_difference)),
                #('Quadratic regression dif a', ['A','B'], MultiColumnTransform(quadratic_regression_a_difference)),
                #('Quadratic regression dif b', ['A','B'], MultiColumnTransform(quadratic_regression_b_difference)),
                #('Quadratic regression dif c', ['A','B'], MultiColumnTransform(quadratic_regression_c_difference)),
                (['Quadr reg rsqr A B','Quadr reg a A B','Quadr reg b A B','Quadr reg c A B',
                	'Quadr reg rsqr B A','Quadr reg a B A','Quadr reg b B A','Quadr reg c B A',
                	'Quadr reg rsqr dif','Quadr reg a dif','Quadr reg b dif','Quadr reg c dif'], ['A','B'], MultiColumnTransform(quadratic_regression_misc_misc)),
                ('Polyfit3 regression A B rsquared', ['A','B'], MultiColumnTransform(polyfit3_regression_rsquared)),
                ('Polyfit3 regression B A rsquared', ['B','A'], MultiColumnTransform(polyfit3_regression_rsquared)),
                ('Polyfit3 regression A B a', ['A','B'], MultiColumnTransform(polyfit3_regression_a)),
                ('Polyfit3 regression B A a', ['B','A'], MultiColumnTransform(polyfit3_regression_a)),
                ('Polyfit3 regression A B b', ['A','B'], MultiColumnTransform(polyfit3_regression_b)),
                ('Polyfit3 regression B A b', ['B','A'], MultiColumnTransform(polyfit3_regression_b)),
                ('Polyfit regression A B c', ['A','B'], MultiColumnTransform(polyfit3_regression_c)),
                ('Polyfit3 regression B A c', ['B','A'], MultiColumnTransform(polyfit3_regression_c)),
                ('Polyfit3 regression A B d', ['A','B'], MultiColumnTransform(polyfit3_regression_d)),
                ('Polyfit3 regression B A d', ['B','A'], MultiColumnTransform(polyfit3_regression_d)),
                ('Polyfit3 regression dif rsquared', ['A','B'], MultiColumnTransform(polyfit3_regression_rsquared_difference)),
                ('Polyfit3 regression dif a', ['A','B'], MultiColumnTransform(polyfit3_regression_a_difference)),
                ('Polyfit3 regression dif b', ['A','B'], MultiColumnTransform(polyfit3_regression_b_difference)),
                ('Polyfit3 regression dif c', ['A','B'], MultiColumnTransform(polyfit3_regression_c_difference)),
                ('Polyfit3 regression dif d', ['A','B'], MultiColumnTransform(polyfit3_regression_d_difference)),
                ('Polyfit4 regression A B rsquared', ['A','B'], MultiColumnTransform(polyfit4_regression_rsquared)),
                ('Polyfit4 regression B A rsquared', ['B','A'], MultiColumnTransform(polyfit4_regression_rsquared)),
                ('Spearman A B r', ['A','B'], MultiColumnTransform(spearmanr_r)),
                ('Spearman B A r', ['B','A'], MultiColumnTransform(spearmanr_r)),
                ('Spearman dif r', ['A','B'], MultiColumnTransform(spearmanr_r_dif)),
                ('Spearman A B p', ['A','B'], MultiColumnTransform(spearmanr_p)),
                ('Spearman B A p', ['B','A'], MultiColumnTransform(spearmanr_p)),
                ('Spearman dif p', ['A','B'], MultiColumnTransform(spearmanr_p_dif)),
                ('Moving avg A B', ['A','B'], MultiColumnTransform(moving_average_error)),
                ('Moving avg B A', ['B','A'], MultiColumnTransform(moving_average_error)),
                ('Moving avg dif', ['A','B'], MultiColumnTransform(moving_average_error_dif)),
                ('Moving avg logdif', ['A','B'], MultiColumnTransform(moving_average_error_logdif)),
                ('IGCI Uniform Entropy', ['A','B'], MultiColumnTransform(igci_uniform_entropy)),
                ('IGCI Uniform Integral', ['A','B'], MultiColumnTransform(igci_uniform_integral)),
                ('IGCI Gauss Entropy', ['A','B'], MultiColumnTransform(igci_gauss_entropy)),
                ('IGCI Gauss Integral', ['A','B'], MultiColumnTransform(igci_gauss_integral)),
                (['Logistic regression AB score', 'Logistic regression AB coef'], ['A','B', 'AT', 'BT'], MultiColumnTransform(logistic_regression_both)),
                (['Logistic regression BA score', 'Logistic regression BA coef'], ['B','A', 'BT', 'AT'], MultiColumnTransform(logistic_regression_both)),
                #('Logistic regression AB score', ['A','B', 'AT', 'BT'], MultiColumnTransform(logistic_regression_score)),
                #('Logistic regression AB coef', ['A','B', 'AT', 'BT'], MultiColumnTransform(logistic_regression_coef)),
                #('Logistic regression BA score', ['B','A', 'BT', 'AT'], MultiColumnTransform(logistic_regression_score)),
                #('Logistic regression BA coef', ['B','A', 'BT', 'AT'], MultiColumnTransform(logistic_regression_coef)),
                #('Kendalltau A B t', ['A','B'], MultiColumnTransform(kendalltau_t)),
                #('Kendalltau B A t', ['B','A'], MultiColumnTransform(kendalltau_t)),
                #('Kendalltau dif t', ['A','B'], MultiColumnTransform(kendalltau_t_dif)),
                #('Kendalltau A B p', ['A','B'], MultiColumnTransform(kendalltau_p)),
                #('Kendalltau B A p', ['B','A'], MultiColumnTransform(kendalltau_p)),
                #('Kendalltau dif p', ['A','B'], MultiColumnTransform(kendalltau_p_dif)),
                (['Outliers A percbelow', 'Outliers A percabove', 'Outliers A pertotal', 'Outliers A nrtotal'], 'A', SimpleTransform(transformer=find_outliers)),
                (['Outliers B percbelow', 'Outliers B percabove', 'Outliers B pertotal', 'Outliers B nrtotal'], 'B', SimpleTransform(transformer=find_outliers)),
                (['Outlier diff total', 'Outlier diff percent'], ['A','B'], MultiColumnTransform(transformer=find_outlier_dif)),
                ('Inversible AB', ['A','B'], MultiColumnTransform(transformer=check_inversibility_quad)),
                ('Inversible BA', ['B','A'], MultiColumnTransform(transformer=check_inversibility_quad)),
                ('Entropy Difference', ['A','B'], MultiColumnTransform(entropy_difference)),
                ('Min precision A', 'A', SimpleTransform(transformer=minPrecision)),
                ('Min precision B', 'B', SimpleTransform(transformer=minPrecision)),
                (['Best poly AB r squared', 'Best poly AB degree'], ['A','B'], MultiColumnTransform(transformer=polyfit_best_rsquared)),
                (['Best poly BA r squared', 'Best poly BA degree'], ['B','A'], MultiColumnTransform(transformer=polyfit_best_rsquared)),
                (['Best poly train AB r squared', 'Best poly train AB degree'], ['A','B'], MultiColumnTransform(transformer=polyfit_best_rsquared_2)),
                (['Best poly train BA r squared', 'Best poly train BA degree'], ['B','A'], MultiColumnTransform(transformer=polyfit_best_rsquared_2))
                ]
    
    recalculateFeatures = [ 
                ]
    
    combined = FeatureMapper(features, recalculateFeatures)
    return ( combined, features )

def get_pipeline():
    features, featuresdummy = feature_extractor() # gets featureMapper and list of it's features
    # adding extra step? Don't forget to change index in classifier.steps[...]
    steps = [("preprocess_log", PreprocessLogarithmizer(enableLogarithm)),
             ("preprocess_categorical", PreprocessCatOrderizer(enableOrderCat)),
             ("extract_features", features),
             ("save_features_file", SaveFeatures()),
             ("classify", MyClassifier2(n_estimators=nEstimators, 
                                                verbose=2,
                                                n_jobs=nJobs,
                                                compute_importances=True,
                                                oob_score=True,
                                                min_samples_split=10,
                                                random_state=1))]
    return Pipeline(steps)

def train():
    if (enablePrintNewCalculations):
    	print_new_calculations()
    print("Reading in the training data")
    train = read_train_pairs()
    target = read_train_target()
    traininfo = read_train_info()
    # and join them
    train = train.join(traininfo)
    train = train.join(target)
    
    # filter optionally
    if onlyBinaryBinary:
    	 train = train[train.AT==0]
    	 train = train[train.BT==0]
    if onlyCatCat:
    	 train = train[train.AT==1]
    	 train = train[train.BT==1]
    if onlyNumericalNumerical:
    	 train = train[train.AT==2]
    	 train = train[train.BT==2]
    if onlyCatbinCatbin:
    	 train = train[train.AT<2]
    	 train = train[train.BT<2]
    if trainOnlyCatbinToNumerical:
    	 train = train[train.AT<2]
    	 train = train[train.BT==2]
    if trainOnlyNumericalToCatbin:
    	 train = train[train.AT==2]
    	 train = train[train.BT<2]
    
    print("Extracting features and training model")
    classifier = get_pipeline()
    if (enableSymmetryAB) :
    	classifier.fit(train, train.AB)
    else:
    	classifier.fit(train, train.Target)

    print("Saving the classifier")
    save_model(classifier)
    
    print("Score the trained training set")
    labeldummy, boostedt = classifier.steps[-1] # index of last step in pipeline
    oobscore = boostedt.oob_score_
    print "oob score = ", oobscore
    #out_of_bag_prediction_for_x = boostedt.oob_prediction_
    #oobpredmean = np.mean(out_of_bag_prediction_for_x)
    #print "oob pred score = ", oobpredmean


########################################################################################
#-# PREDICT.PY
########################################################################################

import numpy as np
import pickle

def historic():
    print("Calculating correlations")
    calculate_pearsonr = lambda row: abs(pearsonr(row["A"], row["B"])[0])
    correlations = valid.apply(calculate_pearsonr, axis=1)
    correlations = np.array(correlations)

    print("Calculating causal relations")
    calculate_causal = lambda row: causal_relation(row["A"], row["B"])
    causal_relations = valid.apply(calculate_causal, axis=1)
    causal_relations = np.array(causal_relations)

    scores = correlations * causal_relations

def predictAll():
	validAB = read_valid_pairs()
	validinfo = read_valid_info()
	# and join them
	validAB = validAB.join(validinfo)
	validBA = validAB.rename(columns = dict(zip(validAB.columns, ["B", "A", "BTstr", "ATstr", "BT", "AT"])))
  
	validAB['resultAB'] = 0
	validBA['resultBA'] = 0
	validAB['resultAB_full'] = 0
	validBA['resultBA_full'] = 0
	
	print "*** PREDICTING PART 1 OF 5 *** catcat ***"
	classifiercatcat = pickle.load(open(get_paths()["model_path_catcat"]))
	predictionsABcatcat = classifiercatcat.predict(validAB[(validAB.AT<2)&(validAB.BT<2)])
	df_AB_catcat = pd.DataFrame(predictionsABcatcat ,index=validAB[(validAB.AT<2)&(validAB.BT<2)].index)
	df_AB_catcat = df_AB_catcat.rename(columns = dict(zip(df_AB_catcat.columns, ["resultAB"])))
	validAB.update(df_AB_catcat)
	predictionsBAcatcat = classifiercatcat.predict(validBA[(validBA.AT<2)&(validBA.BT<2)])
	df_BA_catcat = pd.DataFrame(predictionsBAcatcat ,index=validBA[(validBA.AT<2)&(validBA.BT<2)].index)
	df_BA_catcat = df_BA_catcat.rename(columns = dict(zip(df_BA_catcat.columns, ["resultBA"])))
	validBA.update(df_BA_catcat)
	
	print "*** PREDICTING PART 2 OF 5 *** catnum ***"
	classifiercatnum = pickle.load(open(get_paths()["model_path_catnum"]))
	predictionsABcatnum = classifiercatnum.predict(validAB[(validAB.AT<2)&(validAB.BT==2)])
	df_AB_catnum = pd.DataFrame(predictionsABcatnum ,index=validAB[(validAB.AT<2)&(validAB.BT==2)].index)
	df_AB_catnum = df_AB_catnum.rename(columns = dict(zip(df_AB_catnum.columns, ["resultAB"])))
	validAB.update(df_AB_catnum)
	predictionsBAcatnum = classifiercatnum.predict(validBA[(validBA.AT<2)&(validBA.BT==2)])
	df_BA_catnum = pd.DataFrame(predictionsBAcatnum ,index=validBA[(validBA.AT<2)&(validBA.BT==2)].index)
	df_BA_catnum = df_BA_catnum.rename(columns = dict(zip(df_BA_catnum.columns, ["resultBA"])))
	validBA.update(df_BA_catnum)
	
	print "*** PREDICTING PART 3 OF 5 *** catnum ***"
	classifiernumcat = pickle.load(open(get_paths()["model_path_numcat"]))
	predictionsABnumcat = classifiernumcat.predict(validAB[(validAB.AT==2)&(validAB.BT<2)])
	df_AB_numcat = pd.DataFrame(predictionsABnumcat ,index=validAB[(validAB.AT==2)&(validAB.BT<2)].index)
	df_AB_numcat = df_AB_numcat.rename(columns = dict(zip(df_AB_numcat.columns, ["resultAB"])))
	validAB.update(df_AB_numcat)
	predictionsBAnumcat = classifiernumcat.predict(validBA[(validBA.AT==2)&(validBA.BT<2)])
	df_BA_numcat = pd.DataFrame(predictionsBAnumcat ,index=validBA[(validBA.AT==2)&(validBA.BT<2)].index)
	df_BA_numcat = df_BA_numcat.rename(columns = dict(zip(df_BA_numcat.columns, ["resultBA"])))
	validBA.update(df_BA_numcat)
	
	
	print "*** PREDICTING PART 4 OF 5 *** numnum ***"
	classifiernumnum = pickle.load(open(get_paths()["model_path_numnum"]))
	predictionsABnumnum = classifiernumnum.predict(validAB[(validAB.AT==2)&(validAB.BT==2)])
	df_AB_numnum = pd.DataFrame(predictionsABnumnum ,index=validAB[(validAB.AT==2)&(validAB.BT==2)].index)
	df_AB_numnum = df_AB_numnum.rename(columns = dict(zip(df_AB_numnum.columns, ["resultAB"])))
	validAB.update(df_AB_numnum)
	predictionsBAnumnum = classifiernumnum.predict(validBA[(validBA.AT==2)&(validBA.BT==2)])
	df_BA_numnum = pd.DataFrame(predictionsBAnumnum ,index=validBA[(validBA.AT==2)&(validBA.BT==2)].index)
	df_BA_numnum = df_BA_numnum.rename(columns = dict(zip(df_BA_numnum.columns, ["resultBA"])))
	validBA.update(df_BA_numnum)
	
	validAB.to_csv(locationFolder+"temp_predictions_per_category_AB.csv")
	validBA.to_csv(locationFolder+"temp_predictions_per_category_BA.csv")
	
	print "*** PREDICTING PART 5 OF 5 *** full ***"
	classifierfull = pickle.load(open(get_paths()["model_path_full"]))
	predictionsABfull = classifierfull.predict(validAB)
	df_AB_full = pd.DataFrame(predictionsABfull ,index=validAB.index)
	df_AB_full = df_AB_full.rename(columns = dict(zip(df_AB_full.columns, ["resultAB_full"])))
	validAB.update(df_AB_full)
	predictionsBAfull = classifierfull.predict(validBA)
	df_BA_full = pd.DataFrame(predictionsBAfull ,index=validBA.index)
	df_BA_full = df_BA_full.rename(columns = dict(zip(df_BA_full.columns, ["resultBA_full"])))
	validBA.update(df_BA_full)
	
	print "Save extra info of the predictions"
	#get rid of some columns
	del validAB['A']
	del validAB['B']
	del validBA['A']
	del validBA['B']
	validAB.to_csv(locationFolder+"tmp_predictions_per_category_AB.csv")
	validBA.to_csv(locationFolder+"tmp_predictions_per_category_BA.csv")
	
	# result
	# doesn't work: predictions = validAB.resultAB.tolist() - validBA.resultBA.tolist()
	abcat = validAB.resultAB.tolist()
	abfull = validAB.resultAB_full.tolist()
	abat = validAB.AT.tolist(); abbt = validAB.BT.tolist()
	bacat = validBA.resultBA.tolist()
	bafull = validBA.resultBA_full.tolist()
	baat = validBA.AT.tolist(); babt = validBA.BT.tolist()
	#predictions_cat  = [a-b for a,b in zip(abcat,bacat)] # cat, don't write this, because we can only do 3 submissions
	predictions_full = [c-d for c,d in zip(abfull,bafull)] # full
	predictions_both = [a-b+c-d for a,b,c,d in zip(abcat,bacat,abfull,bafull)] # both
	predictions_weighted = [f8(a1,b1,c1,d1)-f8(a2,b2,c2,d2) for a1,b1,c1,d1,a2,b2,c2,d2 in zip(abcat,abfull,abat,abbt,bacat,bafull,baat,babt)]
	print("Writing predictions to file")
	#write_submission_cat(predictions_cat)
	write_submission_full(predictions_full)
	write_submission_both(predictions_both)
	write_submission_weighted(predictions_weighted)
	print("Finished predicting!")
	
	
	
def f8(ct, fll, at, bt):
	if at == 2 and bt == 2:
		return 1.0*(fll + ct)
	else:
		return 0.9*(fll + ct)

def predictSubcats():
	validAB = read_valid_pairs()
	validinfo = read_valid_info()
	# and join them
	validAB = validAB.join(validinfo)
	validBA = validAB.rename(columns = dict(zip(validAB.columns, ["B", "A", "BTstr", "ATstr", "BT", "AT"])))
  
	validAB['resultAB'] = 0
	validBA['resultBA'] = 0
	
	
	
	print "*** PREDICTING PART 1 OF 4 *** catcat ***"
	classifiercatcat = pickle.load(open(get_paths()["model_path_catcat"]))
	predictionsABcatcat = classifiercatcat.predict(validAB[(validAB.AT<2)&(validAB.BT<2)])
	df_AB_catcat = pd.DataFrame(predictionsABcatcat ,index=validAB[(validAB.AT<2)&(validAB.BT<2)].index)
	df_AB_catcat = df_AB_catcat.rename(columns = dict(zip(df_AB_catcat.columns, ["resultAB"])))
	validAB.update(df_AB_catcat)
	predictionsBAcatcat = classifiercatcat.predict(validBA[(validBA.AT<2)&(validBA.BT<2)])
	df_BA_catcat = pd.DataFrame(predictionsBAcatcat ,index=validBA[(validBA.AT<2)&(validBA.BT<2)].index)
	df_BA_catcat = df_BA_catcat.rename(columns = dict(zip(df_BA_catcat.columns, ["resultBA"])))
	validBA.update(df_BA_catcat)
	
	print "*** PREDICTING PART 2 OF 4 *** catnum ***"
	classifiercatnum = pickle.load(open(get_paths()["model_path_catnum"]))
	predictionsABcatnum = classifiercatnum.predict(validAB[(validAB.AT<2)&(validAB.BT==2)])
	df_AB_catnum = pd.DataFrame(predictionsABcatnum ,index=validAB[(validAB.AT<2)&(validAB.BT==2)].index)
	df_AB_catnum = df_AB_catnum.rename(columns = dict(zip(df_AB_catnum.columns, ["resultAB"])))
	validAB.update(df_AB_catnum)
	predictionsBAcatnum = classifiercatnum.predict(validBA[(validBA.AT<2)&(validBA.BT==2)])
	df_BA_catnum = pd.DataFrame(predictionsBAcatnum ,index=validBA[(validBA.AT<2)&(validBA.BT==2)].index)
	df_BA_catnum = df_BA_catnum.rename(columns = dict(zip(df_BA_catnum.columns, ["resultBA"])))
	validBA.update(df_BA_catnum)
	
	print "*** PREDICTING PART 3 OF 4 *** catnum ***"
	classifiernumcat = pickle.load(open(get_paths()["model_path_numcat"]))
	predictionsABnumcat = classifiernumcat.predict(validAB[(validAB.AT==2)&(validAB.BT<2)])
	df_AB_numcat = pd.DataFrame(predictionsABnumcat ,index=validAB[(validAB.AT==2)&(validAB.BT<2)].index)
	df_AB_numcat = df_AB_numcat.rename(columns = dict(zip(df_AB_numcat.columns, ["resultAB"])))
	validAB.update(df_AB_numcat)
	predictionsBAnumcat = classifiernumcat.predict(validBA[(validBA.AT==2)&(validBA.BT<2)])
	df_BA_numcat = pd.DataFrame(predictionsBAnumcat ,index=validBA[(validBA.AT==2)&(validBA.BT<2)].index)
	df_BA_numcat = df_BA_numcat.rename(columns = dict(zip(df_BA_numcat.columns, ["resultBA"])))
	validBA.update(df_BA_numcat)
	
	print "*** PREDICTING PART 4 OF 4 *** numnum ***"
	classifiernumnum = pickle.load(open(get_paths()["model_path_numnum"]))
	predictionsABnumnum = classifiernumnum.predict(validAB[(validAB.AT==2)&(validAB.BT==2)])
	df_AB_numnum = pd.DataFrame(predictionsABnumnum ,index=validAB[(validAB.AT==2)&(validAB.BT==2)].index)
	df_AB_numnum = df_AB_numnum.rename(columns = dict(zip(df_AB_numnum.columns, ["resultAB"])))
	validAB.update(df_AB_numnum)
	predictionsBAnumnum = classifiernumnum.predict(validBA[(validBA.AT==2)&(validBA.BT==2)])
	df_BA_numnum = pd.DataFrame(predictionsBAnumnum ,index=validBA[(validBA.AT==2)&(validBA.BT==2)].index)
	df_BA_numnum = df_BA_numnum.rename(columns = dict(zip(df_BA_numnum.columns, ["resultBA"])))
	validBA.update(df_BA_numnum)
	
	validAB.to_csv(locationFolder+"temp_predictions_per_category_AB.csv")
	validBA.to_csv(locationFolder+"temp_predictions_per_category_BA.csv")
	
	
	print "Save extra info of the predictions"
	#get rid of some columns
	del validAB['A']
	del validAB['B']
	del validBA['A']
	del validBA['B']
	validAB.to_csv(locationFolder+"predictions_per_category_AB.csv")
	validBA.to_csv(locationFolder+"predictions_per_category_BA.csv")
	
	# result
	# doesn't work: predictions = validAB.resultAB.tolist() - validBA.resultBA.tolist()
	r1 = validAB.resultAB.tolist()
	r2 = validBA.resultBA.tolist()
	predictions_cat  = [a-b for a,b in zip(r1,r2)] # cat
	print("Writing predictions to file")
	write_submission_cat(predictions_cat)
	
	
	print("Finished predicting!")
	
	return validAB, validBA
	
	# my score
	#
	# cat cat 0.810275852056
	# num num 0.867594585541
	# cat num 0.863320900705
	# num cat 0.829972216912
	
	# oob score
	# 
	# cat cat 0.212277595951
	# num num 0.402962770416
	# cat num 0.403837242092
	# num cat 0.261142468197
	

def predictAllWithoutCats():
	validAB = read_valid_pairs()
	validinfo = read_valid_info()
	# and join them
	validAB = validAB.join(validinfo)
	validBA = validAB.rename(columns = dict(zip(validAB.columns, ["B", "A", "BTstr", "ATstr", "BT", "AT"])))
  
	validAB['resultAB'] = 0
	validBA['resultBA'] = 0
	
	print "*** PREDICTING PART 5 OF 5 *** full ***"
	classifierfull = pickle.load(open(get_paths()["model_path_full"]))
	predictionsABfull = classifierfull.predict(validAB)
	df_AB_full = pd.DataFrame(predictionsABfull ,index=validAB.index)
	df_AB_full = df_AB_full.rename(columns = dict(zip(df_AB_full.columns, ["resultAB"])))
	validAB.update(df_AB_full)
	predictionsBAfull = classifierfull.predict(validBA)
	df_BA_full = pd.DataFrame(predictionsBAfull ,index=validBA.index)
	df_BA_full = df_BA_full.rename(columns = dict(zip(df_BA_full.columns, ["resultBA"])))
	validBA.update(df_BA_full)
	
	print "Save extra info of the predictions"
	#get rid of some columns
	del validAB['A']
	del validAB['B']
	del validBA['A']
	del validBA['B']
	validAB.to_csv("C:/Kaggle/KaggleCauseEffect/PYTHON/predictions_WITHOUT_category_AB.csv")
	validBA.to_csv("C:/Kaggle/KaggleCauseEffect/PYTHON/predictions_WITHOUT_category_BA.csv")
	
	# result
	predictions = validAB.resultAB.tolist() - validBA.resultBA.tolist()
	print("Writing predictions to file")
	write_submission(predictions)


def evaluateWeights():
	validAB = pd.read_csv("C:/Kaggle/KaggleCauseEffect/PYTHON/predictions_per_category_AB.csv", index_col=0)
	validBA = pd.read_csv("C:/Kaggle/KaggleCauseEffect/PYTHON/predictions_per_category_BA.csv", index_col=0)
	noAB = pd.read_csv("C:/Kaggle/KaggleCauseEffect/PYTHON/predictions_WITHOUT_category_AB.csv", index_col=0)
	noBA = pd.read_csv("C:/Kaggle/KaggleCauseEffect/PYTHON/predictions_WITHOUT_category_BA.csv", index_col=0)
	
	validAB = validAB.rename(columns = dict(zip(validAB.columns, ["ATstr_catAB", "BTstr_catAB", "AT_catAB", "BT_catAB", "resultAB_catAB"])))
	validBA = validAB.rename(columns = dict(zip(validAB.columns, ["BTstr_catBA", "ATstr_catBA", "BT_catBA", "AT_catBA", "resultBA_catBA"])))
	noAB = validAB.rename(columns = dict(zip(validAB.columns, ["ATstr_fullAB", "BTstr_fullAB", "AT_fullAB", "BT_fullAB", "resultAB_fullAB"])))
	noBA = validAB.rename(columns = dict(zip(validAB.columns, ["BTstr_fullBA", "ATstr_fullBA", "BT_fullBA", "AT_fullBA", "resultBA_fullBA"])))
	
	m = validAB.join(validBA, how='inner')
	m = m.join(noAB, how='inner')
	m = m.join(noBA, how='inner')
	
	return m
	
	
def predict():
    print("Reading the valid pairs") 
    
    if (enableSymmetryAB) :
    	validAB = read_valid_pairs()
    	validinfo = read_valid_info()
    	# and join them
    	validAB = validAB.join(validinfo)
    	
    	validBA = validAB.rename(columns = dict(zip(validAB.columns, ["B", "A", "BTstr", "ATstr", "BT", "AT"])))
    	
    	# reverse A or B to check if it matters
    	if (reverseA) :
    		validAB['A'] = validAB.A.apply(lambda x: -x)
    		validBA['A'] = validBA.A.apply(lambda x: -x)
    			
    	if (reverseB) :
    		validAB['B'] = validAB.B.apply(lambda x: -x)
    		validBA['B'] = validBA.B.apply(lambda x: -x)
    	
    	# filter optionally
    	if onlyBinaryBinary:
    		validAB = validAB[validAB.AT==0]
    		validAB = validAB[validAB.BT==0]
    		validBA = validBA[validBA.AT==0]
    		validBA = validBA[validBA.BT==0]
    	if onlyCatCat:
    		validAB = validAB[validAB.AT==0]
    		validAB = validAB[validAB.BT==0]
    		validBA = validBA[validBA.AT==0]
    		validBA = validBA[validBA.BT==0]
    	if onlyNumericalNumerical:
    		validAB = validAB[validAB.AT==2]
    		validAB = validAB[validAB.BT==2]
    		validBA = validBA[validBA.AT==2]
    		validBA = validBA[validBA.BT==2]
    	if onlyCatbinCatbin:
    		validAB = validAB[validAB.AT<2]
    		validAB = validAB[validAB.BT<2]
    		validBA = validBA[validBA.AT<2]
    		validBA = validBA[validBA.BT<2]
    	if trainOnlyCatbinToNumerical:
    		assert False
    	if trainOnlyNumericalToCatbin:
    		assert False
    		
    	# when making the prediction split per type, we can do this:
    	# 
    	# predictionsABcat = classifier.predict(validAB[validAB.AT==1])
    	# validAB['resultAB'] = 0
    	# newcol = pd.Series(predictionsABcat,index=validAB[validAB.AT==1].index)
    	# validAB['resultAB'] = newcol
    	
    	print("Loading the classifier")
    	classifier = load_model()
    	
    	print("Making predictions") 
    	predictionsAB = classifier.predict(validAB)
    	predictionsBA = classifier.predict(validBA)
    	predictionsAB = predictionsAB.flatten() # collapse ndarray to 1 dimension
    	predictionsBA = predictionsBA.flatten() # collapse ndarray to 1 dimension
    	
    	predictions = predictionsAB - predictionsBA
    
    
    else:
    	valid = read_valid_pairs()
    	validinfo = read_valid_info()
    	# and join them
    	valid = valid.join(validinfo)
    	
    	# filter optionally 
    	if onlyBinaryBinary:
    		valid = valid[valid.AT==0]
    		valid = valid[valid.BT==0]
    	if onlyCatCat:
    		valid = valid[valid.AT==1]
    		valid = valid[valid.BT==1]
    	if onlyNumericalNumerical:
    		valid = valid[valid.AT==2]
    		valid = valid[valid.BT==2]
    	if onlyCatbinCatbin:
    		valid = valid[valid.AT<2]
    		valid = valid[valid.BT<2]
    	if trainOnlyCatbinToNumerical:
    		assert False
    	if trainOnlyNumericalToCatbin:
    		assert False
    	
    	
    	print("Loading the classifier")
    	classifier = load_model()
    	
    	print("Making predictions") 
    	predictions = classifier.predict(valid)
    	predictions = predictions.flatten() # collapse ndarray to 1 dimension
    	
    print("Writing predictions to file")
    write_submission(predictions)
    


########################################################################################
#-# SCORE.PY
########################################################################################

def forward_auc(labels, predictions):
    return 0

def reverse_auc(labels, predictions):
    return 0

def bidirectional_auc(labels, predictions):
    score_forward = forward_auc(labels, predictions)
    score_reverse = reverse_auc(labels, predictions)
    score = (score_forward + score_reverse) / 2.0
    return score

def score():
    import pandas as pd
    
    solution = read_solution()
    submission = read_submission() 
    
    if onlyBinaryBinary:
    	validinfo = read_valid_info()
    	solution = solution.join(validinfo)
    	solution = solution[solution.AT==0]
    	solution = solution[solution.BT==0]
    if onlyCatCat:
    	validinfo = read_valid_info()
    	solution = solution.join(validinfo)
    	solution = solution[solution.AT==1]
    	solution = solution[solution.BT==1]
    if onlyNumericalNumerical:
    	validinfo = read_valid_info()
    	solution = solution.join(validinfo)
    	solution = solution[solution.AT==2]
    	solution = solution[solution.BT==2]
    if onlyCatbinCatbin:
    	validinfo = read_valid_info()
    	solution = solution.join(validinfo)
    	solution = solution[solution.AT<2]
    	solution = solution[solution.BT<2]
    if trainOnlyCatbinToNumerical:
    	assert False
    if trainOnlyNumericalToCatbin:
    	assert False
    	
    	
    

    score_forward = forward_auc(solution.Target, submission.Target)
    print("Forward Auc: %0.6f" % score_forward)

    score_reverse = reverse_auc(solution.Target, submission.Target)
    print("Reverse Auc: %0.6f" % score_reverse)

    score = bidirectional_auc(solution.Target, submission.Target)
    print("Bidirectional AUC: %0.6f" % score)
    
    with open("C:/Kaggle/KaggleCauseEffect/PYTHON/bidirectional_auc.txt", "a") as myfile:
        myfile.write(theLabel)
        myfile.write("\t")
        myfile.write(str(score))
        myfile.write("\n") 
    
    if enableScoreSeparate:
    	# write logging with separate scores, but don't print it
    	scoreSeparate(False)
    
def scoreSeparate(printit=True):
    solution = read_solution()
    submission = read_submission()
    
    validinfo = read_valid_info()
    sol = solution.join(validinfo)
    sub = submission.join(validinfo)
    
    if onlyBinaryBinary:
    	sol = sol[sol.AT==0]
    	sol = sol[sol.BT==0]
    if onlyBinaryBinary:
    	sol = sol[sol.AT==1]
    	sol = sol[sol.BT==1]
    if onlyNumericalNumerical:
    	sol = sol[sol.AT==2]
    	sol = sol[sol.BT==2]
    if onlyCatbinCatbin:
    	sol = sol[sol.AT<2]
    	sol = sol[sol.BT<2]
    if trainOnlyCatbinToNumerical:
    	assert False
    if trainOnlyNumericalToCatbin:
    	assert False
    
    forwardScore = [[1,2,3],[1,2,3],[1,2,3]]
    for i in range(0, 3):
    	for j in range(0, 3):
    		forwardScore[i][j] = forward_auc(sol[(sol.AT==i) & (sol.BT==j)].Target, sub[(sub.AT==i) & (sub.BT==j)].Target)
    		if (printit):
    			print "forward AUC", dict_int_to_type[i], dict_int_to_type[j], forwardScore[i][j]
    
    with open("C:/Kaggle/KaggleCauseEffect/PYTHON/bidirectional_auc_separated.txt", "a") as myfile:
    	for i in range(0, 3):
    		for j in range(0, 3):
    			myfile.write(theLabel)
    			myfile.write("\t")
    			myfile.write("forward AUC"+"\t"+str(dict_int_to_type[i])+"\t"+str(dict_int_to_type[j])+"\t"+str(forwardScore[i][j]))
    			myfile.write("\n") 


def scoreDifferentiateTypes(printit=True):
    global scoreDifferentiateTypes_differtiation
    
    solution = read_solution()
    submission = read_submission()
    
    validinfo = read_valid_info()
    sol = solution.join(validinfo)
    sub = submission.join(validinfo)
    
    sub['TargetNew'] = sub.apply(scoreDifferentiateTypes_function, axis=1)
    for i in range(0,200):
    	ifloat = (0.0+i)/200
    	scoreDifferentiateTypes_differtiation = ifloat
    	sub['TargetNew'] = sub.apply(scoreDifferentiateTypes_function, axis=1)
    	score = bidirectional_auc(sol.Target, sub.TargetNew)
    	print(str(ifloat)+"\tBidirectional AUC:\t%0.4f" % score)

scoreDifferentiateTypes_differtiation = 0.5

def scoreDifferentiateTypes_function(x):
	if (x[3] == 0 & x[4] == 0):
		return scoreDifferentiateTypes_differtiation * x[0]
	else:
		return x[0]
		
########################################################################################
#-# DATA_IO.PY
########################################################################################


import csv
import json
import numpy as np
import os
import pandas as pd
import pickle

def get_paths():
    paths = json.loads(open(locationFolder+filenameJSON).read())
    for key in paths:
        paths[key] = os.path.expandvars(paths[key])
    return paths

def parse_dataframe(df):
    parse_cell = lambda cell: np.fromstring(cell, dtype=np.float, sep=" ")
    df = df.applymap(parse_cell)
    return df

def read_train_pairs():
    train_path = get_paths()["train_pairs_path"]
    df = parse_dataframe(pd.read_csv(train_path, index_col="SampleID"))
    #add A<->B
    if (enableSymmetryAB) :
    	df = addReverse2Cols(df)
    return df

def read_train_target():
    path = get_paths()["train_target_path"]
    df = pd.read_csv(path, index_col="SampleID")
    df = df.rename(columns = dict(zip(df.columns, ["Target", "Details"])))
    if (enableSymmetryAB) :
    	df = addReverse2ColsTarget(df)
    	# indication if A->B
    	df['AB'] = df.Target.apply(lambda x: 1 if x>0 else 0)
    	# indication if B->A
    	df['BA'] = df.Target.apply(lambda x: 1 if x<0 else 0)
    return df

def read_train_info():
    path = get_paths()["train_info_path"]
    df = pd.read_csv(path, index_col="SampleID")
    df = df.rename(columns = dict(zip(df.columns, ["ATstr", "BTstr"])))
    # add A<->B
    if (enableSymmetryAB) :
    	df = addReverse2Cols(df)
    # and change string to int
    df['AT'] = df.ATstr.apply(lambda x: type_to_num(x))
    df['BT'] = df.BTstr.apply(lambda x: type_to_num(x))
    return df

def read_valid_pairs():
    valid_path = get_paths()["valid_pairs_path"]
    return parse_dataframe(pd.read_csv(valid_path, index_col="SampleID"))

def read_valid_info():
    path = get_paths()["valid_info_path"]
    df = pd.read_csv(path, index_col="SampleID")
    df = df.rename(columns = dict(zip(df.columns, ["ATstr", "BTstr"])))
    # and change string to int
    df['AT'] = df.ATstr.apply(lambda x: type_to_num(x))
    df['BT'] = df.BTstr.apply(lambda x: type_to_num(x))
    return df

def read_solution():
    path = get_paths()["solution_path"]
    df = pd.read_csv(path, index_col="SampleID")
    df = df.rename(columns = dict(zip(df.columns, ["Target", "Details"])))
    return df

def save_model(model):
    out_path = get_paths()["model_path"]
    # append label
    if (enableLabelInPickle):
    	out_path += "."
    	out_path += theLabel
    	out_path += ".pickle"
    pickle.dump(model, open(out_path, "w"))

def load_model():
    in_path = get_paths()["model_path"]
    # append label
    if (enableLabelInPickle):
    	in_path += "."
    	in_path += theLabel
    	in_path += ".pickle"
    return pickle.load(open(in_path))
    # one line: pickle.load(open(get_paths()["model_path"]))

def read_submission():
    submission_path = get_paths()["submission_path"]
    return pd.read_csv(submission_path, index_col="SampleID")

def write_submission(predictions):
    submission_path = get_paths()["submission_path"]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    valid = read_valid_pairs()
    rows = [x for x in zip(valid.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)

def write_submission_cat(predictions):
    submission_path = get_paths()["submission_path_cat"]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    valid = read_valid_pairs()
    rows = [x for x in zip(valid.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)

def write_submission_full(predictions):
    submission_path = get_paths()["submission_path_full"]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    valid = read_valid_pairs()
    rows = [x for x in zip(valid.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)

def write_submission_both(predictions):
    submission_path = get_paths()["submission_path_both"]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    valid = read_valid_pairs()
    rows = [x for x in zip(valid.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)

def write_submission_weighted(predictions):
    submission_path = get_paths()["submission_path_weighted"]
    writer = csv.writer(open(submission_path, "w"), lineterminator="\n")
    valid = read_valid_pairs()
    rows = [x for x in zip(valid.index, predictions)]
    writer.writerow(("SampleID", "Target"))
    writer.writerows(rows)



########################################################################################
#-# FEATURES.PY
########################################################################################

import numpy as np
from sklearn.base import BaseEstimator
from scipy.special import psi
from scipy.stats.stats import pearsonr

class FeatureMapper:
    def __init__(self, features, recalcfeatures):
        #print ("FeatureMapper.constructor")
        self.features = features
        self.recalcfeatures = recalcfeatures

    def fit(self, X, y=None):
        #print ("FeatureMapper.fit")
        trStart = time.clock()
        for feature_name, column_names, extractor in self.features:
            extractor.fit(X[column_names], y)
        trEnd = time.clock()
        print ("FeatureMapper.fit took "+str(int(trEnd-trStart))+" seconds or over "+str(int((trEnd-trStart)/60))+" minutes\n")

    def transform(self, X):
        print ("FeatureMapper.transform")
        # type of X: pandas.core.frame.DataFrame
        if not recalculateAllFeatures:
        	#start from the previous version of features
        	df = pd.read_csv(locationFolder+"features_previous_run.csv", index_col=0)
        	#df['A: Number of Samples'] = df['A: Number of Samples']*10
        	# and only recalculate the features in the recalculation list
        	self.features = self.recalcfeatures
        	firsttime = False
        else:
        	 # not recalculating -> start from empty dataframe
        	df = pd.DataFrame()
        	firsttime = True
        
        trStart = time.clock()
        extracted = []
        # type of self.features: list
        for feature_name, column_names, extractor in self.features:
            # type of feature_name: str or list
            # type of column_names: str (or list?)
            # type of extractor: ce_mult.SimpleTransform
            print "* transforming feature "+str(feature_name)
            # calculate the feature.
            fea_ = extractor.transform(X[column_names])
            #print ("type of fea_ is "+str(type(fea_)))
            #print ("result of transformation: "+str(fea_))
            
            # if the feature returned, is a list (nr of subfeatures), then we need to perform the code
            # below for every element in the list. Make it easier, by making it
            # a list anyway, but just with one single element
            if type(feature_name) is list:
            	subfeatures = np.transpose(fea_)
            	feature_names = feature_name
            else:
            	subfeatures = fea_
            	feature_names = [feature_name]
            #print "shape of subfeatures = "+str(shape(subfeatures))
            #print "feature_names = ", feature_names
            
            # check if the fea size (nr of subfeatures) equals the number of titles feature_names
            dummy, nr_subfeatures = np.shape(subfeatures)
            if nr_subfeatures != len(feature_names):
            	print "Number of features names is ", len(feature_names), ", shape is ", shape(subfeatures)
            	assert nr_subfeatures == len(feature_names)
            
            # loop through all subfeatures that this feature has calculated
            #print str(subfeatures)
            for idx, current_feature_name in enumerate(feature_names):
            	#print "idx", idx
            	fea = subfeatures[:,idx]
            	if hasattr(fea, "toarray"):
            	    #print current_feature_name
            	    extracted.append(fea.toarray()) # @TO_DO, can be erased
            	    if (firsttime):
            	    	df = pd.DataFrame(fea.toarray(), columns=[current_feature_name])
            	    else:
            	    	df[current_feature_name] = fea.toarray()
            	else:
            	    #print (str(type(current_feature_name)))
            	    #print (str(current_feature_name)+" does NOT have toarray")
            	    extracted.append(fea) # @TO_DO, can be erased
            	    if (firsttime):
            	    	df = pd.DataFrame(fea, columns=[current_feature_name])
            	    	firsttime = False
            	    else:
            	    	df[current_feature_name] = fea
        trEnd = time.clock()
        #print ("FeatureMapper.transform took "+str(int(trEnd-trStart))+" seconds or over "+str(int((trEnd-trStart)/60))+" minutes\n")
        #if len(extracted) > 1:
        #    return np.concatenate(extracted, axis=1)
        #else: 
        #    return extracted[0]
        #print "FeatureMapper is returning a "+str(type(df))
        return df
    
    def fit_transform(self, x, y=None):
        #print ("FeatureMapper.fit_transform")
        return self.transform(x)

def identity(x):
    return x

####################################################
#-# Count unique

def count_unique(x):
    return len(set(x))

def count_unique_difference(x, y):
    return count_unique(x) - count_unique(y)
    
def count_unique_fraction(x, y):
    return ( (0.0 + count_unique(x)) / ( count_unique(x) + count_unique(y)) )

def fraction_unique(x):
    return (0.0+len(set(x)))/(len(x))
    
####################################################
#-# Entropy

def normalized_entropy(x):
    x = (x - np.mean(x)) / np.std(x)
    x = np.sort(x)
    
    hx = 0.0;
    for i in range(len(x)-1):
        delta = x[i+1] - x[i];
        if delta != 0:
            hx += np.log(np.abs(delta));
    hx = hx / (len(x) - 1) + psi(len(x)) - psi(1);

    return hx

def entropy_difference(x, y):
    return normalized_entropy(x) - normalized_entropy(y)
    
def entropy_fraction(x, y):
    return ( (0.0 + normalized_entropy(x)) / ( normalized_entropy(x) + normalized_entropy(y)) )


####################################################
#-# avg/mean

def my_median(x):
    return np.median(x)
    
def my_mean(x):
    return np.mean(x)
    
def my_range(x):
    return np.max(x) - np.min(x)
    
def my_min(x):
    return np.min(x)
    
def my_max(x):
    return np.max(x)
    
def my_std(x):
    return np.std(x)
    
def normalized_median(x):
    x = (x - np.mean(x)) / np.std(x)
    return np.median(x)
    
def normalized_range(x):
    x = (x - np.mean(x)) / np.std(x)
    return np.max(x) - np.min(x)
    
def skewness(x):
    # is already normalised
    return stats.skew(x, axis=0, bias=True)
    
def skewtest_z(x):
    # is already normalised
    # requires at least 8 data points
    if (len(x)>=8):
    	zval, pval = stats.skewtest(x, axis=0)
    	return zval
    else:
    	return 0
    	
def skewtest_p(x):
    # is already normalised
    # requires at least 8 data points
    if (len(x)>=8):
    	zval, pval = stats.skewtest(x, axis=0)
    	return pval
    else:
    	return 0

# combination of the two functions above:
# [skewtest_z, skewtest_p]
def skewtest_both(x):
    # is already normalised
    # requires at least 8 data points
    if (len(x)>=8):
    	zval, pval = stats.skewtest(x, axis=0)
    	return [zval, pval]
    else:
    	return [0, 0]
    
def kurtosis(x):
    # is already normalised
    return stats.kurtosis(x, axis=0, fisher=True, bias=True)
    
def normalized_median_difference(x, y):
    return normalized_median(x) - normalized_median(y)

			
####################################################
#-# percentile

def percentile01(x):
    x = (x - np.mean(x)) / np.std(x)
    return stats.scoreatpercentile(x,  1)

def percentile02(x):
    x = (x - np.mean(x)) / np.std(x)
    return stats.scoreatpercentile(x,  2)

def percentile05(x):
    x = (x - np.mean(x)) / np.std(x)
    return stats.scoreatpercentile(x,  5)
    
def percentile10(x):
    x = (x - np.mean(x)) / np.std(x)
    return stats.scoreatpercentile(x, 10)
    
def percentile25(x):
    x = (x - np.mean(x)) / np.std(x)
    return stats.scoreatpercentile(x, 25)
    
def percentile75(x):
    x = (x - np.mean(x)) / np.std(x)
    return stats.scoreatpercentile(x, 75)
    
def percentile90(x):
    x = (x - np.mean(x)) / np.std(x)
    return stats.scoreatpercentile(x, 90)
    
def percentile95(x):
    x = (x - np.mean(x)) / np.std(x)
    return stats.scoreatpercentile(x, 95)

def percentile010205(x):
    x = (x - np.mean(x)) / np.std(x)
    return stats.scoreatpercentile(x,  1) + stats.scoreatpercentile(x,  2) + stats.scoreatpercentile(x,  5)

####################################################
#-# minimal precision
def minPrecision(x):
	x = np.sort(x)
	minPrec = 1e10
	for i in range(len(x)-1):
		delta = x[i+1] - x[i]
		if delta != 0:
			if delta < minPrec:
				minPrec = delta
	return minPrec

####################################################
#-# maxValueInHistogram
# related with cursosis?

def maxValueInHistogram(x):
		xx = (x - np.mean(x)) / np.std(x)
		xxx = xx[xx<=5.0*np.std(xx)]
		xxxx = xxx[xxx>=-5.0*np.std(xxx)]
		nrBinsX = int(len(xxxx)/50)
		if (nrBinsX > count_unique(xxxx)/2):
			nrBinsX = count_unique(xxxx)/2
		if(nrBinsX<=0):
			nrBinsX = 1
		if (len(xxxx)>1):
			(histnrs, histbins) = np.histogram(xxxx, bins=nrBinsX)
			maxhistnrs = max(histnrs)
			maxpercentage = 100. * maxhistnrs / len(x)
			return maxpercentage
		else :
			return 0

def maxValueInHistogramEq(x):
		xx = (x - np.mean(x)) / np.std(x)
		xxx = xx[xx<=5.0*np.std(xx)]
		xxxx = xxx[xxx>=-5.0*np.std(xxx)]
		nrBinsX = 50 # @TO_DO tune
		if (len(xxxx)>1):
			(histnrs, histbins) = np.histogram(xxxx, bins=nrBinsX)
			maxhistnrs = max(histnrs)
			maxpercentage = 100. * maxhistnrs / len(x)
			return maxpercentage
		else :
			return 0
			
####################################################
#-# fraction near maximum minimum

def fractionNearMin(x):
    m = my_min(x)
    
    factor_fractionNearMinMax = 0.25
    zone = factor_fractionNearMinMax * my_std(x)
    xx = x[x>=m]
    xxx = xx[xx<=m+zone]
    nr = len(xxx)
    r7 = 100.*nr/len(x)
    
    return r7
    
def fractionNearMax(x):
    m = my_max(x)
    
    factor_fractionNearMinMax = 0.25
    zone = factor_fractionNearMinMax * my_std(x)
    xx = x[x<=m]
    xxx = xx[xx>=m-zone]
    nr = len(xxx)
    r7 = 100.*nr/len(x)
    
    return r7
    
def fractionNearMinMax(x):
    r7 = fractionNearMin(x)
    s7 = fractionNearMax(x)
    return r7+s7
    
def fractionNearMinMax_dif(x, y):
    r7 = fractionNearMin(x)
    s7 = fractionNearMax(x)
    return r7-s7

####################################################
#-# type

def my_type(x):
    return x
    
####################################################
# Correlation

def correlation(x, y):
    return pearsonr(x, y)[0]

def correlation_magnitude(x, y):
    return abs(correlation(x, y))




nSplit = 4

def correlation_magnitude_splitted_1(x, y):
	
    # sort both x and y by x (and by y if equal x)
    a = np.vstack([y,x]) # Alternatively: all data in one big matrix
    indices = np.lexsort(a) # Sort on last row, then on 2nd last row, etc.
    b= a.take(indices, axis=-1)
    
    x = b[1]
    y = b[0]
    
    xParts = []
    yParts = []
    cors = []
    
    for i in range(0, nSplit):
        xParts += [x[int(i*len(x)/nSplit):int((i+1)*len(x)/nSplit)]]
        yParts += [y[int(i*len(y)/nSplit):int((i+1)*len(y)/nSplit)]]
        # calculate correlation
        cors += [abs(correlation(xParts[i], yParts[i]))]
        if (np.isnan(cors[i])):
            cors[i] = 99
        if (cors[i] > 99):
            cors[i] = 99
    
    # return standard error on correlation
    return np.std(cors)

def correlation_magnitude_splitted_2(x, y):
	
    # sort both x and y by x (and by y if equal x)
    a = np.vstack([y,x]) # Alternatively: all data in one big matrix
    indices = np.lexsort(a) # Sort on last row, then on 2nd last row, etc.
    b= a.take(indices, axis=-1)
    
    x = b[1]
    y = b[0]
    
    xParts = []
    yParts = []
    cors = []
    
    for i in range(0, nSplit):
        xParts += [x[int(i*len(x)/nSplit):int((i+1)*len(x)/nSplit)]]
        yParts += [y[int(i*len(y)/nSplit):int((i+1)*len(y)/nSplit)]]
        # calculate correlation
        cors += [abs(correlation(xParts[i], yParts[i]))]
        if (np.isnan(cors[i])):
            cors[i] = 99
        if (cors[i] > 99):
            cors[i] = 99
    
    # return standard error on correlation
    return np.min(cors)

def correlation_magnitude_splitted_3(x, y):
	
    # sort both x and y by x (and by y if equal x)
    a = np.vstack([y,x]) # Alternatively: all data in one big matrix
    indices = np.lexsort(a) # Sort on last row, then on 2nd last row, etc.
    b= a.take(indices, axis=-1)
    
    x = b[1]
    y = b[0]
    
    xParts = []
    yParts = []
    cors = []
    
    for i in range(0, nSplit):
        xParts += [x[int(i*len(x)/nSplit):int((i+1)*len(x)/nSplit)]]
        yParts += [y[int(i*len(y)/nSplit):int((i+1)*len(y)/nSplit)]]
        # calculate correlation
        cors += [abs(correlation(xParts[i], yParts[i]))]
        if (np.isnan(cors[i])):
            cors[i] = 99
        if (cors[i] > 99):
            cors[i] = 99
    
    # return standard error on correlation
    return np.max(cors)
    

















nLargeSplit = 32

def correlation_magnitude_splitted_11(x, y):
	
    # sort both x and y by x (and by y if equal x)
    a = np.vstack([y,x]) # Alternatively: all data in one big matrix
    indices = np.lexsort(a) # Sort on last row, then on 2nd last row, etc.
    b= a.take(indices, axis=-1)
    
    x = b[1]
    y = b[0]
    
    xParts = []
    yParts = []
    cors = []
    
    for i in range(0, nLargeSplit):
        xParts += [x[int(i*len(x)/nLargeSplit):int((i+1)*len(x)/nLargeSplit)]]
        yParts += [y[int(i*len(y)/nLargeSplit):int((i+1)*len(y)/nLargeSplit)]]
        # calculate correlation
        cors += [abs(correlation(xParts[i], yParts[i]))]
        if (np.isnan(cors[i])):
            cors[i] = 99
        if (cors[i] > 99):
            cors[i] = 99
    
    # return standard error on correlation
    return np.std(cors)

def correlation_magnitude_splitted_12(x, y):
	
    # sort both x and y by x (and by y if equal x)
    a = np.vstack([y,x]) # Alternatively: all data in one big matrix
    indices = np.lexsort(a) # Sort on last row, then on 2nd last row, etc.
    b= a.take(indices, axis=-1)
    
    x = b[1]
    y = b[0]
    
    xParts = []
    yParts = []
    cors = []
    
    for i in range(0, nLargeSplit):
        xParts += [x[int(i*len(x)/nLargeSplit):int((i+1)*len(x)/nLargeSplit)]]
        yParts += [y[int(i*len(y)/nLargeSplit):int((i+1)*len(y)/nLargeSplit)]]
        # calculate correlation
        cors += [abs(correlation(xParts[i], yParts[i]))]
        if (np.isnan(cors[i])):
            cors[i] = 99
        if (cors[i] > 99):
            cors[i] = 99
    
    # return standard error on correlation
    return np.min(cors)

def correlation_magnitude_splitted_13(x, y):
	
    # sort both x and y by x (and by y if equal x)
    a = np.vstack([y,x]) # Alternatively: all data in one big matrix
    indices = np.lexsort(a) # Sort on last row, then on 2nd last row, etc.
    b= a.take(indices, axis=-1)
    
    x = b[1]
    y = b[0]
    
    xParts = []
    yParts = []
    cors = []
    
    for i in range(0, nLargeSplit):
        xParts += [x[int(i*len(x)/nLargeSplit):int((i+1)*len(x)/nLargeSplit)]]
        yParts += [y[int(i*len(y)/nLargeSplit):int((i+1)*len(y)/nLargeSplit)]]
        # calculate correlation
        cors += [abs(correlation(xParts[i], yParts[i]))]
        if (np.isnan(cors[i])):
            cors[i] = 99
        if (cors[i] > 99):
            cors[i] = 99
    
    # return standard error on correlation
    return np.max(cors)
    

def correlation_magnitude_splitted_1_dif(x, y):
    return correlation_magnitude_splitted_1(x, y) - correlation_magnitude_splitted_1(y, x)
def correlation_magnitude_splitted_2_dif(x, y):
    return correlation_magnitude_splitted_2(x, y) - correlation_magnitude_splitted_2(y, x)
def correlation_magnitude_splitted_3_dif(x, y):
    return correlation_magnitude_splitted_3(x, y) - correlation_magnitude_splitted_3(y, x)
def correlation_magnitude_splitted_11_dif(x, y):
    return correlation_magnitude_splitted_11(x, y) - correlation_magnitude_splitted_11(y, x)
def correlation_magnitude_splitted_12_dif(x, y):
    return correlation_magnitude_splitted_12(x, y) - correlation_magnitude_splitted_12(y, x)
def correlation_magnitude_splitted_13_dif(x, y):
    return correlation_magnitude_splitted_13(x, y) - correlation_magnitude_splitted_13(y, x)
    
####################################################
#-# Linear regression

def linear_regression_slope(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return slope
    
def linear_regression_intercept(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return intercept
    
def linear_regression_r(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return r_value
    
def linear_regression_p(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return p_value
    
def linear_regression_stderr(x, y):
    slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
    return std_err
    

def linear_regression_rsquared(x, y):
    degree = 1
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    sstot = max( np.sum((y-ybar)**2), 0.001 )
    ssreg = np.sum((y-ybar)**2)
    ssres = np.sum((y-yhat)**2)
    rSquared = 1 - ssres / sstot
    return rSquared
    
    
####################################################
#-# Quadratic regression

def quadratic_regression_rsquared(x, y):
    degree = 2
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    sstot = max( np.sum((y-ybar)**2), 0.001 )
    ssreg = np.sum((y-ybar)**2)
    ssres = np.sum((y-yhat)**2)
    rSquared = 1 - ssres / sstot
    return rSquared
    
def quadratic_regression_a(x, y):
    degree = 2
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    sstot = max( np.sum((y-ybar)**2), 0.001 )
    ssreg = np.sum((y-ybar)**2)
    ssres = np.sum((y-yhat)**2)
    rSquared = 1 - ssres / sstot
    # highest degree
    return coeffs[0]
    
def quadratic_regression_b(x, y):
    degree = 2
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    sstot = max( np.sum((y-ybar)**2), 0.001 )
    ssreg = np.sum((y-ybar)**2)
    ssres = np.sum((y-yhat)**2)
    rSquared = 1 - ssres / sstot
    # middle degree
    return coeffs[1]
    
def quadratic_regression_c(x, y):
    degree = 2
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    sstot = max( np.sum((y-ybar)**2), 0.001 )
    ssreg = np.sum((y-ybar)**2)
    ssres = np.sum((y-yhat)**2)
    rSquared = 1 - ssres / sstot
    # lowest degree
    return coeffs[2]

# returning multiple values
def quadratic_regression_misc(x, y):
    degree = 2
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    sstot = max( np.sum((y-ybar)**2), 0.001 )
    ssreg = np.sum((y-ybar)**2)
    ssres = np.sum((y-yhat)**2)
    rSquared = 1 - ssres / sstot
    return [ rSquared, coeffs[0], coeffs[1], coeffs[2] ]

# combination of many functions
def quadratic_regression_misc_misc(x, y):
    [rsq_xy, coef0_xy, coef1_xy, coef2_xy] = quadratic_regression_misc(x, y)
    [rsq_yx, coef0_yx, coef1_yx, coef2_yx] = quadratic_regression_misc(y, x)
    return [rsq_xy, coef0_xy, coef1_xy, coef2_xy, rsq_yx, coef0_yx, coef1_yx, coef2_yx, rsq_xy-rsq_yx, coef0_xy-coef0_yx, coef1_xy-coef1_yx, coef2_xy-coef2_yx]

def quadratic_regression_rsquared_difference(x, y):
    return quadratic_regression_rsquared(x, y) - quadratic_regression_rsquared(y, x)
    
def quadratic_regression_a_difference(x, y):
    return quadratic_regression_a(x, y) - quadratic_regression_a(y, x)
    
def quadratic_regression_b_difference(x, y):
    return quadratic_regression_b(x, y) - quadratic_regression_b(y, x)
    
def quadratic_regression_c_difference(x, y):
    return quadratic_regression_c(x, y) - quadratic_regression_c(y, x)
    

####################################################
#-# Polyfit 3 regression

def polyfit3_regression_rsquared(x, y):
    degree = 3
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    sstot = max( np.sum((y-ybar)**2), 0.001 )
    ssreg = np.sum((y-ybar)**2)
    ssres = np.sum((y-yhat)**2)
    rSquared = 1 - ssres / sstot
    return rSquared
    
def polyfit3_regression_a(x, y):
    degree = 3
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    sstot = max( np.sum((y-ybar)**2), 0.001 )
    ssreg = np.sum((y-ybar)**2)
    ssres = np.sum((y-yhat)**2)
    rSquared = 1 - ssres / sstot
    # highest degree
    return coeffs[0]
    
def polyfit3_regression_b(x, y):
    degree = 3
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    sstot = max( np.sum((y-ybar)**2), 0.001 )
    ssreg = np.sum((y-ybar)**2)
    ssres = np.sum((y-yhat)**2)
    rSquared = 1 - ssres / sstot
    # middle degree
    return coeffs[1]
    
def polyfit3_regression_c(x, y):
    degree = 3
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    sstot = max( np.sum((y-ybar)**2), 0.001 )
    ssreg = np.sum((y-ybar)**2)
    ssres = np.sum((y-yhat)**2)
    rSquared = 1 - ssres / sstot
    # lowest degree
    return coeffs[2]

def polyfit3_regression_d(x, y):
    degree = 3
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    sstot = max( np.sum((y-ybar)**2), 0.001 )
    ssreg = np.sum((y-ybar)**2)
    ssres = np.sum((y-yhat)**2)
    rSquared = 1 - ssres / sstot
    # lowest degree
    return coeffs[3]

def polyfit3_regression_rsquared_difference(x, y):
    return polyfit3_regression_rsquared(x, y) - polyfit3_regression_rsquared(y, x)
    
def polyfit3_regression_a_difference(x, y):
    return polyfit3_regression_a(x, y) - polyfit3_regression_a(y, x)
    
def polyfit3_regression_b_difference(x, y):
    return polyfit3_regression_b(x, y) - polyfit3_regression_b(y, x)
    
def polyfit3_regression_c_difference(x, y):
    return polyfit3_regression_c(x, y) - polyfit3_regression_c(y, x)
    
def polyfit3_regression_d_difference(x, y):
    return polyfit3_regression_d(x, y) - polyfit3_regression_d(y, x)

####################################################
#-# Polyfit 4 regression

def polyfit4_regression_rsquared(x, y):
    degree = 4
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    ybar = np.mean(y)
    sstot = max( np.sum((y-ybar)**2), 0.001 )
    ssreg = np.sum((y-ybar)**2)
    ssres = np.sum((y-yhat)**2)
    rSquared = 1 - ssres / sstot
    return rSquared
    
####################################################
#-# Polyfit 4 regression

def polyfit_best_rsquared(x, y):
    best_degree = 0;
    best_sum_res = 1e10;
    ybar = np.sum(y)/len(y)
    sstot = np.sum((y - ybar)**2)
    for degree in range(2, 10):
    	coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    	# calculate r-squared
    	# source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    	p = np.poly1d(coeffs)
    	yhat = p(x)
    	sum_of_residuals_squared = np.sum((y-yhat)**2)
    	rquare_accent = sum_of_residuals_squared / (len(x)-degree-1)
    	if rquare_accent < best_sum_res:
    		best_degree = degree
    		best_sum_res = rquare_accent
    return [best_sum_res, best_degree]

# help function
def shuffle_in_unison_inplace(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def polyfit_best_rsquared_2(x, y):
    xx, yy = shuffle_in_unison_inplace(x, y)
    ind_mid = (int) ( 0.6666666 * len(x) )
    xtrain = xx[:ind_mid]
    ytrain = yy[:ind_mid]
    xtest = xx[ind_mid:]
    ytest = yy[ind_mid:]
    best_degree = 0;
    best_rquare = -1;
    for degree in range(1, 10):
    	try:
    		coeffs, residuals, rank, singular_values, rcond  = np.polyfit (xtrain, ytrain, degree, full=True)
    		# calculate r-squared
    		# source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    		p = np.poly1d(coeffs)
    		yhat = p(xtest)
    		ybar = np.mean(ytest)
    		sstot = max( np.sum((ytest-ybar)**2), 0.001 )
    		ssreg = np.sum((yhat-ybar)**2)
    		ssres = np.sum((ytest-yhat)**2)
    		rSquared = 1 - ssres / sstot
    		if rSquared > best_rquare:
    			best_degree = degree
    			best_rquare = rSquared
    	except Exception:
    		pass
    return [best_rquare, best_degree]

####################################################
#-# Is remainder of fitter degree 4 uniform or gaussian?

def polyfit4_remainder(x, y):
    degree = 4
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    p = np.poly1d(coeffs)
    yhat = p(x)
    r1 = uniform_test_chi(yhat)
    r2 = normaltest_p(yhat)
    # SWITCH
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (y, x, degree, full=True)
    # calculate r-squared
    # source: http://stackoverflow.com/questions/893657/how-do-i-calculate-r-squared-using-python-and-numpy
    q = np.poly1d(coeffs)
    xhat = q(y)
    r3 = uniform_test_chi(xhat)
    r4 = normaltest_p(xhat)
    return [r1, r2, r3, r4, r1-r3, r2-r4]
    
####################################################
#-# Other fitters

def spearmanr_r(x, y):
    rho, pval = stats.spearmanr(x, y, axis=0)
    return rho

def spearmanr_p(x, y):
    rho, pval = stats.spearmanr(x, y, axis=0)
    return pval

def spearmanr_r_dif(x, y):
    return spearmanr_r(x, y) - spearmanr_r(y, x)

def spearmanr_p_dif(x, y):
    return spearmanr_p(x, y) - spearmanr_p(y, x)


def kendalltau_t(x, y):
    tau, pval = stats.kendalltau(x, y, initial_lexsort=True)
    return tau

def kendalltau_p(x, y):
    tau, pval = stats.kendalltau(x, y, initial_lexsort=True)
    return pval

def kendalltau_t_dif(x, y):
    return kendalltau_t(x, y) - kendalltau_t(y, x)

def kendalltau_p_dif(x, y):
    return kendalltau_p(x, y) - kendalltau_p(y, x)


####################################################
#-# Normal test

def normaltest_z(x):
	z,pval = stats.normaltest(x)
	return z
	
def normaltest_p(x):
	z,pval = stats.normaltest(x)
	return pval*100.

def normaltest_z_dif(x, y):
    return normaltest_z(x) - normaltest_z(y)

def normaltest_p_dif(x, y):
    return normaltest_p(x) - normaltest_p(y)

####################################################
#-# Uniform test

def uniform_test_chi(x):
	nrBins = 20
	if (count_unique(x)<5*20):
		nrBins = int(1+count_unique(x)/5)
	histo, myBins = np.histogram(x, bins=nrBins)
	chi,pval = stats.chisquare(histo)
	if (np.isnan(chi)):
		return 10000 # very bad chi
	else:
		return chi
	
def uniform_test_p(x):
	nrBins = 20
	if (count_unique(x)<5*20):
		nrBins = int(1+count_unique(x)/5)
	histo, myBins = np.histogram(x, bins=nrBins)
	chi,pval = stats.chisquare(histo)
	if (np.isnan(pval)):
		return 0.0000001 # very bad p value
	else:
		return pval*100

def uniform_test_chi_dif(x, y):
    return uniform_test_chi(x) - uniform_test_chi(y)

def uniform_test_p_dif(x, y):
    return uniform_test_p(x) - uniform_test_p(y)


####################################################
#-# Normal test after substracting linear regression

degree_normal_poly = 2

def normal_substract_poly_z(x, y):
    degree = degree_normal_poly
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    p = np.poly1d(coeffs)
    yhat = p(x)
    y_noise = y - yhat
    return normaltest_z(y_noise)
    
def normal_substract_poly_p(x, y):
    degree = degree_normal_poly
    coeffs, residuals, rank, singular_values, rcond  = np.polyfit (x, y, degree, full=True)
    p = np.poly1d(coeffs)
    yhat = p(x)
    y_noise = y - yhat
    return normaltest_p(y_noise)
    
def normal_substract_poly_z_dif(x, y):
    return normal_substract_poly_z(x, y) - normal_substract_poly_z(y, x)
    
def normal_substract_poly_p_dif(x, y):
    return normal_substract_poly_p(x, y) - normal_substract_poly_p(y, x)


####################################################
#-# IGCI
# Information Geometric Causal Inference
#
# [1]  P. Daniusis, D. Janzing, J. Mooij, J. Zscheischler, B. Steudel,
#      K. Zhang, B. Scholkopf:  Inferring deterministic causal relations.
#      Proceedings of the 26th Annual Conference on Uncertainty in Artificial 
#      Intelligence (UAI-2010).  
#      http://event.cwi.nl/uai2010/papers/UAI2010_0121.pdf

# based on entropy

def igci_entropy_helpfunction(x):
    n = len(x)
    xsorted = np.sort(x)
    hx = 0.
    for i in range(1, n):
    	delta = xsorted[i]-xsorted[i-1]
    	if (delta != 0):
    		hx += np.log(abs(delta))
    result = hx / (n - 1) + psi(n) - psi(1)
    return result
    
def igci_entropy(x, y):
    return igci_entropy_helpfunction(y) - igci_entropy_helpfunction(x)
    
def igci_gauss_entropy(x, y):
    xnorm = (x - np.mean(x)) / np.std(x)
    ynorm = (y - np.mean(y)) / np.std(y)
    return igci_entropy(xnorm, ynorm)

def igci_uniform_entropy(x, y):
    if ( ( my_max(x) - my_min(x) > 0 ) & (my_max(y) - my_min(y) > 0 ) ):
    	x_uniform = (x - my_min(x)) / (my_max(x) - my_min(x))
    	y_uniform = (y - my_min(y)) / (my_max(y) - my_min(y))
    	return igci_entropy(x_uniform, y_uniform)
    else:
    	return igci_entropy(x, y)
    
# integral-approximation based estimator
def igci_integral(x, y):
    n = len(x) # = len(y)
    ind1 = np.argsort(x)
    ind2 = np.argsort(y)
    a = 0
    b = 0
    for i in range(1, n):
    	X1 = x[ind1[i-1]];  X2 = x[ind1[i]]
    	Y1 = y[ind1[i-1]];  Y2 = y[ind1[i]]
    	if (X2 != X1) & (Y2 != Y1) :
    		a += + np.log(abs((0.0 + Y2 - Y1) / (X2 - X1)))
    	X1 = x[ind2[i-1]];  X2 = x[ind2[i]]
    	Y1 = y[ind2[i-1]];  Y2 = y[ind2[i]]
    	if (X2 != X1) & (Y2 != Y1) :
    		b += np.log(abs((0.0 + Y2 - Y1) / (X2 - X1)))
    result = ( 0.0 + a - b)/n
    return result
    
def igci_gauss_integral(x, y):
    xnorm = (x - np.mean(x)) / np.std(x)
    ynorm = (y - np.mean(y)) / np.std(y)
    return igci_integral(xnorm, ynorm)
    
def igci_uniform_integral(x, y):
    if ( ( my_max(x) - my_min(x) > 0 ) & (my_max(y) - my_min(y) > 0 ) ):
    	x_uniform = (x - my_min(x)) / (my_max(x) - my_min(x))
    	y_uniform = (y - my_min(y)) / (my_max(y) - my_min(y))
    	return igci_integral(x_uniform, y_uniform)
    else:
    	return igci_integral(x, y)

####################################################
#-# Moving average

from scipy.interpolate import interp1d

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    return (ret[n - 1:] - ret[:1 - n]) / n

def moving_average_error(x, y):
		windowsize = 50
		x = 100. * (x - np.mean(x)) / np.std(x)
		y = 100. * (y - np.mean(y)) / np.std(y)
		a = np.vstack([y,x])
		indices = np.lexsort(a)
		b= a.take(indices, axis=-1)
		x2 = b[1]
		y2 = b[0]
		yma = moving_average(y2, windowsize)
		xma = x2[int(windowsize/2):len(yma)+int(windowsize/2)] # remove first and last windowsize/2 x's
		#plot(x2, y2, "b^", xma, yma, "r-")
		#show()
		yinter1 = interp1d(xma, yma)
		#yinter2 = interp1d(xma, yma, kind='cubic') # terrible! Also slow
		#plt.plot(xma,yma,'o',xma,yinter1(xma),'-', xma, yinter2(xma),'--')
		#plt.legend(['data', 'linear', 'cubic'])
		#plt.show()
		# calculate average difference squared
		npsquare = np.square(yma-yinter1(xma))
		npsquare = np.nan_to_num(npsquare) #replace nan by zeros and infinities by numbers
		ret = np.average(npsquare)
		return ret
		
def moving_average_error_dif(x, y):
		return moving_average_error(x, y) - moving_average_error(y, x)
		
def moving_average_error_logdif(x, y):
		return np.log(max(moving_average_error(x, y),0.0001)) - np.log(max(moving_average_error(y, x),0.0001))


####################################################
#-# Logistic regression

from sklearn.linear_model import LogisticRegression
def logistic_regression_score(x, y, xt, yt):
		# {'Binary': 0, 'Categorical': 1, 'Numerical': 2}
		#print ("yt = "+str(yt))
		if (yt<1.5):
			# convert Series to array
			if hasattr(y, "tolist"):
				y = y.tolist()
			# perform logistic regression
			lr = LogisticRegression(penalty='l2', tol=0.01)
			dfx = pd.DataFrame(x)
			lr.fit(dfx, y)
			#print "fit performed, now get the score"
			#print ("shape of x: "+str(shape(dfx)))
			#print ("shape of y: "+str(shape(y)))
			#print ("count_unique x, y: "+str(count_unique(x))+", "+str(count_unique(y)))
			#print ("type x, y: "+str(type(dfx))+", "+str(type(y)))
			#coefff = lr.coef_[0]
			score = lr.score(dfx, y)
			#print "logistic_regression_score = "+str(score)
			if (np.isnan(score)):
				return -1
			return score
		else:
			# don't perform logistic regression, instead return -1
			return -2
			
def logistic_regression_coef(x, y, xt, yt):
		# {'Binary': 0, 'Categorical': 1, 'Numerical': 2}
		if (yt<1.5):
			# perform logistic regression
			lr = LogisticRegression(penalty='l2', tol=0.01)
			dfx = pd.DataFrame(x)
			lr.fit(dfx, y)
			coefff = lr.coef_[0]
			#score = lr.score(dfx, y)
			if (np.isnan(coefff)):
				return -1
			return coefff[0] # first element of first element of lr.coef_
		else:
			# don't perform logistic regression, instead return -1
			return -2
		
# combination of the two upper functions
def logistic_regression_both(x, y, xt, yt):
		# {'Binary': 0, 'Categorical': 1, 'Numerical': 2}
		if (yt<1.5):
			# convert Series to array
			if hasattr(y, "tolist"):
				y = y.tolist()
			# perform logistic regression
			lr = LogisticRegression(penalty='l2', tol=0.01)
			dfx = pd.DataFrame(x)
			lr.fit(dfx, y)
			coefff = lr.coef_[0]
			score = lr.score(dfx, y)
			if np.isnan(coefff) or np.isnan(score):
				return [-1,-1]
			return [score, coefff[0]] # first element of first element of lr.coef_
		else:
			# don't perform logistic regression, instead return -1
			return [-2,-2]
		

####################################################
#-# Outlier detection

# we find outliers by 
def find_outliers(x):
    x = (x - np.mean(x)) / np.std(x)
    treshold = thesholdForOutlier
    total = len(x)
    nrbelow = len(x[x<-treshold])
    nrabove = len(x[x>treshold])
    nrtotal = nrbelow+nrabove
    percbelow = 1000. * (0.+nrbelow)/total
    percabove = 1000. * (0.+nrabove)/total
    perctotal = 1000. * (0.+nrtotal)/total
    return [ percbelow , percabove , perctotal , nrtotal ]

def find_outlier_dif(x, y):
    [ percbelowx , percabovex , perctotalx , nrtotalx ] = find_outliers(x)
    [ percbelowy , percabovey , perctotaly , nrtotaly ] = find_outliers(y)
    return [perctotaly-perctotalx, nrtotaly-nrtotalx]
    

####################################################
#-# Check inversibility using quadratic regression

def check_inversibility_quad(x, y):
	# normalize
	x = (x - np.mean(x)) / np.std(x)
	y = (y - np.mean(y)) / np.std(y)
	# remove outliers => failed, because we were removing certain x values without removing the corresponding y values
	#treshold = thesholdForOutlier
	#x = x[x>-treshold]
	#y = y[y>-treshold]
	#x = x[x<treshold]
	#y = y[y<treshold]
	## normalize again
	#x = (x - np.mean(x)) / np.std(x)
	#y = (y - np.mean(y)) / np.std(y)
	# quadratic fit
	[rsq_xy, coef0_xy, coef1_xy, coef2_xy] = quadratic_regression_misc(x, y)
	# if coef 0 = 0, the function is reversible, so return 0
	if coef0_xy == 0:
		return 0
	# find minimum or maximum
	localoptx = -coef1_xy/2./coef0_xy
	# if this is not in the range of x, return 0, function is inversible
	if localoptx <= np.min(x):
		return 0
	if localoptx >= np.max(x):
		return 0
	# we have found a non inversible function!
	# now we calculate the size of a square of the minimum overlap
	f2 = lambda xx: coef2_xy + coef1_xy * xx + coef0_xy * xx * xx
	mindistancex = min(abs(np.max(x)-localoptx), abs(localoptx-np.min(x)))
	mindistancey = min(abs(f2(np.max(x))-f2(localoptx)), abs(f2(localoptx)-f2(np.min(x))))
	minsquare = mindistancex * mindistancey
	# multiply with r squared, because not accurate -> less important
	if rsq_xy < 0:
		rsq_xy = 0
	result = minsquare * rsq_xy
	# second scenario, only count this when r squared > 0.5
	# results were not so good for this second scenario, so delete it.
	#rsq_xy2 = 2*rsq_xy-1
	#if rsq_xy2 < 0:
	#	rsq_xy2 = 0
	#result2 = minsquare * rsq_xy2
	#return [result, result2]
	return result


####################################################
#-# Theil-Sen estimator
# http://en.wikipedia.org/wiki/Theil%E2%80%93Sen_estimator
# https://github.com/CamDavidsonPilon/Python-Numerics/blob/master/Estimators/theil_sen.py
#
#
# can be tested like this, settings 500
#
# import sys; sys.path.append('C:/Kaggle/KaggleCauseEffect/PYTHON/'); from ce import *;
# train = read_train_pairs(); target = read_train_target(); traininfo = read_train_info(); train = train.join(traininfo); i = 5; x = train.ix[i]['A']; y = train.ix[i]['B']
# slope, intercept, r_value, p_value, std_err = stats.linregress(x,y)
# [slop_, interc_] = theil_sen(x, y, n_samples = 1000)
# f1 = lambda y: slope*y+intercept # normal lin regression
# f2 = lambda y: slop_*y+interc_ # teil sen
# xr = arange(min(x), max(x), (max(x)-min(x))/100)
# plot(x, y, "b^", xr, f1(xr), "r-", xr, f2(xr), "g-"); show()

import itertools

def theil_sen(x,y, sample= "auto", n_samples = 1e7):
    """
    Computes the Theil-Sen estimator for 2d data.
    parameters:
        x: 1-d np array, the control variate
        y: 1-d np.array, the ind variate.
        sample: if n>100, the performance can be worse, so we sample n_samples.
                Set to False to not sample.
        n_samples: how many points to sample.
    
    This complexity is O(n**2), which can be poor for large n. We will perform a sampling
    of data points to get an unbiased, but larger variance estimator. 
    The sampling will be done by picking two points at random, and computing the slope,
    up to n_samples times.
    
    """
    assert x.shape[0] == y.shape[0], "x and y must be the same shape."
    n = x.shape[0]
    
    if n < 100 or not sample:
        ix = np.argsort( x )
        slopes = np.empty( n*(n-1)*0.5 )
        for c, pair in enumerate(itertools.combinations( range(n),2 ) ): #it creates range(n) =( 
            i,j = ix[pair[0]], ix[pair[1]]
            slopes[c] = slope( x[i], x[j], y[i],y[j] )
    else:
        i1 = np.random.randint(0, n, n_samples)
        i2 = np.random.randint(0, n, n_samples)
        slopes = slope( x[i1], x[i2], y[i1], y[i2] )
        #pdb.set_trace()
    
    slope_ = np.median( slopes )
    #find the optimal b as the median of y_i - slope*x_i
    intercepts = np.empty( n )
    for c in xrange(n):
        intercepts[c] = y[c] - slope_*x[c]
    intercept_ = np.median( intercepts )

    return np.array( [slope_, intercept_] )
    
def slope( x_1, x_2, y_1, y_2):
    if x_2 == x_1: # please don't crash due to division by zero
    	return 1e5
    else:
    	return (1 - 2*(x_1>x_2) )*( (y_2 - y_1)/np.abs((x_2-x_1)) )
    
####################################################
#-# Binary - binary

def statsBinaryBinary (x, y):
	# assumpsion: y = f(x), so
	# - x is the actual class
	# - y is the prediction
	x = np.array(x, int)
	y = np.array(y, int)
	xy = np.array(2*x+y, int)
	xbincount = np.bincount(x)
	ybincount = np.bincount(y)
	xybincount = np.bincount(xy)
	# it is possible that there are missing values in these bins, because for instance 1-1 does not occur -> add zeros
	while len(xbincount) < 2 :
		xbincount = np.append(xbincount, 0)
	while len(ybincount) < 2 :
		ybincount = np.append(ybincount, 0)
	while len(xybincount) < 4 :
		print "add 1 zero to xybincount = ", len(xybincount)
		xybincount = np.append(xybincount, 0)
	# when calculating precision and recall we use '1' as
	# the least occurring class. This can be different as
	# the '1' in the samples. Therefore we define 'hit'
	# as this least occuring value.
	hit_x = 0 if xbincount[0] <= xbincount[1] else 1
	hit_y = 0 if ybincount[0] <= ybincount[1] else 1
	nohit_x = 1 - hit_x
	nohit_y = 1 - hit_y
	tp=xybincount[2*hit_x+hit_y] # true positive
	fp=xybincount[2*nohit_x+hit_y] # false positive
	fn=xybincount[2*hit_x+nohit_y] # false negative
	tn=xybincount[2*nohit_x+nohit_y] # true negative
	# precision, recall and accuracy (last doesn't depend on x<->y
	precision = 100.*tp/(tp+fp)
	recall = 100. * tp / (tp+fn)
	accuracy = 100. * (tp+tn) / (tp+tn+fp+fn)
	# maat voor afstand tot bissectrice ROC = lineair 
	# met verschil sensitivity en specificity.
	# We verwachten positieve waarde indien causale
	# relatie correct, ander negatieve waarde
	sensitivity = 100. * tp / ( tp + fp )
	specificity = 100. * fp / ( tp + fp )
	afstandROC = sensitivity - specificity
	# Dit houdt geen rekening met het feit dat dit getal eigenlijk inherent 
	# kleiner is bij heel grote of heel kleine waarden
	min_sens_spec = min (sensitivity/100., specificity/100., 0.5)
	max_sens_spec = max (sensitivity/100., specificity/100., 0.5)
	min_sens_spec = max (min_sens_spec, 0) #safety
	max_sens_spec = min (min_sens_spec, 1.) #safety
	afstandROC_accent = afstandROC
	afstandROC_accent *= 8 * min_sens_spec * min_sens_spec - 10 * min_sens_spec + 4 # correctieterm indien lage waarden
	afstandROC_accent *= 8 * max_sens_spec * max_sens_spec - 6 * max_sens_spec + 2 # correctieterm indien hoge waarden
	# return all these values
	return [precision, recall, accuracy, afstandROC, afstandROC_accent]
	
def statsBinaryBinary_dif (x, y):
	[precision1, recall1, accuracy1, afstandROC1, afstandROC_accent1] = statsBinaryBinary(x, y)
	[precision2, recall2, accuracy2, afstandROC2, afstandROC_accent2] = statsBinaryBinary(y, x)
	return [
		precision1, recall1, accuracy1, afstandROC1, afstandROC_accent1,
		precision2, recall2,            afstandROC2, afstandROC_accent2,
		precision2-precision1, recall2-recall1, afstandROC2-afstandROC1, 
		afstandROC_accent2-afstandROC_accent1
		]
	
	
####################################################
#-# Features important to BINARY - BINARY

def entropy_information_gain_YX (A0B0,A0B1,A1B0,A1B1):
	if A0B0 <= 0:
		A0B0 = 1
	if A0B1 <= 0:
		A0B1 = 1
	if A1B0 <= 0:
		A1B0 = 1
	if A1B1 <= 0:
		A1B1 = 1
	cnt = float(A0B0+A0B1+A1B0+A1B1)
	a00 = float(A0B0/cnt)
	a01 = float(A0B1/cnt)
	a10 = float(A1B0/cnt)
	a11 = float(A1B1/cnt)
	p_x0 = a00+a01
	p_x1 = a10+a11
	p_y0 = a00+a10
	p_y1 = a01+a11
	# (probability of x=0) * (entropy of y with x=0)
	lenx = cnt
	if lenx==0: # TODO, dit klopt eigenlijk niet, toegevoegd voor Inf
		lenx = 1
	resultingentropy = 0
	resultingentropy += p_x0 * entropy_binary(A0B0, A0B1)
	resultingentropy += p_x1 * entropy_binary(A1B0, A1B1)
	# the information gain is the gain in entropy
	entropy_y = entropy_binary(A0B0+A1B0, A0B1+A1B1)
	return entropy_y-resultingentropy


def entropy_binary(A0, A1):
	if A0 <= 0:
		A0 = 1
	if A1 <= 0:
		A1 = 1
	cnt = float(A0+A1)
	a0 = float(A0/cnt)
	a1 = float(A1/cnt)
	lenx = cnt
	if lenx==0: # TODO, dit klopt eigenlijk niet, toegevoegd voor Inf (maar lost het probleem precies niet op)
		lenx = 1
	# entropy is the average of all possible [ -ln(Px) ]
	result = 0
	result += -1. * a0 * np.log(a0)
	result += -1. * a1 * np.log(a1)
	return result

def binary_max_cumprob(A0B0,A0B1,A1B0,A1B1):
	cnt = float(A0B0+A0B1+A1B0+A1B1)
	a00 = float(A0B0/cnt)
	a01 = float(A0B1/cnt)
	a10 = float(A1B0/cnt)
	a11 = float(A1B1/cnt)
	max_x = max(a00+a01, a10+a11)
	max_y = max(a00+a10, a01+a11)
	return [ max_x, max_y ]

import random as rd
rd.seed(123)

# convert categories randomly to binary
def convertCatToBin(x):
	# how many categories will we select from x
	nr_sel_cat_from_x = rd.randint(1, count_unique(x)-1)
	# select the random categories
	samples_from_x = rd.sample(set(x), nr_sel_cat_from_x)
	# return 0 when class in this sample, else 1
	return np.where(np.in1d(x, samples_from_x), 0, 1)
	
def to_bin_bin_feats(x, y, xt, yt):
	# {'Binary': 0, 'Categorical': 1, 'Numerical': 2}
	if xt == 1:
		x = convertCatToBin(x)
	if yt == 1:
		y = convertCatToBin(y)
	nr_samples = 25
	df = pd.DataFrame(np.array([x, y])).T
	rows = rd.sample(df.index, nr_samples)
	df_10 = df.ix[rows]
	feat1 = 0
	feat2 = 0
	feat3 = 0
	feat4 = 0
	for i in range(nr_samples):
		lowX = df_10.iloc[i,0]
		hihX = lowX + 0.1 * my_std(x)
		lowY = df_10.iloc[i,1]
		hihY = lowY + 0.1 * my_std(y)
		if xt < 2:
			lowX = .5
			hihX = 100
		if yt < 2:
			lowY = .5
			hihY = 100
		A1B1 = len(df[ (df[0] >= lowX) & (df[0] < hihX) & (df[1] >= lowY) & (df[1] < hihY) ].index)
		A1B0 = len(df[ (df[0] >= lowX) & (df[0] < hihX) ].index) - A1B1
		A0B1 = len(df[ (df[1] >= lowY) & (df[1] < hihY) ].index) - A1B1
		A0B0 = len(df.index) - A1B1 - A1B0 - A0B1
		[ maxcum1, maxcum2 ] = binary_max_cumprob(A0B0,A0B1,A1B0,A1B1)
		feat1 += entropy_information_gain_YX (A0B0,A0B1,A1B0,A1B1)
		feat2 += entropy_information_gain_YX (A0B0,A1B0,A0B1,A1B1)
		feat3 += maxcum1
		feat4 += maxcum2
	return [feat1, feat2, feat3, feat4]

def to_bin_bin_feats_minmax(x, y, xt, yt):
	# {'Binary': 0, 'Categorical': 1, 'Numerical': 2}
	if xt == 1:
		x = convertCatToBin(x)
	if yt == 1:
		y = convertCatToBin(y)
	nr_samples = 25
	df = pd.DataFrame(np.array([x, y])).T
	rows = rd.sample(df.index, nr_samples)
	df_10 = df.ix[rows]
	feat1 = -1e5
	feat2 = -1e5
	feat3 = -1e5
	feat4 = -1e5
	feat01 = 1e5
	feat02 = 1e5
	feat03 = 1e5
	feat04 = 1e5
	for i in range(nr_samples):
		lowX = df_10.iloc[i,0]
		hihX = lowX + 0.1 * my_std(x)
		lowY = df_10.iloc[i,1]
		hihY = lowY + 0.1 * my_std(y)
		if xt < 2:
			lowX = .5
			hihX = 100
		if yt < 2:
			lowY = .5
			hihY = 100
		A1B1 = len(df[ (df[0] >= lowX) & (df[0] < hihX) & (df[1] >= lowY) & (df[1] < hihY) ].index)
		A1B0 = len(df[ (df[0] >= lowX) & (df[0] < hihX) ].index) - A1B1
		A0B1 = len(df[ (df[1] >= lowY) & (df[1] < hihY) ].index) - A1B1
		A0B0 = len(df.index) - A1B1 - A1B0 - A0B1
		[ maxcum1, maxcum2 ] = binary_max_cumprob(A0B0,A0B1,A1B0,A1B1)
		ent1 = entropy_information_gain_YX (A0B0,A0B1,A1B0,A1B1)
		ent2 = entropy_information_gain_YX (A0B0,A1B0,A0B1,A1B1)
		feat1 = max(feat1, ent1)
		feat2 = max(feat2, ent2)
		feat3 = max(feat3, maxcum1)
		feat4 = max(feat4, maxcum2)
		feat01 = min(feat01, ent1)
		feat02 = min(feat02, ent2)
		feat03 = min(feat03, maxcum1)
		feat04 = min(feat04, maxcum2)
	return [feat1, feat2, feat3, feat4, feat01, feat02, feat03, feat04]


	
####################################################
#-# K-Means Clustering

from scipy.cluster.vq import kmeans,vq

def find_kmeans(x, y):
    # see 
    # http://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans.html#scipy.cluster.vq.kmeans
    # http://glowingpython.blogspot.be/2012/04/k-means-clustering-with-scipy.html
  try:
    data = np.array([x, y]).T
    centroids, distortionscore = kmeans(data,2)
    deltax = abs(centroids[0,0] - centroids[1,0])
    deltay = abs(centroids[0,1] - centroids[1,1])
    deltaxy = abs(deltax-deltay)
    slope1 = deltay/max(deltax,1e-5)
    slope2 = deltax/max(deltay,1e-5)
    return [distortionscore, deltax, deltay, deltaxy, slope1, slope2]
  except Exception,e: 
  	#print "Error in find_kmeans:", str(e)
  	return [0, 0, 0, 0, 0, 0]
    
####################################################
#-# Information theory

from sklearn import metrics

def information_theory_sklearn_metrics_ew (x, y, xt, yt):
	# {'Binary': 0, 'Categorical': 1, 'Numerical': 2}
	if xt == 2:
		x = discretize(x, method="ew", nbins=10)
	if yt == 2:
		y = discretize(y, method="ew", nbins=10)
	mutual_info_score = metrics.adjusted_mutual_info_score(x, y)
	homogeneity_xy, completeness_xy, vmeasure = metrics.homogeneity_completeness_v_measure(x, y)
	homogeneity_yx, completeness_yx, vmeasure = metrics.homogeneity_completeness_v_measure(y, x)
	homogeneity_dif = homogeneity_xy-homogeneity_yx
	completeness_dif = completeness_xy-completeness_yx
	return [mutual_info_score, homogeneity_xy, homogeneity_yx, homogeneity_dif, completeness_dif, vmeasure]

def information_theory_sklearn_metrics_ef (x, y, xt, yt):
	# {'Binary': 0, 'Categorical': 1, 'Numerical': 2}
	if xt == 2:
		x = discretize(x, method="ef", nbins=10)
	if yt == 2:
		y = discretize(y, method="ef", nbins=10)
	mutual_info_score = metrics.adjusted_mutual_info_score(x, y)
	homogeneity_xy, completeness_xy, vmeasure = metrics.homogeneity_completeness_v_measure(x, y)
	homogeneity_yx, completeness_yx, vmeasure = metrics.homogeneity_completeness_v_measure(y, x)
	homogeneity_dif = homogeneity_xy-homogeneity_yx
	completeness_dif = completeness_xy-completeness_yx
	return [mutual_info_score, homogeneity_xy, homogeneity_yx, homogeneity_dif, completeness_dif, vmeasure]
	

def calculate_condentropy(px, py, pxpy, logbase=2):
	result = 0
	for ix in range(len(px)):
		for iy in range(len(py)):
			result += pxpy[ix][iy] * np.nan_to_num(np.log2(py[iy]/pxpy[ix][iy]))
	if logbase == 2:
		return result
	else:
		return logbasechange(2, logbase) * result

def discretize(X, method="ef", nbins=None):
    nobs = len(X)
    if nbins == None:
        nbins = np.floor(np.sqrt(nobs))
    if method == "ef":
        discrete = np.ceil(nbins * stats.rankdata(X)/nobs)
    if method == "ew":
        width = np.max(X) - np.min(X)
        width = np.floor(width/nbins)
        svec, ivec = stats.fastsort(X)
        discrete = np.zeros(nobs)
        binnum = 1
        base = svec[0]
        discrete[ivec[0]] = binnum
        for i in xrange(1,nobs):
            if svec[i] < base + width:
                discrete[ivec[i]] = binnum
            else:
                base = svec[i]
                binnum += 1
                discrete[ivec[i]] = binnum
    return discrete

def shannonentropy(px, logbase=2):
#TODO: haven't defined the px,py case?
    px = np.asarray(px)
    if not np.all(px <= 1) or not np.all(px >= 0):
        raise ValueError, "px does not define proper distribution"
    entropy = -np.sum(np.nan_to_num(px*np.log2(px)))
    if logbase != 2:
        return logbasechange(2,logbase) * entropy
    else:
        return entropy

def logbasechange(a,b):
    return np.log(b)/np.log(a)

def corrent(px,py,pxpy,logbase=2):
    return mutualinfo(px,py,pxpy,logbase=logbase)/shannonentropy(py,
            logbase=logbase)

def mutualinfo(px,py,pxpy, logbase=2):
    return shannonentropy(px, logbase=logbase) - calculate_condentropy(px,py,pxpy,
            logbase=logbase)
            
def information_theory_other (x, y, xt, yt):
	# {'Binary': 0, 'Categorical': 1, 'Numerical': 2}
	if xt == 2:
		x = discretize(x, method="ew", nbins=10)
	if yt == 2:
		y = discretize(y, method="ew", nbins=10)
	
	# convert to int array
	# perform -1 (for categorical) because bincount will count occurences from 0 on
	x = np.array(x-np.min(x), int); y = np.array(y-np.min(y), int)
	xbincount = np.bincount(x); ybincount = np.bincount(y)
	xbincount = np.array(xbincount, float); ybincount = np.array(ybincount, float)
	px = xbincount/np.sum(xbincount); py = ybincount/np.sum(ybincount)
	
	# create joined distribution
	pxpy = np.array(np.zeros((len(px), len(py)), dtype=int), int)
	for ix in range(len(px)):
		for iy in range(len(py)):
			counter = 0
			# len(x) and len(y) should be equal.
			# Just to be sure, take the min
			for i in range(min(len(x), len(y))):
				if ( ix == x[i] ) and ( iy == y[i] ) :
					counter += 1
			#print ix, iy, counter
			pxpy[ix][iy] = counter
	#pxpy = pxpy.T
	pxpy = np.array(pxpy, float);
	pxpy = pxpy/np.sum(pxpy)
	pypx = pxpy.T
	
	# assertions
	shapex, shapey = np.shape(pxpy)
	assert shapex == len(px)
	assert shapey == len(py)
	
	shannonentropy_x = shannonentropy(px)
	shannonentropy_y = shannonentropy(py)
	shannonentropy_dif = shannonentropy_y - shannonentropy_x
	
	condentropy_xy = calculate_condentropy(px, py, pxpy)
	condentropy_yx = calculate_condentropy(py, px, pypx)
	condentropy_dif = condentropy_xy - condentropy_yx
	
	mutualinfo_xy = mutualinfo(px, py, pxpy)
	corrent_xy = corrent(px, py, pxpy)
	covent_xy = condentropy_xy + condentropy_yx
	
	return [shannonentropy_x,shannonentropy_y,shannonentropy_dif,condentropy_xy,condentropy_yx,condentropy_dif,mutualinfo_xy,corrent_xy,covent_xy]
	
####################################################
#-# NOT - USED - Orthogonal distance regression

# not used, not correctly implemented
# example: http://www.physics.utoronto.ca/~phy326/python/odr_fit_to_data.py
def orthogonal_distance_regression(x, y):
    p_guess = (1, 1)
    model = scipy.odr.Model(func)
    fit = scipy.odr.ODR(data, model, p_guess, maxit=5000,job=10)
    output = fit.run()
    p = output.beta
    output.cov_beta
    return 0




####################################################
# NOT - USED - Check if the values of Categorical
# have some meaning --> they don't!

#from pylab import *
def check_order_of_categorical(x, y):
    i = 97
    x = train.ix[i]['A']
    y = train.ix[i]['B']
    count_unique(y)
    l_counts = [ (y.tolist().count(x), x) for x in set(y)]
    plot(l_counts)
    show()

#check with the following code:
#for i in range(1, 100):
#	x = train.ix[i]['A']
#	y = train.ix[i]['B']
#	c = check_logaritmic(x)
#	d = check_logaritmic(y)
#	if (c>0):
#		print 'x', i, c, traininfo.ix[i]['A type']
#	if (d>0):
#		print 'y', i, d, traininfo.ix[i]['B type']
		
def check_logaritmic(x):
    # check1: check if ascending (or descending) counts
    mn = np.min(x)
    mx = np.max(x)
    rg = mx-mn
    nrTranches = 5
    n1 = ((mn+0*rg/nrTranches < x) & (x < mn+1*rg/nrTranches)).sum()
    n2 = ((mn+1*rg/nrTranches < x) & (x < mn+2*rg/nrTranches)).sum()
    n3 = ((mn+2*rg/nrTranches < x) & (x < mn+3*rg/nrTranches)).sum()
    n4 = ((mn+3*rg/nrTranches < x) & (x < mn+4*rg/nrTranches)).sum()
    n5 = ((mn+4*rg/nrTranches < x) & (x < mn+5*rg/nrTranches)).sum()
    if ( n1 > n2 and n2 > n3 and n3 > n4 and n4 > n5):
    	check1 = True
    elif ( n1 < n2 and n2 < n3 and n3 < n4 and n4 < n5):
    	check1 = True
    else:
    	check1 = False
    
    # check2: check if many uniques
    if ( count_unique(x) > 100):
    	check2 = True
    else:
    	check2 = False

    # check3: check if logarithmic has a good linear regression, compared to the normal
    xsorted = np.sort(x)
    yrange = np.arange(len(xsorted), dtype=float)
    r_squared_xsorted = linear_regression_rsquared(xsorted, yrange)
    xsorted_log = xsorted - mn + 0.1
    xsorted_log = np.log(xsorted_log)
    r_squared_xsorted_log = linear_regression_rsquared(xsorted_log, yrange)
    if (r_squared_xsorted_log-r_squared_xsorted > 0.4):
    	check3 = True
    else:
    	check3 = False
    
    return check1 and check2 and check3
    

class SimpleTransform(BaseEstimator):
    def __init__(self, transformer=identity):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(x) for x in X], ndmin=2).T

class MultiColumnTransform(BaseEstimator):
    def __init__(self, transformer):
        self.transformer = transformer

    def fit(self, X, y=None):
        return self

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def transform(self, X, y=None):
        return np.array([self.transformer(*x[1]) for x in X.iterrows()], ndmin=2).T



########################################################################################
# MISC.PY
########################################################################################

def trainPredict():
	train()
	predict()

def trainPredictScore():
	
    print "Starting with " + theLabel
    start = time.clock()
    train()
    predict()
    score()
    ipt=getImportances()
    end = time.clock()
    writeTime(start, end)
    print "Elapsed time: " + str(int(end-start)) + " seconds " + theLabel

def getImportances():
	classifier = load_model()
	labeldummy, boostedt = classifier.steps[-1] # index of last step in pipeline
	myimportances = boostedt.feature_importances_
	
	combined_dummy, features = feature_extractor()
	myIndex = 0
	
	with open("C:/Kaggle/KaggleCauseEffect/PYTHON/importances.txt", "a") as myfile:
		for thisFeature in features:
			(featureName, dummy1, dummy2) = thisFeature
			if (writeImportances):
				if type(featureName) is list:
					subfeatures = featureName
				else:
					subfeatures = [featureName]
				for currentSubFeatureName in subfeatures:
					if printImportances:
						print currentSubFeatureName, myimportances[myIndex]
					myfile.write(theLabel)
					myfile.write("\t")
					myfile.write(currentSubFeatureName)
					myfile.write("\t")
					myfile.write(str(myimportances[myIndex]))
					myfile.write("\n")
					myIndex += 1
			
	return myimportances

def addReverse2Cols(df):
	newA = df.ix[:,0].tolist() + df.ix[:,1].tolist()
	newB = df.ix[:,1].tolist() + df.ix[:,0].tolist()
	dfnew = pd.DataFrame([newA, newB]).T
	# rename columns back to original
	columnslist = df.columns.tolist()
	dfnew  = dfnew.rename(columns = dict(zip(dfnew.columns, columnslist)))
	return dfnew

def addReverse2ColsTarget(df):
	dfrev = df.copy(deep=True)
	dfrev['Target'] = dfrev.Target.apply(lambda x: 0 - int(x))
	dfrev['Details'] = df.Details.apply(lambda x: reverseDetails(x))
	newA = df.ix[:,0].tolist() + dfrev.ix[:,0].tolist()
	newB = df.ix[:,1].tolist() + dfrev.ix[:,1].tolist()
	dfnew = pd.DataFrame([newA, newB]).T
	# rename columns back to original
	columnslist = df.columns.tolist()
	dfnew  = dfnew.rename(columns = dict(zip(dfnew.columns, columnslist)))
	return dfnew

# A->B 1
# B->A 2
# other: 3 or 4
def reverseDetails(x):
	if (x==1):
		return 2
	elif (x==2):
		return 1
	elif (x==3):
		return 4
	else:
		return 3


def logaritmize(df):
	df['A'] = df.A.apply(lambda x: logaritmizeFunction(x))
	df['B'] = df.B.apply(lambda x: logaritmizeFunction(x))
	return df


def logaritmizeFunction(x):
	if (check_logaritmic(x)):
		if (np.min(x) <= 0):
			x = x - np.min(x) + 0.001
		return np.log(x)
	else:
		return x


dict_type_to_int = {'Binary': 0, 'Categorical': 1, 'Numerical': 2}
dict_int_to_type = {v:k for k, v in dict_type_to_int.items()}
	
	
# convert type string (Binary, Categorical, ...) to an int
def type_to_num(st):
	if st in dict_type_to_int.keys():
		return dict_type_to_int[st]
	else:
		return 3
		
def print_new_calculations():
	train = read_train_pairs(); target = read_train_target(); traininfo = read_train_info(); train = train.join(traininfo)
	with open("C:/Kaggle/KaggleCauseEffect/PYTHON/"+theLabel+".txt", "w") as myfile:
		myfile.write("i"+"\t"+"scoreXY"+"\t"+"scoreXY"+"\t"+"coefXY"+"\t"+"coefYX"+"\t"+"XT"+"\t"+"YT"+"\t"+"target"+"\n")
		myfile.flush()
		for i in range(0, len(train)):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			xt = train.ix[i]['AT']
			yt = train.ix[i]['BT']
			tr = target.ix[i]['Target']
			[tt, ttt] = polyfit_best_rsquared_2(x, y)
			myfile.write(str(i))
			myfile.write("\t")
			myfile.write(str(tt))
			myfile.write("\t")
			myfile.write(str(ttt))
			myfile.write("\t")
			myfile.write(dict_int_to_type[xt])
			myfile.write("\t")
			myfile.write(dict_int_to_type[yt])
			myfile.write("\t")
			myfile.write(str(tr))
			myfile.write("\n")
			myfile.flush()



def print_timing():
	train = read_train_pairs(); target = read_train_target(); traininfo = read_train_info(); train = train.join(traininfo)
	
	#nrcalculations = len(train)
	nrcalculations = 2500
	
	with open("C:/Kaggle/KaggleCauseEffect/PYTHON/print_timing_2500_calculations.txt", "w") as myfile:
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			a = len(x)
			b = count_unique(x)
			c = fraction_unique(x)
			d = count_unique_difference(x, y)
			e = count_unique_fraction(x, y)
		end = time.clock()
		myfile.write("count"+"\t"+str(int(end-start))+"\n")
		print("count"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			b = normalized_entropy(x)
			d = entropy_difference (x, y)
			e = entropy_fraction(x, y)
		end = time.clock()
		myfile.write("entropy"+"\t"+str(int(end-start))+"\n")
		print("entropy"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			a = normaltest_z (x)
			b = normaltest_p (x)
			d = normaltest_z_dif (x, y)
			e = normaltest_p_dif (x, y)
		end = time.clock()
		myfile.write("normaltest"+"\t"+str(int(end-start))+"\n")
		print("normaltest"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			d = spearmanr_r (x, y)
			e = spearmanr_r_dif (x, y)
			f = spearmanr_p (x, y)
			g = spearmanr_p_dif (x, y)
		end = time.clock()
		myfile.write("spearman"+"\t"+str(int(end-start))+"\n")
		print("spearman"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			d = polyfit3_regression_rsquared (x, y)
			e = polyfit3_regression_a (x, y)
			h = polyfit3_regression_b (x, y)
			j = polyfit3_regression_c (x, y)
			k = polyfit3_regression_d (x, y)
			l = polyfit3_regression_rsquared_difference (x, y)
			o = polyfit3_regression_a_difference (x, y)
			u = polyfit3_regression_b_difference (x, y)
			r = polyfit3_regression_c_difference (x, y)
			a = polyfit3_regression_d_difference (x, y)
		end = time.clock()
		myfile.write("polyfit"+"\t"+str(int(end-start))+"\n")
		print("polyfit"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			d = quadratic_regression_rsquared  (x, y)
			e = quadratic_regression_a  (x, y)
			h = quadratic_regression_b  (x, y)
			j = quadratic_regression_c  (x, y)
			l = quadratic_regression_rsquared_difference  (x, y)
			o = quadratic_regression_a_difference  (x, y)
			u = quadratic_regression_b_difference  (x, y)
			r = quadratic_regression_c_difference   (x, y)
		end = time.clock()
		myfile.write("quadratic"+"\t"+str(int(end-start))+"\n")
		print("quadratic"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			d = normal_substract_poly_z (x, y)
			e = normal_substract_poly_p (x, y)
			a = normal_substract_poly_z_dif  (x, y)
			b = normal_substract_poly_p_dif  (x, y)
		end = time.clock()
		myfile.write("normsubstract"+"\t"+str(int(end-start))+"\n")
		print("normsubstract"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			d = linear_regression_p  (x, y)
			e = linear_regression_stderr  (x, y)
			a = linear_regression_rsquared   (x, y)
			b = linear_regression_slope   (x, y)
			b = linear_regression_intercept    (x, y)
		end = time.clock()
		myfile.write("linregression"+"\t"+str(int(end-start))+"\n")
		print("linregression"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			b = correlation (x, y)
			d = correlation_magnitude (x, y)
			a = correlation_magnitude_splitted_1 (x, y)
			e = correlation_magnitude_splitted_2 (x, y)
			f = correlation_magnitude_splitted_3 (x, y)
			a = correlation_magnitude_splitted_1_dif  (x, y)
			e = correlation_magnitude_splitted_2_dif  (x, y)
			f = correlation_magnitude_splitted_3_dif  (x, y)
		end = time.clock()
		myfile.write("pearson"+"\t"+str(int(end-start))+"\n")
		print("pearson"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			b = skewness (x)
			c = kurtosis (x)
			d = skewtest_z (x)
			a = skewtest_p (x)
		end = time.clock()
		myfile.write("skewkurt"+"\t"+str(int(end-start))+"\n")
		print("skewkurt"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			b = percentile05 (x)
			c = percentile10 (x)
			b = percentile25 (x)
			c = percentile75 (x)
			b = percentile90 (x)
			c = percentile95 (x)
		end = time.clock()
		myfile.write("percentile"+"\t"+str(int(end-start))+"\n")
		print("percentile"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			b = uniform_test_chi (x)
			c = uniform_test_p (x)
			d = uniform_test_chi_dif (x, y)
			a = uniform_test_p_dif (x, y)
		end = time.clock()
		myfile.write("uniform"+"\t"+str(int(end-start))+"\n")
		print("uniform"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			d = normalized_median_difference  (x, y)
		end = time.clock()
		myfile.write("normedian"+"\t"+str(int(end-start))+"\n")
		print("normedian"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			b1 = my_mean (x)
			c2 = my_median (x)
			b3 = my_range (x)
			c4 = my_min (x)
			b5 = my_max (x)
			c6 = my_std (x)
		end = time.clock()
		myfile.write("simplestats"+"\t"+str(int(end-start))+"\n")
		print("simplestats"+"\t"+str(int(end-start))+"\n")
		
		start = time.clock()
		for i in range(0, nrcalculations):
			x = train.ix[i]['A']
			y = train.ix[i]['B']
			d = kendalltau_t (x, y)
			e = kendalltau_t_dif (x, y)
			f = kendalltau_p (x, y)
			g = kendalltau_p_dif (x, y)
		end = time.clock()
		myfile.write("kendalltau"+"\t"+str(int(end-start))+"\n")
		print("kendalltau"+"\t"+str(int(end-start))+"\n")
		

def prepare_new_train_files():
	# concatenate
	filenames = ["C:/Kaggle/KaggleCauseEffect/DATA/SUP1_CEdata_train_pairs.csv", "C:/Kaggle/KaggleCauseEffect/DATA/SUP2_CEdata_train_pairs.csv"]
	with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_pairs.csv", 'w') as outfile:
		for fname in filenames:
			with open(fname) as infile:
				for line in infile:
					outfile.write(line)
	# split into two pieces
	i = 0
	with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_p1_publicinfo.csv", 'w') as outfile:
		with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_publicinfo.csv") as infile:
			for line in infile:
				if (i%2 == 0):
					outfile.write(line)
				i += 1
	i = 0
	with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_p2_publicinfo.csv", 'w') as outfile:
		with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_publicinfo.csv") as infile:
			for line in infile:
				if (i%2 == 1):
					outfile.write(line)
				i += 1
	i = 0
	with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_p1_target.csv", 'w') as outfile:
		with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_target.csv") as infile:
			for line in infile:
				if (i%2 == 0):
					outfile.write(line)
				i += 1
	i = 0
	with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_p2_target.csv", 'w') as outfile:
		with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_target.csv") as infile:
			for line in infile:
				if (i%2 == 1):
					outfile.write(line)
				i += 1
	i = 0
	with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_p1_pairs.csv", 'w') as outfile:
		with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_pairs.csv") as infile:
			for line in infile:
				if (i%2 == 0):
					outfile.write(line)
				i += 1
	i = 0
	with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_p2_pairs.csv", 'w') as outfile:
		with open("C:/Kaggle/KaggleCauseEffect/DATA/SUP12_CEdata_train_pairs.csv") as infile:
			for line in infile:
				if (i%2 == 1):
					outfile.write(line)
				i += 1
	print "dont forget to add a header row to all files, now only halve of them have headers"

def writeTime(s, e):
    with open("C:/Kaggle/KaggleCauseEffect/PYTHON/timing.txt", "a") as myfile:
        myfile.write(theLabel)
        myfile.write("\t")
        myfile.write(str(int(e-s)))
        myfile.write("\t")
        myfile.write(str(int((e-s)/60)))
        myfile.write("\n") 

def saveAllFigures():
	causalRelation = ['dummy', 'A-B', 'B-A', 'commonCause', 'independent']
	train = read_train_pairs()
	target = read_train_target()
	traininfo = read_train_info()
	train = train.join(traininfo)
	hold(False)
	for i in range(0, len(train)):
		figure()
		x = train.ix[i]['A']
		y = train.ix[i]['B']
		xcat = train.ix[i]['ATstr']
		ycat = train.ix[i]['BTstr']
		res = target.ix[i]['Details']
		plot(x, y, "b^")
		savefig("C:/Kaggle/KaggleCauseEffect/FIGURES/figure_"+str(i)+"_"+xcat+"_"+ycat+"_"+causalRelation[res]+".png", bbox_inches=0)
		clf()
		close()
		if (i%10 == 0):
			print ("i = "+str(i))

def saveAllHistograms():
	causalRelation = ['dummy', 'A-B', 'B-A', 'commonCause', 'independent']
	train = read_train_pairs()
	target = read_train_target()
	traininfo = read_train_info()
	train = train.join(traininfo)
	plt.hold(True)
	for i in range(0, len(train)):
		plt.figure()
		x = train.ix[i]['A']
		y = train.ix[i]['B']
		xcat = train.ix[i]['ATstr']
		ycat = train.ix[i]['BTstr']
		xat = train.ix[i]['AT']
		yat = train.ix[i]['BT']
		res = target.ix[i]['Details']
		xx = (x - np.mean(x)) / np.std(x)
		yy = (y - np.mean(y)) / np.std(y)

		xxx = xx[xx<=5.0*np.std(xx)]
		yyy = yy[yy<=5.0*np.std(yy)]
		xxxx = xxx[xxx>=-5.0*np.std(xxx)]
		yyyy = yyy[yyy>=-5.0*np.std(yyy)]
		nrBinsX = int(len(xxxx)/50)
		nrBinsY = int(len(yyyy)/50)
		if (nrBinsX > count_unique(xxxx)/2):
			nrBinsX = count_unique(xxxx)/2
		if (nrBinsY > count_unique(yyyy)/2):
			nrBinsY = count_unique(yyyy)/2
		if(nrBinsX<=0):
			nrBinsX = 1
		if(nrBinsY<=0):
			nrBinsY = 1
		if (xat==2 and len(xxxx)>1):
			plt.hist(xxxx, bins=nrBinsX, histtype='stepfilled', normed=True, color='b', alpha=0.5, label='X')
		if (yat==2 and len(yyyy)>1):
			plt.hist(yyyy, bins=nrBinsY, normed=True, color='r', alpha=0.5, label='Y')
		if (xat==2|yat==2):
			plt.savefig("C:/Kaggle/KaggleCauseEffect/FIGURES/histogram_"+str(i)+"_"+causalRelation[res]+".png", bbox_inches=0)
		plt.clf()
		plt.close()
		if (i%10 == 0):
			print ("i = "+str(i))


def saveAllCausalHistograms():
	causalRelation = ['dummy', 'A-B', 'B-A', 'commonCause', 'independent']
	train = read_train_pairs()
	target = read_train_target()
	traininfo = read_train_info()
	train = train.join(traininfo)
	plt.hold(True)
	for i in range(0, len(train)):
		plt.figure()
		x = train.ix[i]['A']
		y = train.ix[i]['B']
		xcat = train.ix[i]['ATstr']
		ycat = train.ix[i]['BTstr']
		xat = train.ix[i]['AT']
		yat = train.ix[i]['BT']
		res = target.ix[i]['Details']
		xx = (x - np.mean(x)) / np.std(x)
		yy = (y - np.mean(y)) / np.std(y)

		xxx = xx[xx<=5.0*np.std(xx)]
		yyy = yy[yy<=5.0*np.std(yy)]
		xxxx = xxx[xxx>=-5.0*np.std(xxx)]
		yyyy = yyy[yyy>=-5.0*np.std(yyy)]
		nrBinsX = int(len(xxxx)/50)
		nrBinsY = int(len(yyyy)/50)
		if (nrBinsX > count_unique(xxxx)/2):
			nrBinsX = count_unique(xxxx)/2
		if (nrBinsY > count_unique(yyyy)/2):
			nrBinsY = count_unique(yyyy)/2
		if(nrBinsX<=0):
			nrBinsX = 1
		if(nrBinsY<=0):
			nrBinsY = 1
		if (res == 2):
			if (xat==2):
				plt.hist(xxxx, bins=nrBinsX, histtype='stepfilled', normed=True, color='b', alpha=0.5, label='X')
			if (yat==2):
				plt.hist(yyyy, bins=nrBinsY, normed=True, color='r', alpha=0.5, label='Y')
			if (xat==2):
				plt.savefig("C:/Kaggle/KaggleCauseEffect/FIGURES/causal_histogram_"+str(i)+"_"+causalRelation[res]+".png", bbox_inches=0)
		if (res == 1):
			if (xat==2):
				plt.hist(yyyy, bins=nrBinsY, histtype='stepfilled', normed=True, color='b', alpha=0.5, label='Y')
			if (yat==2):
				plt.hist(xxxx, bins=nrBinsX, normed=True, color='r', alpha=0.5, label='X')
			if (yat==2):
				plt.savefig("C:/Kaggle/KaggleCauseEffect/FIGURES/causal_histogram_reverse_"+str(i)+"_"+causalRelation[res]+".png", bbox_inches=0)
		plt.clf()
		plt.close()
		if (i%10 == 0):
			print ("i = "+str(i))



class PreprocessLogarithmizer:
	def __init__(self, enableLog):
		self.enableLog = enableLog

	def fit_transform(self, x, y=None):
		print ("PreprocessLogarithmizer fit_transform")
		if (self.enableLog):
			return logaritmize(x)
		else:
			return x
	
	def transform(self, x):
		print ("PreprocessLogarithmizer transform")
		return self.fit_transform(x)



class PreprocessCatOrderizer:
	def __init__(self, enableLog):
		self.enableLog = enableLog

	def fit_transform(self, xx, y=None):
		print ("PreprocessCatOrderizer fit_transform")
		if (self.enableLog):
				train = xx # rename to enable copy paste
				#for i in range(0, len(train)):
				for i, row in train.iterrows():
					try:
						if (train.loc[i,'BT'] == 1):
							x = train.loc[i,'A']
							y = train.loc[i,'B']
							df = pd.DataFrame(np.array([x, y])).T
							df = df.rename(columns = dict(zip(df.columns, ["X", "Y"])))
							dfgr = df.groupby(['Y']).mean()
							dfgr = dfgr.sort_index(by='X')
							ls = dfgr.index.to_series().tolist()
							df['Ynew'] = df.Y.apply(lambda x: int(ls.index(int(x))))
							train.at[i, 'B'] = df.Y.apply(lambda x: int(ls.index(int(x))))
							#train.iat[i,1] = df.Y.apply(lambda x: int(ls.index(int(x))))
					except Exception:
						pass
				for i, row in train.iterrows():
					try:
						if (train.loc[i,'AT'] == 1):
							x = train.loc[i,'B']
							y = train.loc[i,'A']
							df = pd.DataFrame(np.array([x, y])).T
							df = df.rename(columns = dict(zip(df.columns, ["X", "Y"])))
							dfgr = df.groupby(['Y']).mean()
							dfgr = dfgr.sort_index(by='X')
							ls = dfgr.index.to_series().tolist()
							df['Ynew'] = df.Y.apply(lambda x: int(ls.index(int(x))))
							train.at[i, 'A'] = df.Y.apply(lambda x: int(ls.index(int(x))))
					except Exception:
						pass
				return train
		else: # if not enabled
			return xx
	
	def transform(self, x):
		print ("PreprocessCatOrderizer transform")
		return self.fit_transform(x)



from time import gmtime, strftime
class SaveFeatures:
	def __init__(self):
		self.enabled = 1

	def fit_transform(self, x, y=None):
		# x has type pandas.core.frame.DataFrame
		x.to_csv(locationFolder+theLabel+"_features_"+strftime("%Y-%m-%d_%H-%M-%S", gmtime())+".csv")
		x.to_csv(locationFolder+"features_previous_run.csv")
		return x
	
	def transform(self, x):
		print ("SaveFeatures transform")
		return self.fit_transform(x)

	       
          
class MyClassifier2(RandomForestRegressor):
	def fit(self, x, y=None):
		print "start fitting"
		# safety check for INF and for NaN
		# to avoid crashing of the fitter
		# Replace NaN by 0 and Inf by a very large (positive or negative) number
		y = np.nan_to_num(y)
		x = x.fillna(0)
		x = x.replace([ np.inf],  1e5)
		x = x.replace([-np.inf], -1e5)
		super(MyClassifier2, self).fit(x, y)
		print "end fitting"
		print "start scoring"
		myscore = super(MyClassifier2, self).score(x, y)
		print "my score = ", myscore
		
	def extraMethod(a):
		print "extra"


if __name__=="__main__":
    predictAll()


