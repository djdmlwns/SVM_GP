# %%
from sklearn import svm
import numpy as np

from auxil.testfunc import Hosaki2d, Branin2d, HARTMANN4D, HARTMANN6D, Dette8d
from auxil.initialization import Initialization
from core.asopt import MyGPR, ActiveSampling
from auxil.samplesvm import Sampling_based_SVM
from auxil.auxilfunc import check_data_feasible
from plotting.plots import progress_plot, required_sample, plot_svm_boundary, plot_heatmap_uncertainty

# %%
# Benchmark Function Test Main script
case = 'benchmark' # case can be 'benchmark' or 'simulation'
dim = 4 # number of features for benchmark function

# Set condition for feasible region  
# This is only needed for benchmark function (case = 'benchmark')
condition = lambda x: HARTMANN4D(x) <= -0.5

# Initial samples
num_samples = 16    # number of initial samples
Initializer = Initialization(dim, num_samples, case = case, condition = condition) # Class for initial sampling

# Check data feasibility
# Raise ValueError if only one classification (e.g., only 1 and no -1) is included in the initial sample
# Sampling again if the error is raised
for itr in range(10):
    try: 
        X, y = Initializer.sample('lhs')
        check_data_feasible(y) 
        break
    except ValueError:
        print('Need to resample') 

#%%
max_main_loop = 1             # number of main loop to calculate mean/variance of the svm accuracy of the proposed algorithm -> need for plot
accuracy_method = 'F1'        # method to calculate svm accuracy {'F1', 'MCC', 'Simple'} are available. Default is F1-score
max_itr = 500                 # maximum number of samples
report_frq = 10               # frequency to test svm accuracy and print

# Variable bound
bounds = []
for i in range(dim):
    bounds.append((0.0, 1.0))   # [0, 1]^n. For simulation, normalization of X is needed

# Start Active Learning/Sampling algorithm 
opt_score_list = [] # initialize score list 

for _itr in range(max_main_loop):
    # initialize SVM
    SVM_classifier = svm.SVC(kernel='rbf', C = 10000, random_state = 42) 
    
    # initialize GP regressor
    GP_regressor = MyGPR(normalize_y = True, n_restarts_optimizer = 0, alpha = 1e-7) 
    
    # initialize Active sampling algorithm class 
    AS = ActiveSampling(X, y, SVM_classifier, GP_regressor, bounds, 
                        max_itr = max_itr, case = case, C1=1, accuracy_method = accuracy_method, report_frq = report_frq,
                        condition = condition, p_check = 0.0, threshold = 1, C1_schedule='None', acq_type = 'f1')
    # training start
    AS.train()

    # append accuracy score to the list
    opt_score_list.append(AS.score_list)

# change the shape of list for future plotting
opt_score_list = (np.array(opt_score_list).T).tolist()

# %%
# SVM trained without using Active sampling algorithm
# 1) LHS Sampling-based SVM

# Initialize class instance
# sampling_method = 'lhs'
SS_LHS = Sampling_based_SVM(X, max_itr = max_itr,
                        report_frq = report_frq, iteration = 3, sampling_method = 'lhs', accuracy_method = accuracy_method, 
                        case = case, svm_random_state = 42, condition = condition)

# train
SS_LHS.train()

# save the svm accuracy score
lhs_score_list = SS_LHS.score_list

# 2) Random Sampling-based SVM
# Initialize class instance
# sampling_method = 'random'
SS_Rand = Sampling_based_SVM(X, max_itr = max_itr, 
                        report_frq = report_frq, iteration = 3, sampling_method = 'random', accuracy_method = accuracy_method, 
                        case = case, svm_random_state = 42, condition = condition)

# train
SS_Rand.train()

# save the svm accuracy score
rand_score_list = SS_Rand.score_list

#########################################################################################
# Plot section
#########################################################################################
#%%
# Plot progress plot 
# set title (e.g., C1=100 / Score = F1)
title = 'C1_' + str(AS.C1) + '_Score_' + accuracy_method + '_Acquisition type_' + AS.acq_type + '_C1_schedule_' + str(AS.C1_schedule)# set title

# Plot svm accuracy improvement w.r.t. number of samples
# Require three score lists of SVM trained by AS/LHS/RANDOM
progress_plot(num_iter_list= AS.num_iter_list, X_initial = AS.X_initial, opt_score_list=opt_score_list,
                    lhs_score_list = lhs_score_list, rand_score_list= rand_score_list,
                    title = title, method = accuracy_method, path = '.')


# Plot required sample numbers in terms of desired accuracy
required_sample(0.1, 1.0, opt_score_list = opt_score_list, lhs_score_list=lhs_score_list, rand_score_list=rand_score_list, num_iter_list = AS.num_iter_list,
                        X_initial = X, accuracy_method = accuracy_method, title = title, path = '.')

#%%

# Plot heatmap of uncertainty to check only for 2-D
if dim == 2:
    plot_heatmap_uncertainty(AS.GP_regressor)

# %%
# Plot SVM boundary of the proposed algorithm if needed only for 2-D
if dim == 2:
    plot_svm_boundary(AS.SVM_classifier, AS.X, AS.y)
