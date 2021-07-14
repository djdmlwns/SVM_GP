#################################################################################################################

#################################################################################################################
'''
Functions for plotting
'''
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sn

def progress_plot(num_iter_list, X_initial, opt_score_list, lhs_score_list, rand_score_list, title, method, path = None):
    '''
    Plot progress plot
    ############################################
    INPUT:

    num_iter_list : array of number of samples

    X_initial : Initial samples

    opt_score_list : score list using active sampling  

    lhs_score_list : score list using LHS

    rand_score_list : score list using Random sampling

    title : title of plot

    method : Method to calculate SVM accuracy {'F1', 'MCC', 'Simple'}
    '''
    # To calculate total number of samples 
    extended_num_iter_list = np.array(num_iter_list) + X_initial.shape[0]

    # Plot for the result using optimization
    plt.fill_between(extended_num_iter_list, np.max(opt_score_list, axis=1), np.min(opt_score_list, axis=1), alpha=0.3, color = 'g')
    plt.scatter(extended_num_iter_list, np.mean(opt_score_list, axis=1), color='g')
    plt.plot(extended_num_iter_list, np.mean(opt_score_list, axis=1), color='g', label='optimization')

    # Plot for the result using LHS
    plt.fill_between(extended_num_iter_list, np.max(lhs_score_list, axis=1), np.min(lhs_score_list, axis=1), alpha = 0.1, color='r')
    plt.scatter(extended_num_iter_list, np.mean(lhs_score_list, axis=1), color='r')
    plt.plot(extended_num_iter_list, np.mean(lhs_score_list, axis=1), color = 'r', label = 'LHS')

    # Plot for the result using random sampling
    plt.fill_between(extended_num_iter_list, np.max(rand_score_list, axis=1), np.min(rand_score_list, axis=1), alpha = 0.1, color='b')
    plt.scatter(extended_num_iter_list, np.mean(rand_score_list, axis=1), color='b')
    plt.plot(extended_num_iter_list, np.mean(rand_score_list, axis=1), color = 'b', label = 'Random')

    # Plot formatting
    plt.title(title)
    plt.xlabel('Samples')
    _ylabel = 'SVM accuracy (' + str(method) + ')'
    plt.ylabel(_ylabel)
    plt.legend()
    
    if path != None:
        plt.savefig(path + '/pgp_' + title + '.png')

    plt.show()


def required_sample(threshold_start, threshold_stop, opt_score_list, lhs_score_list, rand_score_list, num_iter_list, X_initial, 
                    accuracy_method, title = None, path=None):
    '''
    Plot desired accuracy vs required number of samples
    If mean score is above the threshold, then choose the number of samples to achieve that accuarcy
    ##################################################################
    INPUT:
    
    threshold_start : start value for desired svm accuracy

    threshold_stop : final value for desired svm accuracy

    opt_score_list : score list using active sampling  

    lhs_score_list : score list using LHS

    rand_score_list : score list using Random sampling

    num_iter_list: number of iteration list

    X_initial : initial training data

    accuracy method : Method to calculate SVM accuracy {'F1', 'MCC', 'Simple'}

    '''
    # To calculate total number of samples 
    extended_num_itr_list = np.array(num_iter_list) + X_initial.shape[0]
    
    threshold_accuracy = np.arange(threshold_start, threshold_stop, 0.01)
    mean_score_opt = np.mean(opt_score_list, axis=1)
    mean_score_lhs = np.mean(lhs_score_list, axis=1)
    mean_score_rand = np.mean(rand_score_list, axis=1)

    sample_opt = []
    sample_lhs = []
    sample_rand = []

    thr_valid_opt = []
    thr_valid_lhs = []
    thr_valid_rand = []

    for thr in threshold_accuracy:

        mean_score_opt_filtered = mean_score_opt[mean_score_opt < thr]
        opt_size = mean_score_opt_filtered.shape[0]
        mean_score_lhs_filtered = mean_score_lhs[mean_score_lhs < thr]
        lhs_size = mean_score_lhs_filtered.shape[0]
        mean_score_rand_filtered = mean_score_rand[mean_score_rand < thr]
        rand_size = mean_score_rand_filtered.shape[0]
        
        if (opt_size == 0 or opt_size == mean_score_opt.shape[0]):
            thr_valid_opt.append(False)
        else:
            itr_opt = max(extended_num_itr_list[mean_score_opt < thr])
            sample_opt.append(itr_opt)
            thr_valid_opt.append(True)

        if (lhs_size == 0 or lhs_size == mean_score_lhs.shape[0]):
            thr_valid_lhs.append(False)
        else:
            itr_lhs = max(extended_num_itr_list[mean_score_lhs < thr])
            sample_lhs.append(itr_lhs)
            thr_valid_lhs.append(True)

        if (rand_size == 0 or rand_size == mean_score_rand.shape[0]):
            thr_valid_rand.append(False)
        else:
            itr_rand = max(extended_num_itr_list[mean_score_rand < thr])
            sample_rand.append(itr_rand)
            thr_valid_rand.append(True)

#    minimum_plot_size = min(len(sample_opt), len(sample_lhs), len(sample_rand))

    plt.figure()        
    plt.plot(threshold_accuracy[thr_valid_opt], sample_opt, 'g-', label = 'Optimization')
    plt.plot(threshold_accuracy[thr_valid_lhs], sample_lhs, 'r--', label = 'LHS', alpha = 0.4)
    plt.plot(threshold_accuracy[thr_valid_rand], sample_rand, 'b--', label = 'Random', alpha = 0.4)

    if title == None:
        title = 'Number of samples for desired accuracy (' + accuracy_method + ')'

    plt.title(title)
    plt.xlabel('Desired accuracy')
    plt.ylabel('Number of samples needed')
    plt.legend()
    
    if path != None:
        plt.savefig(path + '/sample_' + title + '.png')
    
    plt.show()



##################################################################################################################################
# Functions for plotting only for 2-D problem

def plot_heatmap_uncertainty(Gaussian_regressor):
    ''' 
    Plot heat map of uncertainty calculated by Gaussian regressor 
    Only for 2-D problem
    '''
    n_points = 30
    # Assume x1 and x2 are within [0,1]
    x1 = np.linspace(0,1,n_points)
    x2 = np.linspace(1,0,n_points)

    for i, _x2 in enumerate(x2):
        y_value = []
        for _x1 in x1:
            # Gaussian_regressor.predict can calculate uncertainty if return_std is True
            y_value.append(Gaussian_regressor.predict(np.atleast_2d([_x1,_x2]), return_std = True)[1][0])
        if i == 0:
            heatmap_data = np.array(y_value).reshape(1,n_points)
        else:
            heatmap_data = np.vstack([heatmap_data, np.array(y_value).reshape(1,n_points)])  
    sn.heatmap(heatmap_data)
    plt.show()


def plot_svm_boundary(svm_classifier, X, y, **kwargs):
    ''' 
    Plot svm decision boundary of svm_classifier with data X and y 
    Only for 2-D problem
    '''
    xx, yy = np.meshgrid(np.linspace(0, 1, 500),
                        np.linspace(0, 1, 500))

    # plot the decision function for each datapoint on the grid
    Z = svm_classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.imshow(Z, interpolation='none',
            extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='equal',
            origin='lower', cmap=plt.cm.PuOr_r)
    contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                        linestyles='dashed')
    plt.scatter(X[:, 0], X[:, 1], s=30, c=y, cmap=plt.cm.Paired,
                edgecolors='k')
    plt.xticks(())
    plt.yticks(())
    plt.axis([-0.1, 1.1, -0.1, 1.1])
    # If want to save the decision boundary plot
    if 'path' in kwargs.keys():
        path = kwargs['path'] + '/final_decisionboundary.png'
        plt.savefig(path)
    plt.show()


def plot_scatter_data(svm_classifier, X, y, num_initial_sample):
    ''' 
    Scatter plot for data 
    Only for 2-D problem
    '''

    X_initial = X[:num_initial_sample, :]
    y_initial = y[:num_initial_sample]

    new_points = X[num_initial_sample:, :]
    
    # Initial samples
    plt.scatter(X_initial[:,0], X_initial[:,1], c=y_initial, s=30, alpha = 0.3)
    # New points
    plt.scatter(new_points[:,0], new_points[:,1], s=50, c = 'r', marker = '*')
    # Support vectors
    plt.scatter(svm_classifier.support_vectors_[:,0], 
                svm_classifier.support_vectors_[:,1], 
                s=15, marker='x')
    plt.xlim((-0.1,1.1))
    plt.ylim((-0.1,1.1))
    plt.legend(['Initial points', 'new points', 'support vectors'])
    plt.show()
