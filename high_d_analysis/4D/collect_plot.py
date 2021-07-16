# %%
import matplotlib.pyplot as plt
import numpy as np
import pickle

index = np.arange(5)
dict = {}

for i in index:
    initialsample = 2 ** (i+2)
    y_title = "SampleMethod_lhs_Initial_" + str(initialsample) + "_C1_1_AC_f1.pkl"
    x_title = "SampleMethod_lhs_Initial_" + str(initialsample) + "_C1_1_AC_f1.npy"
    
    x = np.load(x_title)

    with open(y_title, "rb") as f:
        y = pickle.load(f)

    dict['lhs', 'x', initialsample] = x
    dict['lhs', 'y', initialsample] = np.mean(np.array(y), axis=1) 

    plt.plot(dict['lhs', 'x', initialsample], dict['lhs', 'y', initialsample], label = 'lhs' + str(initialsample), alpha = 0.7)

plt.title('Hartmann 4D with LHS initial samples')
plt.xlabel('Total samples')
plt.ylabel('SVM F1 score')
plt.legend()
plt.savefig('LHS.pdf')
plt.savefig('LHS.png')
plt.show()

# %%
for i in index:
    initialsample = 2 ** (i+2)
    y_title = "SampleMethod_sobol_Initial_" + str(initialsample) + "_C1_1_AC_f1.pkl"
    x_title = "SampleMethod_sobol_Initial_" + str(initialsample) + "_C1_1_AC_f1.npy"
    
    x = np.load(x_title)

    with open(y_title, "rb") as f:
        y = pickle.load(f)

    dict['sobol', 'x', initialsample] = x
    dict['sobol', 'y', initialsample] = np.mean(np.array(y), axis=1) 

    plt.plot(dict['sobol', 'x', initialsample], dict['sobol', 'y', initialsample], '-', label = 'sobol' + str(initialsample), alpha = 0.7)

plt.title('Hartmann 4D with sobol initial samples')
plt.xlabel('Total samples')
plt.ylabel('SVM F1 score')
plt.legend()
plt.savefig('Sobol.pdf')
plt.savefig('Sobol.png')

plt.show()

# %%
for i in index:
    initialsample = 2 ** (i+2)
    plt.plot(dict['lhs', 'x', initialsample], dict['lhs', 'y', initialsample], label = 'lhs' + str(initialsample), alpha = 0.7)
    plt.plot(dict['sobol', 'x', initialsample], dict['sobol', 'y', initialsample], '--', label = 'sobol' + str(initialsample), alpha = 0.7)

plt.title('Hartmann 4D with both initial samples')
plt.xlabel('Total samples')
plt.ylabel('SVM F1 score')
plt.legend()
plt.savefig('LHS_Sobol.pdf')
plt.savefig('LHS_Sobol.png')

plt.show()
