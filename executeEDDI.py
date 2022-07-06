import active_learning_functions 
#### Import libraries
from sklearn.model_selection import train_test_split
import pandas as pd, numpy as np
import p_vae, tensorflow as tf
import sklearn.preprocessing as preprocessing
tfd = tf.contrib.distributions
import argparse, datetime, pickle




parser = argparse.ArgumentParser(description='Train and evaluate the hierarchy for active feature selection using our exploration heuristic.')
# Data arguments
parser.add_argument('pathTrain', help = 'XLS file with the train samples. Last feature is the label.')
parser.add_argument('pathTest', help = 'XLS file with the test samples. Last feature is the label.')
parser.add_argument('pathOutput', help = 'Pickle file with the predictions and costs.')
parser.add_argument(
    '--epochs',
    type=int,
    default=3000,
    metavar='N_eps',
    help='number of epochs to train (default: 3000)')
parser.add_argument(
    '--latent_dim',
    type=int,
    default=10,
    metavar='LD',
    help='latent dimension (default: 10)')
parser.add_argument(
    '--p',
    type=float,
    default=0.7,
    metavar='probability',
    help='dropout probability of artificial missingness during training')
parser.add_argument(
    '--batch_size',
    type=int,
    default=100,
    metavar='batch',
    help='Mini Batch size per epoch.  ')
parser.add_argument(
    '--K',
    type=int,
    default=20,
    metavar='K',
    help='Dimension of PNP feature map ')
parser.add_argument(
    '--M',
    type=int,
    default=50,
    metavar='M',
    help='Number of MC samples when perform imputing')
parser.add_argument('--name', default = 'No name given', help = 'Name of the dataset')

#Settings of the algorithm
parser.add_argument('-maxNumAcquisitions', type=int, nargs='+', default = [-1], help = 'Maximal number of acquisitions that can be acquired')

args = parser.parse_args()

p = args.p # Prob of missing
epochs = args.epochs
batch_size = args.batch_size
latent_variable_dim = args.latent_dim
K = args.K






### load data
Data = pd.read_excel(args.pathTrain)
Data = Data.values

### data preprocess
max_Data = 1  #
min_Data = 0  #
dMin =  Data.min(axis=0)
dMax =  Data.max(axis=0)
Data_std = (Data - dMin) / (dMax - dMin)
Data_train = Data_std * (max_Data - min_Data) + min_Data
Data_test = pd.read_excel(args.pathTest)
Data_test = Data_test.values
Data_std = (Data_test - dMin) / (dMax - dMin)
Data_test = Data_std * (max_Data - min_Data) + min_Data

mask_train = np.ones_like(Data_train)
mask_test = np.ones_like(Data_test)


print(mask_train.shape)

# Training
tf.reset_default_graph()

p = args.p # Prob of missing
epochs = args.epochs
batch_size = args.batch_size
latent_variable_dim = args.latent_dim
K = args.K
vae = active_learning_functions.train_p_vae(Data_train, mask_train, epochs, latent_variable_dim, batch_size, p, K, batch_size)


# Testing
n_test = Data_test.shape[0]
n_train = Data_train.shape[0]
OBS_DIM = Data_test.shape[1]

for nMax in args.maxNumAcquisitions:
    # personalized active feature selection strategy
    ## create arrays to store data and missingness
    x = Data_test[:, :]  #
    x = np.reshape(x, [n_test, OBS_DIM])
    y_test = x[:, -1].flatten().astype(bool)
    mask = np.zeros((n_test, OBS_DIM))  # this stores the mask of missingness (stems from both test data missingness and unselected features during active learing)
    mask2 = np.zeros((n_test,OBS_DIM))  # this stores the mask indicating that which features has been selected of each data
    mask[:,-1] = 0  # Note that no matter how you initialize mask, we always keep the target variable (last column) unobserved.
    im_saved = []
    M = 1
    actions = []
    mask_saved = []
    if nMax == -1:
        T = OBS_DIM - 1
    else:
        T = nMax

    for t in range(T): # t is a indicator of step
        R = -1e4 * np.ones((n_test, OBS_DIM - 1))
        im = p_vae.completion(x, mask, M, vae)
        for u in range(OBS_DIM - 1): # u is the indicator for features. calculate reward function for each feature candidates
            loc = np.where(mask[:, u] == 0)[0]

            R[loc, u] = p_vae.R_lindley_chain(u, x, mask, M, vae, im,
                                        loc)
        i_optimal = R.argmax(axis=1)
        actions.append(i_optimal)
        io = np.eye(OBS_DIM)[i_optimal]


        mask = mask + io # this mask takes into account both data missingness and missingness of unselected features
        negative_predictive_llh, uncertainty = vae.predictive_loss(x, mask, 'rmse',M)
        mask2 = mask2 + io # this mask only stores missingess of unselected features, i.e., which features has been selected of each data
        im = np.mean(p_vae.completion(x, mask, M, vae), axis = 0)

        im_saved.append(im)
        mask_saved.append(mask2)

    pred = im_saved[-1][:, -1].flatten()

    # Save results
    result = {}
    result['DatasetName'] = args.name
    result['DateEnd'] = str(datetime.datetime.now())
    result['Arguments'] = vars(args) #Save everything, just in case
    result['nMax'] = nMax

    result['Accuracy'] = (pred > 0.5) == y_test
    result['YPred'] = pred
    result['AccuracyMean'] = np.mean((pred > 0.5) == y_test)


    path = args.pathOutput.split('.')[0] + f'_{nMax}.pkl' 
    with open(path, 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
