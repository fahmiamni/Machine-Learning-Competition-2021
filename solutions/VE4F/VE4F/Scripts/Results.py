# import libraries
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.preprocessing import RobustScaler, normalize, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.multioutput import RegressorChain
#from sklearn.impute import IterativeImputer
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cluster import KMeans
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow import keras
from sklearn.linear_model import LinearRegression
import pickle
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


import random
from sklearn.impute import KNNImputer

def result_plot(y_predict, y_real, n_points=5000):
    names = ['PHIF', 'SW', 'VSH']
    RMSE, R2 = [], []

    for i, name in enumerate(names):
        RMSE.append(np.sqrt(mean_squared_error(y_real[:,i], y_predict[:,i])))
        R2.append(r2_score(y_real[i], y_predict[i]))

    # check the accuracy of predicted data and plot the result
    print('RMSE:', '{:.5f}'.format(np.sqrt(mean_squared_error(y_real, y_predict))))
    for i, name in enumerate(names):
        print(f'    {name:5s} : {RMSE[i]:.5f}')
    #     print("-"*65)

    print('R^2: ', r2_score(y_real, y_predict))
    for i, name in enumerate(names):
        print(f'    {name:5s} : {R2[i]:.5f}')

    plt.subplots(nrows=3, ncols=2, figsize=(16, 16))

    for i, name in enumerate(names):
        plt.subplot(3, 2, i * 2 + 1)
        plt.plot(y_real[:n_points, i])
        plt.plot(y_predict[:n_points, i])
        plt.legend(['True', 'Predicted'])
        plt.xlabel('Sample')
        plt.ylabel(name)
        plt.title(name + ' Prediction Comparison')

        plt.subplot(3, 2, i * 2 + 2)
        plt.scatter(y_real[:, i], y_predict[:, i], alpha=0.01)
        plt.xlabel('Real Value')
        plt.ylabel('Predicted Value')
        plt.title(name + ' Prediction Comparison')

    plt.show()


# Open the training data
df1 = pd.read_csv('test.csv')

# Replace the large values with NaN
df1.replace(['-9999', -9999], np.nan, inplace=True)
# Drop all the rows that have just NaN
# Drop the rows with NaN at the outputs
df1.drop(['WELLNUM','DEPTH','BS','DTC', 'DTS', 'ROP'], axis=1, inplace=True)

# Log10 of the resistivity data due to large variations
df1['RMED']=np.log10(df1['RMED'])
df1['RDEP']=np.log10(df1['RDEP'])



dat = np.array(df1)



# Impute the data for missing values
imp_mean = IterativeImputer(n_nearest_features=None, imputation_order='ascending', random_state=10)
imp_mean.fit(dat)
dat2 = imp_mean.transform(dat)
imputer = KNNImputer(n_neighbors=10, weights="uniform")
dat3 = imputer.fit_transform(dat)
dat2 = (dat2 + dat3) / 2

df1 = pd.DataFrame(dat2, columns=['CALI', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED'])

# Augment the data using first and second derivative
Par=np.array(df1[['CALI', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED']])
D_Par=np.zeros(np.shape(Par))
for i in range(0,8):
    D_Par[:,i]=np.hstack((np.diff(Par[:,i]),0))

DD_Par=np.zeros(np.shape(Par))
for i in range(0,8):
    DD_Par[:,i]=np.hstack((np.diff(D_Par[:,i]),0))

p=np.hstack((Par , D_Par, DD_Par))

df1 = pd.DataFrame(p, columns=['CALI', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED', \
                                 'D_CALI', 'D_DEN', 'D_DENC', 'D_GR', 'D_NEU', 'D_PEF', 'D_RDEP', 'D_RMED', \
                                 'DD_CALI', 'DD_DEN', 'DD_DENC', 'DD_GR', 'DD_NEU', 'DD_PEF', 'DD_RDEP', 'DD_RMED'], index=df1.index)


filehandler = open('trans', 'rb')
trans = pickle.load(filehandler)
filehandler = open('n', 'rb')
n = pickle.load(filehandler)

gk=np.array(df1)
data = trans.transform(gk)
data = n.transform(data)

filehandler = open('estimator1', 'rb')
estimator1 = pickle.load(filehandler)

filehandler = open('trans2', 'rb')
trans2 = pickle.load(filehandler)
filehandler = open('n2', 'rb')
n2 = pickle.load(filehandler)

y=estimator1.predict(data)


yyd=n2.inverse_transform(y)
test_predict2=trans2.inverse_transform(yyd)

test_predict2=np.where(test_predict2 > 0.015, test_predict2, 0.015)
ttg=test_predict2[:,1]
ttg=np.where(ttg < 1, ttg, 1)
test_predict2[:,1]=ttg



col_names = ['PHIF', 'SW', 'VSH']
# Replace team_name and num_submit
team_name = 'VE4F'
num_submit = 1

# Please don't change codes below
N_SAMPLES = 11275
n_sub_dict = {1: 1, 2: 2, 3: 3}

# Check submission number is correct
try:
    n_sub = n_sub_dict[num_submit]
except KeyError:
    print(f"ERROR!!! Sumbmission Number must be in 1, 2 or 3")

# Check number of samples are correct
if test_predict2.shape[0] != N_SAMPLES:
    raise ValueError(f"Number of samples {test_predict2.shape[0]} doesn't matches with the correct value {N_SAMPLES}")

# Write results to csv file
output_result = pd.DataFrame(
    {col_names[-3]: test_predict2[:, 0], col_names[-2]: test_predict2[:, 1], col_names[-1]: test_predict2[:, 2]})
output_result.to_csv(path_or_buf=f'./{team_name}_{n_sub}.csv', index=False)







