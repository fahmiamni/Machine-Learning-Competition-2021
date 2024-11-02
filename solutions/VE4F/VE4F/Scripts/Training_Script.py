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
df1 = pd.read_csv('train.csv')

# Replace the large values with NaN
df1.replace(['-9999', -9999], np.nan, inplace=True)
# Drop all the rows that have just NaN
df1 = df1.dropna(how='all')
# Drop the rows with NaN at the outputs
df1 = df1.dropna(subset=['PHIF', 'SW', 'VSH'])

# Keep the following inputs
col_names =  ['DEN', 'GR', 'NEU', 'PEF', 'RDEP', 'ROP'] + list(df1.columns.values[-3:])
df1.dropna(axis=0, subset=col_names, inplace=True)
df1.drop(['BS','DTC', 'DTS'], axis=1, inplace=True)

# Log10 of the resistivity data due to large variations
df1['RMED']=np.log10(df1['RMED'])
df1['RDEP']=np.log10(df1['RDEP'])

# Remove outliers
Is = IsolationForest(random_state=0).fit(df1)
clf = Is.predict(df1)
df1=df1[clf==1]

# Augment the data using first and second derivative
Par=np.array(df1[['CALI', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED']])
D_Par=np.zeros(np.shape(Par))
for i in range(0,8):
    D_Par[:,i]=np.hstack((np.diff(Par[:,i]),0))

DD_Par=np.zeros(np.shape(Par))
for i in range(0,8):
    DD_Par[:,i]=np.hstack((np.diff(D_Par[:,i]),0))

p=np.hstack((Par , D_Par, DD_Par))


ppp=np.hstack((df1[['WELLNUM']], Par , D_Par, DD_Par, df1[['PHIF', 'SW', 'VSH']]))

df1 = pd.DataFrame(ppp, columns=['WELLNUM', 'CALI', 'DEN', 'DENC', 'GR', 'NEU', 'PEF', 'RDEP', 'RMED', \
                                 'D_CALI', 'D_DEN', 'D_DENC', 'D_GR', 'D_NEU', 'D_PEF', 'D_RDEP', 'D_RMED', \
                                 'DD_CALI', 'DD_DEN', 'DD_DENC', 'DD_GR', 'DD_NEU', 'DD_PEF', 'DD_RDEP', 'DD_RMED', 'PHIF', 'SW', 'VSH' ], index=df1.index)

f_d=df1

# Scale the data
w_num = f_d['WELLNUM']
col=f_d.columns
f_d.drop(['WELLNUM'], axis=1, inplace=True)
col2=f_d.columns
trans = RobustScaler()
n=MinMaxScaler()

gk=np.array(f_d)
data_w = trans.fit_transform(gk[:,:-3])
data_w = n.fit_transform(data_w)

trans2 = RobustScaler()
n2=MinMaxScaler()
data_q = trans2.fit_transform(gk[:,-3:])
data_q = n2.fit_transform(data_q)

data=np.hstack((data_w,data_q))

f_d = pd.DataFrame(data, columns=col2, index=f_d.index)
f_d2 = pd.DataFrame(np.hstack((np.reshape(np.array(w_num),(-1,1)),data)), columns=col)



# Use SMOTE between boreholes to balance the data between boreholes
smt=SMOTE(random_state=10)
d , w =smt.fit_resample(np.array(f_d),np.array(w_num))

w2=np.reshape(w,(-1,1))
da=np.hstack((w2,d))
df11 = pd.DataFrame(da,columns=col)

df11.drop(['WELLNUM'], axis=1, inplace=True)

# Train the model using Random forest
d2 = np.array(df11)
X_train = d2[:, :-3]
Y_train = d2[:, -3:]



estimator1 = RegressorChain(GradientBoostingRegressor(n_estimators= 125, random_state=100))
estimator1.fit(X_train,Y_train)

filehandler = open('estimator1', 'wb')
pickle.dump(estimator1, filehandler)
filehandler = open('trans', 'wb')
pickle.dump(trans, filehandler)
filehandler = open('n', 'wb')
pickle.dump(n, filehandler)


filehandler = open('trans2', 'wb')
pickle.dump(trans2, filehandler)
filehandler = open('n2', 'wb')
pickle.dump(n2, filehandler)





