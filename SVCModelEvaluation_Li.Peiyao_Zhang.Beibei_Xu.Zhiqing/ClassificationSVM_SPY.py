'''
import os, os.path
# import winsound
from sys import platform
if os.name == 'nt' or platform == 'win32':
    os.chdir(os.path.dirname(__file__))
    '''

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from fwTimeSeriesSplit import fwTimeSeriesSplit
import pandas as pd
import numpy as np
import talib as ta

import seaborn
import matplotlib.pyplot as plt
#%matplotlib inline

def SVC_model(n=10, cv_model="tscv",random_state_model=None):
    #Df = pd.read_csv('SPY_800.csv') 
    Df = pd.read_csv('NSE_cash.csv') 
    Df=Df[['Time','Open','High','Low','Close','Volume']]
    Df.head()
    Df=Df.drop(Df[Df['Volume']==0].index)
    Df['Time']=pd.to_datetime(Df['Time'])
    Df.head()

    t=.8
    split = int(t*len(Df))

    Df['RSI']=ta.RSI(np.array(Df['Close'].shift(1)), timeperiod=n)

    Df['SMA'] = Df['Close'].shift(1).rolling(window=n).mean()

    Df['Corr'] = Df['Close'].shift(1).rolling(window=n).corr(Df['SMA'].shift(1))


    Df['SAR']=ta.SAR(np.array(Df['High'].shift(1)),np.array(Df['Low'].shift(1)), 0.2,0.2)
    Df['ADX']=ta.ADX(np.array(Df['High'].shift(1)),np.array(Df['Low'].shift(1)), np.array(Df['Open']), timeperiod =n)

    Df['close'] = Df['Close'].shift(1)
    Df['high'] = Df['High'].shift(1)
    Df['low'] = Df['Low'].shift(1)

    Df['OO']= Df['Open']-Df['Open'].shift(1)
    Df['OC']= Df['Open']-Df['close']

    Df['Ret']=np.log(Df['Open'].shift(-1)/Df['Open'])
    for i in range(1,n):
        Df['return%i'%i]=Df['Ret'].shift(i)

    Df.head()

    #the correlation function used above to generate the Df['Corr'] column
    #should not be generating any values outside -1 and 1; 
    #however, something is going on that makes the function generate infinite or NAN values sometimes and
    #we have to get rid of these errors

    Df.loc[Df['Corr']<-1,'Corr']=-1
    Df.loc[Df['Corr']>1,'Corr']=1
    Df=Df.dropna()

    Df['Signal']=0
    Df.loc[Df['Ret']>Df['Ret'][:split].quantile(q=0.66),'Signal']=1
    Df.loc[Df['Ret']<Df['Ret'][:split].quantile(q=0.34),'Signal']=-1
    X=Df.drop(['Close','Signal','Time','High','Low','Volume','Ret'],axis=1)
    y=Df['Signal']
    X2D = X[['SMA','RSI']]
    #print("Our selected features: \n", X2D.head(1))
    X2D_arr=X2D.values
    #print("Our X2D, first three rows: \n", X2D_arr[0:3,:])
    y_arr=y.values
    #print("Our y values, first six rows: ", y_arr[0:6])
    #print("X2D.shape:", X2D_arr.shape)
    X2D = X[['ADX','Corr']]
    #print("Our selected features: \n", X2D.head(1))
    X2D_arr=X2D.values
    #print("Our X2D, first three rows: \n", X2D_arr[0:3,:])
    y_arr=y.values
    #print("Our y values, first six rows: ", y_arr[0:6])
    #print("X2D.shape:", X2D_arr.shape)
    X2D = X[['SAR','ADX']]
    #print("Our selected features: \n", X2D.head(1))
    X2D_arr=X2D.values
    #print("Our X2D, first three rows: \n", X2D_arr[0:3,:])
    y_arr=y.values
    #print("Our y values, first six rows: ", y_arr[0:6])
    #print("X2D.shape:", X2D_arr.shape)
    steps = [
         		    ('scaler',StandardScaler()),
        		    ('svc',SVC())
                  ]
    pipeline =Pipeline(steps)

    tscv = TimeSeriesSplit(n_splits=7)
    tscv_fw = fwTimeSeriesSplit(n_splits=7)
    CV7=7

    #for grid search
    c_gs =[10,100,1000,10000]
    g_gs = [1e-2,1e-1,1e0] 

    #for random search
    c_rs = np.linspace(10, 10000, num=40, endpoint=True)
    g_rs = np.linspace(1e-2, 1e0, num=30, endpoint=True)



    #set of parameters for grid search
    parameters_gs = {
              		    'svc__C':c_gs,
              		    'svc__gamma':g_gs,
              		    'svc__kernel': ['rbf', 'poly']
             	               }
    #set of parameters for random search
    parameters_rs = {
              		    'svc__C':c_rs,
              		    'svc__gamma':g_rs,
              		    'svc__kernel': ['rbf', 'poly']
             	               }

    Df.groupby("Signal").count() #shows -1, 0 and 1 occur in roughly equal numbers


    from sklearn.metrics import SCORERS
    #sorted(SCORERS.keys()) #to get available scorers

    #cvo = GridSearchCV(pipeline, parameters_gs,cv=7, scoring=None)
    #cvo = RandomizedSearchCV(pipeline, parameters_rs,cv=7, scoring=None, n_iter=50, random_state=None)
    #cvo = RandomizedSearchCV(pipeline, parameters_rs,cv=tscv, scoring=None, n_iter= 50, random_state=70) #7 
    #cvo = RandomizedSearchCV(pipeline, parameters_rs,cv=tscv_fw, scoring=None, n_iter=50, random_state=None)


    cvo = RandomizedSearchCV(pipeline, parameters_rs, cv=locals()[cv_model], scoring=None, n_iter= 50, random_state=random_state_model)


    cvo.fit(X.iloc[:split],y.iloc[:split])
    best_C = cvo.best_params_['svc__C']
    best_kernel =cvo.best_params_['svc__kernel']
    best_gamma=cvo.best_params_['svc__gamma']
    cls = SVC(C =best_C,kernel=best_kernel, gamma=best_gamma)
    ss1= StandardScaler()
    cls.fit(ss1.fit_transform(X.iloc[:split]),y.iloc[:split])
    y_predict =cls.predict(ss1.transform(X.iloc[split:]))


    Df['Pred_Signal']=0
    Df.iloc[:split,Df.columns.get_loc('Pred_Signal')]\
           =pd.Series(cls.predict(ss1.transform(X.iloc[:split])).tolist())
    Df.iloc[split:,Df.columns.get_loc('Pred_Signal')]=y_predict

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import plot_confusion_matrix
    from sklearn.utils.multiclass import unique_labels
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import classification_report

    cls_d = DummyClassifier(strategy='uniform') #can substitute for some other strategy
    cls_d.fit(ss1.fit_transform(X.iloc[:split]),y.iloc[:split])
    y_predict_d =cls_d.predict(ss1.transform(X.iloc[split:]))

    #print("DummyStrategy=\"uniform\", accuracy=",round(classification_report(Df['Signal'].iloc[split:], y_predict_d,output_dict=True)["accuracy"],3))
    print("    SVC Classifier, accuracy=",round(classification_report(Df['Signal'].iloc[split:], y_predict,output_dict=True)["accuracy"],3))


    Df['Ret1']=Df['Ret']*Df['Pred_Signal'] 
    Df['Cu_Ret1']=0.

    Df['Cu_Ret1']=np.cumsum(Df['Ret1'].iloc[split:])

    Df['Cu_Ret']=0.
    Df['Cu_Ret']=np.cumsum(Df['Ret'].iloc[split:])

    Std =np.std(Df['Cu_Ret1'])
    Sharpe = (Df['Cu_Ret1'].iloc[-1]-Df['Cu_Ret'].iloc[-1])/Std #will not annualize this because the data is intraday data
    #print('Sharpe Ratio:',Sharpe)

    #Having run the code a number of times with fixed window and with growing window TimeSeriesSplit, that is:
    #tscv = TimeSeriesSplit(n_splits=7)
    #tscv = fwTimeSeriesSplit(n_splits=7)
    #The maximum values are:
    #Sharpe Ratio: --- with growing window (maximum)
    #Sharpe Ratio: --- with fixed window (maximum)

    import WhiteRealityCheckFor1_noprint
    import detrendPrice 
    #Detrend prices before calculating detrended returns
    Df['DetOpen'] = detrendPrice.detrendPrice(Df['Open']).values 
    #these are the detrended returns to be fed to White's Reality Check
    Df['DetRet']=np.log(Df['DetOpen'].shift(-1)/Df['DetOpen'])
    Df['DetStrategy']=Df['DetRet']*Df['Pred_Signal'] 
    WhiteRealityCheck_p_value=WhiteRealityCheckFor1_noprint.bootstrap(Df['DetStrategy'].iloc[split:])

    return Sharpe, WhiteRealityCheck_p_value



if __name__ == '__main__':
    SVC_model(n=2)