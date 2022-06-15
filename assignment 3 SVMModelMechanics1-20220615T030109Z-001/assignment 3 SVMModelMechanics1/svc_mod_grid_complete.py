####################################################################################################################
#NO SCALING NO GRID
####################################################################################################################
import warnings
warnings.filterwarnings("ignore")

# Machine learning
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt

# Fetch the Data
#from iexfinance.stocks import get_historical_data 
from datetime import datetime

start = datetime(2017, 1, 1) # starting date: year-month-date
end = datetime(2018, 1, 1) # ending date: year-month-date

#Df = get_historical_data('SPY', start=start, end=end, output_format='pandas')  
Df = pd.read_csv("SPY.csv", parse_dates=['Date'])
Df = Df.sort_values(by='Date')
Df.set_index('Date', inplace = True)        
Df= Df.dropna()
Df = Df.rename (columns={'open':'Open', 'high':'High','low':'Low', 'close':'Close'})

#Df.Close.plot(figsize=(10,5))
#plt.ylabel("S&P500 Price")
#plt.show()

y = np.where(Df['Close'].shift(-1) > Df['Close'],1,-1)

Df['Open-Close'] = Df.Open - Df.Close
Df['High-Low'] = Df.High - Df.Low

X=Df[['Open-Close','High-Low']]
X.head()


#visualizing the data
plt.figure(figsize=(15, 5))

plt.scatter(Df['Open-Close'], Df['High-Low'], c=y)
plt.xlabel('Open-Close')
plt.ylabel('High-Low')

split_percentage = 0.8
split = int(split_percentage*len(Df))

# Train data set
X_train = X[:split]
y_train = y[:split] 

#there is another way of doing this split with:
#splitting into training and testing
#X_train, X_text, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=42, shuffle=False) #default=0.25 for test_size

# Test data set
X_test = X[split:]
y_test = y[split:]

cls = SVC().fit(X_train, y_train)
print(SVC()) #running with default parameters C=1.0, kernel=’rbf’, degree=3, gamma=’auto_deprecated’

accuracy_train = accuracy_score(y_train, cls.predict(X_train))
accuracy_test = accuracy_score(y_test, cls.predict(X_test))

print('\nTrain Accuracy no scaling no grid:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy no scaling no grid:{: .2f}%'.format(accuracy_test*100))

#trading the signals

Df['Predicted_Signal'] = cls.predict(X) #this predicts all X data, train and test, but only the test predictions will be analyzed

# Calculate log returns
Df['Return'] = np.log(Df.Close.shift(-1) / Df.Close)
Df['Strategy_Return'] = Df.Return * Df.Predicted_Signal
return_mean= Df['Strategy_Return'].iloc[split:].mean() #only for the test set (split onwards)
return_std= Df['Strategy_Return'].iloc[split:].std() #only for the test set (split onwards)
start_date = Df.iloc[split].name
end_date = Df.iloc[-1].name
days=(end_date - start_date).days
period=365
years=days/period
annualization_factor= period**.5
cumulative_returns = (Df.Strategy_Return.iloc[split:].cumsum()+1)[0:-1] #only for the test set (split onwards)
begin_balance=cumulative_returns.iloc[0]
end_balance=cumulative_returns.iloc[-1]
#calculate the CAGR and print the CAGR here (Financial Metrics.pptx)
CAGR=(end_balance/begin_balance)**(1/years) - 1
print("CAGR: ",CAGR)
#calculate the Sharpe_ratio and print the Sharpe_ratio here (Financial Metrics.pptx)
Sharpe_ratio = annualization_factor*return_mean/return_std
print("Sharpe Ratio: ", Sharpe_ratio)

#Plot the cumulative returns in Df["Cumulative_returns"] from the split onwards (test set)
#graph
Df.dropna()
Df['Strategy_Return'] = Df['Strategy_Return']*100 #multiplying by 100 for visibility of the graph
Df['Cumulative_returns'] = (Df.Strategy_Return.iloc[split:].cumsum()+1)[0:-1]
df_plot = Df['Cumulative_returns']
df_plot.plot(figsize=(10,5))
plt.xlabel("Date")
plt.ylabel("Cumulative Returns")
plt.show()



####################################################################################################################
#NO SCALING WITH GRID
####################################################################################################################

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn", lineno=193)


param_grid = [{'kernel': ['rbf'], 
               'C': [0.001, 0.01, 0.1, 1, 10, 100], 
                'gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'kernel': ['linear'],
              'C': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'kernel': ['poly'],
              'degree': [0, 1, 2, 3, 4, 5, 6]}]

grid_search = GridSearchCV(SVC(), param_grid, cv=5, return_train_score=True, n_jobs=-1)
grid_search.fit(X_train, y_train)

#results = pd.DataFrame(grid_search.cv_results_)
#print(results.T)
best_parameters = grid_search.best_params_
print("Best parameters no scaling grid: {}".format(best_parameters))
print("Best cross-validation score no scaling grid: {:.2f}%".format(grid_search.best_score_*100)) #default is accuracy
print("Test score no scaling grid (grid object): {:.2f}%".format(grid_search.score(X_test,y_test)*100)) #default is accuracy

results = pd.DataFrame(grid_search.cv_results_)
#print(results.T)
results.to_csv("results_svc.csv")
#set up and run the model with the best parameters
cls = SVC(**best_parameters)
cls.fit(X_train, y_train)
test_score = accuracy_score(y_test, cls.predict(X_test))
#print("model parameters no scaling grid: ", cls.get_params)
print("test score no scaling grid (model): {:.2f}%", test_score*100)
Df['Predicted_Signal'] = cls.predict(X)
Df['Return'] = np.log(Df.Close.shift(-1) / Df.Close)
Df['Strategy_Return'] = Df.Return * Df.Predicted_Signal
return_mean = Df['Strategy_Return'].mean()
return_std = Df['Strategy_Return'].std()
start_date = Df.iloc[split].name
end_date = Df.iloc[-1].name
days=(end_date - start_date).days
period=365
years=days/period
annualization_factor= period**.5
cumulative_returns = (Df.Strategy_Return.iloc[split:].cumsum()+1)[0:-1] #only for the test set (split onwards)
begin_balance=cumulative_returns.iloc[0]
end_balance=cumulative_returns.iloc[-1]
#calculate the CAGR and print the CAGR here (Financial Metrics.pptx)
CAGR=(end_balance/begin_balance)**(1/years) - 1
print("CAGR: ",CAGR)
#calculate the Sharpe_ratio and print the Sharpe_ratio here (Financial Metrics.pptx)
Sharpe_ratio = annualization_factor*return_mean/return_std
print("Sharpe Ratio: ", Sharpe_ratio)

#Plot the cumulative returns in Df["Cumulative_returns"] from the split onwards (test set)
#graph
Df.dropna()
Df['Strategy_Return'] = Df['Strategy_Return']*100
Df['Cumulative_returns'] = (Df.Strategy_Return.iloc[split:].cumsum()+1)[0:-1]
df_plot = Df['Cumulative_returns']
df_plot.plot(figsize=(10,5))
plt.xlabel("Date")
plt.ylabel("Cumulative Returns SVC")
plt.show()



'''
####################################################################################################################
#WITH SCALING WITH GRID
####################################################################################################################

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


pipe = Pipeline([("scaler", MinMaxScaler()),("svm", SVC())])

param_grid = [{'svm__kernel': ['rbf'], 
               'svm__C': [0.001, 0.01, 0.1, 1, 10, 100], 
                'svm__gamma': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'svm__kernel': ['linear'],
              'svm__C': [0.001, 0.01, 0.1, 1, 10, 100]},
              {'svm__kernel': ['poly'],
               'svm__degree': [0, 1, 2, 3, 4, 5, 6]}]

grid_search = GridSearchCV(pipe, param_grid, cv=5, return_train_score=True, n_jobs=-1)

grid_search.fit(X_train, y_train)

#results = pd.DataFrame(grid_search.cv_results_)
#print(results.T)
best_parameters = grid_search.best_params_
print("Best parameters scaling grid: {}".format(best_parameters))
print("Best cross-validation score scaling grid: {:.2f}".format(grid_search.best_score_*100)) #default is accuracy
print("Test score scaling grid (grid object): {:.2f}".format(grid_search.score(X_test,y_test)*100)) #default is accuracy

results = pd.DataFrame(grid_search.cv_results_)
#print(results.T)
results.to_csv("results_svc_norm.csv")

#set up and run the model with the best parameters
scl = MinMaxScaler()
scl.fit(X_train)
X_train = scl.transform(X_train)
X_test = scl.transform(X_test)
cls = SVC(kernel=best_parameters['svm__kernel'], C=best_parameters['svm__C'], gamma=best_parameters['svm__gamma'])
cls.fit(X_train, y_train)
test_score = accuracy_score(y_test, cls.predict(X_test))
#print("model parameters scaling grid: ", cls.get_params)
print("Test score scaling grid (model): ", test_score*100)
'''

#We compare the results generated by two different SVC settings. 
#The first one has smaller CAGR value but larger Sharpe Ratio.
#According to the definition of CAGR: the rate of return that would be required 
#for an investment to grow from its beginning balance to its ending balance, 
#assuming the profits were reinvested at the end of each year of the investment's lifespan 
#and the definition of Sharpe Ratio: CAGR divided by the average annual standard deviation of returns. 
#It is an indicator of the relative trade-off of risk and reward. The first one has less returns but is less risky.
#The second one has larger CAGR value but smaller Sharpe Ratio. 
#This means the second model has more returns but is also more risky. The plots of cumulative return also indicates that the second one has an overall higher return cumulatively than the first one
#Thus, the first one is more suitable for conservative investors who prefer low risk lower return and the second one is more suitable for investors who
#concentrate more on returns and is willing to take a bigger risk. Therefore, overall the second one is slightly better as the cumulative return is higher.