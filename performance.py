import matplotlib.pyplot as plt
import pandas as pd
from ucimlrepo import fetch_ucirepo 
  
#fetch dataset 
student_performance = fetch_ucirepo(id=320) 
  
#data (as pandas dataframes) 
X = student_performance.data.features 
y = student_performance.data.targets 
  
#dropping two target variables to focus on just the final grades
y = y.drop(['G1','G2'], axis=1)

#dropping categorical data to avoid dummy variable trap
X = X.iloc[:,[2,6,7,12,13,14,23,24,25,26,27,28,29]]

#split into test and train 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#linear regression model
from sklearn.linear_model import LinearRegression
linearRegressor = LinearRegression()
linearRegressor.fit(X_train, Y_train)

#predict linear regression
y_pred_Linear = linearRegressor.predict(X_test)


#evaluating the model performance for linear
print("\nLinear Regression Metrics:")

from sklearn.metrics import r2_score,mean_absolute_error
R2_Linear = r2_score(Y_test, y_pred_Linear)
print("R^2=",R2_Linear) #r^2

MAE_Linear= mean_absolute_error(Y_test, y_pred_Linear)
print("MAE=", MAE_Linear) #mae

import statsmodels.api as sm
model = sm.OLS(endog = Y_train, exog = X_train).fit()
AR2_Linear = model.rsquared_adj
print("Adj R^2=", AR2_Linear) #adj r^2


#scale data
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
X_train= sc_x.fit_transform(X_train)
X_test= sc_x.fit_transform(X_test)

sc_y = StandardScaler()
Y_train = sc_y.fit_transform(Y_train)

#SVR model
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train, Y_train.ravel())

y_pred_scaled = regressor.predict(X_test)

y_pred_SVR = sc_y.inverse_transform(y_pred_scaled.reshape(len(y_pred_scaled),1))


#evaluating the model performance svr
print("\nSupport Vector Metrics:")

from sklearn.metrics import r2_score,mean_absolute_error
R2_SVR = r2_score(Y_test, y_pred_SVR)
print("R^2=", R2_SVR) #r^2

MAE_SVR = mean_absolute_error(Y_test, y_pred_SVR)
print("MAE=", MAE_SVR) #mae

import statsmodels.api as sm
model = sm.OLS(endog = Y_train, exog = X_train).fit()
AR2_SVR = model.rsquared_adj
print("Adj R^2=", AR2_SVR) # adj r^2

#bar graph to compare between the adjusted R^2 for each regression model
plt.bar(['Linear Regression', 'SVR'], [AR2_Linear, AR2_SVR], color=['blue', 'green'])
plt.ylabel('R^2')
plt.title('Regression Adjusted R^2 Comparison')
plt.ylim(0, 1)
plt.show()

#bar graph to compare between the  MAE for each regression model
plt.bar(['Linear Regression', 'SVR'], [MAE_Linear, MAE_SVR], color=['blue', 'green'])
plt.ylabel('Mean Squared Error')
plt.title('Regression Mean Squared Error Comparison')
plt.show()
