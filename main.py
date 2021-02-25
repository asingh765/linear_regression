
# coding: utf-8

# In[5]:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
get_ipython().magic('matplotlib inline')


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error


# In[6]:

Location = '/Users/adarshsingh/Downloads/kc_house_data.csv'
df = pd.read_csv(Location)


# In[7]:

df.head()


# In[8]:

df.isnull().sum() 


# In[189]:

corr = df.corr().abs()
sns.heatmap(data=corr,mask = corr < 0.4)
plt.show()


# In[55]:

#sqft_living = df['sqft_living'] 
#price = df['price']

#x.values.reshape(1, -1)


# In[140]:

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2) 
columns = ['sqft_living','sqft_lot','bedrooms','bathrooms','floors','waterfront','view','condition',
           'sqft_above','sqft_basement','yr_built','yr_renovated','sqft_living15','sqft_lot15']
x = df[columns] # predictor
y = df['price'] # response


# In[162]:

#df.values.reshape((999,1))
linreg = LinearRegression()
linreg.fit(x_train, y_train)


# In[163]:

print (linreg.intercept_)
print (linreg.coef_)


# In[188]:

# the model
y_predict = linreg.predict(x_test)
print ('Actual Values: ', y_test.values)
print ('Predicted Values: ', y_predict)


# In[179]:

mse = mean_squared_error(y_test, linreg.predict(x_test))
np.sqrt(mse)


# In[180]:

linreg.score(x_test,y_test)


# In[181]:

df_1 = pd.DataFrame()
df_1['predicted'] = y_predict
df_1['Actual Price'] = y_test
df_1[['Actual Price','predicted']].plot(alpha=0.5)
plt.show()


# In[ ]:



