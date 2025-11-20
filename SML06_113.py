#!/usr/bin/env python
# coding: utf-8

# In[1]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("zahidmughal2343/patients-satisfaction-data")

print("Path to dataset files:", path)


# In[4]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.graphics.regressionplots import plot_leverage_resid2
from statsmodels.stats.outliers_influence import OLSInfluence

data = pd.read_csv('Patient.csv')
data.head()

y = data['Satisfaction']  
X = data[['Age', 'Severity']]  
data.head()

X = sm.add_constant(X)

model = sm.OLS(y, X).fit()

print(model.summary())

plt.figure(figsize=(10, 5))
plt.scatter(model.fittedvalues, model.resid)
plt.axhline(0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residual Plot')
plt.show()

sm.qqplot(model.resid, line='45', fit=True)
plt.title('Q-Q Plot')
plt.show()

influence = OLSInfluence(model)

cooks_d = influence.cooks_distance[0]
plt.figure(figsize=(10, 5))
plt.stem(np.arange(len(cooks_d)), cooks_d, markerfmt=",")
plt.axhline(0.5, color='red', linestyle='--', label="Threshold")
plt.xlabel('Observation Index')
plt.ylabel("Cook's Distance")
plt.title("Cook's Distance Plot")
plt.legend()
plt.show()

plot_leverage_resid2(model)
plt.title('Leverage vs. Residuals')
plt.show()

dffits = influence.dffits[0]
plt.figure(figsize=(10, 5))
plt.stem(np.arange(len(dffits)), dffits, markerfmt=",")
plt.axhline(2 * np.sqrt(2 / len(y)), color='red', linestyle='--', label='Threshold')
plt.xlabel('Observation Index')
plt.ylabel('DFFITS')
plt.title('DFFITS Plot')
plt.legend()
plt.show()

dfbetas = influence.dfbetas
for i, predictor in enumerate(X.columns):
    plt.figure(figsize=(10, 5))
    plt.stem(np.arange(len(dfbetas)), dfbetas[:, i], markerfmt=",")
    plt.axhline(2 / np.sqrt(len(y)), color='red', linestyle='--', label='Threshold')
    plt.xlabel('Observation Index')
    plt.ylabel(f'DFBETAS for {predictor}')
    plt.title(f'DFBETAS Plot for {predictor}')
    plt.legend()
    plt.show()

leverage = influence.hat_matrix_diag
plt.figure(figsize=(10, 5))
plt.stem(np.arange(len(leverage)), leverage, markerfmt=",")
plt.axhline(2 * (X.shape[1] / len(y)), color='red', linestyle='--', label='Threshold')
plt.xlabel('Observation Index')
plt.ylabel('Leverage')
plt.title('Leverage Points Plot')
plt.legend()
plt.show()


# In[ ]:




