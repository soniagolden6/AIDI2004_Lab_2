#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install ucimlrepo


# In[2]:


from ucimlrepo import fetch_ucirepo 


# In[3]:


# fetch dataset 
breast_cancer_wisconsin_diagnostic = fetch_ucirepo(id=17) 
  
# data (as pandas dataframes) 
X = breast_cancer_wisconsin_diagnostic.data.features 
y = breast_cancer_wisconsin_diagnostic.data.targets 
  
# metadata 
print(breast_cancer_wisconsin_diagnostic.metadata) 
  
# variable information 
print(breast_cancer_wisconsin_diagnostic.variables) 


# In[4]:


X.head()


# In[5]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
# Assuming the variables X and y are already loaded
data = pd.concat([X, y], axis=1)

# Display the first few rows of the dataset
print(data.head())


# In[6]:


# Basic information
print(data.info())

# Summary statistics
print(data.describe())


# In[7]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, classification_report


# In[8]:



# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)


# In[9]:


# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Classification Report:\n{report}')


# In[ ]:





# In[12]:


import sweetviz as sv

report = sv.analyze(data)

# Display the report
report.show_html('sweetviz_report.html')


# In[ ]:




