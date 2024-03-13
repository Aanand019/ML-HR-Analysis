#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# A large company named XYZ, employs, at any given point of time, around 4000 employees. However, every year, around 15% of its employees leave the company and need to be replaced with the talent pool available in the job market. The management believes that this level of attrition (employees leaving, either on their own or because they got fired) is bad for the company, because of the following reasons -
# 
# 1.The former employees’ projects get delayed, which makes it difficult to meet timelines, resulting in a reputation loss among consumers and partners
# 
# 2.A sizeable department has to be maintained, for the purposes of recruiting new talent
# 
# 3.More often than not, the new employees have to be trained for the job and/or given time to acclimatise themselves to the company
# 
# Hence, the management has contracted an HR analytics firm to understand what factors they should focus on, in order to curb attrition. In other words, they want to know what changes they should make to their workplace, in order to get most of their employees to stay. Also, they want to know which of these variables is most important and needs to be addressed right away.
# 
# Since you are one of the star analysts at the firm, this project has been given to you.

# # Goal of the case study
# 
# You are required to model the probability of attrition using a logistic regression.
# 
# The results thus obtained will be used by the management to understand what changes they should make to their workplace, in order to get most of their employees to stay.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[ ]:





# # Step 1: Importing and Merging Data

# In[2]:


manager_survey_data = pd.read_csv(r"C:\Users\hp\Downloads\archive (11)\manager_survey_data.csv")
out_time = pd.read_csv(r"C:\Users\hp\Downloads\archive (11)\out_time.csv")
data_dictionary = pd.read_excel(r"C:\Users\hp\Downloads\archive (11)\data_dictionary.xlsx")
employee_survey_data = pd.read_csv(r"C:\Users\hp\Downloads\archive (11)\employee_survey_data.csv")
general_data = pd.read_csv(r"C:\Users\hp\Downloads\archive (11)\general_data.csv")
in_time = pd.read_csv(r"C:\Users\hp\Downloads\archive (11)\in_time.csv")


# In[3]:


in_time.head()


# In[4]:


in_time=in_time.replace(np.nan,0)
in_time.head()


# In[5]:


in_time.iloc[:, 1:] = in_time.iloc[:, 1:].apply(pd.to_datetime, errors='coerce')


# In[6]:


out_time=out_time.replace(np.nan,0)
out_time.head()


# In[7]:


out_time.iloc[:, 1:] = out_time.iloc[:, 1:].apply(pd.to_datetime, errors='coerce')


# In[8]:


in_time=in_time.append(out_time)


# In[9]:


in_time.shape


# In[10]:


in_time=in_time.diff(periods=4410)
in_time=in_time.iloc[4410:]
in_time.reset_index(inplace=True)
in_time.head()


# In[11]:


in_time.drop(columns=['index','Unnamed: 0'],axis=1,inplace=True)
in_time.head()


# In[12]:


in_time.shape


# In[13]:


in_time.drop(['2015-01-01', '2015-01-14','2015-01-26','2015-03-05',
             '2015-05-01','2015-07-17','2015-09-17','2015-10-02',
              '2015-11-09','2015-11-10','2015-11-11','2015-12-25'
             ], axis = 1,inplace=True) 


# In[14]:


in_time['Actual Time']=in_time.mean(axis=1)


# In[15]:


in_time['hrs']=in_time['Actual Time']/np.timedelta64(1, 'h')
in_time.head()


# In[16]:


in_time.reset_index(inplace=True)
in_time.head()


# In[17]:


in_time.drop(in_time.columns.difference(['index','hrs']), 1, inplace=True)


# In[18]:


in_time.rename(columns={'index': 'EmployeeID'},inplace=True)
in_time.head()


# In[19]:


in_time.shape


# In[20]:


general_data.head()


# In[21]:


employee_survey_data.head()


# # Combining all data files into one consolidated dataframe

# In[22]:


df_1 = pd.merge(employee_survey_data, general_data, how='inner', on='EmployeeID')
hr = pd.merge(manager_survey_data, df_1, how='inner', on='EmployeeID')
hr = pd.merge(in_time, hr, how='inner', on='EmployeeID')
hr.head()


# In[23]:


hr.info()


# ## Drop Non Required Columns

# In[24]:


hr.drop(['EmployeeID', 'EmployeeCount','StandardHours','Over18'], axis = 1,inplace=True)


# In[25]:


hr.isnull().sum()[hr.isnull().sum()>0] 


# In[26]:


hr.EnvironmentSatisfaction.value_counts()


# In[27]:


hr.EnvironmentSatisfaction.fillna(hr.EnvironmentSatisfaction.mean(),inplace=True)
hr.JobSatisfaction.fillna(hr.JobSatisfaction.mean(),inplace=True)
hr.WorkLifeBalance.fillna(hr.WorkLifeBalance.mean(),inplace=True)
hr.NumCompaniesWorked.fillna(hr.NumCompaniesWorked.mean(),inplace=True)
hr.TotalWorkingYears.fillna(hr.TotalWorkingYears.mean(),inplace=True)


# In[ ]:





# In[28]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[29]:


hr[hr.select_dtypes(include='object').columns]=hr[hr.select_dtypes(include='object').columns].apply(le.fit_transform)


# In[30]:


hr.head()


# # train_test_split

# In[31]:


from sklearn.model_selection import train_test_split


# In[32]:


hr_train ,hr_test = train_test_split(hr,test_size=.2)


# In[33]:


# over sampling
df1=hr_train[hr_train.Attrition==1]
hr_train=pd.concat([hr_train,df1,df1,df1])
hr_train.shape


# In[34]:


hr_train_x = hr_train.iloc[::,hr.columns!='Attrition']
hr_train_y = hr_train.Attrition

hr_test_x = hr_test.iloc[::,hr.columns!='Attrition']
hr_test_y = hr_test.Attrition


# # LogisticRegression

# In[35]:


from sklearn.linear_model import LogisticRegression 


# In[36]:


logreg = LogisticRegression() 


# In[37]:


logreg.fit(hr_train_x,hr_train_y) 


# In[38]:


pred_test = logreg.predict(hr_test_x) 


# In[39]:


from sklearn.metrics import confusion_matrix 


# In[40]:


matrix_test = confusion_matrix(hr_test_y , pred_test) 
matrix_test 


# In[41]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


# In[42]:


accuracy_score(hr_test_y , pred_test)


# In[43]:


recall_score(hr_test_y , pred_test)


# In[44]:


precision_score(hr_test_y , pred_test)


# In[45]:


f1_score(hr_test_y , pred_test)


# In[46]:


from sklearn.metrics import roc_auc_score , roc_curve


# In[47]:


pred_prob_test=logreg.predict_proba(hr_test_x)
len(pred_prob_test)


# In[48]:


fpr , tpr , ther = roc_curve(hr_test_y , pred_prob_test[: , 1])


# In[49]:


plt.plot(fpr , tpr, marker = "." )
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC plot on fpr and tpr")
plt.grid()


# # DecisionTreeClassifier
# with gini
# 
# It is a type of supervised learning algorithm that is mostly used for classification problems. Surprisingly, it works for both categorical and continuous dependent variables. In this algorithm, we split the population into two or more homogeneous sets. This is done based on most significant attributes/ independent variables to make as distinct groups as possible.

# In[50]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()


# In[51]:


dt.fit(hr_train_x,hr_train_y) 


# In[52]:


pred_test = dt.predict(hr_test_x)


# In[53]:


from sklearn.metrics import confusion_matrix


# In[54]:


matrix_test = confusion_matrix(hr_test_y , pred_test)
matrix_test


# In[55]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


# In[56]:


accuracy_score(hr_test_y , pred_test)


# In[57]:


recall_score(hr_test_y , pred_test)


# In[58]:


precision_score(hr_test_y , pred_test)


# In[59]:


f1_score(hr_test_y , pred_test)


# In[60]:


from sklearn.metrics import roc_auc_score , roc_curve


# In[61]:


pred_prob_test=dt.predict_proba(hr_test_x)


# In[62]:


fpr , tpr , ther = roc_curve(hr_test_y , pred_prob_test[: , 1])


# In[63]:


plt.plot(fpr , tpr, marker = "." )
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC plot on fpr and tpr")
plt.grid()


# In[ ]:





# # DecisionTreeClassifier
# with entropy

# In[64]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion ='entropy')


# In[65]:


dt.fit(hr_train_x,hr_train_y) 


# In[66]:


pred_test = dt.predict(hr_test_x)


# In[67]:


from sklearn.metrics import confusion_matrix


# In[68]:


matrix_test = confusion_matrix(hr_test_y , pred_test)
matrix_test


# In[69]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


# In[70]:


accuracy_score(hr_test_y , pred_test)


# In[71]:


recall_score(hr_test_y , pred_test)


# In[72]:


precision_score(hr_test_y , pred_test)


# In[73]:


f1_score(hr_test_y , pred_test)


# In[74]:


from sklearn.metrics import roc_auc_score , roc_curve


# In[75]:


pred_prob_test=dt.predict_proba(hr_test_x)


# In[76]:


fpr , tpr , ther = roc_curve(hr_test_y , pred_prob_test[: , 1])


# In[77]:


plt.plot(fpr , tpr, marker = "." )
plt.xlabel("fpr")
plt.ylabel("tpr")
plt.title("ROC plot on fpr and tpr")
plt.grid()


# # RandomForestClassifier
# 

# In[78]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=250,criterion ='entropy')


# In[79]:


rfc.fit(hr_train_x,hr_train_y)


# In[80]:


pred_rfc_hr = rfc.predict(hr_test_x)


# In[82]:


tab_rfc_hr = confusion_matrix(hr_test_y, pred_rfc_hr)


# In[83]:


pred_test =rfc.predict(hr_test_x)
pred_test = rfc.predict(hr_test_x)
mat_test = confusion_matrix(hr_test_y , pred_test)
mat_test


# In[84]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


# In[85]:


accuracy_score(hr_test_y, pred_rfc_hr)


# In[86]:


recall_score(hr_test_y, pred_rfc_hr)


# In[87]:


precision_score(hr_test_y, pred_rfc_hr)


# In[88]:


f1_score(hr_test_y, pred_rfc_hr)


# In[ ]:





# # kNN (k- Nearest Neighbors)
# It can be used for both classification and regression problems. However, it is more widely used in classification problems in the industry. K nearest neighbors is a simple algorithm that stores all available cases and classifies new cases by a majority vote of its k neighbors.

# In[98]:


from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=9, leaf_size=20)


# In[99]:


kn.fit(hr_train_x,hr_train_y)


# In[100]:


pred_kn_hr = kn.predict(hr_test_x)


# In[101]:


tab_kn_hr = confusion_matrix(hr_test_y, pred_rfc_hr)


# In[102]:


pred_test = kn.predict(hr_test_x)
pred_test = kn.predict(hr_test_x)
mat_test = confusion_matrix(hr_test_y , pred_test)
mat_test


# In[103]:


from sklearn.metrics import accuracy_score,recall_score,precision_score,f1_score


# In[104]:


accuracy_score(hr_test_y, pred_kn_hr)


# In[105]:


recall_score(hr_test_y, pred_kn_hr)


# In[106]:


f1_score(hr_test_y, pred_kn_hr)


# # SVM (Support Vector Machine)
# It is a classification method. In this algorithm, we plot each data item as a point in n-dimensional space (where n is number of features you have) with the value of each feature being the value of a particular coordinate.

# In[107]:


from sklearn.svm import SVC
svc = SVC(kernel="linear")


# In[108]:


svc.fit(hr_train_x,hr_train_y)


# In[109]:


pred_svc_hr = svc.predict(hr_test_x)


# In[111]:


tab_svc_hr = confusion_matrix(hr_test_y, pred_svc_hr)


# In[112]:


pred_test = svc.predict(hr_test_x)
pred_test = svc.predict(hr_test_x)
mat_test = confusion_matrix(hr_test_y , pred_test)
mat_test


# In[113]:


accuracy_score(hr_test_y, pred_svc_hr)


# In[114]:


recall_score(hr_test_y, pred_svc_hr)


# In[115]:


f1_score(hr_test_y, pred_svc_hr)


# In[ ]:





# In[ ]:





# In[ ]:





# # feature_importances_ in rfc

# In[116]:


rf_model = rfc.fit(hr_train_x,hr_train_y)
rf_model.feature_importances_


# In[117]:


importances = pd.DataFrame(hr_train_x.columns,columns=['Features'])
importances


# In[119]:


imp = importances.sort_values(by='Feature_importances',ascending=False)
imp


# In[120]:


sns.barplot(y=imp['Features'],x=imp['Feature_importances'])
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




