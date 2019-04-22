# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:26:08 2019

@author: ASUS
"""
import spacy
import pdfminer
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import Imputer
data = pd.read_csv('C:\\Users\\ASUS\\Desktop\\DATA_SCIENCE_FOLDER\\Data_Set.csv\\IBM_DATA.csv')
d=data.corr()
print (d)
data = data.drop(["EmployeeCount","Over18"],axis = 1)
#a = 1752 -1470
#check if there is any null value
data.isnull().values.any()
#data.isnull().sum() - this will return the count of NULLs/NaN values in each column.
data.isnull().sum()

#Finding Outlier
std= np.std(data["MonthlyIncome"])
max1 = np.max(data["MonthlyIncome"])
min1= np.min(data["MonthlyIncome"])
avg_monthlyincome = np.mean(data["MonthlyIncome"])

std1= np.std(data["PerformanceRating"])
max2 = np.max(data["PerformanceRating"])
min2= np.min(data["PerformanceRating"])
avg_performance_rating = np.mean(data["PerformanceRating"])
std1
max2
min2

std2= np.std(data["WorkLifeBalance"])
max3 = np.max(data["WorkLifeBalance"])
min3= np.min(data["WorkLifeBalance"])
avg_performance_rating = np.mean(data["WorkLifeBalance"])
std2
max3
min3

#plt.boxplot(data["MonthlyIncome"])
#plt.show()
data["MonthlyIncome"].quantile([0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])


#pre data processing

data.iloc[:,0:8] = data.fillna(method='ffill')
data.iloc[:,32] = data.fillna(method='ffill')
imputer = Imputer(missing_values="NaN",strategy= "mean",axis = 0)
columns_to_impute = ["Age","DailyRate","DistanceFromHome","Education","EnvironmentSatisfaction","HourlyRate","JobInvolvement","JobLevel","JobSatisfaction","MonthlyIncome","MonthlyRate","NumCompaniesWorked","PercentSalaryHike","PerformanceRating","RelationshipSatisfaction","StandardHours","StockOptionLevel","TotalWorkingYears","TrainingTimesLastYear","WorkLifeBalance","YearsAtCompany","YearsInCurrentRole","YearsSinceLastPromotion","YearsWithCurrManager"]
data[columns_to_impute] = Imputer().fit_transform(data[columns_to_impute])
data.isnull().sum()

data = data.drop_duplicates(subset=['EmployeeNumber'], keep=False)
len(data)


#BEST EMPLOYEE
coeff6= np.corrcoef(data['JobLevel'],data['MonthlyIncome'])[0,1]
coeff6
coeff6= np.corrcoef(data['JobLevel'],data['MonthlyIncome'])[0,1]

coeff8 = np.corrcoef(data['Education'],data['JobLevel'])[0,1]
coeff8
coeff9 = np.corrcoef(data['Education'],data['JobLevel'])[0,1]
coeff8
coeff2 = np.corrcoef(data['WorkLifeBalance'],data['JobLevel'])[0,1]
coeff2
coeff3 = np.corrcoef(data['JobSatisfaction'],data['JobLevel'])[0,1]
coeff3
coeff4 = np.corrcoef(data['DailyRate'],data['MonthlyIncome'])[0,1]
coeff4
coeff5 = np.corrcoef(data['PerformanceRating'],data['MonthlyIncome'])[0,1]
coeff5
coeff10 = np.corrcoef(data['PerformanceRating'],data['MonthlyRate'])[0,1]
coeff10
coeff11 = np.corrcoef(data['JobInvolvement'],data['MonthlyIncome'])[0,1]
coeff11

a = set(data["Education"])
a
b= set(data["WorkLifeBalance"])
b
c = set (data["PerformanceRating"])
c
#Best Employee

education = data["Education"]>=3
WorkLifeBalance= data["WorkLifeBalance"]>=3
PerformanceRating =data["PerformanceRating"]>=3
best_employee= data[education & WorkLifeBalance & PerformanceRating]




female_employee =best_employee[best_employee["Gender"]== 'Female' ]
avg_monthlyincome_female = np.mean(female_employee["MonthlyIncome"])
avg_performance_rating_female = np.mean(female_employee["PerformanceRating"])
avg_performance_rating_female
len(female_employee)
job_satisfaction_female= female_employee[female_employee["JobSatisfaction"] < 3]
len(job_satisfaction_female)
job_satisfaction_female_avg= np.mean(job_satisfaction_female["JobSatisfaction"])
len(job_satisfaction_female)
job_satisfaction_female_avg


male_employee =best_employee[best_employee["Gender"]== 'Male' ]
avg_monthlyincome_male = np.mean(male_employee["MonthlyIncome"])
avg_performance_rating_male = np.mean(male_employee["PerformanceRating"])
avg_performance_rating_male
len(male_employee)
job_satisfaction_male= male_employee[male_employee["JobSatisfaction"] < 3]
job_satisfaction_male_avg= np.mean(job_satisfaction_male["JobSatisfaction"])
len(job_satisfaction_male)
job_satisfaction_male_avg




best_employee_salary_higher_avg = best_employee[best_employee["MonthlyIncome"] > avg_monthlyincome]
len(best_employee_salary_higher_avg)



best_employee_salary_lower_avg = best_employee[best_employee["MonthlyIncome"] < avg_monthlyincome]
len(best_employee_salary_lower_avg)

avarage_daily_rate_higher_salary_people = np.mean(best_employee_salary_higher_avg["DailyRate"])
avarage_daily_rate_higher_salary_people
avarage_daily_rate_lower_salary_people = np.mean(best_employee_salary_lower_avg["DailyRate"])
avarage_daily_rate_lower_salary_people

diffrence = 842.7265 - 765.6033
diffrence



    
x = best_employee.iloc[:,:-1]
y=best_employee.iloc[:,-1]

x = pd.get_dummies(x, columns=["Gender","EducationField","MaritalStatus","BusinessTravel","Department","JobRole","OverTime"])
# create training and testing vars
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

##DIMENSION REDUCTION
#from sklearn.decomposition import PCA
#pca = PCA(n_components =2)
#x_train = pca.fit_transform(x_train)
#x_test=pca.transform(x_test)
#explained_variance=pca.explained_variance_ratio_
#imputer = Imputer(missing_values="NaN",strategy= "mean",axis = 0)
#imputer = imputer.fit(x[:,6:])
#x[:,6:]=imputer.transform(x[:,6:])

#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#encoder = LabelEncoder()
#x[:, 1] = encoder.fit_transform(x[:,1])
#onehotencoder = OneHotEncoder(categorical_features= [1])
#x= onehotencoder.fit_transform(x)
#x[:,0]

#1.KNN
from sklearn.neighbors import KNeighborsClassifier
knn_classifier = KNeighborsClassifier(n_neighbors=5, p=2, metric ='minkowski')
knn_classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred_knn = knn_classifier.predict(x_test)

# Making the Confusion Matrix and evaluating the model performance
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


knn_cm = confusion_matrix(y_test, y_pred_knn)
knn_cm
knn_accuracy = accuracy_score(y_test,y_pred_knn)
knn_accuracy
knn_error = 1-knn_accuracy
knn_accuracy
knn_error
knn_precision_macro = precision_score(y_test, y_pred_knn, average='macro')  
knn_precision_macro
knn_precision_micro = precision_score(y_test, y_pred_knn, average='micro')  
knn_precision_micro
knn_precision_weighted = precision_score(y_test, y_pred_knn, average='weighted')  
knn_recall = recall_score(y_test, y_pred_knn, average='macro')  
knn_recall


#
#2.DecisionTree
from sklearn.tree import DecisionTreeClassifier
classifier =DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(x_train,y_train)

# Predicting the Test set results
y_pred_decision = classifier.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
decision_cm = confusion_matrix(y_test, y_pred_decision)
decision_cm
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

decision_accuracy = accuracy_score(y_test,y_pred_decision)
decision_error = 1- decision_accuracy
decision_accuracy
decision_error
decision_precision_macro = precision_score(y_test, y_pred_decision, average='macro')  
decision_precision_micro = precision_score(y_test, y_pred_decision, average='micro')  
decision_precision_weighted = precision_score(y_test, y_pred_decision, average='weighted')  
decision_recall = recall_score(y_test, y_pred_decision, average='macro')  


#
#3.Random_forest
from sklearn.ensemble import RandomForestClassifier
Random_forest =RandomForestClassifier(n_estimators=10,criterion = 'entropy')
Random_forest.fit(x_train,y_train)

# Predicting the Test set results
y_pred_random=list( Random_forest.predict(x_test))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
Random_cm = confusion_matrix(y_test, y_pred_random)
Random_cm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
Random_accuracy = accuracy_score(y_test,y_pred_random)
Random_error = 1- Random_accuracy
Random_accuracy
Random_error
Random_precision_macro = precision_score(y_test, y_pred_random, average='macro')  
Random_precision_micro = precision_score(y_test, y_pred_random, average='micro')  
Random_precision_weighted = precision_score(y_test, y_pred_random, average='weighted')  
Random_recall = recall_score(y_test, y_pred_random, average='macro')  


#
# Making the Confusion Matrix
#4.Logistic_Regression
from sklearn.linear_model import LogisticRegression
Regressor =LogisticRegression(random_state=0)
Regressor.fit(x_train,y_train)

# Predicting the Test set results
y_pred_logistic =list( Regressor.predict(x_test))

from sklearn.metrics import confusion_matrix
Logistic_cm = confusion_matrix(y_test, y_pred_logistic)
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
logistic_accuracy = accuracy_score(y_test,y_pred_logistic)
logistic_error = 1- logistic_accuracy
logistic_accuracy
logistic_precision_macro = precision_score(y_test, y_pred_logistic, average='macro')  
logistic_precision_micro = precision_score(y_test, y_pred_logistic, average='micro')  
logistic_precision_weighted = precision_score(y_test, y_pred_logistic, average='weighted')  
logistic_recall = recall_score(y_test, y_pred_logistic, average='macro')  

#5 SVC
from sklearn.svm import SVC
svc =SVC(kernel = 'linear')
svc.fit(x_train,y_train)

# Predicting the Test set results
y_pred_svc = svc.predict(x_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
svc_cm = confusion_matrix(y_test, y_pred_svc)
svc_cm
from sklearn.metrics import accuracy_score
svc_accuracy = accuracy_score(y_test,y_pred_svc)
svc_error = 1- svc_accuracy
svc_error
svc_accuracy
#
#svc_cm
#Logistic_cm
#knn_cm
#Random_cm
#decision_cm
#
#logistic_accuracy
#Random_accuracy
#knn_accuracy
#decision_accuracy
#svc_accuracy


## Visualising the Training set results
data

data_v=["Education","WorkLifeBalance","PerformanceRating","DailyRate","JobSatisfaction"]
datav_v=best_employee[best_employee_visualization]
best_employee_visualization.hist()


best_employee

best_employee_visualization=["Education","WorkLifeBalance","PerformanceRating","DailyRate","JobSatisfaction"]
best_employee_visualization=best_employee[best_employee_visualization]
best_employee_visualization.hist()
#
#from pandas.plotting import scatter_matrix
#scatter_matrix(best_employee_visualization)
#best_employee.plot(x="joblevel", y=["WorkLifeBalance"], kind="bar")

female_employee

female_employee_visualization=["Education","WorkLifeBalance","PerformanceRating","DailyRate","JobSatisfaction"]
female_employee_visualization=female_employee[female_employee_visualization]
female_employee_visualization.hist()

male_employee

male_employee_visualization=["Education","WorkLifeBalance","PerformanceRating","DailyRate","JobSatisfaction"]
male_employee_visualization=male_employee[male_employee_visualization]
male_employee_visualization.hist()

#visualization
x= best_employee["EmployeeNumber"].head()
print (len(x))
x1= best_employee["Education"].head()
x3= best_employee["JobSatisfaction"].head()
x4= best_employee["PerformanceRating"].head()


ax = plt.subplot(111)
w = 0.29
ax.bar(x-w, x1,width=w,color='b',align='center',label = 'EDUCATION')
ax.bar(x, x3,width=w,color='g',align='center',label = "JobSatisfaction")
ax.bar(x+w, x4,width=w,color='r',align='center',label='PERFORMANCERATING')
ax.autoscale(tight=True)
ax.set_xlabel("employeeid")
plt.legend()
plt.show()


x= female_employee["EmployeeNumber"].head()
print (len(x))
x1= female_employee["Education"].head()
x3= female_employee["JobSatisfaction"].head()
x4= female_employee["PerformanceRating"].head()


ax = plt.subplot(111)
w = 0.5
ax.bar(x-w, x1,width=w,color='b',align='center',label = 'EDUCATION')
ax.bar(x, x3,width=w,color='g',align='center',label = "JobSatisfaction")
ax.bar(x+w, x4,width=w,color='r',align='center',label='PERFORMANCERATING')
ax.autoscale(tight=True)
plt.legend()

plt.show()


x= male_employee["EmployeeNumber"].head()
print (len(x))
x1= male_employee["Education"].head()
x3= male_employee["JobSatisfaction"].head()
x4= male_employee["PerformanceRating"].head()

ax = plt.subplot(111)
w = 0.3
ax.bar(x-w, x1,width=w,color='b',align='center',label = 'EDUCATION')
ax.bar(x, x3,width=w,color='g',align='center',label = "JobSatisfaction")
ax.bar(x+w, x4,width=w,color='r',align='center',label='PERFORMANCERATING')
ax.autoscale(tight=True)
plt.legend()
plt.show()


x= best_employee["JobLevel"].head(20)

x1= best_employee["MonthlyIncome"].head(20)

plt.scatter(x,x1)


x= best_employee["JobLevel"].head(20)

x1= best_employee["MonthlyIncome"].head(20)
 
plt.xlabel("joblevel")
plt.ylabel("monthlysalary")

plt.scatter(x,x1)

x= female_employee["JobLevel"].head(20)

x1= female_employee["MonthlyIncome"].head(20)
plt.xlabel("joblevel")
plt.ylabel("monthlysalary")

plt.scatter(x,x1)


x= male_employee["JobLevel"].head(20)

x1= male_employee["MonthlyIncome"].head(20)

plt.xlabel("joblevel")
plt.ylabel("monthlysalary")

plt.scatter(x,x1)













#from matplotlib.colors import ListedColormap
#X_set, y_set = x_train, y_train
#X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
#    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
#                c = ListedColormap(('red', 'green'))(i), label = j)
#plt.title('Decision_Tree_Classifier(Training set)')
#plt.xlabel('Age')
#plt.ylabel('Estimated Salary')
#plt.legend()
#plt.show()


#Departmentwise
#best_employee_Sales= best_employee[best_employee["Department"] == "Sales"]
#best_employee_Research_Development= best_employee[best_employee["Department"] == "Research & Development"]
#best_employee_Research_Development= best_employee[best_employee["Department"] == "Research & Development"]
#best_employee_Human_Resources= best_employee[best_employee["Department"] == "Human Resources"]
#
#
#
#



#imputer = Imputer(missing_values="NaN",strategy= "mean",axis = 0)
#columns_to_impute = 
#df[columns_to_impute] = Imputer().fit_transform(df[columns_to_impute])

#imputer.fit(data["Age"])
#data.isnull().sum()



