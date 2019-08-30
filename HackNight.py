import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, model_selection, neighbors

train_data = pd.read_csv('trainms.csv',header=0, index_col = 's.no',parse_dates=True)
labels_to_be_dropped = ['s.no','Timestamp','Gender','Country','state','comments']
features = [i for i in train_data.keys() if i not in labels_to_be_dropped]
train_data = train_data[features]

for i in features:
    print(i,train_data[i].unique())

conversion = {'nan':0,'Yes':1,'No':0,"Don't know":0.5,'Not sure':0.5,'Maybe':0.5,'Some of them':0.5,\
              'Often':0.75,'Rarely':0.25,'Never':0,'Sometimes':0.5,'Very easy':1,'Somewhat easy':0.75,'Somewhat difficult':0.25,'Very difficult':0}

non_numerical = ['self_employed','family_history','treatment','work_interfere','remote_work','tech_company','benefits','care_options','wellness_program','seek_help',\
                 'anonymity','leave','mental_health_consequence','phys_health_consequence','coworkers','supervisor','mental_health_interview','phys_health_interview',\
                 'mental_vs_physical','obs_consequence']

for i in features:
    if i in non_numerical:
        lst = [i for i in train_data[i]]
        for j in range(len(lst)):
            lst[j] = conversion[str(lst[j])]
        train_data[i] = lst

lst = [i for i in train_data['no_employees']]
for i in range(len(lst)):
    if '-' in lst[i]:
        num1,num2 = lst[i].split('-')
        lst[i] = num2
    else:
        lst[i] = 1500

train_data['no_employees'] = lst

X = np.array(train_data.drop(['treatment'],1))
y = np.array(train_data['treatment'])
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
print(X_train, X_test, y_train, y_test)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)
#train_data.to_csv('trial.csv')
