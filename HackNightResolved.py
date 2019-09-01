import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, model_selection, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
import warnings

warnings.filterwarnings("ignore")

'''
HackNight solution!

functions requied:
1) read csv(filename):
        return dataframe
2) get labels and drop unwanted labels(dataframe, unwanted labels):
        return dataframe
2) convert dataframe to numbers(dataframe):
        return dataframe
3) train dataset using LogisticRegression(train_dataset):
        return trained model
4) predict dataset with test data(trained model,test_data):
        return prediction
5) convert prediction to output csv(prediction):
        return csvstring
6) write to csv(csvstring)
7) trial with offline result
'''


def read_csv_to_dataframe(filename):
    #reads csv and makes it a pandas dataframe
    df = pd.read_csv(filename,header=0,index_col='s.no', parse_dates=True)
    return df

def drop_labels(dataframe,unwanted_labels):
    #drops unwanted columns from the dataframe
    current_labels = dataframe.columns.tolist()
    drop_labels = []
    for i in unwanted_labels:
        if i in current_labels:
            drop_labels.append(i)
    df = dataframe.drop(drop_labels,1)
    return df 

def convert_to_integer(dataframe):
    #converts all default strings to numbers in the dataframe
    non_numerical = ['self_employed','family_history','treatment','remote_work','work_interfere','tech_company','benefits','seek_help',\
                     'leave','mental_health_consequence','phys_health_consequence','mental_health_interview','phys_health_interview',\
                     'mental_vs_physical','obs_consequence','supervisor']
    current_labels = dataframe.columns.tolist()
    string_feature_list = []
    for i in non_numerical:
        if i in current_labels:
            string_feature_list.append(i)
##    conversion = {'nan':-1,'Yes':1,'No':0,"Don't know":0.5,'Not sure':0.5,'Maybe':0.5,'Some of them':0.5,\
##                  'Often':0.3,'Rarely':0.25,'Never':0.5,'Sometimes':0.5,'Very easy':1,'Somewhat easy':0.75,'Somewhat difficult':1,'Very difficult':-1}
    conversion = {'nan':-1,'Yes':1,'No':0,"Don't know":0.5,'Not sure':0.5,'Maybe':0.5,'Some of them':0.5,\
                  'Often':0.75,'Rarely':0.25,'Never':0,'Sometimes':0.5,'Very easy':1,'Somewhat easy':0.75,'Somewhat difficult':0.25,'Very difficult':0}
    list_in_focus = []
    for i in string_feature_list:
        list_in_focus = list(dataframe[i])
        for j in range(len(list_in_focus)):
            list_in_focus[j] = conversion[str(list_in_focus[j])]
        dataframe[i] = list_in_focus
        
    if 'Gender' in current_labels:
        list_in_focus=list(dataframe["Gender"])
        for i in range(len(list_in_focus)):
            if "cis" in list_in_focus[i].lower():
                if "f" in list_in_focus[i].lower():
                    list_in_focus[i]=1
                else:
                    list_in_focus[i]=0
            elif "f" in list_in_focus[i].lower():
                list_in_focus[i]=1
            elif "m" in list_in_focus[i].lower():
                list_in_focus[i]=0
            else:
                list_in_focus[i]=0.5
        dataframe["Gender"]=list_in_focus

    if 'no_employees' in current_labels:
        list_in_focus = list(dataframe['no_employees'])
        for i in range(len(list_in_focus)):
            if '-' in list_in_focus[i]:
                num1,num2 = list_in_focus[i].split('-')
                list_in_focus[i] = num2
            else:
                list_in_focus[i] = 1500
        dataframe['no_employees'] = list_in_focus
    return dataframe

def train_and_predict(dataframe):
    #training with model
    clf = LogisticRegression(C=4, penalty='l1', verbose=5)              #79
    #clf2 = neighbors.KNeighborsClassifier()  #63
    #clf2 = RandomForestClassifier()          #77
    #clf2 = AdaBoostClassifier()              #77
    #clf = GaussianProcessClassifier()       #65
    #clf2 = DecisionTreeClassifier()          #71
    #clf2 = QuadraticDiscriminantAnalysis()   #68
    #clf2 = SVC()                             #69

    X_train = np.array(train_data.drop(['treatment'],1))
    y_train = np.array(train_data['treatment'])

    #clf = VotingClassifier(estimators=[('LR',clf1), ('AB', clf2)], voting='soft', weights=[1, 1])
    clf.fit(X_train, y_train)
    print(X_train)
    return clf

def predict_with_model(model,test_data):
    #predicting test data
    result = model.predict(test_data)
    return result

def result_to_modified_csv(result,filename):
    #write to csv
    csvstring = 's.no,treatment\n'
    result = list(result)
    for i in range(len(result)):
        if str(result[i]) == '0':
            result[i] = 'No'
        elif str(result[i]) == '1':
            result[i] = 'Yes'
        csvstring+=str(i+1)+','+str(result[i])+'\n'
    csvstring=csvstring.strip()
    with open(filename,'w') as file:
        file.write(csvstring)

def score_model_offline(model,X_test,y_test):
    #score offline
    print(model.score(X_test,y_test))

raw_train_data = read_csv_to_dataframe('trainms.csv')
dropped_columns = ['s.no','Timestamp','state','comments','anonymity','coworkers','supervisor','wellness_program','care_options',"Country"]
dropped_train_data = drop_labels(raw_train_data,dropped_columns)
train_data = convert_to_integer(dropped_train_data)
trained_model = train_and_predict(train_data)

raw_test_data = read_csv_to_dataframe('testms.csv')
dropped_test_data = drop_labels(raw_test_data,dropped_columns)
test_data= convert_to_integer(dropped_test_data)

solution_data = read_csv_to_dataframe('samplems.csv')
conversion = {'nan':-1,'Yes':1,'No':0,"Don't know":0.5,'Not sure':0.5,'Maybe':0.5,'Some of them':0.5,\
                  'Often':0.75,'Rarely':0.25,'Never':0,'Sometimes':0.5,'Very easy':1,'Somewhat easy':0.75,'Somewhat difficult':0.25,'Very difficult':0}
lst = [i for i in solution_data['treatment']]
for j in range(len(lst)):
    lst[j] = conversion[str(lst[j])]
solution_data['treatment'] = lst

result = predict_with_model(trained_model,test_data)
score_model_offline(trained_model,test_data,solution_data)
result_to_modified_csv(result,'predicted2.csv')
#train_data.to_csv('trial.csv')
'''
train_data = pd.read_csv('trainms.csv',header=0, index_col = 's.no',parse_dates=True)
labels_to_be_dropped = ['s.no','Timestamp','state','comments','anonymity','coworkers','supervisor','wellness_program','care_options',"Country"]
features = [i for i in train_data.keys() if i not in labels_to_be_dropped]
train_data = train_data[features]


labels_to_be_dropped = ['s.no','Timestamp','state','comments','anonymity','coworkers','supervisor','wellness_program','care_options',"Country"]
features = [i for i in train_data.keys() if i not in labels_to_be_dropped]
train_data = train_data[features]

#conversion = {'nan':-1,'Yes':1,'No':0,"Don't know":0.5,'Not sure':0.5,'Maybe':0.5,'Some of them':0.5,\
#              'Often':0.3,'Rarely':0.25,'Never':0.5,'Sometimes':0.5,'Very easy':1,'Somewhat easy':0.75,'Somewhat difficult':1,'Very difficult':-1}

conversion = {'nan':-1,'Yes':1,'No':0,"Don't know":0.5,'Not sure':0.5,'Maybe':0.5,'Some of them':0.5,\
              'Often':0.75,'Rarely':0.25,'Never':0,'Sometimes':0.5,'Very easy':1,'Somewhat easy':0.75,'Somewhat difficult':0.25,'Very difficult':0}

non_numerical = ['self_employed','family_history','treatment','remote_work','work_interfere','tech_company','benefits','seek_help',\
                 'leave','mental_health_consequence','phys_health_consequence','mental_health_interview','phys_health_interview',\
                 'mental_vs_physical','obs_consequence','supervisor']

for i in features:
    print(i)
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

lst=list(train_data["Gender"])
for i in range(len(lst)):
    if "cis" in lst[i].lower():
        if "f" in lst[i].lower():
            lst[i]=1
        else:
            lst[i]=0
    elif "f" in lst[i].lower():
        lst[i]=1
    elif "m" in lst[i].lower():
        lst[i]=0
    else:
        lst[i]=0.5#nigga
train_data["Gender"]=lst

country={"united states":0,"canada":0,"united kingdom":0,"bulgaria":0,"france":0,"portugal":0,"netherlandsd":0,"switzerland":0,"poland":0,"australia":0,"germany":0,\
         "russia":0,"mexico":0,"brazil":0,"slovenia":0,"costa rica":0,"austria":0,"ireland":0,"india":0,"south africa":0,"italy":0,"sweden":0,"columbia":0,"latvia":0,\
         "romania":0,"belgium":0,"new zealand":0,"zimbabwe":0,"spain":0,"finland":0,"uruguay":0,"israel":0,"bosnia and herzegovina":0,"hungar":0,"singapore":0,\
         "japan":0,"nigeria":0,"croatia":0,"norway":0,"thailand":0,"denmark":0,"bahamas":0}


##lst=list(train_data["Country"])
##for i in range(len(lst)):
##    try:
##        country[str(lst[i]).lower()]+=1
##    except:
##        pass
##for i in range(len(lst)):
##    try:lst[i]=country[str(lst[i]).lower()]
##    except:lst[i]=-1


##a=-0.42
##for i in country.keys():
##    country[i]=a
##    a+=0.02
##for i in range(len(lst)):
##    try:lst[i]=country[str(lst[i]).lower()]
##    except:lst[i]=-1


##train_data["Country"]=lst


X_train = np.array(train_data.drop(['treatment'],1))
y_train = np.array(train_data['treatment'])
#X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
#print(X_train, X_test, y_train, y_test)

X_test = pd.read_csv('testms.csv',header=0, index_col = 's.no',parse_dates=True)
labels_to_be_dropped = ['s.no','Timestamp','state','comments','anonymity','coworkers','wellness_program','supervisor','care_options',"Country"]
features = [i for i in X_test.keys() if i not in labels_to_be_dropped]
X_test = X_test[features]

y_test = pd.read_csv('samplems.csv',header=0, index_col = 's.no',parse_dates=True)
labels_to_be_dropped = ['s.no','Timestamp','state','comments','anonymity','coworkers','wellness_program','supervisor','care_options',"Country"]
y_test = y_test[[i for i in y_test.keys() if i not in labels_to_be_dropped]]

clf1 = LogisticRegression(C=5, penalty='l1', verbose=5)              #79
#clf2 = neighbors.KNeighborsClassifier()  #63
clf2 = RandomForestClassifier()          #77
#clf2 = AdaBoostClassifier()              #77
#clf = GaussianProcessClassifier()       #65
#clf2 = DecisionTreeClassifier()          #71
#clf2 = QuadraticDiscriminantAnalysis()   #68
#clf2 = SVC()                             #69

clf = VotingClassifier(estimators=[('LR',clf1), ('AB', clf2)],
                        voting='soft',
                        weights=[1, 1])
clf.fit(X_train, y_train)

for i in features:
    if i in non_numerical and i!='treatment':
        lst = [i for i in X_test[i]]
        for j in range(len(lst)):
            lst[j] = conversion[str(lst[j])]
        X_test[i] = lst
lst = [i for i in X_test['no_employees']]
for i in range(len(lst)):
    if '-' in lst[i]:
        num1,num2 = lst[i].split('-')
        lst[i] = num2
    else:
        lst[i] = 1500
X_test['no_employees'] = lst
lst=list(X_test["Gender"])
for i in range(len(lst)):
    if "cis" in lst[i].lower():
        if "f" in lst[i].lower():
            lst[i]=1
        else:
            lst[i]=0
    elif "f" in lst[i].lower():
        lst[i]=1
    elif "m" in lst[i].lower():
        lst[i]=0
    else:
        lst[i]=0.5#nigga
X_test["Gender"]=lst

##lst=list(X_test["Country"])
##
##a=-0.42
##for i in country.keys():
##    country[i]=a
##    a+=0.02
##for i in range(len(lst)):
##    try:lst[i]=country[str(lst[i]).lower()]
##    except:lst[i]=-1
##
##
##
##X_test["Country"]=lst

lst = [i for i in y_test['treatment']]
for j in range(len(lst)):
    lst[j] = conversion[str(lst[j])]
y_test['treatment'] = lst

accuracy = clf.score(X_test, y_test)
print(accuracy)

result = clf.predict(X_test)
result = list(result)
for i in range(len(result)):
    if result[i] == 1:
        result[i] = 'Yes'
    elif result[i] == 0:
        result[i] = 'No'

csvstring = 's.no,treatment\n'
for i in range(len(result)):
    csvstring+=str(i+1)+','+result[i]+'\n'
csvstring=csvstring.strip()

with open('predicted.csv','w') as file:
    file.write(csvstring)

y_test = list(y_test['treatment'])
for i in range(len(y_test)):
    if y_test[i] == 1:
        y_test[i] = 'Yes'
    elif y_test[i] == 0:
        y_test[i] = 'No'

def compare(A,B):
    #print(result,list(y_test['treatment']))
    count_total, count_correct = 0,0
    for i in range(len(A)):
        count_total += 1
        if A[i] == B[i]:
            count_correct += 1
    print(((count_correct/count_total)*100))
compare(result,y_test)

##print(csvstring)
##resultdf = pd.DataFrame()
##resultdf['s.no'] = [i for i in range(1,len(result)+1)]
##resultdf['treatment'] = result
###print(resultdf.head())
###y_te['treatment'] = result
##resultdf.to_csv('predicted.csv')
##print(resultdf.columns)
#print(y_test.keys())

##lst = [i for i in X_test]
##print(len(lst))
##print(len(clf.predict(lst)))
##print(clf.predict(lst))
#train_data.to_csv('trial.csv')
'''
