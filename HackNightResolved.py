import pandas as pd
import numpy as np
import matplotlib.pyplot as pl
from sklearn.model_selection import train_test_split
from sklearn import preprocessing, model_selection, neighbors
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis 
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
                     'mental_vs_physical','obs_consequence','supervisor','care_options','wellness_program','coworkers']
    current_labels = dataframe.columns.tolist()
    string_feature_list = []
    for i in non_numerical:
        if i in current_labels:
            string_feature_list.append(i)
    conversion = {'nan':-1,'Yes':1,'No':0,"Don't know":1.5,'Not sure':2.5,'Maybe':1.5,'Some of them':-0.5,\
                  'Often':0.75,'Rarely':0.25,'Never':0,'Sometimes':0.5,'Very easy':1,'Somewhat easy':-0.75,'Somewhat difficult':0.5,'Very difficult':-1}
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
    dataframe.to_csv('num_csv.csv')

    if 'care_options' in current_labels and 'wellness_program' in current_labels:
        co = list(dataframe['care_options'])
        wp = list(dataframe['wellness_program'])
        combined_co_wp = []
        for i in range(len(co)):
            a = co[i]
            b = wp[i]
            if i == 0 and j == 0:
                combined_co_wp.append(-7)
            elif i == 1 and j == 0:
                combined_co_wp.append(-1)
            elif i == 0 and j == 1:
                combined_co_wp.append(1)
            else:
                combined_co_wp.append(7)
        dataframe['combined_co_wp'] = combined_co_wp
        dataframe = dataframe.drop(['care_options','wellness_program'],1)

    if 'combined_co_wp' in current_labels and 'mental_health_consequence' in current_labels:
        co = list(dataframe['combined_co_wp'])
        wp = list(dataframe['mental_health_consequence'])
        combined_co_wp = []
        for i in range(len(co)):
            a = co[i]
            b = wp[i]
            if i == 0 and j == 0:
                combined_co_wp.append(-7)
            elif i == 1 and j == 0:
                combined_co_wp.append(-1)
            elif i == 0 and j == 1:
                combined_co_wp.append(1)
            else:
                combined_co_wp.append(7)
        dataframe['combined_co_wp_be'] = combined_co_wp
        dataframe = dataframe.drop(['combined_co_wp','mental_health_consequence'],1)

    if 'phys_health_interview' in current_labels and 'phys_health_consequence' in current_labels:
        co = list(dataframe['phys_health_interview'])
        wp = list(dataframe['phys_health_consequence'])
        combined_co_wp = []
        for i in range(len(co)):
            a = co[i]
            b = wp[i]
            if i == 0 and j == 0:
                combined_co_wp.append(-7)
            elif i == 1 and j == 0:
                combined_co_wp.append(-1)
            elif i == 0 and j == 1:
                combined_co_wp.append(1)
            else:
                combined_co_wp.append(7)
        dataframe['combined_co_wp_be_al'] = combined_co_wp
        dataframe = dataframe.drop(['phys_health_interview','phys_health_consequence'],1)

    if 'coworkers' in current_labels and 'supervisor' in current_labels:
        co = list(dataframe['coworkers'])
        wp = list(dataframe['supervisor'])
        combined_co_wp = []
        for i in range(len(co)):
            a = co[i]
            b = wp[i]
            if i == 0 and j == 0:
                combined_co_wp.append(-7)
            elif i == 1 and j == 0:
                combined_co_wp.append(-1)
            elif i == 0 and j == 1:
                combined_co_wp.append(1)
            else:
                combined_co_wp.append(7)
        dataframe['combined_co_wp_be_al_qw'] = combined_co_wp
        dataframe = dataframe.drop(['coworkers','supervisor'],1)

    current_labels = dataframe.columns.tolist()
    print(current_labels)
    if 'combined_co_wp_be_al_qw' in current_labels and 'remote_work' in current_labels:
        print('hi')
        co = list(dataframe['combined_co_wp_be_al_qw'])
        wp = list(dataframe['remote_work'])
        combined_co_wp = []
        for i in range(len(co)):
            a = co[i]
            b = wp[i]
            if i == 0 and j == 0:
                combined_co_wp.append(-7)
            elif i == 1 and j == 0:
                combined_co_wp.append(-1)
            elif i == 0 and j == 1:
                combined_co_wp.append(1)
            else:
                combined_co_wp.append(7)
        dataframe['combined_co_wp_be_al_q'] = combined_co_wp
        dataframe = dataframe.drop(['combined_co_wp_be_al_qw','remote_work'],1)

    if 'mental_vs_physical' in current_labels and 'obs_consequence' in current_labels:
        print('hi')
        co = list(dataframe['mental_vs_physical'])
        wp = list(dataframe['obs_consequence'])
        combined_co_wp = []
        for i in range(len(co)):
            a = co[i]
            b = wp[i]
            if i == 0 and j == 0:
                combined_co_wp.append(-7)
            elif i == 1 and j == 0:
                combined_co_wp.append(-1)
            elif i == 0 and j == 1:
                combined_co_wp.append(1)
            else:
                combined_co_wp.append(7)
        dataframe['combined_co_wp_be_a'] = combined_co_wp
        dataframe = dataframe.drop(['mental_vs_physical','obs_consequence'],1)

##    if 'combined_co_wp_be_a' in current_labels and 'Gender' in current_labels:
##        print('hi')
##        co = list(dataframe['combined_co_wp_be_a'])
##        wp = list(dataframe['Gender'])
##        combined_co_wp = []
##        for i in range(len(co)):
##            a = co[i]
##            b = wp[i]
##            if i == 0 and j == 0:
##                combined_co_wp.append(-7)
##            elif i == 1 and j == 0:
##                combined_co_wp.append(-1)
##            elif i == 0 and j == 1:
##                combined_co_wp.append(1)
##            else:
##                combined_co_wp.append(7)
##        dataframe['combined_co_wp_b'] = combined_co_wp
##        dataframe = dataframe.drop(['combined_co_wp_be_a','Gender'],1)
    
    #getting hours
##    if 'Timestamp' in current_labels:
##        list_in_focus = list(dataframe['Timestamp'])
##        for i in range(len(list_in_focus)):
##            date = list_in_focus[i].split()[0].split('-')
##            time = list_in_focus[i].split()[1].split(':')
##            list_in_focus[i] = int(date[2])*10+int(time[0])+(0.01*int(time[1]))
##        dataframe['Timestamp'] = list_in_focus
    return dataframe

def train_and_predict(dataframe):
    #training with model
    #clf2 = LogisticRegression(C=4, penalty='l1', verbose=5)              #79
    #clf = neighbors.KNeighborsClassifier(n_neighbors=6)  #63
    clf = RandomForestClassifier(n_estimators = 15000,min_samples_leaf = 80)#, max_depth=None)          #77
    #clf2 = AdaBoostClassifier()              #77
    #clf = GaussianProcessClassifier()       #65
    #clf2 = DecisionTreeClassifier()          #71
    #clf2 = QuadraticDiscriminantAnalysis()   #68
    #clf2 = SVC()                             #69

    #clf = LinearDiscriminantAnalysis()
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
dropped_columns = ['s.no','Timestamp','state','comments',"Country",'anonymity']
dropped_train_data = drop_labels(raw_train_data,dropped_columns)
train_data = convert_to_integer(dropped_train_data)

trained_model = train_and_predict(train_data)

raw_test_data = read_csv_to_dataframe('testms.csv')
dropped_test_data = drop_labels(raw_test_data,dropped_columns)
#test_data= convert_to_integer(dropped_test_data)

solution_data = read_csv_to_dataframe('samplems.csv')
conversion = {'nan':-1,'Yes':1,'No':0,"Don't know":0.5,'Not sure':0.5,'Maybe':0.5,'Some of them':0.5,\
                  'Often':0.75,'Rarely':0.25,'Never':0,'Sometimes':0.5,'Very easy':1,'Somewhat easy':0.75,'Somewhat difficult':1,'Very difficult':-1}
lst = [i for i in solution_data['treatment']]
for j in range(len(lst)):
    lst[j] = conversion[str(lst[j])]
solution_data['treatment'] = lst

result = predict_with_model(trained_model,test_data)
score_model_offline(trained_model,test_data,solution_data)
result_to_modified_csv(result,'predicted.csv')
##
##print(len(test_data.columns.tolist()))
##print(len(trained_model.feature_importances_))
##print(trained_model.feature_importances_)
##for i in range(len(trained_model.feature_importances_)):
##    print(test_data.columns.tolist()[i],end=':')
##    print(float(trained_model.feature_importances_[i])*100)
#train_data.to_csv('trial.csv')

'''
clf = RandomForestClassifier(n_estimators = 15000, min_samples_leaf = 80)#, max_depth=None)#, min_samples_split=3)          #77
#clf = ExtraTreesClassifier(n_estimators=15000, max_depth=None,min_samples_split=2, random_state=0)
#clf = LinearDiscriminantAnalysis()

X = train_data.drop(['treatment'],1)
y = train_data['treatment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

#clf = VotingClassifier(estimators=[('LR',clf1), ('AB', clf2)], voting='soft', weights=[1, 1])
clf.fit(X_train, y_train)
print(clf.score(X_test,y_test))

'''
