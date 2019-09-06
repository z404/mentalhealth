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
import tkinter as tk
from tkinter import *
import time

warnings.filterwarnings("ignore")

root = tk.Tk()

final_dict = {'Timestamp':'','Age':'','Gender':'','Country':'','state':'','self_employed':'','family_history':'','work_interfere':'','no_employees':'','remote_work':'',\
              'tech_company':'','benefits':'','care_options':'','wellness_program':'','seek_help':'','anonymity':'','leave':'','mental_health_consequence':'','phys_health_consequence':'',\
              'coworkers':'','supervisor':'','mental_health_interview':'','phys_health_interview':'','mental_vs_physical':'','obs_consequence':'','comments':''}

def clear_window(window):
    for ele in window.winfo_children():
        ele.destroy()

def show_and_back(string):
    clear_window(root)
    label = tk.Label(root,text = string, font= 'Ariel 28 bold')
    label.pack()
    def back():
        clear_window(root)
        start()
    button = tk.Button(root, text = 'Back', command = back, font = 'Ariel 28 bold')
    button.pack()

def finish():
    clear_window(root)
    label = tk.Label(root,text = 'Processing, Please wait.....', font= 'Ariel 28 bold')
    label.pack()
    ######
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
                         'mental_vs_physical','obs_consequence','supervisor','care_options']
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
            print('gender',list_in_focus)
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
    col = [i for i in raw_train_data.columns]    
    dropped_columns = ['s.no','Timestamp','state','comments',"Country",'anonymity']
    dropped_train_data = drop_labels(raw_train_data,dropped_columns)
    train_data = convert_to_integer(dropped_train_data)
    trained_model = train_and_predict(train_data)
    lst = []
    for i in range(len(final_dict.keys())):
        klst = [i for i in final_dict.keys()]
        if klst[i] not in dropped_columns:
            templst = [i for i in final_dict.values()]
            lst.append(templst[i])
    print(lst)

    with open('temp.csv','w') as file:
        string = ''
        for i in col:
            string = string+','+str(i)
        string = string.rstrip(',')
        file.write(string+'\n')
        string = ''
        for i in lst:
            string = string+','+str(i)
        string = string.rstrip(',')
        file.write(string)

    raw_test_data = read_csv_to_dataframe('temp.csv')
    dropped_columns = ['s.no','Timestamp','state','comments',"Country",'anonymity']
    dropped_test_data = drop_labels(raw_test_data,dropped_columns)
    test_data = convert_to_integer(dropped_test_data)
    result = predict_with_model(trained_model,test_data)

    print(result)
    #not working, maybe convert to temp csv?
##    ndf = ndf.append([lst])
##    print('hello',ndf)
##    dropped_columns = ['s.no','Timestamp','state','comments',"Country",'anonymity']
##    dropped_test_data = drop_labels(ndf,dropped_columns)
##    test_data = convert_to_integer(dropped_test_data)
##    result = predict_with_model(trained_model,test_data)
##    print(result)
    
##    conversion = {'nan':-1,'Yes':1,'No':0,"I don't know":1.5,'Not sure':2.5,'Maybe':1.5,'Some of them':-0.5,\
##                      'Often':0.75,'Rarely':0.25,'Never':0,'Sometimes':0.5,'Very easy':1,'Somewhat easy':-0.75,'Somewhat difficult':0.5,'Very difficult':-1}
##    for i in range(len(lst)):
##        try:
##            lst[i] = conversion[lst[i]]
##        except:
##            pass
##    if "cis" in lst[1].lower():
##        if "f" in lst[1].lower():
##            lst[1]=1
##        else:
##            lst[1]=0
##    elif "f" in lst[1].lower():
##        lst[1]=1
##    elif "m" in lst[1].lower():
##        lst[1]=0
##    else:
##        lst[1]=0.5
##    print(lst)
##    for i in range(len(lst)):
##        if '-' in str(lst[i]) or 'More' in str(lst[i]):
##            if '-' in lst:
##                num1,num2 = lst[i].split('-')
##                lst[i] = num2
##            else:
##                lst[i] = 1500
##    lst = [lst[i] for i in range(-1,-len(lst),-1) if i not in [-2,-6,-9,-7]]#,-8]]#,-14]]
##    newlst = [lst]
##    print(newlst)
##    result = trained_model.predict(newlst)
##    print(result)
##    if '0' in str(result):
##        result = 'does not'
##    else:
##        result = 'does'
##    string = 'The employee '+result+' require treatment'
    show_and_back(string)   
    

def fill_form4():
    global final_dict
    clear_window(root)
    Title_Label = tk.Label(root, text = 'Mental Health Survey', font = 'Ariel 40 bold', pady = 12)
    Title_Label.pack()

    Frame = tk.Frame(root, pady=20)
    Frame.pack()

    sup_Label = tk.Label(Frame, text='Would you be willing to discuss a mental\n health issue with your supervisor?', font = 'Ariel 25 bold')
    sup_Label.grid(row=0,column=0, padx = 10, pady = 15)
    sup_c = StringVar()
    sup_c.set('Select')
    list_of_sup = ['Yes','No','Some of them']
    droplist8 = OptionMenu(Frame,sup_c,*list_of_sup)
    droplist8.grid(row=0,column=1)

    mhi_Label = tk.Label(Frame, text='Would you bring up a mental health issue \nwith a potential employer in an interview?', font = 'Ariel 25 bold')
    mhi_Label.grid(row=1,column=0, padx = 10, pady = 15)
    mhi_c = StringVar()
    mhi_c.set('Select')
    list_of_mhi = ['Yes','No','Maybe']
    droplist9 = OptionMenu(Frame,mhi_c,*list_of_mhi)
    droplist9.grid(row=1,column=1)

    phi_Label = tk.Label(Frame, text='Would you bring up a physical health issue \nwith a potential employer in an interview?', font = 'Ariel 25 bold')
    phi_Label.grid(row=2,column=0, padx = 10, pady = 15)
    phi_c = StringVar()
    phi_c.set('Select')
    list_of_phi = ['Yes','No','Maybe']
    droplist10 = OptionMenu(Frame,phi_c,*list_of_phi)
    droplist10.grid(row=2,column=1)

    mvp_Label = tk.Label(Frame, text='Do you feel that your employer takes mental \nhealth as seriously as physical health?', font = 'Ariel 25 bold')
    mvp_Label.grid(row=3,column=0, padx = 10, pady = 15)
    mvp_c = StringVar()
    mvp_c.set('Select')
    list_of_mvp = ['Yes','No',"I don't know"]
    droplist11 = OptionMenu(Frame,mvp_c,*list_of_mvp)
    droplist11.grid(row=3,column=1)

    obc_Label = tk.Label(Frame, text='Have you heard of or observed negative consequences \nfor coworkers with mental health conditions in your workplace?', font = 'Ariel 25 bold')
    obc_Label.grid(row=4,column=0, padx = 10, pady = 15)
    obc_c = StringVar()
    obc_c.set('Select')
    list_of_obc = ['Yes','No',"I don't know"]
    droplist12 = OptionMenu(Frame,obc_c,*list_of_obc) 
    droplist12.grid(row=4,column=1)

    comment = tk.Entry(root, font = 'Ariel 15 bold')
    comment.insert('end','Enter some comments')
    comment.pack()
    
    def submit_button():
        final_dict['supervisor'] = sup_c.get()
        final_dict['mental_health_interview'] = mhi_c.get()
        final_dict['phys_health_interview'] = phi_c.get()
        final_dict['mental_vs_physical'] = mvp_c.get()
        final_dict['obs_consequence'] = obc_c.get()
        final_dict['comments'] = comment.get()
        print(final_dict)
        finish()

    finish_button = tk.Button(root, command = submit_button, text = 'Next',font = 'Ariel 15 bold')
    finish_button.pack()
    
def fill_form3():
    global final_dict
    clear_window(root)
    Title_Label = tk.Label(root, text = 'Mental Health Survey', font = 'Ariel 40 bold', pady = 12)
    Title_Label.pack()

    Frame = tk.Frame(root, pady=8)
    Frame.pack()

    seek_Label = tk.Label(Frame, text='Does your employer provide resources to \nlearn more about mental health issues \nand how to seek help?', font = 'Ariel 25 bold')
    seek_Label.grid(row=0,column=0, padx = 10, pady = 5)
    seek_c = StringVar()
    seek_c.set('Select')
    list_of_seek = ['Yes','No',"I don't know"]
    droplist5 = OptionMenu(Frame,seek_c,*list_of_seek)
    droplist5.grid(row=0,column=1)

    anon_Label = tk.Label(Frame, text='Is your anonymity protected if you choose to\n take advantage of mental health or \nsubstance abuse treatment resources?', font = 'Ariel 25 bold')
    anon_Label.grid(row=1,column=0, padx = 10, pady = 5)
    anon_c = StringVar()
    anon_c.set('Select')
    list_of_anon = ['Yes','No',"I don't know"]
    droplist5 = OptionMenu(Frame,anon_c,*list_of_anon)
    droplist5.grid(row=1,column=1)

    label_Label = tk.Label(Frame, text='How easy is it for you to take medical \nleave for a mental health condition?', font = 'Ariel 25 bold')
    label_Label.grid(row=2,column=0, padx = 10, pady = 5)
    label_c = StringVar()
    label_c.set('Select')
    list_of_label = ['Yes','No',"I don't know"]
    droplist6 = OptionMenu(Frame,label_c,*list_of_label)
    droplist6.grid(row=2,column=1)

    mhc_Label = tk.Label(Frame, text='Do you think that discussing a mental health \nissue with your employer would have negative consequences?', font = 'Ariel 25 bold')
    mhc_Label.grid(row=3,column=0, padx = 10, pady = 5)
    mhc_c = StringVar()
    mhc_c.set('Select')
    list_of_mhc = ['Yes','No','Maybe']
    droplist7 = OptionMenu(Frame,mhc_c,*list_of_mhc)
    droplist7.grid(row=3,column=1)

    phc_Label = tk.Label(Frame, text='Do you think that discussing a physical \nissue with your employer would have negative consequences?', font = 'Ariel 25 bold')
    phc_Label.grid(row=4,column=0, padx = 10, pady = 5)
    phc_c = StringVar()
    phc_c.set('Select')
    list_of_phc = ['Yes','No','Maybe']
    droplist8 = OptionMenu(Frame,phc_c,*list_of_phc)
    droplist8.grid(row=4,column=1)

    cow_Label = tk.Label(Frame, text='Would you be willing to discuss a mental\n health issue with your coworkers?', font = 'Ariel 25 bold')
    cow_Label.grid(row=5,column=0, padx = 10, pady = 5)
    cow_c = StringVar()
    cow_c.set('Select')
    list_of_cow = ['Yes','No','Some of them']
    droplist7 = OptionMenu(Frame,cow_c,*list_of_cow)
    droplist7.grid(row=5,column=1)

    def submit_button():
        final_dict['seek_help'] = seek_c.get()
        final_dict['anonymity'] = anon_c.get()
        final_dict['leave'] = label_c.get()
        final_dict['mental_health_consequence'] = mhc_c.get()
        final_dict['phys_health_consequence'] = phc_c.get()
        final_dict['coworkers'] = cow_c.get()
        fill_form4()

    finish_button = tk.Button(root, command = submit_button, text = 'Next',font = 'Ariel 15 bold')
    finish_button.pack()
    
def fill_form2():
    global final_dict
    clear_window(root)
    Title_Label = tk.Label(root, text = 'Mental Health Survey', font = 'Ariel 40 bold', pady = 12)
    Title_Label.pack()

    Frame = tk.Frame(root, pady=12)
    Frame.pack()

    noempl_Label = tk.Label(Frame, text='How many employees does your \ncompany or organization have?', font = 'Ariel 25 bold')
    noempl_Label.grid(row=0,column=0, padx = 10, pady = 15)
    noempl_c = StringVar()
    noempl_c.set('Select')
    list_of_noempl = ['1-5','6-25','More than 1000','26-100','100-500','500-1000']
    droplist2 = OptionMenu(Frame,noempl_c,*list_of_noempl)
    droplist2.grid(row=0,column=1)

    Frame1 = tk.Frame(Frame, pady=12)
    Frame1.grid(row=1,column=1)
    remote_var = IntVar()
    remote_Label = tk.Label(Frame, text='Do you work remotely at least 50% of the time?', font = 'Ariel 25 bold')
    remote_Label.grid(row=1,column=0, padx = 10, pady = 15)
    remote_r_yes = tk.Radiobutton(Frame1, text='Yes',font = 'Ariel 25 bold', variable = remote_var,value=1)
    remote_r_no = tk.Radiobutton(Frame1, text='No', variable = remote_var, value=0,font = 'Ariel 25 bold')
    remote_r_yes.grid(column=0, row=0)
    remote_r_no.grid(column=1, row=0)

    Frame1 = tk.Frame(Frame, pady=12)
    Frame1.grid(row=2,column=1)
    tech_var = IntVar()
    tech_Label = tk.Label(Frame, text='Is your employer primarily \na tech company/organization?', font = 'Ariel 25 bold')
    tech_Label.grid(row=2,column=0, padx = 10, pady = 15)
    tech_r_yes = tk.Radiobutton(Frame1, text='Yes',font = 'Ariel 25 bold', variable = tech_var,value=1)
    tech_r_no = tk.Radiobutton(Frame1, text='No', variable = tech_var, value=0,font = 'Ariel 25 bold')
    tech_r_yes.grid(column=0, row=0)
    tech_r_no.grid(column=1, row=0)

    benefit_Label = tk.Label(Frame, text='Does your employer provide \nmental health benefits?', font = 'Ariel 25 bold')
    benefit_Label.grid(row=3,column=0, padx = 10, pady = 12)
    benefit_c = StringVar()
    benefit_c.set('Select')
    list_of_benefit = ['Yes','No',"I don't know"]
    droplist3 = OptionMenu(Frame,benefit_c,*list_of_benefit)
    droplist3.grid(row=3,column=1)

    coptions_Label = tk.Label(Frame, text='Do you know the options for mental\n health care your employer provides?', font = 'Ariel 25 bold')
    coptions_Label.grid(row=4,column=0, padx = 10, pady = 12)
    coptions_c = StringVar()
    coptions_c.set('Select')
    list_of_coptions = ['Yes','No',"I don't know"]
    droplist4 = OptionMenu(Frame,coptions_c,*list_of_coptions)
    droplist4.grid(row=4,column=1)

    wellness_Label = tk.Label(Frame, text='Has your employer ever discussed mental\n health as part of an employee wellness program?', font = 'Ariel 25 bold')
    wellness_Label.grid(row=5,column=0, padx = 10, pady = 12
                        )
    wellness_c = StringVar()
    wellness_c.set('Select')
    list_of_wellness = ['Yes','No',"I don't know"]
    droplist3 = OptionMenu(Frame,wellness_c,*list_of_wellness)
    droplist3.grid(row=5,column=1)

    def submit_button():
        final_dict['no_employees'] = noempl_c.get()
        final_dict['remote_work'] = remote_var.get()
        final_dict['tech_company'] = tech_var.get()
        final_dict['benefits'] = benefit_c.get()
        final_dict['care_options'] = coptions_c.get()
        final_dict['wellness_program'] = wellness_c.get()
        print(final_dict)
        fill_form3()

    finish_button = tk.Button(root, command = submit_button, text = 'Next',font = 'Ariel 15 bold')
    finish_button.pack()

def fill_form1():
    global final_dict
    clear_window(root)
    Title_Label = tk.Label(root, text = 'Mental Health Survey', font = 'Ariel 40 bold', pady = 12)
    Title_Label.pack()

    Frame = tk.Frame(root, pady=20)
    Frame.pack()
    
    Name_Label = tk.Label(Frame, text='Age:', font = 'Ariel 25 bold')#, yscrollcommand = s.set)
    Name_Label.grid(row=0,column=0)
    Name_Entry = tk.Entry(Frame, font = 'Ariel 25')
    Name_Entry.grid(row=0, column=1, padx=10, pady = 15)
    
    Gender_Label = tk.Label(Frame, text='Gender:', font = 'Ariel 25 bold')
    Gender_Label.grid(row=1,column=0, padx = 10, pady = 15)
    Gender_Entry = tk.Entry(Frame, font = 'Ariel 25')
    Gender_Entry.grid(row=1,column=1)

    Country_Label = tk.Label(Frame, text='Country:', font = 'Ariel 25 bold')
    Country_Label.grid(row=2,column=0, padx = 10, pady = 15)
    Country_Entry = tk.Entry(Frame, font = 'Ariel 25')
    Country_Entry.grid(row=2,column=1)

    State_Label = tk.Label(Frame, text='State:', font = 'Ariel 25 bold')
    State_Label.grid(row=3,column=0, padx = 10, pady = 15)
    State_Entry = tk.Entry(Frame, font = 'Ariel 25')
    State_Entry.grid(row=3,column=1)

    Frame1 = tk.Frame(Frame, pady=20)
    Frame1.grid(row=4,column=1)
    Self_Employed_var = IntVar()
    Self_Employed_Label = tk.Label(Frame, text='Are you self-employed?', font = 'Ariel 25 bold')
    Self_Employed_Label.grid(row=4,column=0, padx = 10, pady = 15)
    Self_Employed_r_yes = tk.Radiobutton(Frame1, text='Yes',font = 'Ariel 25 bold', variable = Self_Employed_var,value=1)
    Self_Employed_r_no = tk.Radiobutton(Frame1, text='No', variable = Self_Employed_var, value=0,font = 'Ariel 25 bold')
    Self_Employed_r_yes.grid(column=0, row=0)
    Self_Employed_r_no.grid(column=1, row=0)

    Frame1 = tk.Frame(Frame, pady=20)
    Frame1.grid(row=5,column=1)
    Family_Hist_var = IntVar()
    Family_Hist_var = 0
    Family_Hist_Label = tk.Label(Frame, text='Do you have a family history \nof mental illness?', font = 'Ariel 25 bold')
    Family_Hist_Label.grid(row=5,column=0, padx = 10, pady = 15)
    Family_Hist_Employed_r_yes = tk.Radiobutton(Frame1, text='Yes',font = 'Ariel 25 bold', variable = Family_Hist_var,value=1)
    Family_Hist_Employed_r_no = tk.Radiobutton(Frame1, text='No', variable = Family_Hist_var, value=0,font = 'Ariel 25 bold')
    Family_Hist_Employed_r_yes.grid(column=0, row=0)
    Family_Hist_Employed_r_no.grid(column=1, row=0)
    
    Work_inter_Label = tk.Label(Frame, text='If you have a mental health condition, do you \nfeel that it interferes with your work?', font = 'Ariel 25 bold')
    Work_inter_Label.grid(row=6,column=0, padx = 10, pady = 15)
    Work_inter_c = StringVar()
    Work_inter_c.set('Select')
    list_of_work_inter = ['Often','Rarely','Never','Sometimes']
    droplist1 = OptionMenu(Frame,Work_inter_c,*list_of_work_inter)
    droplist1.grid(row=6,column=1)

    def submit_button():
        final_dict['Gender'] = Gender_Entry.get()
        final_dict['Age'] = Name_Entry.get()
        final_dict['Country'] = Country_Entry.get()
        final_dict['state'] = State_Entry.get()
        final_dict['self_employed'] = Self_Employed_var.get()
        final_dict['family_history'] = Family_Hist_var
        final_dict['work_interfere'] = Work_inter_c.get()
        print(final_dict)
        fill_form2()

    finish_button = tk.Button(root, command = submit_button, text = 'Next',font = 'Ariel 15 bold')
    finish_button.pack()

def to_csv():
    clear_window(root)
    Title_Label = tk.Label(root, text = 'Mental Health Predictor', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    Frame = tk.Frame(root, pady=20)
    Frame.pack()
    Company_Label = tk.Label(Frame, text='File Name:', font = 'Ariel 25 bold')
    Company_Label.grid(row=0,column=0)
    Company_Label = tk.Label(Frame, text='The file needs to be in the same directory as the application', font = 'Ariel 15 bold')
    Company_Label.grid(row=1,column=0)
    Company_Entry = tk.Entry(Frame, font = 'Ariel 25')
    Company_Entry.grid(row=0, column=1, padx=10, pady = 15)
    def submit_button():
        filename = Company_Entry.get()
        Company_Label.config(text = 'File "Predicted.csv" has been created in the directory')
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
                             'mental_vs_physical','obs_consequence','supervisor','care_options']
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

        raw_train_data = read_csv_to_dataframe("trainms.csv")
        dropped_columns = ['s.no','Timestamp','state','comments',"Country",'anonymity']
        dropped_train_data = drop_labels(raw_train_data,dropped_columns)
        train_data = convert_to_integer(dropped_train_data)

        trained_model = train_and_predict(train_data)

        raw_test_data = read_csv_to_dataframe(filename)
        dropped_test_data = drop_labels(raw_test_data,dropped_columns)
        test_data= convert_to_integer(dropped_test_data)

##        solution_data = read_csv_to_dataframe('samplems.csv')
##        conversion = {'nan':-1,'Yes':1,'No':0,"Don't know":0.5,'Not sure':0.5,'Maybe':0.5,'Some of them':0.5,\
##                          'Often':0.75,'Rarely':0.25,'Never':0,'Sometimes':0.5,'Very easy':1,'Somewhat easy':0.75,'Somewhat difficult':1,'Very difficult':-1}
##        lst = [i for i in solution_data['treatment']]
##        for j in range(len(lst)):
##            lst[j] = conversion[str(lst[j])]
##        solution_data['treatment'] = lst

        result = predict_with_model(trained_model,test_data)
        #score_model_offline(trained_model,test_data,solution_data)
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


############################################################################################################################
    finish_button = tk.Button(root, command = submit_button, text = 'Finish',font = 'Ariel 15 bold')
    finish_button.pack()

def start():
    global root
    root.attributes("-fullscreen",True)
    def start_button():
        fill_form1()
    def upload_csv():
        to_csv()
    def read_button():
        pass
    def exit_button():
        exit(0)
    Title_Label = tk.Label(root, text = 'Mental Health Predictor', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    Catchphrase_Label = tk.Label(root, text = '', font='Ariel 15', pady = 20)
    Catchphrase_Label.pack()
    Start_Button = tk.Button(root, text = 'Start', font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = start_button)
    Start_Button.pack()
    Csv_Button = tk.Button(root, text = 'Upload Csv', font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = upload_csv)
    Csv_Button.pack()
    Update_Button = tk.Button(root, text = 'Read recorded results',font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = read_button)
    Update_Button.pack()
    Exit_Button = tk.Button(root, text = 'Exit', font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = exit_button)
    Exit_Button.pack()
start()
#print(final_dict)
root.mainloop()
