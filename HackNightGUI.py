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
from tkinter import messagebox

warnings.filterwarnings("ignore")

root = tk.Tk()

def clear_window(window):
    for ele in window.winfo_children():
        ele.destroy()

def fill_form():
    clear_window(root)
    Title_Label = tk.Label(root, text = 'Mental Health Survey', font = 'Ariel 40 bold', pady = 20)
    Title_Label.pack()
    Frame = tk.Frame(root, pady=20)
    Frame.pack()
    
    Name_Label = tk.Label(Frame, text='Name:', font = 'Ariel 25 bold')
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

    Self_Employed_var = IntVar()
    Self_Employed_r_



def start():
    global root
    root.attributes("-fullscreen",True)
    def start_button():
        fill_form()
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
    Update_Button = tk.Button(root, text = 'Read recorded results',font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = read_button)
    Update_Button.pack()
    Exit_Button = tk.Button(root, text = 'Exit', font = 'Ariel 20 bold', pady = 20, width = 30, background='lightgrey', command = exit_button)
    Exit_Button.pack()
start()
root.mainloop()
