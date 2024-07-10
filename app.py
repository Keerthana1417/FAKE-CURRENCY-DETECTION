import tkinter as tk
from tkinter import scrolledtext, filedialog
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

main = tk.Tk()
main.title("Fake currency detection using machine learning")
main.geometry("1300x800")

data = None  # Define data as a global variable
x_train = None
x_test = None
y_train = None
y_test = None
clf = None
scalar = StandardScaler()

def load_dataset():
    global data
    filename = filedialog.askopenfilename(initialdir="dataset", title="Select CSV file", filetypes=(("CSV files", "*.csv"), ("All files", "*.*")))
    if filename:
        try:
            data = pd.read_csv(filename)
            text.insert(tk.END, f"Dataset loaded successfully. Length: {len(data)}\n")
        except Exception as e:
            text.insert(tk.END, f"Error loading dataset: {e}\n")
    else:
        text.insert(tk.END, "No file selected.\n")

font = ('times', 16, 'bold')
title = tk.Label(main, text='Fake currency detection using ml')
title.config(bg='greenyellow', fg='dodger blue')
title.config(font=font)
title.config(height=3, width=120)
title.place(x=0, y=5)

font1 = ('times', 12, 'bold')
text = scrolledtext.ScrolledText(main, height=20, width=150, wrap=tk.WORD)
text.place(x=50, y=120)
text.config(font=font1)

def preprocess_data():
    global data
    if data is None:
        text.insert(tk.END, "Error: Data not loaded.\n")
        return
    try:
        data.columns = ['var', 'skew', 'curt', 'entr', 'auth']
        text.insert(tk.END, "Data loaded successfully.\n")
        text.insert(tk.END, data.head(10))
        text.insert(tk.END, "\n\n")
    except Exception as e:
        text.insert(tk.END, f"Error preprocessing data: {e}\n")

def train_test_split_data():
    global data, x_train, x_test, y_train, y_test, scalar
    if data is None:
        text.insert(tk.END, "Error: Data not loaded.\n")
        return
    try:
        x = data.loc[:, data.columns != 'auth']
        y = data.loc[:, data.columns == 'auth']
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)
        scalar.fit(x_train)
        text.insert(tk.END, "Train-test split completed.\n")
        text.insert(tk.END, f"x_train shape: {x_train.shape}\n")
        text.insert(tk.END, f"x_test shape: {x_test.shape}\n")
        text.insert(tk.END, f"y_train shape: {y_train.shape}\n")
        text.insert(tk.END, f"y_test shape: {y_test.shape}\n")
    except Exception as e:
        text.insert(tk.END, f"Error performing train-test split: {e}\n")


def logistic_regression_accuracy():
    global x_train, x_test, y_train, y_test, clf, scalar
    if x_train is None or x_test is None or y_train is None or y_test is None:
        text.insert(tk.END, "Error: Data not split.\n")
        return
    try:
        x_train_scaled = scalar.transform(x_train)
        x_test_scaled = scalar.transform(x_test)
        clf = LogisticRegression(solver='lbfgs', random_state=42, multi_class='auto')
        clf.fit(x_train_scaled, y_train.values.ravel())
        accuracy = clf.score(x_test_scaled, y_test)*100
        text.insert(tk.END, f"Logistic Regression Accuracy: {accuracy:.4f}\n")
    except Exception as e:
        text.insert(tk.END, f"Error computing logistic regression accuracy: {e}\n")

def fuel_consumption_graph():
    global data
    if data is None:
        text.insert(tk.END, "Error: Data not loaded.\n")
        return
    try:
        plt.figure(figsize=(8,6))
        plt.title('Distribution of Target', size=18)
        sns.countplot(x=data['auth'])
        target_count = data.auth.value_counts()
        plt.ylim(0,900)
        plt.show()
    except Exception as e:
        text.insert(tk.END, f"Error displaying graph: {e}\n")

def bar_graph():
    global data
    if data is None:
        text.insert(tk.END, "Error: Data not loaded.\n")
        return
    try:
        plt.figure(figsize=(8,6))
        plt.title('Distribution of Target', size=18)
        sns.countplot(x=data['auth'])
        target_count = data.auth.value_counts()
        plt.ylim(0,900)
        plt.show()
    except Exception as e:
        text.insert(tk.END, f"Error displaying graph: {e}\n")

def print_logistic_regression_results():
    global x_test, clf, scalar
    if x_test is None or clf is None:
        text.insert(tk.END, "Error: Data not split or model not trained.\n")
        return
    try:
        new_banknote = np.array([4.5, -8.1, 2.4, 1.4], ndmin=2)
        new_banknote = scalar.transform(new_banknote)
        prediction = clf.predict(new_banknote)[0]
        
        if prediction == 0:
            text.insert(tk.END, "Original currency.\n")
        else:
            text.insert(tk.END, "Fake currency.\n")
    except Exception as e:
        text.insert(tk.END, f"Error computing logistic regression results: {e}\n")


preprocessButton = tk.Button(main, text="Preprocess Data", command=preprocess_data)
preprocessButton.place(x=420, y=550)
preprocessButton.config(font=font1)

trainTestButton = tk.Button(main, text="Train Test Split", command=train_test_split_data)
trainTestButton.place(x=760, y=550)
trainTestButton.config(font=font1)

logisticAccuracyButton = tk.Button(main, text="Logistic Regression Accuracy", command=logistic_regression_accuracy)
logisticAccuracyButton.place(x=420, y=600)
logisticAccuracyButton.config(font=font1)


barGraphButton = tk.Button(main, text="Bar Graph", command=bar_graph)
barGraphButton.place(x=760, y=600)
barGraphButton.config(font=font1)

printResultsButton = tk.Button(main, text="Print Logistic Regression Results", command=print_logistic_regression_results)
printResultsButton.place(x=760, y=650)
printResultsButton.config(font=font1)

uploadButton = tk.Button(main, text="Upload Dataset", command=load_dataset)
uploadButton.place(x=50, y=550)
uploadButton.config(font=font1)


main.config(bg='LightSkyBlue')
main.mainloop()
