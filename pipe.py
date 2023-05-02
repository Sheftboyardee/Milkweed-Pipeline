from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler as SS
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans as KM
from sklearn.tree import DecisionTreeClassifier as DTC

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

from tkinter import scrolledtext
from tkinter import filedialog
import tkinter as tk

import pandas as pd
import numpy as np


class Pipeline:
    ''' Requires a dataframe containing only numerical columns except for two, representing ID and species.
        'ID' and 'species' must be labeled in the dataframe, or changed in the code
    '''
    def __init__(self, plot_size = (4,4), species = 'species', ID = 'ID'):
        self.plot_size = plot_size
        self.species = species
        self.ID = ID
        self.filename = None
        
    def GetData(self):
        ''' Load the data and perform z-score scaling on it
            Requires that the last two columns are species and ID
        '''
        data = pd.read_csv(self.filename)
        y = data[self.species]    # Needs to be in last two columns
        ID_names = data[self.ID]  # Needs to be in last two columns
        y_names = y.unique()
        y_names.sort()
        X = data.drop(columns=[self.species, self.ID], axis = 1)
        
        # Reason why ID and species need to be last two columns 
        X = pd.DataFrame(SS().fit_transform(data.iloc[:, 0:-2]), columns = data.columns[0:-2]) 
        X['ID'] = ID_names
        return X, y, y_names
    
    def DoKFold(self, weight = 'balanced', k = 10):
        X, y, y_names = self.GetData()

        # Split the data into features and target
        features = X.drop(['ID'], axis = 1)

        # Create a KFold object with 10 splits
        kf = KFold(n_splits = 10)

        # Initialize an empty list to store incorrect sample names
        incorrect_class = []

        # Iterate through each split
        for train_index, test_index in kf.split(features):
            # Split the data into training and testing sets
            X_train, X_test = features.iloc[train_index], features.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            # Train a decision tree model on the training data
            if weight == 'balanced':
                model = DTC(random_state = 42, max_depth = 2, class_weight = 'balanced')
            elif weight is None:
                model = DTC(random_state = 42, max_depth = 2)
            else:
                raise ValueError('Unknown weight method:', weight)

            model.fit(X_train, y_train)

            # Predict the species on the testing data
            y_pred = model.predict(X_test)

            # Compare the predicted and actual species and store the incorrect sample names
            incorrect = X_test[y_pred != y_test].index.tolist()

            # VERY very cool method to know, appends to end of list
            incorrect_class.extend(incorrect)

        return incorrect_class
    
    def DoClustering(self):
        X, y, y_names = self.GetData()

        # Split the dataset into features and target
        X2 = X.drop(['ID'], axis=1)

        # Perform KMeans clustering with k=2
        model = KM(n_clusters = 2, random_state = 42)
        model.fit(X2)

        # Predict the cluster labels for each data point
        y_pred = model.predict(X2)

        # Add the predicted labels to the original dataset
        X2['cluster'] = y_pred
        X2['species'] = y
        # Relabel the cluster column with the associated species names
        if y[0] == 'exaltata':
            X2['cluster'] = X2['cluster'].map({0: 'syriaca', 1: 'exaltata'})
        elif y[0] == 'syriaca':
            X2['cluster'] = X2['cluster'].map({1: 'syriaca', 0: 'exaltata'})

        incorrect_clust = X2[X2['species'] != X2['cluster']]
        incorrect_clust = incorrect_clust.index.tolist()
        return incorrect_clust
    
    def Comparison(self, incorrect_samples_classification, incorrect_samples_clustering):
        
        intersect = np.intersect1d(incorrect_samples_classification, incorrect_samples_clustering)
        return (intersect)
    
    def RunPipeline(self):
        X, y, y_names = self.GetData()
        incorrect_samples_classification = self.DoKFold()
        incorrect_samples_clustering = self.DoClustering()
        intersect = self.Comparison(incorrect_samples_classification,incorrect_samples_clustering)
        return X, y, y_names, incorrect_samples_classification, incorrect_samples_clustering, intersect
    
    def RunPCA(self):
        X, y, y_names, incorrect_samples_classification, incorrect_samples_clustering, intersect = self.RunPipeline()
        pca = PCA()
        Xpca = pca.fit_transform(X.drop(columns='ID'))

        # Plot the data with the true labels, circle the ones that were always classified/clustered wrong
        fig, ax = plt.subplots(figsize = self.plot_size)
        colors = ['red', 'blue']
        for i, yi in enumerate(np.unique(y)):
            idx = y == yi
            ax.scatter(Xpca[idx,0], Xpca[idx,1], color = colors[i], label=y_names[i], alpha=0.3, ec ='k')
        
        # Highlight Incorrect Points
        ax.scatter(Xpca[intersect,0], Xpca[intersect,1], fc = 'purple', alpha = 0.2, s = 130)
        ax.scatter(Xpca[intersect,0], Xpca[intersect,1], fc = 'none', ec = 'k', s = 130)
        
        # Labels, Title, & Legend
        ax.set_xlabel('PC$_1$')
        ax.set_ylabel('PC$_2$')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc = 'upper right', bbox_to_anchor = [1, 1])
        ax.set_title('Circled Samples Classified Incorrectly')
        return fig, ax
   
    def choose_file(self):
        self.root = tk.Tk()

        self.filename = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV Files", "*.csv")])
        if self.filename:
            self.root.destroy()
            self.Tk() 
            
    def Tk(self):
        ''' Tk is the culmination of this project. It calls RunPipeline() and RunPCA(), placing plot and incorrect labels
            inside of Tk object for easy visualization.
        '''
        
        if not self.filename:
            return
        
        X, y, y_names, incorrect_samples_classification, incorrect_samples_clustering, intersect = self.RunPipeline()
        fig, ax = self.RunPCA()
        
        # Add PCA Plot to tkinter Window 
        root = tk.Tk()
        canvas = FigureCanvasTkAgg(fig, master = root)
        canvas.draw()
        canvas.get_tk_widget().pack()
        
        # Scrolled Textbox for overflow
        label = tk.scrolledtext.ScrolledText(root, height = 1, borderwidth = 0)
        Incorrect_IDs = 'Incorrect ID Labels: ' + str([i for i in X.iloc[intersect]['ID']])[1:-1]
        label.insert(1.0, Incorrect_IDs)
        label.pack()
        
        # Make the text interactive, i.e., copy label IDs
        label.configure(state = "disabled")
        label.configure(inactiveselectbackground=label.cget("selectbackground"))
        
        #self.root.destroy()

        tk.mainloop()

if __name__ == '__main__':
    mydata = Pipeline()
    mydata.choose_file()