import random
import pandas
import numpy as np
from imblearn.combine import SMOTEENN
from sklearn import tree
import pydotplus
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier  # Import Decision Tree Classifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
import lightgbm as lgb
from imblearn.over_sampling import SMOTE

class Balance_Classify_Class:

    def main():
        df = pandas.read_csv("F:\healthcare-dataset-stroke-data.csv", header=0)
        df.head()
        print(df.head())
        d = {'NE': 0, 'E': 1, 'E,NE': 1, 'NE.E': 0}
        df['stroke'] = df['stroke'].map(d)

        # Scaler takes arrays
        scaler = StandardScaler()

        dc_X = np.array(df["age"]).reshape(-1, 1)
        df["age"] = scaler.fit_transform(dc_X)

        nc_X = np.array(df["avg_glucose_level"]).reshape(-1, 1)
        df["avg_glucose_level"] = scaler.fit_transform(nc_X)

        ada2_X = np.array(df["bmi"]).reshape(-1, 1)
        df["bmi"] = scaler.fit_transform(ada2_X)

        X1 = df["gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", "Residence_type", "avg_glucose_level", "bmi", "smoking_status"]
        X1 = df['gender','age','hypertension','heart_disease','ever_married','work_type','Residence_type','avg_glucose_level','bmi','smoking_status']
        features = [10]
        features.clear()
        features.append('gender')
        features.append('age')
        features.append('hypertension')
        features.append('heart_disease')
        features.append('ever_married')
        features.append('work_type')
        features.append('Residence_type')
        features.append('avg_glucose_level')
        features.append('bmi')
        features.append('smoking_status')
        X1=df[features]
        y1 = df["stroke"]

        clf = XGBClassifier()
        clf = lgb.LGBMClassifier()
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=8)
        clf = XGBClassifier()
        clf = RandomForestClassifier(n_estimators=100)
        scores = cross_val_score(estimator=clf, X=X1, y=y1, cv=10, scoring='accuracy')
        predicted_label = cross_val_predict(estimator=clf, X=X1, y=y1, cv=10)
        score = round(scores.mean(), 6)

        # ENSEMBLE CODE STARTS HERE
        estimators = []
        estimators.clear()
        clf = XGBClassifier(max_depth=4, scale_pos_weight=3)
        estimators.append(('XGBoost', clf))

        # Random Forest Classifier Code
        clf2 = RandomForestClassifier(n_estimators=100)
        estimators.append(('RandomForest', clf2))

        # DecisionTree Classifier
        clf3 = lgb.LGBMClassifier()
        estimators.append(('DecisionTree', clf3))

        # ensemble = VotingClassifier(estimators)
        scores = cross_val_score(estimator=ensemble, X=X_smote, y=y_smote, cv=10, scoring='accuracy')
        predicted_label = cross_val_predict(estimator=ensemble, X=X_smote, y=y_smote, cv=10)
        score = round(scores.mean(), 6)
        # ENSEMBLE CODE ENDS HERE

        print(score)

        cm = np.array(confusion_matrix(y1, predicted_label))
        confusion = pd.DataFrame(cm, index=['Healthy', 'Stroke'], columns=['Healthy', 'Stroke'])
        CM = confusion_matrix(y1, predicted_label)
        print(confusion)

        smote = SMOTEENN(sampling_strategy='minority')
        X_smote, y_smote = smote.fit_resample(X1, y1)
        print(y_smote.value_counts())

        clf = XGBClassifier()
        clf = lgb.LGBMClassifier()
        clf = DecisionTreeClassifier(criterion="entropy", max_depth=8)
        clf = XGBClassifier()
        clf = RandomForestClassifier(n_estimators=100)

        scores = cross_val_score(estimator=clf, X=X_smote, y=y_smote, cv=10, scoring='accuracy')
        predicted_label = cross_val_predict(estimator=clf, X=X_smote, y=y_smote, cv=10)
        score = round(scores.mean(), 6)

        # ENSEMBLE CODE STARTS HERE
        estimators = []
        estimators.clear()
        clf = XGBClassifier(max_depth=4, scale_pos_weight=3)
        estimators.append(('XGBoost', clf))

        # Random Forest Classifier Code
        clf2 = RandomForestClassifier(n_estimators=100)
        estimators.append(('RandomForest', clf2))

        # DecisionTree Classifier
        clf3 = lgb.LGBMClassifier()
        estimators.append(('DecisionTree', clf3))

        ensemble = VotingClassifier(estimators)
        scores = cross_val_score(estimator=ensemble, X=X_smote, y=y_smote, cv=10, scoring='accuracy')
        predicted_label = cross_val_predict(estimator=ensemble, X=X_smote, y=y_smote, cv=10)
        score = round(scores.mean(), 6)
        # ENSEMBLE CODE ENDS HERE
        print(score)

        cm = np.array(confusion_matrix(y_smote, predicted_label))
        confusion = pd.DataFrame(cm, index=['Healthy', 'Stroke'], columns=['Healthy', 'Stroke'])
        CM = confusion_matrix(y_smote, predicted_label)
        print(confusion)

    main()


