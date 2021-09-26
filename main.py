import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
#alt.data_transformers.enable('data_server')
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
#from datetime import datetime
from datetime import datetime as dt

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import roc_curve

import seaborn as sns
sns.set(style="whitegrid", color_codes=True)
sns.set(rc={'figure.figsize':(11,6)})

from scipy.stats import spearmanr 
import missingno as msno

from boruta import BorutaPy
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm_notebook, tqdm

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn import tree


st.title("COVID-19 in Malaysia by Ministry of Health, Malaysia")

@st.cache(allow_output_mutation=True)
def load_data(classifier_name):
   data = pd.read_csv('https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_state.csv')

   return data

def conditions(df_cn):
    if (df_cn['total_cases'] == 0):
        return 'no'
    else:
        return 'yes'

def conditions01(df_cn):
    if (df_cn['total_cases'] > df_cn['cases_recovered']):
        return 'yes'
    else:
        return 'no'

def ranking(ranks, names, order=1):
    minmax = MinMaxScaler()
    ranks = minmax.fit_transform(order*np.array([ranks]).T).T[0]
    ranks = map(lambda x: round(x,2), ranks)
    return dict(zip(names, ranks))



# selecting collebration 
choise_state_colleration = st.sidebar.selectbox("Select States Exhibit Correlation",("All State","Pahang","Johor"))
# selecting futureSelection 
choise_futureSelection = st.sidebar.selectbox("Select Future Selection",("Boruta classifier","RFE classifier"))
# selecting classifier
method_name = st.sidebar.selectbox("Select Models",("K-Nearest Neighbors","Random Forest","Linear Regression","CART Regression Tree"))


df = load_data(method_name)
df_fs = df.copy()
st.write("Cases State Datasets:")
st.write(df_fs.head(5))
st.write("Shape of dataset:", df_fs.shape)

if(method_name=="K-Nearest Neighbors"):
    st.sidebar.subheader("Prediction using KNN")
    selecting_states = st.sidebar.selectbox("State",("--Choose State--","Pahang","Kedah","Johor","Selangor"))
   
    
    datasets = df.copy()
    datasets['total_cases'] = datasets['cases_import'] + datasets['cases_new']
    datasets['active_cases'] = datasets.apply(conditions01, axis=1)
    dfs = datasets.copy()

    if(selecting_states=="Pahang"):
        kvalue = st.sidebar.number_input('K Value')
        new_cases = st.sidebar.number_input('New Cases')
        import_cases = st.sidebar.number_input('Import Cases')
        
        total = new_cases + import_cases

        dfs= dfs[dfs["state"] == selecting_states]
        dfs = dfs[['total_cases','active_cases']]
        X = dfs.drop('active_cases', axis=1)#dataset,independanent variable
        y = dfs['active_cases']#dependent variable
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        knum = int(kvalue)
        if(knum <= 0):
            knum=1
        
        knn = KNeighborsClassifier(n_neighbors=knum)
        knn.fit(X_train, y_train)
        
        vall= knn.predict([[total]])
        st.sidebar.write('Active cases : ',vall[0])

    elif(selecting_states=="Johor"):
        kvalue = st.sidebar.number_input('K Value')
        new_cases = st.sidebar.number_input('New Cases')
        import_cases = st.sidebar.number_input('Import Cases')
        
        total = new_cases + import_cases

        dfs= dfs[dfs["state"] == selecting_states]
        dfs = dfs[['total_cases','active_cases']]
        X = dfs.drop('active_cases', axis=1)#dataset,independanent variable
        y = dfs['active_cases']#dependent variable
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        knum = int(kvalue)
        if(knum <= 0):
            knum=1
        
        knn = KNeighborsClassifier(n_neighbors=knum)
        knn.fit(X_train, y_train)
        
        vall= knn.predict([[total]])
        st.sidebar.write('Active cases : ',vall[0])

    elif(selecting_states=="Kedah"):
        kvalue = st.sidebar.number_input('K Value')
        new_cases = st.sidebar.number_input('New Cases')
        import_cases = st.sidebar.number_input('Import Cases')
        
        total = new_cases + import_cases

        dfs= dfs[dfs["state"] == selecting_states]
        dfs = dfs[['total_cases','active_cases']]
        X = dfs.drop('active_cases', axis=1)
        y = dfs['active_cases']
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        knum = int(kvalue)
        if(knum <= 0):
            knum=1
        
        knn = KNeighborsClassifier(n_neighbors=knum)
        knn.fit(X_train, y_train)
        
        vall= knn.predict([[total]])
        st.sidebar.write('Active cases : ',vall[0])

    elif(selecting_states=="Selangor"):
        kvalue = st.sidebar.number_input('K Value')
        new_cases = st.sidebar.number_input('New Cases')
        import_cases = st.sidebar.number_input('Import Cases')
        
        total = new_cases + import_cases

        dfs= dfs[dfs["state"] == selecting_states]
        dfs = dfs[['total_cases','active_cases']]
        X = dfs.drop('active_cases', axis=1)#dataset,independanent variable
        y = dfs['active_cases']#dependent variable
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
        knum = int(kvalue)
        if(knum <= 0):
            knum=1
        
        knn = KNeighborsClassifier(n_neighbors=knum)
        knn.fit(X_train, y_train)
        
        vall= knn.predict([[total]])
        st.sidebar.write('Active cases : ',vall[0])

elif(method_name=="Random Forest"):
    st.sidebar.subheader("Prediction using Random Forest Classifier")
    selecting_states = st.sidebar.selectbox("State",("--Choose State--","Pahang","Kedah","Johor","Selangor"))
   
    
    datasets = df.copy()
    datasets['total_cases'] = datasets['cases_import'] + datasets['cases_new']
    datasets['active_cases'] = datasets.apply(conditions01, axis=1)
    dfs = datasets.copy()

    if(selecting_states=="Pahang"):

        new_cases = st.sidebar.number_input('New Cases')
        import_cases = st.sidebar.number_input('Import Cases')
        
        total = new_cases + import_cases

        dfs= dfs[dfs["state"] == selecting_states]
        dfs = dfs[['total_cases','active_cases']]
        X = dfs.drop('active_cases', axis=1)#dataset,independanent variable
        y = dfs['active_cases']#dependent variable
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        rf = RandomForestClassifier(random_state=5)
        rf.fit(X_train, y_train) 

        vall= rf.predict([[total]])
        st.sidebar.write('Active cases : ', vall[0])
    
    elif(selecting_states=="Kedah"):

        new_cases = st.sidebar.number_input('New Cases')
        import_cases = st.sidebar.number_input('Import Cases')
        
        total = new_cases + import_cases

        dfs= dfs[dfs["state"] == selecting_states]
        dfs = dfs[['total_cases','active_cases']]
        X = dfs.drop('active_cases', axis=1)#dataset,independanent variable
        y = dfs['active_cases']#dependent variable
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        rf = RandomForestClassifier(random_state=5)
        rf.fit(X_train, y_train) 

        vall= rf.predict([[total]])
        st.sidebar.write('Active cases : ', vall[0])

    elif(selecting_states=="Johor"):
        
        new_cases = st.sidebar.number_input('New Cases')
        import_cases = st.sidebar.number_input('Import Cases')
        
        total = new_cases + import_cases

        dfs= dfs[dfs["state"] == selecting_states]
        dfs = dfs[['total_cases','active_cases']]
        X = dfs.drop('active_cases', axis=1)#dataset,independanent variable
        y = dfs['active_cases']#dependent variable
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        rf = RandomForestClassifier(random_state=5)
        rf.fit(X_train, y_train) 

        vall= rf.predict([[total]])
        st.sidebar.write('Active cases : ', vall[0])

    elif(selecting_states=="Selangor"):

        new_cases = st.sidebar.number_input('New Cases')
        import_cases = st.sidebar.number_input('Import Cases')
        
        total = new_cases + import_cases

        dfs= dfs[dfs["state"] == selecting_states]
        dfs = dfs[['total_cases','active_cases']]
        X = dfs.drop('active_cases', axis=1)#dataset,independanent variable
        y = dfs['active_cases']#dependent variable
        X = pd.get_dummies(X, drop_first=True)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        rf = RandomForestClassifier(random_state=5)
        rf.fit(X_train, y_train) 

        vall= rf.predict([[total]])
        st.sidebar.write('Active cases : ', vall[0])
            
else:
        st.sidebar.warning("Select State")

#simple visualization
df_fs = df_fs.groupby(['state']).sum().reset_index()
st.subheader("Visualisation Cases by State")
st.write(df_fs)
# 1)
fig = plt.figure()
fig, ax = plt.subplots(figsize=(11.7, 8.27))
bar = alt.Chart(df_fs).mark_bar().encode(
    x='state',
    y='cases_new',
    color=alt.condition(
        alt.datum.cases_new >= 150000,  
        alt.value('red'),     
        alt.value('steelblue')   
    )
)
rule = alt.Chart(df_fs).mark_rule(color='red').encode(
    y='mean(cases_new)'
)
st.write((bar + rule).properties(width=600))


# 2)
df1 = df.copy()
df1['total_cases'] = df1['cases_import'] + df1['cases_new'] 
df1['dates'] = df1['date'].apply(lambda x: x[:-3])

line = alt.Chart(df1).mark_line().encode(
     x='dates',
     y='total_cases'
)
st.write(line)
#st.pyplot(fig)

#---Correlation
if(choise_state_colleration=="All State"):
    st.subheader("States Exhibit Strong Correlation: ")
    st.success(choise_state_colleration)
    fig = plt.figure()
    sqrt_transform = np.sqrt(df1["total_cases"])
    sns.distplot(sqrt_transform, bins=10)
    st.write("Skewness",sqrt_transform.skew()) 
    st.pyplot(fig)

    fig = plt.figure()
    fig, ax = plt.subplots(figsize=(17.7, 8.27))
    sns.boxplot(data=df1, x="state", y="total_cases", ax=ax)
    plt.title('Boxplot of State With Total Cases')
    st.pyplot(fig)

    fig = plt.figure()
    corr = df1.corr()
    sns.heatmap(corr, vmax=.8, square=True, annot=True, fmt='.2f',
           annot_kws={'size': 15}, cmap=sns.color_palette("Blues"))
    st.pyplot(fig)

elif(choise_state_colleration=="Pahang"):
    st.subheader("States Exhibit Strong Correlation: ")
    st.success(choise_state_colleration)


    df2_state_pahang = df1[df1["state"] == "Pahang"]
    
    fig = plt.figure()
    sqrt_transform = np.sqrt(df2_state_pahang["total_cases"])
    sns.distplot(sqrt_transform, bins=10)
    st.write("Skewness",sqrt_transform.skew()) 
    st.pyplot(fig)

    fig = plt.figure()
    sns.boxplot(data=df2_state_pahang, x="state", y="total_cases")
    plt.title('Boxplot of State of Pahang With Total Cases')
    st.pyplot(fig)

    df2_corr_Bycases_recovered = df2_state_pahang[["cases_import", "cases_recovered", "cases_new","total_cases"]]

    fig = plt.figure()
    corr = df2_corr_Bycases_recovered.corr()
    sns.heatmap(corr, vmax=.8, square=True, annot=True, fmt='.2f',
           annot_kws={'size': 15}, cmap=sns.color_palette("Reds"))
    st.pyplot(fig)

else:
    st.subheader("States Exhibit Strong Correlation: ")
    st.success(choise_state_colleration)


    df2_state_johor = df1[df1["state"] == "Johor"]
    
    fig = plt.figure()
    sqrt_transform = np.sqrt(df2_state_johor["total_cases"])
    sns.distplot(sqrt_transform, bins=10)
    st.write("Skewness",sqrt_transform.skew()) 
    st.pyplot(fig)

    fig = plt.figure()
    sns.boxplot(data=df2_state_johor, x="state", y="total_cases")
    plt.title('Boxplot of State of Pahang With Total Cases')
    st.pyplot(fig)

    df2_corr_Bycases_recovered = df2_state_johor[["cases_import", "cases_recovered", "cases_new","total_cases"]]

    fig = plt.figure()
    corr = df2_corr_Bycases_recovered.corr()
    sns.heatmap(corr, vmax=.8, square=True, annot=True, fmt='.2f',
           annot_kws={'size': 15}, cmap=sns.color_palette("Reds"))
    st.pyplot(fig)

# Selecting futureSelection
df3 = df.copy()
col_list = [col for col in df3.columns.tolist() if df3[col].dtype.name == "object"]
df_oh = df3[col_list]
dfs = df3.drop(col_list, 1)
df_oh = pd.get_dummies(df_oh)
dfs = pd.concat([df3, df_oh], axis=1)
dfs.drop(['date', 'state'], axis='columns', inplace=True)
#st.write(dfs.head())
st.subheader("Features to indicate daily cases: ")
st.success(choise_futureSelection)
# --------------------------------------------------------
if(choise_futureSelection=="Boruta classifier"):
        states = st.selectbox("States ",("Pahang","Kedah","Johor","Selangor"))
        if(states=="Pahang"):
            y = dfs.state_Pahang
            X = dfs.drop("state_Pahang", 1)
            colnames = X.columns
            rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
            feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1)
            feat_selector.fit(X.values,y.values.ravel())

            boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order= -1)
            boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
            boruta_score = boruta_score.sort_values("Score", ascending = False)

            st.write('---------Top 10----------')
            st.write(boruta_score.head(10))

            st.write('---------Bottom 10----------')
            st.write(boruta_score.tail(10))
        
        elif(states=="Kedah"):
            y = dfs.state_Kedah
            X = dfs.drop("state_Kedah", 1)
            colnames = X.columns
            rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
            feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1)
            feat_selector.fit(X.values,y.values.ravel())

            boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order= -1)
            boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
            boruta_score = boruta_score.sort_values("Score", ascending = False)

            st.write('---------Top 10----------')
            st.write(boruta_score.head(10))

            st.write('---------Bottom 10----------')
            st.write(boruta_score.tail(10))

        elif(states=="Johor"):
            y = dfs.state_Johor
            X = dfs.drop("state_Johor", 1)
            colnames = X.columns
            rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
            feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1)
            feat_selector.fit(X.values,y.values.ravel())

            boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order= -1)
            boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
            boruta_score = boruta_score.sort_values("Score", ascending = False)

            st.write('---------Top 10----------')
            st.write(boruta_score.head(10))

            st.write('---------Bottom 10----------')
            st.write(boruta_score.tail(10))
        
        else:
            y = dfs.state_Selangor
            X = dfs.drop("state_Selangor", 1)
            colnames = X.columns
            rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth=5)
            feat_selector = BorutaPy(rf, n_estimators='auto', random_state=1)
            feat_selector.fit(X.values,y.values.ravel())

            boruta_score = ranking(list(map(float, feat_selector.ranking_)), colnames, order= -1)
            boruta_score = pd.DataFrame(list(boruta_score.items()), columns=['Features','Score'])
            boruta_score = boruta_score.sort_values("Score", ascending = False)

            st.write('---------Top 10----------')
            st.write(boruta_score.head(10))

            st.write('---------Bottom 10----------')
            st.write(boruta_score.tail(10))

elif(choise_futureSelection=="RFE classifier"):

        states = st.selectbox("States ",("Pahang","Kedah","Johor","Selangor"))
        if(states=="Pahang"):
            y = dfs.state_Pahang
            X = dfs.drop("state_Pahang", 1)
            colnames = X.columns
            rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 5, n_estimators = 100)
            rf.fit(X, y)
            rfe = RFECV(rf, min_features_to_select = 1, cv = 3)
            rfe.fit(X,y)
            rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
            rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
            rfe_score = rfe_score.sort_values("Score", ascending = False)

            st.write('---------Top 10----------')
            st.write(rfe_score.head(10))

            st.write('---------Bottom 10----------')
            st.write(rfe_score.tail(10))
        
        elif(states=="Kedah"):
            y = dfs.state_Kedah
            X = dfs.drop("state_Kedah", 1)
            colnames = X.columns
            rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 5, n_estimators = 100)
            rf.fit(X, y)
            rfe = RFECV(rf, min_features_to_select = 1, cv = 3)
            rfe.fit(X,y)
            rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
            rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
            rfe_score = rfe_score.sort_values("Score", ascending = False)

            st.write('---------Top 10----------')
            st.write(rfe_score.head(10))

            st.write('---------Bottom 10----------')
            st.write(rfe_score.tail(10))

        elif(states=="Johor"):
            y = dfs.state_Johor
            X = dfs.drop("state_Johor", 1)
            colnames = X.columns
            rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 5, n_estimators = 100)
            rf.fit(X, y)
            rfe = RFECV(rf, min_features_to_select = 1, cv = 3)
            rfe.fit(X,y)
            rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
            rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
            rfe_score = rfe_score.sort_values("Score", ascending = False)

            st.write('---------Top 10----------')
            st.write(rfe_score.head(10))

            st.write('---------Bottom 10----------')
            st.write(rfe_score.tail(10))
        
        else:
            y = dfs.state_Selangor
            X = dfs.drop("state_Selangor", 1)
            colnames = X.columns
            rf = RandomForestClassifier(n_jobs=-1, class_weight="balanced", max_depth = 5, n_estimators = 100)
            rf.fit(X, y)
            rfe = RFECV(rf, min_features_to_select = 1, cv = 3)
            rfe.fit(X,y)
            rfe_score = ranking(list(map(float, rfe.ranking_)), colnames, order=-1)
            rfe_score = pd.DataFrame(list(rfe_score.items()), columns=['Features', 'Score'])
            rfe_score = rfe_score.sort_values("Score", ascending = False)

            st.write('---------Top 10----------')
            st.write(rfe_score.head(10))

            st.write('---------Bottom 10----------')
            st.write(rfe_score.tail(10))

# --------------------------------------------------------


# ---Selecting Classifier
dataset = df.copy()
dataset['active_cases'] = abs(dataset['cases_new'] - dataset['cases_recovered'])
dataset['total_cases'] = dataset['cases_import'] + dataset['cases_new']
dataset['has_cases'] = dataset.apply(conditions, axis=1)
df = dataset.copy()
st.subheader("Comparing model performs well in predicting the daily cases: ")
st.warning(method_name)
st.write(df.head())
if(method_name=="K-Nearest Neighbors"):
    
    state = st.selectbox("Select States",("Pahang","Kedah","Johor","Selangor"))
    if(state=="Pahang"):

        df = df[df["state"] == state]
        categorical = ['date','state','has_cases']
        d = defaultdict(LabelEncoder)
        df[categorical] = df[categorical].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
        X = df.drop('has_cases', axis=1)#dataset,independanent variable
        y = df['has_cases']#dependent variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)

        prob_KNN = knn.predict_proba(X_test)
        prob_KNN = prob_KNN[:, 1]
        fpr_knn, tpr_knn, thresholds_DT = roc_curve(y_test, prob_KNN) 

        fig = plt.figure()
        plt.plot(fpr_knn, tpr_knn, color='red', label='KNN') 
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        st.pyplot(fig)

        fig = plt.figure()
        prec_knn, rec_knn, thresholds_DT = precision_recall_curve(y_test, prob_KNN)
        plt.plot(prec_knn, rec_knn, color='red', label='KNN') 
        plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        st.pyplot(fig)

        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

    elif(state=="Kedah"):
        df = df[df["state"] == state]
        categorical = ['date','state','has_cases']
        d = defaultdict(LabelEncoder)
        df[categorical] = df[categorical].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
        X = df.drop('has_cases', axis=1)#dataset,independanent variable
        y = df['has_cases']#dependent variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)

        prob_KNN = knn.predict_proba(X_test)
        prob_KNN = prob_KNN[:, 1]
        fpr_knn, tpr_knn, thresholds_DT = roc_curve(y_test, prob_KNN) 

        fig = plt.figure()
        plt.plot(fpr_knn, tpr_knn, color='red', label='KNN') 
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        st.pyplot(fig)

        fig = plt.figure()
        prec_knn, rec_knn, thresholds_DT = precision_recall_curve(y_test, prob_KNN)
        plt.plot(prec_knn, rec_knn, color='red', label='KNN') 
        plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        st.pyplot(fig)

        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
    
    elif(state=="Johor"):
        df = df[df["state"] == state]
        categorical = ['date','state','has_cases']
        d = defaultdict(LabelEncoder)
        df[categorical] = df[categorical].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
        X = df.drop('has_cases', axis=1)#dataset,independanent variable
        y = df['has_cases']#dependent variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)

        prob_KNN = knn.predict_proba(X_test)
        prob_KNN = prob_KNN[:, 1]
        fpr_knn, tpr_knn, thresholds_DT = roc_curve(y_test, prob_KNN) 

        fig = plt.figure()
        plt.plot(fpr_knn, tpr_knn, color='red', label='KNN') 
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        st.pyplot(fig)

        fig = plt.figure()
        prec_knn, rec_knn, thresholds_DT = precision_recall_curve(y_test, prob_KNN)
        plt.plot(prec_knn, rec_knn, color='red', label='KNN') 
        plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        st.pyplot(fig)

        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))

    else:
        df = df[df["state"] == state]
        categorical = ['date','state','has_cases']
        d = defaultdict(LabelEncoder)
        df[categorical] = df[categorical].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
        X = df.drop('has_cases', axis=1)#dataset,independanent variable
        y = df['has_cases']#dependent variable
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)

        confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)

        prob_KNN = knn.predict_proba(X_test)
        prob_KNN = prob_KNN[:, 1]
        fpr_knn, tpr_knn, thresholds_DT = roc_curve(y_test, prob_KNN) 

        fig = plt.figure()
        plt.plot(fpr_knn, tpr_knn, color='red', label='KNN') 
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        st.pyplot(fig)

        fig = plt.figure()
        prec_knn, rec_knn, thresholds_DT = precision_recall_curve(y_test, prob_KNN)
        plt.plot(prec_knn, rec_knn, color='red', label='KNN') 
        plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        st.pyplot(fig)

        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
      
elif(method_name=="Random Forest"):
    state = st.selectbox("Select States",("Pahang","Kedah","Johor","Selangor"))
    
    if(state=="Pahang"):
        df= df[df["state"] == state]
        categorical = ['date','state','has_cases']
        d = defaultdict(LabelEncoder)
        df[categorical] = df[categorical].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
        X = df.drop('has_cases', axis=1)
        y = df['has_cases']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        rf = RandomForestClassifier(random_state=10)
        rf.fit(X_train, y_train) 

        prob_RF = rf.predict_proba(X_test)
        prob_RF = prob_RF[:, 1]
        fpr_rf, tpr_rf, thresholds_DT = roc_curve(y_test, prob_RF)

        fig = plt.figure()
        plt.plot(fpr_rf, tpr_rf, color='purple', label='RF') 
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        st.pyplot(fig)

        fig = plt.figure()
        prec_RF, rec_RF, thresholds_DT = precision_recall_curve(y_test, prob_RF)
        plt.plot(prec_RF, rec_RF, color='purple', label='RF') 
        plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        st.pyplot(fig)
    
    elif(state=="Kedah"):

        df= df[df["state"] == state]
        categorical = ['date','state','has_cases']
        d = defaultdict(LabelEncoder)
        df[categorical] = df[categorical].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
        X = df.drop('has_cases', axis=1)
        y = df['has_cases']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        rf = RandomForestClassifier(random_state=10)
        rf.fit(X_train, y_train) 

        prob_RF = rf.predict_proba(X_test)
        prob_RF = prob_RF[:, 1]
        fpr_rf, tpr_rf, thresholds_DT = roc_curve(y_test, prob_RF)

        fig = plt.figure()
        plt.plot(fpr_rf, tpr_rf, color='purple', label='RF') 
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        st.pyplot(fig)

        fig = plt.figure()
        prec_RF, rec_RF, thresholds_DT = precision_recall_curve(y_test, prob_RF)
        plt.plot(prec_RF, rec_RF, color='purple', label='RF') 
        plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        st.pyplot(fig)

        y_pred = rf.predict(X_test)
        confusion_majority=confusion_matrix(y_test, y_pred)
        confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)
        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
    
    elif(state=="Johor"):

        df= df[df["state"] == state]
        categorical = ['date','state','has_cases']
        d = defaultdict(LabelEncoder)
        df[categorical] = df[categorical].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
        X = df.drop('has_cases', axis=1)
        y = df['has_cases']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        rf = RandomForestClassifier(random_state=10)
        rf.fit(X_train, y_train) 

        prob_RF = rf.predict_proba(X_test)
        prob_RF = prob_RF[:, 1]
        fpr_rf, tpr_rf, thresholds_DT = roc_curve(y_test, prob_RF)

        fig = plt.figure()
        plt.plot(fpr_rf, tpr_rf, color='purple', label='RF') 
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        st.pyplot(fig)

        fig = plt.figure()
        prec_RF, rec_RF, thresholds_DT = precision_recall_curve(y_test, prob_RF)
        plt.plot(prec_RF, rec_RF, color='purple', label='RF') 
        plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        st.pyplot(fig)

        y_pred = rf.predict(X_test)
        confusion_majority=confusion_matrix(y_test, y_pred)
        confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)
        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))
    
    else:
        df= df[df["state"] == state]
        categorical = ['date','state','has_cases']
        d = defaultdict(LabelEncoder)
        df[categorical] = df[categorical].apply(lambda x: d[x.name].fit_transform(x.astype(str)))
        X = df.drop('has_cases', axis=1)
        y = df['has_cases']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

        rf = RandomForestClassifier(random_state=10)
        rf.fit(X_train, y_train) 

        prob_RF = rf.predict_proba(X_test)
        prob_RF = prob_RF[:, 1]
        fpr_rf, tpr_rf, thresholds_DT = roc_curve(y_test, prob_RF)

        fig = plt.figure()
        plt.plot(fpr_rf, tpr_rf, color='purple', label='RF') 
        plt.plot([0, 1], [0, 1], color='green', linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend()
        st.pyplot(fig)

        fig = plt.figure()
        prec_RF, rec_RF, thresholds_DT = precision_recall_curve(y_test, prob_RF)
        plt.plot(prec_RF, rec_RF, color='purple', label='RF') 
        plt.plot([1, 0], [0.1, 0.1], color='green', linestyle='--')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend()
        st.pyplot(fig)

        y_pred = rf.predict(X_test)
        confusion_majority=confusion_matrix(y_test, y_pred)
        confussion_m = confusion_matrix(y_true=y_test, y_pred= y_pred)
        st.write('Precision= {:.2f}'.format(precision_score(y_test, y_pred, pos_label=0)))
        st.write('Recall= {:.2f}'. format(recall_score(y_test, y_pred, pos_label=0)))
        st.write('F1= {:.2f}'. format(f1_score(y_test, y_pred, pos_label=0)))
        st.write('Accuracy= {:.2f}'. format(accuracy_score(y_test, y_pred)))




elif(method_name=="Linear Regression"):
    
    df4 = df.copy()

    df4['active_cases'] = abs(df4['cases_new'] - df4['cases_recovered'])
    df4['total_cases'] = df4['cases_import'] + df4['cases_new']
    
    state = st.selectbox("Select States",("Pahang","Kedah","Johor","Selangor"))
    cases = st.selectbox("Select Cases",("Import Cases","New Cases","Total Cases","Active Cases"))
    
    if(cases=="Import Cases"):
        select_cases ='cases_import'
    elif(cases=="New Cases"):
        select_cases ='cases_new'
    elif(cases=="Total Cases"):
        select_cases ='total_cases'
    else:
        select_cases ='active_cases'
    

    if(state=="Pahang"):

        df4= df4[df4["state"] == state]
        X = df4[['cases_new']]
        Y = df4[[select_cases]]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=5)
        lm = LinearRegression()
        lm.fit(X_train, Y_train)
        Y_test_pred = lm.predict(X_test)
        Y_test_pred = pd.DataFrame(Y_test_pred, columns=['PredictedSum'])

        df_tmp = df4.loc[:,['cases_new',select_cases]] 
        df_new = pd.concat([df_tmp.reset_index(drop=True), Y_test_pred], axis=1)
       
        fig = plt.figure()
        plt.plot(X_test, Y_test, 'o')
        plt.plot(X_test, Y_test_pred)
        plt.show()
        st.pyplot(fig)

        errors = mean_squared_error(Y_test, Y_test_pred)
        st.write("Coefficient: ",lm.coef_)
        st.write("Intercept: ",lm.intercept_)
        st.write("Score: ",lm.score(X_test, Y_test))
        st.write("Error: ",errors)
    
    elif(state=="Kedah"):

        df4= df4[df4["state"] == state]
        X = df4[['cases_new']]
        Y = df4[[select_cases]]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=5)
        lm = LinearRegression()
        lm.fit(X_train, Y_train)
        Y_test_pred = lm.predict(X_test)
        Y_test_pred = pd.DataFrame(Y_test_pred, columns=['PredictedSum'])

        df_tmp = df4.loc[:,['cases_new',select_cases]] 
        df_new = pd.concat([df_tmp.reset_index(drop=True), Y_test_pred], axis=1)
       
        fig = plt.figure()
        plt.plot(X_test, Y_test, 'o')
        plt.plot(X_test, Y_test_pred)
        plt.show()
        st.pyplot(fig)

        errors = mean_squared_error(Y_test, Y_test_pred)
        st.write("Coefficient: ",lm.coef_)
        st.write("Intercept: ",lm.intercept_)
        st.write("Score: ",lm.score(X_test, Y_test))
        st.write("Error: ",errors)

    elif(state=="Johor"):

        df4= df4[df4["state"] == state]
        X = df4[['cases_new']]
        Y = df4[[select_cases]]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=5)
        lm = LinearRegression()
        lm.fit(X_train, Y_train)
        Y_test_pred = lm.predict(X_test)
        Y_test_pred = pd.DataFrame(Y_test_pred, columns=['PredictedSum'])

        df_tmp = df4.loc[:,['cases_new',select_cases]] 
        df_new = pd.concat([df_tmp.reset_index(drop=True), Y_test_pred], axis=1)
       
        fig = plt.figure()
        plt.plot(X_test, Y_test, 'o')
        plt.plot(X_test, Y_test_pred)
        plt.show()
        st.pyplot(fig)

        errors = mean_squared_error(Y_test, Y_test_pred)
        st.write("Coefficient: ",lm.coef_)
        st.write("Intercept: ",lm.intercept_)
        st.write("Score: ",lm.score(X_test, Y_test))
        st.write("Error: ",errors)

    else:

        df4= df4[df4["state"] == state]
        X = df4[['cases_new']]
        Y = df4[[select_cases]]
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=5)
        lm = LinearRegression()
        lm.fit(X_train, Y_train)
        Y_test_pred = lm.predict(X_test)
        Y_test_pred = pd.DataFrame(Y_test_pred, columns=['PredictedSum'])

        df_tmp = df4.loc[:,['cases_new',select_cases]] 
        df_new = pd.concat([df_tmp.reset_index(drop=True), Y_test_pred], axis=1)
       
        fig = plt.figure()
        plt.plot(X_test, Y_test, 'o')
        plt.plot(X_test, Y_test_pred)
        plt.show()
        st.pyplot(fig)

        errors = mean_squared_error(Y_test, Y_test_pred)
        st.write("Coefficient: ",lm.coef_)
        st.write("Intercept: ",lm.intercept_)
        st.write("Score: ",lm.score(X_test, Y_test))
        st.write("Error: ",errors)

elif(method_name=="CART Regression Tree"):
   
    cases = st.selectbox("Select Cases",("Import Cases","Total Cases","Active Cases"))
    
    if(cases=="Import Cases"):
        select_cases ='cases_import'
    elif(cases=="Total Cases"):
        select_cases ='total_cases'
    else:
        select_cases ='active_cases'

    df_cart = df.copy()

    del df_cart['date']
    del df_cart['state']
    del df_cart['has_cases']

    dataset = df_cart.copy()
    X = dataset.drop(select_cases, axis=1)  
    y = dataset[select_cases]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2)
    regressor = DecisionTreeRegressor(max_depth=3)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    df=pd.DataFrame({'Actual':y_test, 'Predicted':y_pred})  
    st.write(df.head())
    fn = X.columns
    errors = mean_squared_error(y_test, y_pred)
    st.write("Errors: ",errors)

    fig, axes=plt.subplots(nrows=1,ncols=1,figsize=(4,4),dpi=300)
    tree.plot_tree(regressor,
               feature_names=fn,
               filled=True);
    st.pyplot(fig)

    





 


