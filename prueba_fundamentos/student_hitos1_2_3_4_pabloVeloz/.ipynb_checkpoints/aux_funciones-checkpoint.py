import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
#from sklearn.cluster import KMean

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm
import statsmodels.formula.api as smf
import factor_analyzer as factor
import missingno as msngo
import warnings
warnings.filterwarnings('ignore')

def boxplot_num(dataframe,lista):
    #2 filas 3 columnas
    #var=["age","fnlwgt","educational_num","capital_gain","capital_loss","hours_per_week"]
    var=lista
    fig = plt.figure()
    plt.subplots_adjust(wspace=0.4,right = 2.0,bottom = -1.1)
    
    for i,n in enumerate(var):
        plt.subplot(2,3,i+1)
        sns.boxplot(dataframe["income"],df2[n])
        plt.title(n+" v/s"+" income")
        sns.set_palette("bright")

def plot_bar(df)        
    lista=[df2.groupby('income').workclass_recod,df2.groupby('income').educ_recod,
           df2.groupby('income').educational_num,df2.groupby('income').civstatus,
           df2.groupby('income').collars,df2.groupby('income').relationship,
           df2.groupby('income').race,df2.groupby('income').gender,
           df2.groupby('income').region]
    var=["workclass_recod","educ_recod","educational_num","civstatus","collars","relationship","race","gender","region"]
    fig, axarr = plt.subplots(3, 3, figsize=(15, 8))
    plt.subplots_adjust(wspace=0.4,right = 0.7,bottom = -1.4,hspace = 0.5)
    j=0
    k=0
    for i,n in enumerate(lista):
        #plot.subplot(3,3,i+1)
        lista[i].value_counts().unstack(0).plot.bar(ax=axarr[j][k])
        axarr[j][k].set_title(var[i], fontsize=18)
        j+=1
        if j==3:
            k+=1
            j=0
            if k==3:
                k=0
                j=0        
        

def hist_and_box(col):
    f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw= {"height_ratios": (0.2, 1)})
    mean=df[col].mean()
    median=df[col].median()
    mode=df[col].mode().get_values()[0]
    sns.boxplot(df[col], ax=ax_box)
    ax_box.axvline(mean, color='r', linestyle='--')
    ax_box.axvline(median, color='g', linestyle='-')
    ax_box.axvline(mode, color='b', linestyle='-')

    sns.distplot(df[col], ax=ax_hist)
    ax_hist.axvline(mean, color='r', linestyle='--')
    ax_hist.axvline(median, color='g', linestyle='-')
    ax_hist.axvline(mode, color='b', linestyle='-')

    plt.legend({'Mean':mean,'Median':median,'Mode':mode})
    ax_box.set(xlabel=str(col)+": "+"Mean: "+str(round(mean,2))+" Median: "+str(round(median,2)))
    plt.show()

def cat_plot(varx,vary,var_grouped):
    bins=[0,9,20,30,75]
    names=["0-9","10-20","21-30","31-75"]
    df_copy["absences"]=pd.cut(df_copy["absences"],bins,labels=names)
    sns.catplot(x="sex", y="G1", hue="absences", data=df_copy,height=5, kind="bar", palette="muted")
    
    
    
def plot_columns_behaviour(df, kind='countplot'):
    """Plots the columns of the given dataframe using a countplot or a distplot
    Parameters
    ----------
    df : DataFrame
    kind : str
        countplot or distplot
    """

    cols = list(df.columns)
    n_cols = 3
    n_rows = np.ceil(len(cols) / n_cols)
    plt.figure(figsize=(n_cols * 5, 5 * n_rows))

    for n, col_name in enumerate(cols):
        plt.subplot(n_rows, n_cols, n + 1)

        col = df[col_name]

        if kind == 'countplot':
            sns.countplot(y=col)
            plt.title(humanize(col_name))
            plt.xlabel("")
        else:
            sns.distplot(col, rug=True)
            plt.title(humanize(col_name))
            plt.xlabel("")
            plt.axvline(col.mean(), color='tomato',
                        linestyle='--', label='mean')
            plt.axvline(col.median(), color='green',
                        linestyle='--', label='median')
            plt.legend()
        plt.tight_layout()    
    
    
    
def show_correlaciones(df, value=0.7):
    plt.figure(figsize=(15, 6))
    M = df.corr()
    value_corr = M[((M > value) & (M < 1) | (M < -value))
                  ].dropna(axis=0, how='all').dropna(axis=1, how='all')
    ax=sns.heatmap(value_corr, vmin=-1, vmax=1, center=0, cmap=sns.diverging_palette(20, 320, n=250),annot=True)
    ax.set_xticklabels(ax.get_xticklabels(),rotation=45,horizontalalignment='right')
    plt.title("Corralaciones Totales")
    
def binarize(df):
    #df_dummy=df.copy()
    #df_dummy.columns
    #df_dummy.dtypes
    lista_cat=[]
    for i in df.columns:
        tipo_col=df[i].dtype
        if tipo_col==object:
            lista_cat.append(i)
    #print(lista_cat)
    for i in lista_cat:
        df = pd.get_dummies(df, columns=[i],drop_first=True)
    return df

def string_col_for_models(df,var):
    """probar con largo-2"""
    string_cols=var+"~"
    largo=len(df_dummy.columns)
    for i,n in enumerate(df.columns):
        if n==var:
            pass
        else:
            if i!=largo-1:
                string_cols+=n+"+"

            else:
                string_cols+=n
    return string_cols

def string_col_new_for_model(df,lista_vars):
    string_cols=var+"~"
    lista=lista_vars
    largo=len(df.columns)
    for i,n in enumerate(df.columns):
        if n in lista:
            pass
        else:
            if i!=largo-1:
                string_cols+=n+"+"

            else:
                string_cols+=n
    return string_cols


def significant_pvalues(model):
    """Returns the significant pvalues in model (95% significance)"""
    pvalues = model.pvalues[1:]
    return pvalues[pvalues <= 0.025]

def compara_test_predict(y_test,y_predict)

    df3 = pd.DataFrame({'Actual': y_test, 'Prediccion': y_predict})
    df4 = df3.head(25)
    df4.plot(kind='bar',figsize=(16,10))
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')

    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    plt.title("Posicion del dato vs valor en modelo test - prediccion")
    plt.show()
    
    
def predict_target(df_dummy,var):
    y_vec=df_dummy[var]
    X_mat=df_dummy.loc[:,columnas].dropna()

    X_train,X_test,y_train,y_test=train_test_split(X_mat,y_vec,test_size=0.33,random_state=1612399)
    std_scaler=StandardScaler().fit(X_train)
    data_preproc_Xtrain=std_scaler.transform(X_train)
    data_preproc_Xtest=std_scaler.transform(X_test)

    model_1=linear_model.LinearRegression(fit_intercept=True, normalize=False)
    modelo1_entrenado=model_1.fit(X_train,y_train)
    return y_modelo1_pred=modelo1_entrenado.predict(X_test),mean_squared_error(y_test,y_modelo1_pred),r2_score(y_test,y_modelo1_pred)