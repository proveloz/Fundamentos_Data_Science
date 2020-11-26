import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
def descriptiva1(df):
    """
    -Esta funcion obtiene un df para variables continuas y discretas 
    -Su argumento exige un vector pd.Series
    -Retorna el promedio,varianza y desviacion estandar para variable continua y solo la frecuencia para variable discreta
    """
    lista_cuanti=[]
    lista_cuali=[]
    for colname in df.columns:
        if df[colname].dtype == "float64":
            lista_cuanti.append(colname)
        else:
            lista_cuali.append(colname)

    for colname in lista_cuanti:
        print("VARIABLE CONTINUA: "+colname)
        tmp_describe=df[colname].describe()
        print(tmp_describe)
        print("\n")
        
    for colname in lista_cuali:
        print("VARIABLE DISCRETA :",colname)
        tmp_frec=df[colname].value_counts()
        print(tmp_frec)
        print("\n")
                
        return tmp_describe,tmp_frec


def obs_perdidas(dataframe,var,print_list):
    """
    -Esta funcion las observaciones perdidas de una vector df
    -Su argumento exige un vector pd.Series, una varible o columna del dataframe, y un booleano print_list
    -Retorna la variable, la cantidad de elementos nulos y el porcentaje de nulos respecto al largo del vector.
    """
    nulos=dataframe[var].isnull().sum()

    porcentaje=100*(dataframe[var].isnull().sum()/len(dataframe[var]))

    #df_nulos["Porcentaje"]=round(100*porcentaje,3)
    #return nulos,porcentaje
    print("* En",var,"Hay una cantidad de elementos nulos: ",nulos,",Porcentaje: ",porcentaje,"%")
  
    if print_list == True:
        df_vacios=df_base_generada.loc[dataframe[var].isnull()]
        return df_vacios


#def color_rojo(value):
#    if value < 22:
#        color = 'black'
    #elif value > 22:
    #    color = 'red'
#    else:
#        color = 'red'

#    return 'color: %s' % color


def reporte_obs_perdidas(x):
    """
    -Esta funcion reporta todas las observaciones perdidas para un vector pd.Series
    -Su argumento exige solo un vector pd.Series.
    -Retorna la columna, cantidad de elementos nulos por columnas y el porcentaje de nulos respecto al largo del vector.
    """
    df_nulos=x.isnull().sum().to_frame('nulos')
    lista_columnas=[]
    lista_values=[]
    #lista_porcentaje=[]
    for i in x.columns:
        #porcentaje=(x.isnull().sum()/len(x[i]))
        #df_nulos2=df_nulos["Porcentaje"]=round(100*porcentaje,3)
        lista_columnas.append(i)
    for i in df_nulos.values:
        lista_values.append(int(i[0]))
    #for i in df_nulos2.values:
    #    lista.porcentaje.append(i)
    #df_coloreado=df_nulos.style.applymap(color_rojo, subset=['nulos','Porcentaje'])
    tmp=pd.DataFrame({"Columnas Compañera Tania":lista_columnas,"Cantidad Nulos":lista_values})
    return tmp

#def obs_perdidas_total(dataframe,var,print_list):
    #"""
    #-Esta funcion las observaciones perdidas de todas  las columnas de un vector dataframe
    #-Su argumento exige un vector pd.Series, una varible o columna del dataframe, y un booleano print_list
    #-Retorna la variable, la cantidad de elementos nulos y el porcentaje de nulos respecto al largo del vector.
    #"""
    #nulos=dataframe[var].isnull().sum()

    #porcentaje=100*(dataframe[var].isnull().sum()/len(dataframe[var]))

    #print("* En",var,"Hay una cantidad de elementos nulos: ",nulos,",Porcentaje: ",porcentaje,"%")
  
   # if print_list == True:
   #     df_vacios=df_base_generada.loc[dataframe[var].isnull()]
   #     return df_vacios

def graficar_histograma(x,variable,sample_mean,true_mean):
    """
    -Esta funcion muestra el histograma de un dataframe a partir de una variable de estudio
    -Su argumento un dataframe,una variable a estudiar, sample_mean:booleano para mostrar la media de la variable y true mean
    un booleano que muestre la media completa de los datos.
    -Retorna un histograma de la variable solicitada.
    """
    a = list(variable)

    promedio_total=variable.mean()
    print("Promedio total",promedio_total)
    df_dropna=x[variable].dropna()
    g= plt.hist(df_dropna,color="cyan")
    plt.title(label="Histograma de "+variable)
    
    #plt.hist([p, o], color=['g','r'], alpha=0.8, bins=50)
    #plt.show()
    a= plt.axvline(df_dropna.mean(),lw=3, color ="tomato", linestyle="--")
    print("Media Muestral: Roja")
    #a.text(10.1,0,'Media Muestra',rotation=90)
    a2= plt.axvline(promedio_total,lw=4, color ="black", linestyle="--")
    print("Media Total: Negra")
    if sample_mean == False:
        #g= plt.hist(df_dropna,color="cyan")
        return g
    elif sample_mean == True:

        return g,a
    
    elif true_mean == True:
        
        return g,a,a2

#def mediana_total(columna):
#    global df
#    a=df[columna].median()
#    return a

def graficar_dotPlot(dataframe,plot_var,plot_by,global_stat,statistic):
    """
    -Esta funcion muestra el dotPlot de un dataframe a partir de una variable de estudio
    -Su argumento exige:
    * dataframe : La tabla de datos donde buscar las variables.
    * plot_var : La variable a analizar y extraer las medias.
    * plot_by : La variable agrupadora.
    * global_stat : Booleano. Si es True debe graficar la media global de la variable. Por defecto debe ser False .
     statistic: Debe presentar dos opciones. mean para la media y median para la mediana. Por defecto debe ser mean .
    -Retorna un histograma de la variable solicitada.
    
    """
    if statistic=="mean":
        df_groupby=dataframe.groupby(plot_by)[plot_var].mean()
        if global_stat==True:
            media=dataframe[plot_var].mean()
            plt.axvline(media,color="cyan",lw=3,linestyle="--")
    elif statistic == "median":
        df_groupby=dataframe.groupby(plot_by)[plot_var].median()
        if global_stat==True:
            median=dataframe[plot_var].median()
            plt.axvline(median,color="cyan",lw=3,linestyle="--")
    elif statistic == "zscore":
        df_groupby=dataframe.groupby(plot_by)[plot_var].mean()
        df_na = dataframe[plot_var].dropna()
        media_na = df_na.mean()
        std_na= df_na.std()
        df_groupby=(df_groupby-media_na)/std_na
        plt.axvline(0,color="cyan",lw=3,linestyle="--")
    
    plt.plot(df_groupby.values,df_groupby.index,"o")
    plt.title("Grafico "+plot_var)
    plt.xlabel(statistic)
    plt.ylabel(plot_by)

