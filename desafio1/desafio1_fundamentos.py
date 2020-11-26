import numpy as np
import pandas as pd


print("Hola Mundo, esta es mi primera incursión en Python")

name = "Pablo"
edad  = "34"
hobbies=["Futbol","ping-pong","poe"]
mascotas="si"

print(name)
print(edad)
print(hobbies)
print(mascotas)

print(hobbies[1])

print(type(name))
print(type(edad))
print(type(hobbies))
print(type(mascotas))

def presentacion(nombre,edad,hobbies,mascotas):
    print("Mi nombres es "+nombre+" tengo "+edad+" practico"+" "+hobbies[0]+" "+hobbies[1]+" y "+hobbies[2]+" "+mascotas+" tengo mascotas")

presentacion(name,edad,hobbies,mascotas)

#print('Estaba la pájara pinta sentada en el verde limón)

print('Estaba la pájara pinta sentada en el verde limón')

#print('Mi nombre es' name 'y tengo' edad, 'años')
print('Mi nombre es'+name+'y tengo',edad,'años')

#import padnas as pd
import pandas as pd
#import nunnpy as np
import numpy as np

#"Ornitorrinco" + 45
"Ornitorrinco"+str(45)


df = pd.read_csv("flights.csv")
print(df.head(5))
print(df.tail(5))

print(df.year.describe())
print(df.year.value_counts())
print(df.month.value_counts())
#print(df.columns)


p_15 = df.head(15)
u_15 = df.tail(15)

media_p15=p_15["passengers"].mean()
print(media_p15)
mediana_p15=p_15["passengers"].median()
print(mediana_p15)
desviacion_p15=p_15["passengers"].std()
print(desviacion_p15)

promedio_df_pasajeros = df["passengers"].mean()
promedio_df_pasajeros = df["passengers"].median()
promedio_df_pasajeros = df["passengers"].std()