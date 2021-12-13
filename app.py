# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 00:36:56 2021

@author: j.wadhwani
"""

import sklearn
import streamlit as st
import pandas as pd
import numpy as np

import pickle
import math
import matplotlib.pyplot as plt
##Fuctions


st.markdown("<h1 style='text-align: center; color: blue;'>Lemnisk Assignment</h1>", unsafe_allow_html=True)
st.subheader("Scroll Below to see result")
row_1_1, row_1_2, row_1_3, row_1_4 = st.beta_columns(4)

input_1=row_1_1.number_input("Variable 13",min_value=0, step=1, max_value=1000, value=2)
input_2=row_1_2.number_input("Variable 26",min_value=0, step=1, max_value=14,value=3)
input_3=row_1_3.number_input("Variable 9", step=0.01,min_value=0.0,max_value=100000.0,value=1.0)
input_4=row_1_4.number_input("Variable 21",min_value=0, step=1, max_value=7,value=3)

row_2_1, row_2_2, row_2_3, row_2_4 = st.beta_columns(4)

input_5=row_2_1.number_input("Variable 11",min_value=0.0, step=0.05, max_value=25000.0,value=354.0)
input_6=row_2_2.number_input("Variable 30",min_value=0, step=1, max_value=8,value=4)
input_7=row_2_3.number_input("Variable 43",min_value=0, step=1, max_value=1500,value=200)
input_8=row_2_4.number_input("Variable 51",min_value=0.0, step=0.05, max_value=16000.0,value=548.0)

row_3_1, row_3_2, row_3_3, row_3_4 = st.beta_columns(4)

input_9=row_3_1.number_input("Variable 54",min_value=0.0, step=0.05, max_value=16000.0,value=200.0)
input_10=row_3_2.selectbox("var6 = e939124300", (0,1))
input_11=row_3_3.number_input("Variable 34",min_value=0, step=1, max_value=34,value=4)
input_12=row_3_4.selectbox("var1= 02ad845d2f", (0,1))


row_4_1, row_4_2, row_4_3, row_4_4 = st.beta_columns(4)

input_13=row_4_1.selectbox("var1= 30565a8911", (0,1))
input_14=row_4_2.selectbox("var4= c4ca4238a0", (0,1))
input_15=row_4_3.selectbox("var6= 6c096eda6c", (0,1))
input_16=row_4_4.selectbox("var4= 8f14e45fce", (0,1))

row_5_1, row_5_2, row_5_3= st.beta_columns(3)

input_17=row_5_1.selectbox("var3= ec181e8598", (0,1))
input_18=row_5_2.selectbox("var1= e4c2e8edac", (0,1))
input_19=row_5_3.number_input("Variable 39",min_value=0, step=1, max_value=96,value=60)

input_3= math.log(input_3+0.01)
input_5= math.log(input_5+0.01)
input_8= math.log(input_8+0.01)
input_9= math.log(input_9+0.01)


scalerfile = 'scaler_9.sav'
scaler_9 = pickle.load(open(scalerfile, 'rb'))

scalerfile = 'scaler_11.sav'
scaler_11 = pickle.load(open(scalerfile, 'rb'))

scalerfile = 'scaler_51.sav'
scaler_51 = pickle.load(open(scalerfile, 'rb'))

scalerfile = 'scaler_54.sav'
scaler_54 = pickle.load(open(scalerfile, 'rb'))

dummy_9= np.array([[input_3]])
output_dummy_9= scaler_9.transform(dummy_9)

dummy_11= np.array([[input_5]])
output_dummy_11= scaler_9.transform(dummy_11)

dummy_51= np.array([[input_8]])
output_dummy_51= scaler_9.transform(dummy_51)

dummy_54= np.array([[input_9]])
output_dummy_54= scaler_9.transform(dummy_54)


input_3=output_dummy_9[0][0]
input_5=output_dummy_11[0][0]
input_8=output_dummy_51[0][0]
input_9=output_dummy_54[0][0]



input_array= np.array([input_1, input_2, 
                       input_3, input_4,input_5,input_6,input_7,input_8,input_9,input_10,
                       input_11,input_12,input_13,input_14,input_15,input_16,input_17,input_18,input_19
                       ])

input_array= input_array.reshape(1,-1)
filename = 'Logistic.sav'
loaded_model = pickle.load(open(filename, 'rb'))

result= loaded_model.predict_proba(input_array)[:,1]
st.set_option('deprecation.showPyplotGlobalUse', False)

result_array= np.array([1-result,result])

result_array= result_array.reshape(1,2)

X=['0','1']
Y=[1-result[0], result[0]]

plt.bar(X, Y,
        width = 0.4)
plt.show()
st.pyplot()

statement_1= "Chances of 0: "+ str(round(   (1-result[0])*100   ,2)) + "%"
statement_2= "Chances of 1: "+ str(round(   (result[0])*100   ,2)) + "%"
st.write(statement_1)
st.write(statement_2)







