import streamlit as st
import pickle
l=[]
st.title("POKEMON LEGENDARY PREDICTATION")
st.title("PLEASE ENTER THE RESPECTIVE VALUES BELOW:")
l.append(st.input_text('name',4,50,format="%s"))
l.append(st.number_input('generation',1,5,format="%f"))
l.append(st.input_text('species',4,50,format="%s"))
l.append(st.number_input('type_number',1,5,format="%f"))
l.append(st.input_text('type_1',2,20,format="%s"))
l.append(st.number_input('height_m',2,10,format="%.2f"))
l.append(st.number_input('weight_kg',1,10,format="%.5f"))
l.append(st.number_input('abilities_number',1,5,format="%f"))
l.append(st.number_input('total_points',1,5,format="%.4f"))
l.append(st.number_input('catch_rate',2,20,format="%.6f"))
l.append(st.number_input('base_experience',1,10,format="%.6f"))
l.append(st.number_input('egg_type_number',1,5,format="%f"))
l.append(st.number_input('percentage_male',1,5,format="%.6f"))
l.append(st.number_input('egg_cycles',2,20,format="%.3f"))
target=['POKEMON IS LEGENDARY','POKEMON IS NOT LEGENDARY']
new=pickle.load(open('model3_pkl','rb'))
y_pred=new.predict([1])
y_pred=target[y_pred[0]]
st.title(y_pred)
