import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle
import numpy as np
st.set_page_config(page_title='Diamond Price Prediction', page_icon='icon2.png')
st.write("# Diamond Price Prediction")

df = pd.read_csv('Diamonds Prices2022.csv')
df_num = df.drop(['cut', 'color', 'clarity', 'price'], axis=1)


cut_d = {col: index+1 for index, col in enumerate(['Fair','Good','Very Good','Premium','Ideal'])}
color_d = {col: index+1 for index, col in enumerate(['J', 'I', 'H', 'G', 'F', 'E', 'D'])}
clarity_d = {col: index+1 for index, col in enumerate(['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])}

st.sidebar.header('Specify Input Parameters')

def user_input():
    carat = st.sidebar.slider('Carat Value', min_value=df['carat'].min(), max_value=df['carat'].max(), step=df['carat'].mean(), value=0.23)
    cut = st.sidebar.selectbox('Cut Type', ['Fair','Good','Very Good','Premium','Ideal'])
    color = st.sidebar.selectbox('Color', ['J', 'I', 'H', 'G', 'F', 'E', 'D'])
    clarity = st.sidebar.selectbox('Clarity', ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF'])
    depth = st.sidebar.slider('Depth Value', min_value=df['depth'].min(), max_value=df.depth.max(), step=df.depth.mean(), value=61.0)
    table = st.sidebar.slider('Table Value', min_value=df.table.min(), max_value=df.table.max(), step=df.table.mean(), value=55.1)
    x = st.sidebar.slider('X (Length mm)', min_value=df['x'].min(), max_value=df['x'].max(), step=df['x'].mean(), value=3.85)
    y = st.sidebar.slider('Y (Width mm)', min_value=df['y'].min(), max_value=df['y'].max(), step=df['y'].mean(), value=4.23)
    z = st.sidebar.slider('Z (Depth mm)', min_value=df['z'].min(), max_value=df['z'].max(), step=df['z'].mean(), value=2.4)

    sample = pd.DataFrame({'carat':carat, 'cut':cut, 'color':color,
    'clarity':clarity, 'depth':depth, 'table':table, 'x':x, 'y':y, 'z':z}, index=[0])
    

    return sample

input_df = user_input()
st.header('Specified Input parameters')
st.write(input_df)

for col_d, cat_col in zip([cut_d, color_d, clarity_d],['cut', 'color', 'clarity']):
    input_df[cat_col] = input_df[cat_col].replace(col_d)

scaler = pickle.load(open('scaler.pkl', 'rb'))
input_data = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)


filename = 'dtmodel.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.predict(input_data)
st.write(f"The price of the diamond with selected features is {result[0]:.3f}")
st.markdown("""<iframe src="https://giphy.com/embed/kTUfHKuvw4juDpibQq" width="480" height="270" frameBorder="0" class="giphy-embed" allowFullScreen></iframe><p><a href="https://giphy.com/gifs/007-kTUfHKuvw4juDpibQq">via GIPHY</a></p>
""", unsafe_allow_html=True)