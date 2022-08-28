import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title='Data analysis', page_icon='icon2.png')
st.write("# EDA")
plt.style.use('tableau-colorblind10')
# Interactive EDA
# Ideas
data = pd.read_csv('Diamonds Prices2022.csv', index_col=0)
st.write('## Raw Data')
st.dataframe(data.head(20))
st.write('### Number of unique values')
st.write(dict(data.nunique()))
st.write('### Unique values for Categorical Features')
st.write(dict(data[['cut', 'clarity', 'color']].apply(lambda x: list(x.unique()))))
st.write('''
**Price** is in dollars. x, y and z are length, width and depth respectively.
Rest columns are features of a diamond. Most of which are numeric in nature and some 
are categorical. Categorical features need to be converted to ordered categorical.  
* For Cut: Fair, Good, Very Good, Premium, Ideal
* For Clarity: I1, SI2, SI1, VS2, VS1, VVS2, VVS1, IF
* For Color: J, I, H, G, F, E, D (More values for color but these are the only present in data)  

You can read more about these features on the links given below:
* For clarity: https://www.diamonds.pro/education/clarity/
* For cut: https://www.diamonds.pro/education/cuts/
* For color: https://www.diamonds.pro/education/color/
''')

st.write('### Data Types')
st.write(dict(data.dtypes))
st.write('### Null Values')
st.write(dict(data.isnull().sum()))

ordinal_data = {'cut': ['Fair','Good','Very Good','Premium','Ideal'],
                    'color': ['J', 'I', 'H', 'G', 'F', 'E', 'D'],
                    'clarity': ['I1', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF']}

for var in ordinal_data:
    ordered_var = pd.api.types.CategoricalDtype(ordered = True,
                                                categories = ordinal_data[var])
    data[var] = data[var].astype(ordered_var)
st.write('### Data Description')
st.write(data.describe())

st.write('## Data Exploration')
st.write('### Univariate Exploration')
fig1, ax1 = plt.subplots(figsize=(12, 8))
bins = st.slider('Number of Bins', 250, 750, 250)
ax1.hist(data=data, x='price', bins=bins)
ax1.set_xlabel('Price in Dollars')
ax1.set_title('Price Distribution')
st.pyplot(fig1)

st.write('#### Categorical countplots') 
fig3, ax3 = plt.subplots(figsize=(12, 8))
col = st.selectbox('Select Categorical Column', ['Cut', 'Color', 'Clarity'])
sns.countplot(data=data, x=col.lower(), ax=ax3, color="#3281a8")
ax3.set_xlabel(None)
ax3.set_ylabel(None)
st.pyplot(fig3)

st.markdown('### Bi-variate Exploration')
st.markdown('#### Scatterplots')
fig4, ax4 = plt.subplots(figsize=(12, 8))
sns.scatterplot(x='carat', y='price', data=data, ax=ax4)
ax4.set_xlabel('Carat', fontsize=15)
ax4.set_ylabel('Price', fontsize=15)
ax4.set_title('Price vs. Carat', fontsize=20, pad=10)
st.pyplot(fig4)

numeric_vars = ['price', 'carat', 'depth', 'table', 'x', 'y', 'z']
categoric_vars = ['cut', 'color', 'clarity']
st.markdown('#### Heatmap Plot')
fig2, ax2 = plt.subplots(figsize=(12, 8))
sns.heatmap(data[numeric_vars].corr(), annot=True, fmt='.3f', cmap='rocket', ax=ax2)
ax2.set_title('Correlation Matrix', fontsize=20, pad=10)
st.pyplot(fig2)


