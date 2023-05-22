#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm


# In[2]:


st.title('BoomBikes Usage Prediction')


# In[3]:


st.sidebar.header('User Input Parameteres')


# In[4]:


st.subheader('Instructions to be followed in input data filling')


# In[5]:


st.text('Year:- 0:2018,1:2019;\n Holiday:- 0: Means its not an holiday, 1: Means its an holiday;\n Season:- 2:it means Spring, 1: it means winter;\n Weathersit:- 1:Good 3:Bad;\n Month:- 12:Dec, 7:july , 11:Nov , 9:Sep ;\n')


# In[6]:


def user_input_features():
  CONSTANT = st.sidebar.selectbox('Constant',('1.0'))
  YEAR = st.sidebar.selectbox('Year',('1.0','0'))
  HOLIDAY = st.sidebar.selectbox('Holiday',('1.0','0'))
  TEMPERATURE = st.sidebar.number_input('Insert the temperature')
  SPRING = st.sidebar.selectbox('Spring',('0','1'))
  WINTER = st.sidebar.selectbox('Winter',('1','0'))
  BAD = st.sidebar.selectbox('Bad', ('0','1'))
  GOOD = st.sidebar.selectbox('Good',('1','0'))
  DEC = st.sidebar.selectbox('Dec',('0','1'))
  JULY = st.sidebar.selectbox('July',('1','0'))
  NOV = st.sidebar.selectbox('Nov',('0','1'))
  SEP = st.sidebar.selectbox('Sep',('0','1'))
  data = { 'CONSTANT': CONSTANT,
            'YEAR': YEAR,
            'HOLIDAY': HOLIDAY,
            'TEMPERATURE': TEMPERATURE,
            'SPRING': SPRING,
            'WINTER': WINTER,
            'BAD': BAD,
            'GOOD': GOOD,
            'DEC':DEC,
            'JULY':JULY,
            'NOV':NOV,
            'SEP':SEP}
  features = pd.DataFrame(data,index=[0])
  return features   




# In[7]:


df = user_input_features()   
st.subheader('User Input Parameters')
st.write(df)


# In[8]:


data = pd.read_csv('/Users/apple/Downloads/day.csv',parse_dates=['dteday'])
#data= pd.read_csv('/content/day.csv',parse_dates=['dteday'])
data.drop(['instant','dteday','casual','registered'],axis=1,inplace=True)
data.season.replace((1,2,3,4),('Spring','Summer','Fall','Winter'),inplace=True)
data.mnth.replace((1,2,3,4,5,6,7,8,9,10,11,12),('Jan','Feb','March','April','May','June','July','Aug','Sept','Oct','Nov','Dec'),inplace=True)
data.weekday.replace((0,1,2,3,4,5,6),('Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday'),inplace=True)
data.weathersit.replace((1,2,3,4),('Good','Avg','Bad','Very Bad'),inplace=True)
season=pd.get_dummies(data['season'],drop_first=True) # Falls Drop Out 
weather=pd.get_dummies(data['weathersit'],drop_first=True) # Avg Dropout 
month=pd.get_dummies(data['mnth'],drop_first=True) # April Dropped 
weekday=pd.get_dummies(data['weekday'],drop_first=True) # Friday got Dropout
data_new=pd.concat([data,season,weather,month,weekday],axis=1)
data_new.drop(['season','mnth','weekday','weathersit'],axis=1,inplace=True)


# In[9]:


scaler=MinMaxScaler()
num_var=['temp','atemp','hum','windspeed','cnt']

data_new[num_var] = scaler.fit_transform(data_new[num_var])
y=data_new['cnt']
X=data_new
X = sm.add_constant(X)
X.drop(['workingday', 'Summer', 'Aug', 'Feb', 'Jan', 'June', 'March', 'May',
       'Oct', 'Monday', 'Saturday', 'Sunday', 'Thursday', 'Tuesday',
       'Wednesday','atemp','hum','windspeed','cnt'],axis=1,inplace=True)

lr=sm.OLS(y,X).fit()
y_pred=lr.predict(X)


# In[10]:


df1 = pd.DataFrame(data=df,columns=[ 'CONSTANT','YEAR','HOLIDAY','TEMPERATURE','SPRING','WINTER','BAD','GOOD','DEC','JULY','NOV','SEP' ], dtype=float)
print(df1)
y_pred_input = lr.predict(df1)


# In[11]:


st.subheader('Predicted Usage')
st.write(y_pred_input*100)

