import streamlit as st
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

header = st.container()
dataset = st.container()
features = st.container()
model_training = st.container()

st.markdown(
    """
    <style>
    .main {
    background-color: #D9EDBF;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_data   #to save the file result one time and not to run the whole program everytime we make any changes
def get_data(filename):
    food_data = pd.read_csv(filename)

    return food_data

with header:
    st.title('Welcome to my awesome data science project!')
    st.text('In this project I look into the transaction of taxis in NYC,...')

with dataset:
    st.header('Dhaka taxi dataet')
    st.text('I found this dataset on blabla.com,...')

    food_data = get_data('data/foodss.csv') #to read the csv file
    st.write(food_data.head(45)) #if head is empty then it will show first 5 rows

    st.subheader('Came by in a week distribution on the NYC dataset')
    customers = pd.DataFrame(food_data['Came by in a week'].value_counts()).head(20) #showing as a bar chat the LocationID
    st.bar_chart(customers)

with features:
    st.header('The features I created')

    st.markdown('* **First feature:** I created this feature because of this') #first star creates bullet point , the other four with no space creates bold character 
    st.markdown('* **Second feature:** I created this feature because of that')

with model_training:
    st.header('Time to train the model!')
    st.text('Here you get to choose the hyperparameters of the model and see how the parameters change!')

    sel_col, disp_col = st.columns(2) #making columns

    max_depth = sel_col.slider('What should be the max_depth of the model?', min_value=10, max_value=100, value=20, step=10) #slider as user input

    n_estimators = sel_col.selectbox('How many trees should there be?', options=[100,200,300,400,'No limit'], index=0) #estimator

    sel_col.text('Here is a list of features in my data:')
    sel_col.write(food_data.columns) 
    input_feature = sel_col.text_input('Which feature should be used as the input feature?','Spend') #user text input

    if n_estimators == 'No limit': #to make the 'No limit' work
        regr = RandomForestRegressor(max_depth=max_depth)

    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    X = food_data[[input_feature]]
    y = food_data[['Spend']]

    regr.fit(X, y)
    prediction = regr.predict(y)

    disp_col.subheader('Mean absolute error of the model is:')
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader('Mean squared error of the model is:')
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader('R squared score of the model is:')
    disp_col.write(r2_score(y, prediction))


