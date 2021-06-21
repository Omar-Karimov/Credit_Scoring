import streamlit as st
import pandas as pd
import numpy as np
import joblib
import base64
import os
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt


st.set_option('deprecation.showPyplotGlobalUse', False)



st.set_page_config( 
layout="wide",  
initial_sidebar_state="auto",
page_title= "Credit Scoring Model",  
page_icon= "Images/2.png", 
)


main_bg = "Images/Backs.png"
main_bg_ext = "Images/Backs.png"


st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)




st.sidebar.image("Images/logo.png",  width = 350)





# Collects user input features into dataframe
# Collects user input features into dataframe
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
  input_df = pd.read_csv(uploaded_file)
else:
  def user_input_features():
    CUS_GENDER = st.sidebar.selectbox("Gender",("Male","Female"))
    REQ_PURPOSE = st.sidebar.selectbox("Purpose",("Education","Office Equipment","Other","Personal Needs","Purchase of Car",
                                                     "Purchase of Electronics","Purchase of Furniture", "Repair", "Shopping",
                                                     "Travel", "Treatment", "Wedding"))
    FLAG_OWN_CAR = st.sidebar.selectbox("Own car",("N", "Y"))
    FLAG_OWN_REALTY = st.sidebar.selectbox("Own realty",("N", "Y"))
    
    NAME_CONTRACT_TYPE = st.sidebar.selectbox("Contract type",("Cash loans", "Revolving loans"))
    
    CREDIT_ACTIVE = st.sidebar.selectbox("Status of the Credit Bureau reported credits", ("Closed", "Active", "Sold"))
    NAME_INCOME_TYPE = st.sidebar.selectbox("Customer income type",("Working", "Commercial associate", "Pensioner", "State servant",
                                                               "Unemployed", "Student"))
    
    NAME_EDUCATION_TYPE = st.sidebar.selectbox("Level of highest education the customer achieved",("Secondary / secondary special", 
                                                                      "Higher education","Incomplete higher",
                                                                      "Lower secondary","Academic degree"))
    CUS_MARITIAL_STATUS = st.sidebar.selectbox("Marital status",("Divorced","Married","Single","Widow"))
    CUS_LIVE_PLACE = st.sidebar.selectbox("Place of residence",("Apartment", "House", "Other", "Parents Property",
                                                                    "Rented Apartment"))
    
    ORGANIZATION_TYPE = st.sidebar.selectbox("Type of organization where customer works",("Business Entity Type 3", 
                                                                                        "XNA", "Self-employed", "Other",
                                                                 "Medicine", "Business Entity Type 2", "Government", "School",
                                                                 "Trade: type 7", "Kindergarten", "Construction", "Business Entity Type 1",
                                                                 "Transport: type 4", "Trade: type 3", "Industry: type 9",
                                                                  "Industry: type 3", "Security", "Military", 
                                                                  "Housing", "Industry: type 11", "Bank", "Transport: type 2", 
                                                                 "Police", "Agriculture", "Security Ministries", "Postal",
                                                                 "Trade: type 2", "Services", "Restaurant", "Industry: type 7", 
                                                                  "University", "Hotel", "Transport: type 3", "Industry: type 1",
                                                                 "Electricity", "Industry: type 4", "Trade: type 6", "Telecom",
                                                                 "Emergency", "Industry: type 5", "Insurance", "Industry: type 2",
                                                                 "Advertising", "Industry: type 12", "Culture", "Trade: type 1",
                                                                 "Mobile", "Realtor", "Legal Services", "Transport: type 1",
                                                                 "Cleaning", "Industry: type 10", "Religion", "Industry: type 13",
                                                                 "Industry: type 6", "Trade: type 5", "Industry: type 8",
                                                                 "Trade: type 4"))
        
    OWN_CAR_AGE = st.sidebar.slider("Age of customer's car", 0, 100, 20)
    DAYS_ID_PUBLISH = st.sidebar.slider("How many days before the application did customer change the identity document ?", 0, 7000, 20)
    CNT_CREDIT_PROLONG = st.sidebar.slider("How many times was the Credit Bureau credit prolonged ?", 0, 5, 1)
    REQ_RATE = st.sidebar.slider("Current required credit rate %", 1.0,40.0,18.0)
    CUS_CHILDS = st.sidebar.slider("Number of children the customer has", 0, 5, 1)
    CUS_JOB_ALL_STAJ = st.sidebar.slider("Experience of the client (monthly)", 0, 1000, 1)
    fifd_value = st.sidebar.slider("FIFD", 0.0, 50.0, 16.0)
    REQ_AMOUNT = st.sidebar.text_input("Current required credit amount", "1000")
    REQ_PERIOD = st.sidebar.text_input("Credit period (Monthly)", "18")
    V_CUS_AGE = st.sidebar.text_input("Age of the customer", "42")
    JOB_INCOME = st.sidebar.text_input("Customer income", "1500")
    DTI_VALUE = st.sidebar.text_input("DTI value", "34")
    AMT_CREDIT_SUM = st.sidebar.text_input("Total available credit which defined from Credit Bureau for a customer ", "1000")
    AMT_CREDIT_SUM_DEBT = st.sidebar.text_input("Current debt on Credit Bureau credit", "5000")
    AMT_CREDIT_SUM_LIMIT = st.sidebar.text_input("Current credit limit of a credit card reported in Credit Bureau", "1000")
    
    
    data = {"CUS_GENDER": CUS_GENDER,
            "REQ_PURPOSE": REQ_PURPOSE,
            "NAME_CONTRACT_TYPE": NAME_CONTRACT_TYPE,
            "CNT_CREDIT_PROLONG": CNT_CREDIT_PROLONG,
            "OWN_CAR_AGE" : OWN_CAR_AGE,
            "AMT_CREDIT_SUM_DEBT" : AMT_CREDIT_SUM_DEBT,
            "AMT_CREDIT_SUM" : AMT_CREDIT_SUM,
            "AMT_CREDIT_SUM_LIMIT" : AMT_CREDIT_SUM_LIMIT,
            "DAYS_ID_PUBLISH":DAYS_ID_PUBLISH,
            "NAME_EDUCATION_TYPE": NAME_EDUCATION_TYPE,
            "FLAG_OWN_CAR":FLAG_OWN_CAR,
            "FLAG_OWN_REALTY": FLAG_OWN_REALTY,
            "CREDIT_ACTIVE": CREDIT_ACTIVE,
            "NAME_INCOME_TYPE": NAME_INCOME_TYPE,
            "ORGANIZATION_TYPE":ORGANIZATION_TYPE,
            "CUS_MARITIAL_STATUS": CUS_MARITIAL_STATUS,
            "CUS_LIVE_PLACE": CUS_LIVE_PLACE,
            "REQ_RATE": REQ_RATE,
            "CUS_CHILDS": CUS_CHILDS,
            "CUS_JOB_ALL_STAJ": CUS_JOB_ALL_STAJ,
            "fifd_value": fifd_value,
            "REQ_AMOUNT": REQ_AMOUNT,
            "REQ_PERIOD": REQ_PERIOD,
            "V_CUS_AGE": V_CUS_AGE,
            "JOB_INCOME" : JOB_INCOME,
            "DTI_VALUE": DTI_VALUE,
               }
    features = pd.DataFrame(data, index=[0])
    return features
  input_df = user_input_features()
    
    
# Combines user input features with entire dataset
df_raw = pd.read_csv("final_csv")
data = df_raw.drop(columns = ["TARGET","Unnamed: 0"])
df = pd.concat([input_df,data],axis=0)


# Create a label encoder object
le = LabelEncoder()

# Iterate through the columns
encode = ["FLAG_OWN_CAR", "FLAG_OWN_REALTY", "CUS_GENDER", "NAME_CONTRACT_TYPE"]

for col in encode:
    df[col] = le.fit_transform(df[col])
 

 #Encoding of ordinal features
labels = ["NAME_EDUCATION_TYPE", "NAME_INCOME_TYPE", "ORGANIZATION_TYPE", "CUS_LIVE_PLACE",
          "CREDIT_ACTIVE", "REQ_PURPOSE", "CUS_MARITIAL_STATUS"]

for col in labels:
    dummy = pd.get_dummies(df[col], prefix = col)
    df = pd.concat([df,dummy], axis = 1)
    del df[col]


df = df[:1] # Selects only the first row (the user input data)
    
# Displays the use input features
st.write(""" <p style=" color:#1567e8; font-size: 30px; font-weight:bold"> User Input Features</p>
  """, unsafe_allow_html=True)

st.write("")

if uploaded_file is not None:
    st.write(df)
else:
    st.write("Awaiting CSV file to be uploaded. Currently using example input parameters (shown below).")
    st.write(df)

st.write(" ")
    
    
    
    
# Reads in saved classification model
rf = joblib.load("rf_model")

# Apply model to make predictions
prediction = rf.predict(df)
prediction_proba = rf.predict_proba(df)

st.write(" ")

# Prediction
st.write(""" <p style=" color:#1567e8; font-size: 30px; font-weight:bold">Prediction</p>
  """, unsafe_allow_html=True)
customer_types = np.array([0,1])
st.write(customer_types[prediction])

st.write(" ")

if customer_types[prediction] == 0:
    st.info("Customer is not in the risk group")
else:
    st.error("Customer is in the risk group")


st.write(" ")

st.write(""" <p style=" color:#1567e8; font-size: 30px; font-weight:bold">Prediction Probability</p>
  """, unsafe_allow_html=True)
st.write(prediction_proba)

st.write(" ")

st.write(""" <p style=" color:#1567e8; font-size: 30px; font-weight:bold">Evaluation Scores</p>
  """, unsafe_allow_html=True)

eva1 = pd.read_csv("evaluation_table.csv")
eva2 = pd.read_csv("evaluation_table2.csv")
col1, col2 = st.beta_columns(2)
col1.write(eva1)
col2.write(eva2)

st.write(" ")



if customer_types[prediction] == 0:
    if  0.50 <= np.max(prediction_proba) <= 0.60:
        st.info("Fair")
    elif  0.60 < np.max(prediction_proba) <= 0.70:
        st.info("Good")
    elif 0.70 < np.max(prediction_proba) <= 0.80:
        st.info("Very good")
    else:
        st.info("Excellent")
        
elif customer_types[prediction] == 1:
    if 0.50 <= np.max(prediction_proba) <= 0.60:
        st.error("Bad")
    elif 0.60 < np.max(prediction_proba) <= 0.70:
        st.error("Risky")
    elif 0.70 < np.max(prediction_proba) <= 0.80:
        st.error("Higher Risk")
    else:
        st.error("Terrible")
        
        
      

    

st.write(" ")

st.write(""" <p style=" color:#1567e8; font-size: 30px; font-weight:bold">Feature importances</p>
  """, unsafe_allow_html=True)

feat_importances = pd.Series(rf.feature_importances_, index=df.columns)
feat_importances.nlargest(15).plot(kind='barh', figsize = (12,5))
st.pyplot()
      
