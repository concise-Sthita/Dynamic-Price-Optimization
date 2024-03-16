import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import warnings
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings("ignore")
import time

REGIONS = ["Tier 1", "Tier 2", "Tier 3", "Rural"]

st.set_page_config(layout="wide")
custom_html = """
<div class="banner">
    <img src="https://srisriuniversity.edu.in/wp-content/uploads/2020/11/Untitled-design-min-1536x535.png" />
    <h1 class="heading">PRICEWISE</h1>
    <img src="https://transstadiainstitute.in/wp-content/uploads/ibm-ice-768x266.webp"/>
</div>
<style>
.heading {
    font-size:39px;
    color: white;
}

.banner {
    display: flex;
    justify-content: space-between;
    width: 100%;
}

img {
    width: 240px;
    background-color: white;
    z-index: 99999;
}
</style>
"""
st.components.v1.html(custom_html)

df = st.file_uploader("Choose a file")

@st.cache_data
def train_test_split_(products):
    X, y = products.drop(['product_id', 'unit_price'], axis=1), products['unit_price']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.20)
    return (X_train, X_test, y_train, y_test)

@st.cache_data
def load():
  
    global df
    if df is not None:
        df = pd.read_csv(df)
        return df
def deff(df):
    df['comp1_diff'] = df['unit_price'] - df['comp_1']
    df['comp2_diff'] = df['unit_price'] - df['comp_2']
    df['comp3_diff'] = df['unit_price'] - df['comp_3']
    df['fp1_diff'] = df['freight_price'] - df['fp1']
    df['fp2_diff'] = df['freight_price'] - df['fp2']
    df['fp3_diff'] = df['freight_price'] - df['fp3']
    
    cols_to_mean = ['product_id', 'comp1_diff', 'comp2_diff', 'comp3_diff',
                'fp1_diff', 'fp2_diff', 'fp3_diff', 'product_score', 'unit_price']
    cols_to_sum = ['product_id', 'total_price', 'freight_price', 'customers']
    mean_df = df[cols_to_mean]
    sum_df = df[cols_to_sum]
    products_mean = mean_df.groupby(by='product_id').mean(numeric_only=True)
    products_sum = sum_df.groupby(by='product_id').sum()
    products = pd.concat([products_sum, products_mean],
                     axis=1, join='inner').reset_index()
    return products


def show_elements():
    df = load()
    product_name = st.sidebar.multiselect("Product Category Name", [i for i in df['product_category_name'].unique()])
    region_type = st.sidebar.selectbox("Region Type", REGIONS)

    customer_id = st.sidebar.text_input("Customer ID", placeholder="c0000x", max_chars=6)
    
    dc = pd.read_csv("./consumer.csv")
    ls = 0
    
    
   

    if product_name:
        df2 = df[df['product_category_name'].isin(product_name)]
    
        products = deff(df)
        products2 = deff(df2)

        X, y = products.drop(['product_id', 'unit_price'], axis=1), products['unit_price']
        A, b = products2.drop(['product_id', 'unit_price'], axis=1), products2['unit_price']
        
        dtrain_reg = xgb.DMatrix(X, y, enable_categorical=True)
        dtest_reg = xgb.DMatrix(A, b, enable_categorical=True)

        model = xgb.train(params={"objective": "reg:linear"}, dtrain=dtrain_reg)
        y_pred = model.predict(dtest_reg)


        if region_type == "Tier 1":
            y_pred += (8/100)*y_pred
        if region_type == "Tier 2":
            y_pred += (4/100)*y_pred
        if region_type == "Tier 3":
            pass
        if region_type == "Rural":
            y_pred -= (4/100)*y_pred
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y, y=y_pred, mode='markers', 
                                marker=dict(color='blue'), 
                                name='Actual Retail Unit Price'))
        fig.add_trace(go.Scatter(x=[min(y), max(y)], y=[min(y), max(y)], 
                                mode='lines', 
                                marker=dict(color='red'), 
                                name='Prediction'))
        fig.update_layout(
            xaxis_title='Actual Retail Price',
            yaxis_title='Predicted Retail Price'
        )
        st.plotly_chart(fig)    
    else:
        products = deff(df)
        X_train, X_test, y_train, y_test = train_test_split_(products)

        
        dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
        dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)
        
        model = xgb.train(params={"objective": "reg:linear"},dtrain=dtrain_reg)
        y_pred = model.predict(dtest_reg)


        fig = go.Figure()
        fig.add_trace(go.Scatter(x=X_train, y=y_train, mode='markers', 
                                marker=dict(color='blue'), 
                                name='Actual Retail Unit Price'))
        fig.add_trace(go.Scatter(x=X_train, y=y_pred, mode='markers', 
                                marker=dict(color='red'), 
                                name='Predicted Retail Unit Price'))
       
        fig.update_layout(
            xaxis_title='Actual Retail Price',
            yaxis_title='Predicted Retail Price'
        )
        st.plotly_chart(fig)

    if customer_id:
        ls = dc.loc[dc['C_ID'] == customer_id]['L_Score'].values[0]
        
        if ls == 1:
            y_pred -= (2/100)*y_pred
        elif ls == 2:
            y_pred -= (4/100)*y_pred
        elif ls == 3:
            y_pred -= (5/100)*y_pred
        elif ls == 4:
            y_pred -= (6/100)*y_pred
        elif ls == 5:
            y_pred -= (8/100)*y_pred
        else:
            pass

if df is not None:
    show_elements()