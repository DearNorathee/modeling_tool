import pandas as pd
from modeling_tool.dataprep import *

def test_upsampling():
    df_path = r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Modeling\Modeling 01\Dataset Classification\04 Credit Risk Customer.csv"
    df = pd.read_csv(df_path)
    y_name = 'property_magnitude'
    X_data = df.drop(columns = [y_name])
    y_data = df[y_name]
    strategy = {
        'real estate':1
        ,'life insurance':2
        ,'no known property':3
        ,'car':4
        }
    
    X_train_oversampled, y_train_oversampled = upsampling(X_data,y_data,strategy=strategy)
    print()

test_upsampling()