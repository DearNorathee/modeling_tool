import pandas as pd
from modeling_tool.dataprep import *
import dataframe_short as ds
import inspect_py as inp

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
    expect_count = {
        'real estate':1128
        ,'life insurance':846
        ,'no known property':564
        ,'car':282
        }
    X_train_oversampled, y_train_oversampled = upsampling(X_data,y_data,strategy=strategy)
    actual_count = ds.value_counts(y_train_oversampled, return_type=dict)
    assert actual_count == expect_count, inp.assert_message(actual_count,expect_count)
    print()

def test_upsampling_2():
    df_path = r"C:\Users\Heng2020\OneDrive\D_Code\Python\Python Modeling\Modeling 01\Dataset Binary Classification\02 InsuranceFraud_train.csv"
    df = pd.read_csv(df_path)
    y_name = 'fraud_reported'
    X_data = df.drop(columns = [y_name])
    y_data = df[y_name]
    strategy1 = 'equal'

    strategy2 = {
        1:0.5,
        0:0.5
        }
    
    strategy3 = {
        1:0.2,
        0:0.8
        }
    
    strategy4 = {
        1:0.3,
        0:0.7
        }

    expect_count1 = {
        1: 519,
        0: 519
    }

    expect_count2 = {
        1: 519,
        0: 519
    }
    expect_count3 = {
        1: 181,
        0: 724
    }
    expect_count4 = {
        1: 223,
        0: 519
    }

    X_train_oversampled1, y_train_oversampled1 = upsampling(X_data,y_data,strategy=strategy1)
    actual_counts1 = ds.value_counts(y_train_oversampled1, return_type=dict)
    X_train_oversampled2, y_train_oversampled2 = upsampling(X_data,y_data,strategy=strategy2)
    actual_counts2 = ds.value_counts(y_train_oversampled2, return_type=dict)
    X_train_oversampled3, y_train_oversampled3 = upsampling(X_data,y_data,strategy=strategy3)
    actual_counts3 = ds.value_counts(y_train_oversampled3, return_type=dict)
    X_train_oversampled4, y_train_oversampled4 = upsampling(X_data,y_data,strategy=strategy4)
    actual_counts4 = ds.value_counts(y_train_oversampled4,return_type=dict)

    assert actual_counts1 == expect_count1, inp.assert_message(actual_counts1,expect_count1)
    assert actual_counts2 == expect_count2, inp.assert_message(actual_counts2,expect_count2)
    assert actual_counts3 == expect_count3, inp.assert_message(actual_counts3,expect_count3)
    assert actual_counts4 == expect_count4, inp.assert_message(actual_counts4,expect_count4)
    print()

test_upsampling_2()
test_upsampling()