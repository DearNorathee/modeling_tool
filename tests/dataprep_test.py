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
        'real estate':282
        ,'life insurance':564
        ,'no known property':846
        ,'car':1128
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

    # this should throw an error
    strategy5 = {
        'one': 0.5,
        'two': 0.5
    }

    X_train_oversampled1, y_train_oversampled1 = upsampling(X_data,y_data,strategy=strategy1, concat=False)
    actual_counts1 = ds.value_counts(y_train_oversampled1, return_type=dict)
    X_train_oversampled2, y_train_oversampled2 = upsampling(X_data,y_data,strategy=strategy2, concat=False)
    actual_counts2 = ds.value_counts(y_train_oversampled2, return_type=dict)
    X_train_oversampled3, y_train_oversampled3 = upsampling(X_data,y_data,strategy=strategy3, concat=False)
    actual_counts3 = ds.value_counts(y_train_oversampled3, return_type=dict)
    X_train_oversampled4, y_train_oversampled4 = upsampling(X_data,y_data,strategy=strategy4, concat=False)
    actual_counts4 = ds.value_counts(y_train_oversampled4,return_type=dict)

    try:
        X_train_oversampled5, y_train_oversampled5 = upsampling(X_data,y_data,strategy=strategy5)
    except Exception as err05:
        assert isinstance(err05, ValueError)
    
    data_concat06 = upsampling(X_data,y_data,strategy=strategy4, concat=True)
    

    assert actual_counts1 == expect_count1, inp.assert_message(actual_counts1,expect_count1)
    assert actual_counts2 == expect_count2, inp.assert_message(actual_counts2,expect_count2)
    assert actual_counts3 == expect_count3, inp.assert_message(actual_counts3,expect_count3)
    assert actual_counts4 == expect_count4, inp.assert_message(actual_counts4,expect_count4)
    assert isinstance(data_concat06,pd.DataFrame)
    # check if column of the last column is still the same
    assert data_concat06.columns[-1] == y_name
    print()



def test_smote_upsampling():
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

    # this should throw an error
    strategy5 = {
        'one': 0.5,
        'two': 0.5
    }
    drop_cols = ["policy_bind_date","incident_location"]
    X_data_drop = X_data.drop(columns = drop_cols)
    X_data_drop = X_data_drop.fillna('Missing')
    X_train_oversampled1, y_train_oversampled1 = smote_nc_upsampling(X_data_drop,y_data,strategy=strategy1, concat=False)
    actual_counts1 = ds.value_counts(y_train_oversampled1, return_type=dict)
    X_train_oversampled2, y_train_oversampled2 = smote_nc_upsampling(X_data_drop,y_data,strategy=strategy2, concat=False)
    actual_counts2 = ds.value_counts(y_train_oversampled2, return_type=dict)
    X_train_oversampled3, y_train_oversampled3 = smote_nc_upsampling(X_data_drop,y_data,strategy=strategy3, concat=False)
    actual_counts3 = ds.value_counts(y_train_oversampled3, return_type=dict)
    X_train_oversampled4, y_train_oversampled4 = smote_nc_upsampling(X_data_drop,y_data,strategy=strategy4, concat=False)
    actual_counts4 = ds.value_counts(y_train_oversampled4,return_type=dict)

    try:
        X_train_oversampled5, y_train_oversampled5 = smote_nc_upsampling(X_data,y_data,strategy=strategy5)
    except Exception as err05:
        assert isinstance(err05, ValueError)
    
    data_concat06 = smote_nc_upsampling(X_data,y_data,strategy=strategy4, concat=True)
    

    assert actual_counts1 == expect_count1, inp.assert_message(actual_counts1,expect_count1)
    assert actual_counts2 == expect_count2, inp.assert_message(actual_counts2,expect_count2)
    assert actual_counts3 == expect_count3, inp.assert_message(actual_counts3,expect_count3)
    assert actual_counts4 == expect_count4, inp.assert_message(actual_counts4,expect_count4)
    assert isinstance(data_concat06,pd.DataFrame)
    # check if column of the last column is still the same
    assert data_concat06.columns[-1] == y_name
    print()

test_smote_upsampling()
test_upsampling_2()
test_upsampling()