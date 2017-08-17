import pandas as pd
import numpy as np

def make_baseline(df_train, df_test, out):
    mean_price = df_train.SalePrice.mean()
    with open(out, 'w') as f:
        f.write('SalesID,SalePrice\n')
        for sale_id in df_test.index:
            f.write(str(sale_id)+','+str(mean_price)+'\n')

def score(y_true, y_pred):
    log_diff = np.log(y_pred+1) - np.log(y_true+1)
    return np.sqrt(np.mean(log_diff**2))

def evaluate(outfile):
    df_pred = pd.read_csv(outfile)
    df_pred.set_index('SalesID')
    df_true = pd.read_csv('data/do_not_open/test_soln.csv')
    df_true.set_index('SalesID')
    return score(df_true.SalePrice, df_pred.SalePrice)

BASELINE_OUT = 'out_baseline.csv'
PREDICTION_OUT = 'out_prediction.csv'

df_train = pd.read_csv('data/Train.csv')
df_test = pd.read_csv('data/Test.csv')
df_train.set_index('SalesID', inplace=True)
df_test.set_index('SalesID', inplace=True)

if __name__ == '__main__':
    make_baseline(df_train, df_test, BASELINE_OUT)

    # RMSLE score for just predicting mean price is 0.747448469261
    print evaluate(BASELINE_OUT)
