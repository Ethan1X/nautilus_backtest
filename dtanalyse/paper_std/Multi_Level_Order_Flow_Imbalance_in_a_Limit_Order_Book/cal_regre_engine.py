import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, recall_score, classification_report
from sklearn.linear_model import Ridge, RidgeClassifier
from tqdm import tqdm
import random


class CalRegreEngine:
    # calculate the mid-price 
    @staticmethod
    def cal_mid_price(depth_data):
        mid_price = (depth_data['bid1'] + depth_data['ask1']) / 2
        return mid_price


    # calculate the diff of mid-price
    @staticmethod
    def cal_mid_price_diff(depth_data, window_len=300, test_len=None):
        if test_len == None:
            test_len = depth_data.shape[0]
        else:
            if test_len > depth_data.shape[0]:
                test_len = depth_data.shape[0]
            else:    
                test_len = test_len
        mid_price = CalRegreEngine.cal_mid_price(depth_data[:test_len])
        diff = mid_price - mid_price.shift(1)
        mid_price_diff = pd.DataFrame(diff.groupby(np.arange(diff.shape[0]) // window_len).sum(), 
                                      columns=['mid_price_diff'])
        return mid_price_diff


    # calculate the generator
    @staticmethod
    def cal_spread(depth_data):
        spread = depth_data['ask1'] - depth_data['bid1']
        return spread


    # calculate the e-statistics
    @staticmethod
    def cal_e_stats(depth_data, M=1, test_len=None):
        if test_len == None:
            test_len = depth_data.shape[0]
        else:
            if test_len > depth_data.shape[0]:
                test_len = depth_data.shape[0]
            else:    
                test_len = test_len
        W_stats, V_stats = pd.DataFrame(np.zeros((test_len, M))), pd.DataFrame(np.zeros((test_len, M)))
        
        for i in tqdm(range(test_len)):
            for j in range(M):
                if i == 0:
                    W_stats.iloc[i, j], V_stats.iloc[i, j] = np.nan, np.nan
                else:
                    if depth_data["bid"+str(j+1)].iloc[i] > depth_data["bid"+str(j+1)].iloc[i-1]:
                        W_stats.iloc[i, j] = depth_data["bid_qty"+str(j+1)].iloc[i]
                    elif depth_data["bid"+str(j+1)].iloc[i] < depth_data["bid"+str(j+1)].iloc[i-1]:
                        W_stats.iloc[i, j] = -depth_data["bid_qty"+str(j+1)].iloc[i-1]
                    else:
                        W_stats.iloc[i, j] = depth_data["bid_qty"+str(j+1)].iloc[i] - depth_data["bid_qty"+str(j+1)].iloc[i-1]
                    
                    if depth_data["ask"+str(j+1)].iloc[i] > depth_data["ask"+str(j+1)].iloc[i-1]:
                        V_stats.iloc[i, j] = -depth_data["ask_qty"+str(j+1)].iloc[i-1]
                    elif depth_data["ask"+str(j+1)].iloc[i] < depth_data["ask"+str(j+1)].iloc[i-1]:
                        V_stats.iloc[i, j] = depth_data["ask_qty"+str(j+1)].iloc[i]
                    else:
                        V_stats.iloc[i, j] = depth_data["ask_qty"+str(j+1)].iloc[i] - depth_data["ask_qty"+str(j+1)].iloc[i-1]
        e_stats = W_stats - V_stats
        e_stats.index = depth_data.index[:test_len]
        return e_stats


    # calculate the MLOFI (source: https://doi.org/10.1142/S2382626619500114)
    @staticmethod
    def cal_mlofi(depth_data, M=1, window_len=300, test_len=None): # window_len mimics various options, e.g. 5s, 60s, 300s(5min), ...
        M = 5 if M > 5 else M
        e_stats = CalRegreEngine.cal_e_stats(depth_data, M, test_len=test_len)
        mlofi = e_stats.groupby(np.arange(e_stats.shape[0]) // window_len).sum()
        mlofi.columns = [f"mlofi{i+1}" for i in range(M)]
        return mlofi
    
    
    # OLS Regression
    @staticmethod
    def ols_regre(mid_price_diff, mlofi, lag=0, with_const=False):
        mid_price_diff = mid_price_diff.iloc[lag:]  # Only set the lag for the mid_price_diff
        yx_data = pd.concat([mid_price_diff, mlofi], axis=1, ignore_index=True)
        yx_data.rename(columns={"mlofi":"mlofi_lag"+str(lag)}, inplace=True)
        yx_data.dropna(inplace=True)
        if with_const:
            x_with_const = sm.add_constant(yx_data.iloc[:,1:])
            res = sm.OLS(yx_data.iloc[:,0], x_with_const).fit()
            return res
        else:
            res = sm.OLS(yx_data.iloc[:,0], yx_data.iloc[:,1:]).fit()
            return res
    

    # Calculate the best alpha for Ridge Regression
    @staticmethod
    def cal_best_params(engine, X_train, y_train, alphas, cv=5):
        model = engine
        param_grid = {'alpha': alphas}
        search_engine = GridSearchCV(model, param_grid, cv=5)
        search_engine.fit(X_train, y_train)
        return search_engine.best_params_
    
    
    # Calculate the adjusted-R2
    @staticmethod
    def cal_adj_r2(engine, X_test, y_test):
        r2 = engine.score(X_test, y_test)
        adjusted_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)
        return adjusted_r2
    
    
    # Ridge Regression
    @staticmethod
    def ridge_regre(mid_price_diff, mlofi, lag=0, alphas=np.linspace(0.01, 1000)):
        diff = mid_price_diff.iloc[lag:]
        yx_data = pd.concat([diff, mlofi], axis=1, ignore_index=True)
        yx_data.rename(columns={"mlofi": "mlofi_lag" + str(lag)}, inplace=True)
        yx_data.dropna(inplace=True)
        X_train, X_test, y_train, y_test = train_test_split(yx_data.iloc[:,1:], yx_data.iloc[:,0], 
                                                            test_size=0.2, random_state=42)
        
        if isinstance(alphas, np.ndarray):    # No alpha is given
            best_alpha = CalRegreEngine.cal_best_params(engine=Ridge(), X_train=X_train, y_train=y_train, 
                                                            alphas=alphas, cv=5)['alpha']
        elif isinstance(alphas, int) and alphas >= 0:  # Only a positive alpha is given
            best_alpha = alphas
        else:   # Other circumstances
            best_alpha = CalRegreEngine.cal_best_params(engine=Ridge(), X_train=X_train, y_train=y_train, 
                                                                    alphas=np.linspace(0.01, 1000), cv=5)['alpha']
            
        
        ridge = Ridge(alpha=best_alpha).fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        metrics = {
            "coef": ridge.coef_,
            "intercept": ridge.intercept_,
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)), 
            "adj-r2": CalRegreEngine.cal_adj_r2(ridge, X_test, y_test)
        }
        return best_alpha, metrics
    
    
    # Ridge Classifier Train Only
    @staticmethod
    def ridge_classifier_train(mid_price_diff, mlofi, first_bar, second_bar, alphas=None, lag=0, balance=False):
        diff = mid_price_diff.iloc[lag:]
        diff = diff[((diff < -np.abs(first_bar)) | (diff > np.abs(first_bar))) | 
                    ((diff > -np.abs(second_bar)) & (diff < np.abs(second_bar)))]
        diff["mid_price_diff"] = np.where(diff > np.abs(first_bar), 1, np.where(diff < -np.abs(first_bar), -1, 0))
        diff.index = np.arange(diff.shape[0])
        yx_data = pd.concat([diff, mlofi], axis=1).dropna()

        if balance:
            zero_num = (diff["mid_price_diff"].value_counts()[-1] + diff["mid_price_diff"].value_counts()[1]) / 2
            random_index = random.sample(diff[diff["mid_price_diff"] == 0].index.tolist(), zero_num.astype(int))
            diff_processed = pd.concat([diff.loc[random_index], diff[diff["mid_price_diff"] != 0]])
            yx_data = yx_data.loc[diff_processed.index]


        X_train, X_test, y_train, y_test = train_test_split(yx_data.iloc[:,1:], 
                                                            yx_data.iloc[:,0], 
                                                            test_size=0.2, 
                                                            random_state=42)
        if isinstance(alphas, np.ndarray):    # No alpha is given
            best_alpha = CalRegreEngine.cal_best_params(engine=RidgeClassifier(), X_train=X_train, y_train=y_train, 
                                                            alphas=alphas, cv=5)['alpha']
        elif isinstance(alphas, int) and alphas >= 0:  # Only a positive alpha is given
            best_alpha = alphas
        else:   # Other circumstances
            best_alpha = CalRegreEngine.cal_best_params(engine=RidgeClassifier(), X_train=X_train, y_train=y_train, 
                                                                    alphas=np.linspace(0.01, 1000), cv=5)['alpha']
            
        ridge = RidgeClassifier(alpha=best_alpha).fit(X_train, y_train)
        y_pred = ridge.predict(X_test)
        metrics = {
            "No. Observations": len(y_train) + len(y_test),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2_score": r2_score(y_test, y_pred),
            "report" : classification_report(y_test, y_pred, output_dict=True)
        }
        return best_alpha, ridge, metrics
    
    
    # Ridge Classifier Test Only
    @staticmethod
    def ridge_classifier_test(mid_price_diff, mlofi, first_bar, second_bar, ridge, lag=0):
        diff = mid_price_diff.iloc[lag:]
        diff = diff[((diff < -np.abs(first_bar)) | (diff > np.abs(first_bar))) | 
                    ((diff > -np.abs(second_bar)) & (diff < np.abs(second_bar)))]
        diff["mid_price_diff"] = np.where(diff > np.abs(first_bar), 1, np.where(diff < -np.abs(first_bar), -1, 0))
        diff.index = np.arange(diff.shape[0])
        yx_data = pd.concat([diff, mlofi], axis=1).dropna()
        
        X_test, y_test = yx_data.iloc[:,1:], yx_data.iloc[:,0]
        y_pred = ridge.predict(X_test)
        metrics = {
            "No. Observations": len(y_test),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
            "r2_score": r2_score(y_test, y_pred),
            "report" : classification_report(y_test, y_pred, output_dict=True)
        }
        return metrics, y_pred, y_test
