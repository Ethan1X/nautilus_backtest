# data_download.ipynb
- Fucntion: Download the specifically intraday depth data (LoadS3Data.get_cex_depth) into local folder "/data/btc_usdt" in a form of JSON, and transform into a DataFrame (convert_depth_format.convert_depth_format3)

# regre.ipynb
## (1) Generate the variables
- a. mid_price_diff: the difference of mid-price （the arithmetic average of ask1 price and bid1 price）within every 2 adjacent timestamps.
- b. MLOFI： a statistic calculated from the essay (source web: https://doi.org/10.1142/S2382626619500114)

## (2) Test the multicollinearity
- Method 1: Correlation Matrix
- Method 2: Eigenvalues
- Method 3: VIF (Variance Inflation Factor)

## (3) OLS Regression

## (4) Ridge Regression

# ridge_classifier.ipynb
## (1) Generate the variables
- Function: use RidgeClassifier model to classify the labels defined in the ridgeclassifier_res.md