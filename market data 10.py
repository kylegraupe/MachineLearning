# THIS FILE TAKES IN HISTORICAL OPTIONS DATA FOR $AAPL, CONVERTS IT TO PANDAS DATAFRAME, IMPUTES ZEROES FOR MISSING
# VALUES, USES MINMAXSCALER AND PREDICTS PUT PREMIUMS.

# Revision Updates:
# Included Binomial Model for Puts in training set. This creates two different options for the price at expiration
# based on variables. Maybe try a binary classifier for in the money vs. out of the money?

# Notes:
# an underscore is used to denote the variables that are used as inputs in the functions so the variables are not
# disturbed later in the program.
# Many of the 'actual' values are zeros as you can see in the output plot 'actual vs. predicted'. Later iterations may
# try to reduce the scope of the program to eliminate the strikes that will likely end without value (larger than some
# probability threshold)
# Maybe also try to predict future volatility rather than the actual price of the premium.

# Dropped options past 60 days exp.
# If negative prediction does that mean it will expire worthless (essentially a zero prediction?)
# How does a stock split affect this program?

# Need to incoporate current price and price at expiration in dataset. Currently not predicting the price at a point,
# but what the current price should be.

# Incorporate price at exp in the dataset and binomial model in the dataset. maybe one column for each option (up/down)

#


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from yahoo_fin import options  # Also need to install 'html5lib' package
from yahoo_fin import stock_info as sti
import datetime
import scipy.stats as si
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from scipy.stats import gaussian_kde


def strike_index(chain_, strike_price_):
    """This function finds the index of the strike price input. This will be used to find all other values in the option
    chain for the corresponding strike price."""
    strike_ix = int(chain_[chain_["Strike"] == strike_price_].index.values)
    return strike_ix


def black_scholes(month_, day_, year_, live_price_, strike_price_, iv_float_, risk_free_rate_):
    """The Black-Scholes model is a series of equations used for pricing options contracts and other derivatives using
    time and other variables. This function calculates the greeks for an individual option contract and returns the fair
    market price given its inputs."""
    today = datetime.date.today()
    future = datetime.date(year_, month_, day_)
    left = future - today
    days_left = left.days
    time_to_exp_yrs = days_left / 365

    this_year = today.year
    this_month = today.month
    this_day = today.day

    start = datetime.date(this_year, this_month, this_day)
    exp = datetime.date(year_, month_, day_)
    bus_days = np.busday_count(start, exp)

    # Variables for Black Scholes equations. Used for fair price premium calc and greeks calcs.
    S = float(live_price_.copy())
    K = float(strike_price_)
    sigma = float(iv_float_)
    r = float(risk_free_rate_)
    T = float(time_to_exp_yrs)
    q = 0

    d1 = (np.log(S / K) + (sigma ** 2) / 2) / sigma
    d2 = ((np.log(S / K) + (r - q - (sigma ** 2) / 2)) * T) / (sigma * np.sqrt(T))
    # d1_exp = (np.log(S / K) + (sigma ** 2) / 2) / sigma
    # d2_exp = ((np.log(S / K) + (r - q - (sigma ** 2) / 2)) * 0) / (sigma * np.sqrt(0))

    # Greeks
    delta = -si.norm.cdf(-d1, 0, 1)  # Delta for put = -Phi(-d1)
    gamma = si.norm.pdf(d1, 0, 1) / (S * sigma * np.sqrt(T))
    vega = S * si.norm.pdf(d1, 0, 1) * np.sqrt(T)
    theta = (- S * si.norm.pdf(d1, 0, 1) * sigma) / (2 * T) + (r * K * np.exp(-r * T) * si.norm.cdf(-d2, 0, 1))
    rho = -K * T * np.exp(-r * T) * si.norm.cdf(-d2, 0, 1)

    # Black Scholes equation for fair put price
    put_fair = K * np.exp(-r * T) * si.norm.cdf(-d2) - (S * si.norm.cdf(-d1))
    # put_fair_exp = K * np.exp(-r * 0) * si.norm.cdf(-d2_exp, 0, 1) - S * si.norm.cdf(-d1_exp, 0, 1)
    return put_fair, delta, gamma, vega, theta, rho, days_left, bus_days


def inputs(chain_, strike_ix_, strike_price_):
    """This function pulls values from the option chain for a given strike price and expiration date."""
    bid_col_ix = chain_.columns.get_loc("Bid")
    bid = chain_.iat[strike_ix_, bid_col_ix]

    ask_col_ix = chain_.columns.get_loc("Ask")
    ask = chain_.iat[strike_ix_, ask_col_ix]

    last_col_ix = chain_.columns.get_loc("Last Price")
    last_premium = chain_.iat[strike_ix_, last_col_ix]

    volume_col_ix = chain_.columns.get_loc("Volume")
    volume = float(chain_.iat[strike_ix_, volume_col_ix])

    int_value = abs(live_price - strike_price_)
    ext_value = abs(last_premium - int_value)
    strike_dist_pct = (strike_price_ - live_price) / live_price

    return bid, ask, volume, int_value, ext_value, strike_dist_pct


def bin_model_inputs(S_, K_, Up_, Dn_, prob_up_, prob_dn_, rfr_):
    """This function is a work in progress implementation of the binary model used to price american derivatives using
    a binary tree approach."""
    exp_up = S_ * Up_
    exp_dn = S_ * Dn_

    put_up = max(0, K_ - exp_up)
    put_dn = max(0, K_ - exp_dn)

    put_exp = (put_up * prob_up_) + (put_dn * prob_dn_)
    p_0 = put_exp / (1 + rfr_)
    return put_up, put_dn, p_0


# PREPARE THE DATA

# Drop columns. I have decided these features are not important, or are represented in a similar column.

def drop_columns(df):
    """This function removes columns that are unnecessary inputs for the model or have low correlation to the premium.
    This step will be revisited using other dimensionality reduction algorithms for regression tasks."""
    df = df.drop("[QUOTE_UNIXTIME]", axis=1)
    df = df.drop("[QUOTE_READTIME]", axis=1)
    df = df.drop("[QUOTE_DATE]", axis=1)
    df = df.drop("[QUOTE_TIME_HOURS]", axis=1)
    # df = df.drop("[EXPIRE_DATE]", axis=1)
    df = df.drop("[EXPIRE_UNIX]", axis=1)
    df = df.drop("[STRIKE_DISTANCE]", axis=1)
    df = df.drop("[C_SIZE]", axis=1)
    df = df.drop("[P_SIZE]", axis=1)
    df = df.drop("[C_DELTA]", axis=1)
    df = df.drop("[C_GAMMA]", axis=1)
    df = df.drop("[C_VEGA]", axis=1)
    df = df.drop("[C_THETA]", axis=1)
    df = df.drop("[C_RHO]", axis=1)
    df = df.drop("[C_IV]", axis=1)
    df = df.drop("[C_VOLUME]", axis=1)
    df = df.drop("[C_LAST]", axis=1)
    df = df.drop("[C_BID]", axis=1)
    df = df.drop("[C_ASK]", axis=1)
    df = df.drop("[P_THETA]", axis=1)
    df = df.drop("[P_GAMMA]", axis=1)
    df = df.drop("[P_VEGA]", axis=1)
    return df


def feature_extraction(market_data_, p_up_, p_dn_, U_, D_, r_):
    """This function extracts new features from the original chain, including intrinsic and extrinsic values and columns
    to be used in the binary model."""
    # Create new data for predictions. We will be predicting the premium of Puts.
    market_data_['P_int_value'] = abs(market_data_['[UNDERLYING_LAST]'] - market_data_['[STRIKE]'])
    market_data_['P_ext_value'] = abs(market_data_['[P_LAST]'] - market_data_['P_int_value'])
    market_data_['exp_up'] = market_data_['[UNDERLYING_LAST]'] * U_
    market_data_['exp_dn'] = market_data_['[UNDERLYING_LAST]'] * D_
    market_data_['up_temp'] = market_data_['[STRIKE]'] - market_data_['exp_up']
    market_data_['dn_temp'] = market_data_['[STRIKE]'] - market_data_['exp_dn']
    market_data_['zero'] = 0
    market_data_['put_up'] = market_data_[['up_temp', 'zero']].max(axis=1)
    market_data_['put_dn'] = market_data_[['dn_temp', 'zero']].max(axis=1)
    market_data_['put_exp'] = (market_data_['put_up'] * p_up_) + (market_data_['put_dn'] * p_dn_)
    market_data_['p_0'] = market_data_['put_exp'] / r_
    return market_data_


def remove_long_exp(market_data_, col_name='[DTE]', threshold=60):
    """This function removes options contracts that have long term expiration dates. Given the scope of this project is
    to not predict the long term performance of the underlying stock, we drop any contract outside of a threshold of 60
    days."""
    return market_data_[market_data_[col_name] < threshold]


def expected_move_threshold_pct():
    """This function returns an expected move threshold based on historical volatility. It is a work in progress."""
    expected_pct = 0.10  # For now, working with a 10% threshold for price action moves. (underlying will not move more than 10% in a period)
    return expected_pct


def add_binary_class(market_data_):
    market_data_['[EXPIRE_DATE]'].astype(str)
    market_data_['[EXPIRE_DATE]'] = pd.to_datetime(market_data_['[EXPIRE_DATE]'], format='%Y/%m/%d')
    market_data_['[EXPIRE_DATE]'].astype(str)
    exp_list = market_data_['[EXPIRE_DATE]'].to_list()
    a = []
    for x in exp_list:
        if x not in a:
            a.append(x)

    market_data_['itm_exp'] = 0
    market_data_.sort_values(by=['[EXPIRE_DATE]', '[DTE]'], ascending=False)
    market_data_.dropna()

    strike_col_ix = market_data_.columns.get_loc("[STRIKE]")
    dte_col_ix = market_data_.columns.get_loc("[DTE]")
    p_last_col_ix = market_data_.columns.get_loc("[P_LAST]")
    exp_col_ix = market_data_.columns.get_loc("[EXPIRE_DATE]")
    itm_col_ix = market_data_.columns.get_loc("itm_exp")

    for i in range(len(market_data_['[EXPIRE_DATE]'])):
        if market_data_.iat[i, dte_col_ix] == 0 and market_data_.iat[i, p_last_col_ix] > 0.01:
            market_data_.iat[i, itm_col_ix] = 1
        else:
            market_data_.iat[i, itm_col_ix] = 0

    for j in range(len(market_data_['[EXPIRE_DATE]'])):
        for k in range(j + 1, len(market_data_['[EXPIRE_DATE]'])):
            if market_data_.iat[j, exp_col_ix] == market_data_.iat[k, exp_col_ix] and market_data_.iat[
                j, strike_col_ix] == market_data_.iat[k, strike_col_ix] and market_data_.iat[k, dte_col_ix] > 0 and \
                    market_data_.iat[j, itm_col_ix] == 1:
                market_data_.iat[k, itm_col_ix] = 1
                break
            else:
                market_data_.iat[k, itm_col_ix] = 0

    return market_data_


# def remove_far_out_of_money(market_data_, col_name='[UNDERLYING_LAST]', threshold=live_price - (live_price*expected_pct)):
#     return market_data_[market_data_[col_name] > threshold]


# # market_data = remove_far_out_of_money(market_data)
# print(market_data.info())


def train_val(market_data_):
    """This function takes in the processed market data and creates training and validation sets. It then copies the
    training sets as to not taint the original dataset during training."""
    # Create X and y dataset for splitting the data into test and trains sets. X for train and y for validation.
    X = market_data_.iloc[:, [0, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 13, 19, 20, 22]]
    y = market_data_.iloc[:, [6]]  # Target data for P_Last

    # Split market data into train and test (validation)
    X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=1)

    # Work on copies to not disturb original dataset
    market_data_X_train_ = X_train.copy()
    market_data_X_val_ = X_validation.copy()
    market_data_y_train_ = y_train.copy()
    market_data_y_val_ = y_validation.copy()
    return market_data_X_train_, market_data_X_val_, market_data_y_train_, market_data_y_val_, X_train, X_validation, y_train, y_validation


def data_prep(market_data_X_train_, market_data_X_val_, market_data_y_train_, market_data_y_val_, user_array_):
    """This function performs processing on the market data. It drops rows where there are missing values and scales and
    fits the training data. It is in progress to determine whether missing values should be dropped or filled with a
    zero value. Performance during validation will determine this. The working datasets are then converted back into
    pandas dataframes. It also reshapes the user inputs and creates a pandas dataframe."""
    # Drop rows where values are missing.
    market_data_X_train_.dropna()
    market_data_y_train_.dropna()
    market_data_X_val_.dropna()
    market_data_y_val_.dropna()

    imputer = SimpleImputer(missing_values=np.nan, strategy="constant",
                            fill_value=0)  # replace the missing value with the zero
    imputer.fit(market_data_X_train_)
    imputer.fit(market_data_X_val_)

    imputer_2 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    imputer_2.fit(market_data_y_train_)
    imputer_2.fit(market_data_y_val_)

    # Scale data
    scaler = MinMaxScaler()
    a = scaler.fit_transform(market_data_X_train_.to_numpy())
    b = scaler.fit_transform(market_data_X_val_.to_numpy())
    c = scaler.fit_transform(market_data_y_train_.to_numpy())
    d = scaler.fit_transform(market_data_y_val_.to_numpy())

    # Convert X train and validation back into pandas dataframe
    blah = imputer.transform(a)
    blah2 = imputer.transform(b)
    blah3 = imputer_2.transform(c)
    blah4 = imputer_2.transform(d)

    market_data_transform_X_train_ = pd.DataFrame(blah, columns=X_train.columns, index=X_train.index)
    market_data_transform_X_val_ = pd.DataFrame(blah2, columns=market_data_X_val.columns, index=market_data_X_val.index)
    market_data_transform_y_train_ = pd.DataFrame(blah3, columns=y_train.columns, index=y_train.index)
    market_data_transform_y_val_ = pd.DataFrame(blah4, columns=y_validation.columns, index=y_validation.index)

    # Create dataframe from user array
    reshaped_array = np.reshape(user_array_, (1, 15))
    e = scaler.fit_transform(reshaped_array)
    user_df_ = pd.DataFrame(e, columns=market_data_X_train_.columns)

    return market_data_transform_X_train_, market_data_transform_X_val_, market_data_transform_y_train_, market_data_transform_y_val_, user_df_


def mlp(market_data_transform_x_train_, market_data_transform_x_val_, market_data_transform_y_train_, user_df_):
    """This function uses a MultiLayer Perceptron with a grid search for determining model hyperparameters."""

    param_grid_mlp = [
        {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'],
         'shuffle': [True], 'learning_rate': ['invscaling', 'constant', 'adaptive'],
         'max_iter': [5000]}]
    # Expand for grid search
    # mlp_grid = MLPRegressor()
    # grid_search_mlp = GridSearchCV(mlp_grid, param_grid_mlp, cv=3, verbose=2, scoring='neg_mean_squared_error').fit(market_data_transform_X_train, market_data_transform_y_train.values.ravel())
    # print("-----------------------")
    # print("Best params (MLP): ")
    # print(grid_search_mlp.best_params_)
    # print("Best estimator (MLP): ")
    # print(grid_search_mlp.best_estimator_)
    mlp = MLPRegressor(activation='relu', learning_rate='invscaling', max_iter=5000, shuffle=True, solver='adam').fit(
        market_data_transform_x_train_, market_data_transform_y_train_.values.ravel())
    mlp_predictions_ = mlp.predict(market_data_transform_x_val_)
    mlp_user_prediction_ = mlp.predict(user_df_)
    return mlp_predictions_, mlp_user_prediction_


def errors(mlp_predictions_, market_data_transform_y_val_):
    """This function calculates the mean absolute error, mean squared error, r^2, and root MSE values."""
    mae_mlp_ = mean_absolute_error(market_data_transform_y_val_.values.ravel(), mlp_predictions_.ravel())
    mse_mlp_ = mean_squared_error(market_data_transform_y_val_.values.ravel(), mlp_predictions_.ravel())
    r2_mlp_ = r2_score(market_data_transform_y_val_.values.ravel(), mlp_predictions_.ravel())
    final_mse_mlp_ = mean_squared_error(market_data_transform_y_val_.values.ravel(), mlp_predictions_.ravel())
    final_rmse_mlp_ = np.sqrt(final_mse_mlp_)
    return mae_mlp_, mse_mlp_, r2_mlp_, final_rmse_mlp_


def plots(mlp_predictions_, market_data_transform_y_val_):
    """This function plots the ... is a work in progress."""
    fig, ax = plt.subplots()
    # xy = np.vstack([mlp_predictions_, market_data_transform_y_val_.values.ravel()])
    # z = gaussian_kde(xy)(xy)
    ax.scatter(mlp_predictions_, market_data_transform_y_val_, edgecolors=(0, 0, 1), alpha=0.1)
    ax.plot([mlp_predictions_.min(), mlp_predictions_.max()],
            [market_data_transform_y_val_.min(), market_data_transform_y_val_.max()], 'r--', lw=3)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    # plt.plot(sgd_predictions, market_data_transform_X_val["[DTE]"])
    plt.show()
    # scatter_matrix(market_data_transform_X_val, grid=True)  # Trying a smaller subset for scatter matrix


if __name__ == '__main__':
    # USER: Only input values here.
    ticker = "aapl"
    strike_price = 165  # Choose the strike price you wish to analyze. Cannot be less than 10% below current price.
    risk_free_rate = 0.015  # Found online
    p_up = 0.6
    p_dn = 0.4
    U = 1.25
    D = 1 / U

    month = 11
    day = 12
    year = 2021

    exp_date = str(month) + "/" + str(day) + "/" + str(year)
    chain = options.get_puts(ticker, exp_date)
    live_price = sti.get_live_price(ticker)

    strike_ix = strike_index(chain, strike_price)

    iv_col_ix = chain.columns.get_loc("Implied Volatility")
    iv_str = chain.iat[strike_ix, iv_col_ix]
    imp_vol_str_drop = iv_str.replace("%", "")
    iv_float = float(imp_vol_str_drop) / 100

    put_fair, delta, gamma, vega, theta, rho, days_left, bus_days = black_scholes(month, day, year, live_price,
                                                                                  strike_price,
                                                                                  iv_float, risk_free_rate)

    bid, ask, volume, int_value, ext_value, strike_dist_pct = inputs(chain, strike_ix, strike_price)

    put_up, put_dn, p_0 = bin_model_inputs(live_price, strike_price, U, D, p_up, p_dn, risk_free_rate)

    user_input = [live_price, days_left, strike_price, bid, ask, delta, rho, iv_float, volume, strike_dist_pct,
                  int_value,
                  ext_value, put_up, put_dn, p_0]

    user_array = np.array(user_input)

    # GET THE DATA (change path if this was sent to you)
    csv_path = '/Users/kylegraupe/Documents/Python/Machine Learning/Market ML Project/CSV working/CSV_working.csv'
    df = pd.read_csv(csv_path, skipinitialspace=True)
    df.reset_index()
    df.columns.astype(str)

    df = drop_columns(df)
    market_data = df

    market_data = feature_extraction(market_data, p_up, p_dn, U, D, risk_free_rate)

    market_data = remove_long_exp(market_data)

    market_data = add_binary_class(market_data)

    csv_path_2 = "/Users/kylegraupe/Documents/Python/Machine Learning/Market ML Project/Transformed CSV/marketdata.csv"
    csv_2 = market_data.to_csv(path_or_buf=csv_path_2, index=False, encoding='utf-8-sig')

    market_data_X_train, market_data_X_val, market_data_y_train, market_data_y_val, X_train, X_validation, y_train, y_validation = train_val(
        market_data)

    market_data_transform_X_train, market_data_transform_X_val, market_data_transform_y_train, market_data_transform_y_val, user_df = data_prep(
        market_data_X_train, market_data_X_val, market_data_y_train, market_data_y_val, user_array)

    mlp_predictions, mlp_user_prediction = mlp(market_data_transform_X_train, market_data_transform_X_val,
                                               market_data_transform_y_train, user_df)

    mae_mlp, mse_mlp, r2_mlp, final_rmse_mlp = errors(mlp_predictions, market_data_transform_y_val)

    print("----------------------------")
    print(market_data_transform_X_train.info())
    print(market_data_transform_y_train.info())
    print(market_data_transform_X_val.info())
    print(market_data_transform_y_val.info())
    print("----------------------------")
    print("\nPut Fair Price: " + str(put_fair))
    print("----------------------------")
    print("Last Price: " + str(live_price))
    print("Delta: " + str(delta))
    print("Gamma: " + str(gamma))
    print("Vega: " + str(vega))
    print("Theta: " + str(theta))
    print("Rho: " + str(rho))
    print("----------------------------")
    print("User input array: ")
    print(user_array)
    print("----------------------------")
    print("Correlation Matrix: ")
    corr_matrix = market_data.corr()
    print(corr_matrix["[P_LAST]"].sort_values(ascending=False))
    print("----------------------------")
    print("Predictions (MLP): ")
    print(mlp_predictions)
    print("Prediction from user input (MLP): ")
    print("** " + str(mlp_user_prediction) + " **")
    # print("Fair Put at exp: " + str(put_fair_exp))
    print("Put Fair Price: " + str(put_fair))
    print("----------------------------")
    print("The model performance for testing set (MLP))")
    print('MAE is {}'.format(mae_mlp))
    print('MSE is {}'.format(mse_mlp))
    print('R2 score is {}'.format(r2_mlp))
    print("\nFinal Root Mean Squared Error (MLP): " + str(final_rmse_mlp))
    plots(mlp_predictions, market_data_transform_y_val)
    plt.show()
