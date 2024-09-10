#
import random


#
import numpy
import pandas
from sklearn.model_selection import train_test_split


#
from x_1000.skeleton import X1010


#
rs = 999
random.seed(rs)
numpy.random.seed(rs)


data = pandas.read_csv('./data/house_price_data.csv')
data = data.set_index('No')
data = data.dropna()
data['constant'] = 1

# x_exclusions = []
# x_exclusions = ['constant', 'X5 latitude', 'X6 longitude', 'X1 transaction date']
# x_exclusions = ['constant', 'X5 latitude', 'X1 transaction date', 'X4 number of convenience stores']
# x_exclusions = ['constant', 'X5 latitude', 'X1 transaction date', 'X3 distance to the nearest MRT station']
x_exclusions = ['X5 latitude', 'X1 transaction date', 'X3 distance to the nearest MRT station']

target = 'Y house price of unit area'
x_factors = [x for x in data.columns if x not in x_exclusions + [target]]
x, y = data[x_factors].values, data[target].values

test_size = 0.5
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=rs)


ammo = X1010()

# https://www.statsmodels.org/devel/generated/statsmodels.regression.linear_model.OLSResults.get_robustcov_results.html
model_name = 'OLS'
model_kwargs = {'cov_type': 'nonrobust'}
# model_kwargs = {'cov_type': 'HC3'}
# model_kwargs = {'cov_type': 'HAC', 'cov_kwds': {'maxlags': 1}}

# model_name = 'WLS'
# model_kwargs = {}
# model_kwargs = {'weights_finder': 'fitted_resids'}
# model_kwargs = {'weights_finder': 'abs(err)'}

diagnosed = ammo.diagnose(x=x, y=y, model_code=model_name, x_factors=x_factors, model_kwargs=model_kwargs)
fitted = ammo.fit(x=x, y=y, model_code=model_name, x_factors=x_factors, model_kwargs=model_kwargs)
assessed = ammo.assess(x=x, y=y, model=fitted.model, x_factors=x_factors)
measured = ammo.measure(x=x, y=y, model=fitted.model)

'''
Available methods:

diagnosed.values() returns:
        > multicollinearity: condition number
        > multicollinearity: Pearson: correlated pairs
        > multicollinearity: Pearson: correlation matrix
        > multicollinearity: Kendall Tau: correlated pairs
        > multicollinearity: Kendall Tau: correlation matrix
        > multicollinearity: VIF: estimations
        > multicollinearity: VIF: models built [TBD]
diagnosed.summary() prints:
        Multicollinearity:
            -- condition number
            -- Pearson & Kendall Tau correlated pairs (including significance)
            -- VIF estimations
diagnosed.plot() plots:
        > multicollinearity: Pearson & Kendall Tau correlation matrices colorized


fitted.values() returns:
        model coefficients (starting with intercept)
fitted.summary() prints:
        model formula
fitted.plot() is not implemented


assessed.values() returns:
        Summary tables and tests:
            Summary table 1:
                > linear specification adequacy: aggregated - h0, n passed, significance thresh
                > error terms' distribution
                    -- zero-mean: aggregated - h0, n passed, significance thresh
                    -- normality: aggregated - h0, n passed, significance thresh
                    -- homoskedasticity: aggregated - h0, n passed, significance thresh
                    -- absence of autocorrelation: h0, value, thresh rule
            Summary table 2:
                > individual significance: h0, factor name, pvalue
                > overall model significance: h0, model, pvalue
            All tests' original classes
            All significance values
        Linear Specification test
        Zero Mean test
        Normality tests
        Homoskedasticity tests
        Autocorrelation test
        Individual factor significance
        Group factor significance
assessed.summary() prints:
        Part 1:
            > linear specification adequacy: aggregated - h0, n passed, significance thresh
            > error terms' distribution
                -- zero-mean: aggregated - h0, n passed, significance thresh
                -- normality: aggregated - h0, n passed, significance thresh
                -- homoskedasticity: aggregated - h0, n passed, significance thresh
                -- absence of autocorrelation: h0, value, thresh rule
        Part 2:
            > individual significance: h0, factor name, pvalue
            > overall model significance: h0, model, pvalue
assessed.plot() plots:
        > y: errors // x: n_ob; bounded by outlier border
        > hist: y; bounded by outlier border
        > hist: errors; bounded by outlier border
        > influence/outlier plot
        > y: errors // x: y; bounded by outlier borders for each

measured.values() returns:
    dataframe with the summary
measured.summary() prints:
    estimated values for:
        NSE (standardized MSE, actually R2), higher is better
        NAE (standardized MAE), higher is better
        SMAPE, lower is better
measured.plot() is not implemented
'''
