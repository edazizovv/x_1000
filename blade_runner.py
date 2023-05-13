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

diagnosed = ammo.diagnose(x=x, y=y, model_code='OLSV', x_factors=x_factors)
fitted = ammo.fit(x=x, y=y, model_code='OLSV', x_factors=x_factors)
assessed = ammo.assess(x=x, y=y, model=fitted.model, x_factors=x_factors)
measured = ammo.measure(x=x, y=y, model=fitted.model)
