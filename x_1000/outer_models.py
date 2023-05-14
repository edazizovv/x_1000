#


#
import numpy
import statsmodels.api as sm


#


#
class OLS:
    def __init__(self, x_factors=None, **kwargs):
        self._model = sm.OLS
        self._model_kwargs = {**kwargs}
        self.x_factors = x_factors
        self.model = None

    def fit(self, x, y):
        self.model = self._model(y, x).fit(**self._model_kwargs)

    def predict(self, x):
        return self.model.predict(exog=x)

    @property
    def exog(self):
        return self.model.model.exog

    def specification(self):
        return self.model.params

    def formula(self):
        # TODO: account for absent x0
        s = self.specification()
        if self.x_factors is None:
            return ' + '.join(['{0:.4f}*"x{1}"'.format(s[j], j) for j in range(len(s))])
        else:
            return ' + '.join(['{0:.4f}*"{1}"'.format(s[j], self.x_factors[j]) for j in range(len(s))])

    def copy(self):
        return OLS(x_factors=self.x_factors, **self._model_kwargs)

    @property
    def params(self):
        return self.model.params


class WLS:
    def __init__(self, weights_finder, x_factors=None, ols_kwargs=None, **kwargs):
        self.weights_finder = weights_finder
        self._model = sm.WLS
        self._model_kwargs = {**kwargs}
        self.x_factors = x_factors
        self.model = None
        self.weights = None
        if ols_kwargs is None:
            self.ols_kwargs = {}

    def find_weights(self, x, y, y_hat):
        errors = y - y_hat
        # TODO: implement n bins approach
        if self.weights_finder == 'y':
            self.weights = 1 / y
        elif self.weights_finder == 'y**2':
            self.weights = 1 / y ** 2
        elif self.weights_finder == 'y_hat':
            self.weights = 1 / y_hat
        elif self.weights_finder == 'y_hat**2':
            self.weights = 1 / y_hat ** 2
        elif self.weights_finder == 'abs(err)':
            self.weights = 1 / numpy.abs(errors)
        elif self.weights_finder == 'err**2':
            self.weights = 1 / errors ** 2
        elif self.weights_finder == 'fitted_resids':
            inter_ols = sm.OLS(exog=y_hat, endog=numpy.abs(errors)).fit(**self.ols_kwargs)
            self.weights = 1 / inter_ols.fittedvalues ** 2
        elif self.x_factors is not None:
            if self.weights_finder in self.x_factors:
                self.weights = 1 / x[:, self.x_factors.index(self.weights_finder)]
            else:
                raise ValueError("Invalid weights_finder provided; x_factors is None")
        else:
            raise ValueError("Invalid weights_finder provided")

    def fit(self, x, y):
        inter_model = sm.OLS(y, x).fit(**self.ols_kwargs)
        y_hat = inter_model.fittedvalues
        self.find_weights(x=x, y=y, y_hat=y_hat)
        self.model = self._model(y, x, weights=self.weights).fit(**self._model_kwargs)

    def predict(self, x):
        return self.model.predict(exog=x)

    @property
    def exog(self):
        return self.model.model.exog

    def specification(self):
        return self.model.params

    def formula(self):
        # TODO: account for absent x0
        s = self.specification()
        if self.x_factors is None:
            return ' + '.join(['{0:.4f}*"x{1}"'.format(s[j], j) for j in range(len(s))])
        else:
            return ' + '.join(['{0:.4f}*"{1}"'.format(s[j], self.x_factors[j]) for j in range(len(s))])

    def copy(self):
        copied = WLS(weights_finder=self.weights_finder, x_factors=self.x_factors, **self._model_kwargs)
        copied.weights = self.weights
        return copied

    @property
    def params(self):
        return self.model.params
