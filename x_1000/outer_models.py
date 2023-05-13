#


#
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
        return OLS(**self._model_kwargs)

    @property
    def params(self):
        return self.model.params
