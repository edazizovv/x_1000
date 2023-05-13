#


#
import numpy
import pandas
import seaborn
from matplotlib import pyplot
from scipy.stats import ttest_1samp, shapiro
import statsmodels.stats.api as sms
from statsmodels.stats.stattools import durbin_watson
from statsmodels.stats.diagnostic import normal_ad, het_white
from statsmodels.graphics.regressionplots import plot_leverage_resid2
from statsmodels.stats.outliers_influence import variance_inflation_factor



#
from x_1000.outer_rim import X0000, D0000, F0000, A0000, M0000
from x_1000.outer_models import OLS


#
class X1010(X0000):
    """
    [Operative Unit level]

    An implementation of OLS-related arms

    Namespace reserved: X1010-X1019
    """
    def __init__(self):
        """
        Usual 4-step approach:

            1) diagnose
            2) fit
            3) assess
            4) measure

        Should utilize respective classes for each step-component
        """
        diagnose = D1010
        fit = F1010
        assess = A1010
        measure = M1010

        diagnose_kwargs = {}
        fit_kwargs = {}
        assess_kwargs = {}
        measure_kwargs = {}

        # TODO: TBD 'OLSW', 'OLSN', 'WLS', 'FGLS'
        self.codes_names = {'OLSV': 'Ordinary Least Squares (Vanilla)'}
        self.codes_models = {'OLSV': OLS}
        self.codes_kwargs = {'OLSV': {}}

        super().__init__(diagnose=diagnose, fit=fit, assess=assess, measure=measure,
                         diagnose_kwargs=diagnose_kwargs, fit_kwargs=fit_kwargs,
                         assess_kwargs=assess_kwargs, measure_kwargs=measure_kwargs)

    def available_models(self):
        return self.codes_names

    def get_model(self, model_code):
        return self.codes_models[model_code], self.codes_kwargs[model_code]

    def diagnose(self, x, y, model_code, x_factors, **kwargs):
        model, model_kwargs = self.get_model(model_code=model_code)
        return self._diagnose(x=x, y=y, model=model, x_factors=x_factors,
                              **model_kwargs, **self._diagnose_kwargs, **kwargs)

    def fit(self, x, y, model_code, x_factors, **kwargs):
        model, model_kwargs = self.get_model(model_code=model_code)
        return self._fit(x=x, y=y, model=model, x_factors=x_factors, **model_kwargs, **self._fit_kwargs, **kwargs)

    def assess(self, x, y, model, x_factors, **kwargs):
        return self._assess(x=x, y=y, model=model, x_factors=x_factors, **self._assess_kwargs, **kwargs)

    def measure(self, x, y, model, **kwargs):
        return self._measure(x=x, y=y, model=model, **self._measure_kwargs, **kwargs)


class D1010(D0000):
    """
    [Operative Unit level]

    Diagnosis specific to OLS-related arms

    Checks performed:
        > multicollinearity issue
            -- condition number over X matrix [TBD: test setup]
            -- Pearson & Kendall Tau pair correlation coefficients matrices over X matrix [TBD: significance check]
            -- VIF using base model over X matrix [TBD: test setup]
    """
    # TODO: consider changing all values into test setup yielding pvalues
    def __init__(self, x, y, model, x_factors,
                 condition_number_threshold=10,
                 correlation_coeff_threshold=0.70, correlation_coeff_significance=0.05,
                 vif_threshold=5, **kwargs):

        super().__init__(x=x, y=y)

        self.model = model(**kwargs)
        self.condition_number_threshold = condition_number_threshold
        self.correlation_coeff_threshold = correlation_coeff_threshold
        self.correlation_coeff_significance = correlation_coeff_significance
        self.vif_threshold = vif_threshold

        self.model.fit(x=x, y=y)

        # # multicollinearity issue
        # TODO: reformulate all tests in correct form (with H0)

        # condition number

        # TODO: check if calculations are valid
        # TODO: consider setting up a test here
        self.cn_value = numpy.linalg.cond(self.model.exog)

        # Pearson & Kendall Tau pair correlation coefficient matrix

        self.corr_value_mx_pearson = pandas.DataFrame(data=self.model.exog, columns=x_factors).corr(method='pearson')
        self.corr_value_mx_kendall = pandas.DataFrame(data=self.model.exog, columns=x_factors).corr(method='kendall')

        # significance needed #
        pass
        # significance needed #

        # TODO: group into cliques
        notable_pearson_ixs = numpy.argwhere(numpy.abs(self.corr_value_mx_pearson) > self.correlation_coeff_threshold)
        self.notable_pearson_names = ['"{0}"-"{1}"'.format(self.corr_value_mx_pearson.index[i],
                                                       self.corr_value_mx_pearson.index[j])
                                      for i, j in notable_pearson_ixs if i < j]

        notable_kendall_ixs = numpy.argwhere(numpy.abs(self.corr_value_mx_kendall) > self.correlation_coeff_threshold)
        self.notable_kendall_names = ['"{0}"-"{1}"'.format(self.corr_value_mx_kendall.index[i],
                                                       self.corr_value_mx_kendall.index[j])
                                      for i, j in notable_kendall_ixs if i < j]

        # VIF using base model
        # TODO: consider making it model-dependent // should we account for such model significance?
        # TODO: consider setting up a test here

        self.vif_values = pandas.DataFrame(data=[variance_inflation_factor(x, i)
                                                 for i in range(self.model.exog.shape[1])],
                                           index=x_factors, columns=['VIF values'])

    def plot(self):
        """
        Correlation matrices colorized are plotted
        """
        # TODO: when significance is implemented account for it here
        fig, ax = pyplot.subplots(1, 2)
        ax[0].set_title("Pearson Correlation")
        seaborn.heatmap(self.corr_value_mx_pearson, annot=True, cmap="BuPu", ax=ax[0])
        # fig.rcParams["figure.figsize"] = [8.5, 8.5]
        # fig.rcParams["figure.autolayout"] = True
        ax[1].set_title("Kendall Tau Correlation")
        seaborn.heatmap(self.corr_value_mx_kendall, annot=True, cmap="BuPu", ax=ax[1])
        # fig.rcParams["figure.figsize"] = [8.5, 8.5]
        # fig.rcParams["figure.autolayout"] = True
        pyplot.show()

    def summary(self):
        """
        Multicollinearity:
            -- condition number is printed [TBD: see init]
            -- Pearson & Kendall Tau correlated pairs (grouped into cliques [TBD]) are listed (including significance)
                [TBD: see init]
            -- VIF estimations are printed [TBD: see init]
        """
        print("Muticollinearity")
        print("Condition Number:\t{0:.4f} | \t{1:.4f}".format(self.cn_value, self.condition_number_threshold))
        print("Notable Pearson pairs:\t{0}".format(' | '.join(self.notable_pearson_names)))
        print("Notable Kendall pairs:\t{0}".format(' | '.join(self.notable_kendall_names)))
        print("VIF summary [upper_bound={0}]:".format(self.vif_threshold))
        print(self.vif_values)

    def values(self):
        """
        Returned:
            > multicollinearity: condition number
            > multicollinearity: Pearson: correlated pairs
            > multicollinearity: Pearson: correlation matrix
            > multicollinearity: Kendall Tau: correlated pairs
            > multicollinearity: Kendall Tau: correlation matrix
            > multicollinearity: VIF: estimations
            > multicollinearity: VIF: models built [TBD]
        """
        return self.cn_value, \
            self.notable_pearson_names, self.corr_value_mx_pearson, \
            self.notable_kendall_names, self.corr_value_mx_kendall,\
            self.vif_values['VIF values'], None


class F1010(F0000):
    """
    [Operative Unit level]

    Fit specific to OLS-related arms

    Depending on specified conditions, fits:
        > basic OLS
        > OLS (with White heteroskedasticity consistent covariance matrix estimator) [TBD]
        > OLS (with Newey-West heteroskedasticity and autocorrelation consistent covariance matrix estimator) [TBD]
        > WLS [TBD]
        > FGLS [TBD]
        > M-estimator [TBD]
    """
    def __init__(self, x, y, model, x_factors=None, **kwargs):
        super().__init__(x=x, y=y)

        self.model = model(x_factors=x_factors, **kwargs)

        # direct fit

        self.model.fit(x=x, y=y)

    def plot(self):
        raise NotImplemented()

    def summary(self):
        """
        Model formula printed
        """
        print(self.model.formula())

    def values(self):
        """
        Model coefficients returned
        """
        return self.model.specification()


class A1010(A0000):
    """
    [Operative Unit level]

    Assess specific to OLS-related arms

    Properties assessed
    Part 1:
        > linear specification adequacy
        > error terms' distribution
            -- zero-mean
            -- normality
            -- homoskedasticity
            -- absence of autocorrelation
    Part 2:
        > individual significance
        > overall model significance
    """
    def __init__(self, x, y, model, x_factors=None, linear_spec_threshold=0.05,
                 zero_mean_threshold=0.05, normal_distribution_threshold=0.05, homoskedasticity_threshold=0.05):
        super().__init__(x=x, y=y, model=model)

        self.linear_spec_threshold = linear_spec_threshold
        self.zero_mean_threshold = zero_mean_threshold
        self.normal_distribution_threshold = normal_distribution_threshold
        self.homoskedasticity_threshold = homoskedasticity_threshold

        self.x_factors = x_factors

        self.y_hat = self.model.predict(x=self.x)
        self.errors = self.y - self.y_hat

        # TODO: control that all tests within a group are "one-side" aligned in their H0s
        #  // maybe we can put them all into a specific class?

        # # part 1

        # linear specification

        # TODO: analyze this test
        m = len(self.model.params)
        skip = min([j + m for j in range(100-m) if numpy.linalg.matrix_rank(self.x[:j+m]) == self.x[:j+m].shape[1]] +
                   [100])
        rr = sms.recursive_olsresiduals(self.model.model,
                                        skip=skip, alpha=(1 - self.linear_spec_threshold), order_by=None)
        self.ls = [ttest_1samp(rr[3][skip:], 0)]
        self.ls_n = len(self.ls)
        self.ls_pvalues = [self.ls[0][1]]
        self.ls_h0 = 'H0: correctly specified as linear'
        self.ls_np = '{0} / {1}'.format(
            sum([self.ls_pvalues[j] > self.linear_spec_threshold for j in range(self.ls_n)]), self.ls_n)
        self.ls_summary = '{0}: \t{1} pass at {2:.2f} alpha level'.format(
            self.ls_h0, self.ls_np, self.linear_spec_threshold)

        # zero mean

        self.zm = [ttest_1samp(self.errors, 0)]
        self.zm_n = len(self.zm)
        self.zm_pvalues = [self.zm[0][1]]
        self.zm_h0 = 'H0: zero mean'
        self.zm_np = '{0} / {1}'.format(
            sum([self.zm_pvalues[j] > self.zero_mean_threshold for j in range(self.zm_n)]), self.zm_n)
        self.zm_summary = '{0}: \t{1} pass at {2:.2f} alpha level'.format(
            self.zm_h0, self.zm_np, self.zero_mean_threshold)

        # normality

        self.no = [shapiro(self.errors), normal_ad(self.errors)]
        self.no_n = 2
        self.no_pvalues = [self.no[0][1], self.no[1][1]]
        self.no_h0 = 'H0: normal distribution'
        self.no_np = '{0} / {1}'.format(
            sum([self.no_pvalues[j] > self.normal_distribution_threshold for j in range(self.no_n)]), self.no_n)
        self.no_summary = '{0}: \t{1} pass at {2:.2f} alpha level'.format(
            self.no_h0, self.no_np, self.normal_distribution_threshold)

        # homoskedasticity

        # TODO: tests to be reworked so that they allow using no-constant specification
        self.ho = [sms.het_breuschpagan(self.errors, x), het_white(self.errors, x)]
        self.ho_n = 2
        self.ho_pvalues = [self.ho[0][1], self.ho[1][1]]
        self.ho_h0 = 'H0: errors are homoskedastic'
        self.ho_np = '{0} / {1}'.format(
            sum([self.ho_pvalues[j] > self.homoskedasticity_threshold for j in range(self.ho_n)]), self.ho_n)
        self.ho_summary = '{0}: \t{1} pass at {2:.2f} alpha level'.format(
            self.ho_h0, self.ho_np, self.homoskedasticity_threshold)

        # TODO: upgrade with ACF / PACF testing? or another approach?
        # absense of autocorrelation

        # TODO: reset to test instead of these values
        self.ac_value = (durbin_watson(self.errors) - 2) / -2
        self.ac_summary = 'Autocorrelation check: \t{0:.4f} | from negative -1 to positive 1 autocorr'.format(
            self.ac_value)

        # # part 2

        # TODO: introduce wald tests & analyze their pros & cons
        # individual significance

        r = numpy.identity(self.model.params.shape[0])
        self.ts_pvalues = self.model.model.t_test(r).pvalue

        # overall model significance

        r = numpy.identity(self.model.params.shape[0])
        r = r[1:, :]
        self.fs_pvalue = self.model.model.f_test(r).pvalue

    def plot(self, y_l=0, e_l=0, y_p_low=10, y_p_upp=90, e_p_low=10, e_p_upp=90, hists_n_bins=40):
        # TODO: consider introducing features for MAE-dependent models
        """
        Plots:
            > y: errors // x: n_ob; bounded by outlier border + assumed normal dist border [TBD]
            > hist: y; bounded by outlier border + assumed normal dist border
            > hist: errors; bounded by outlier border + assumed normal dist border
            > influence/outlier plot [TO BE ANALYZED]
            > y: errors // x: y; bounded by outlier borders for each + assumed normal dist border for each
        """

        # outlier identification

        def borders(array, l, p_low=25, p_upp=75):
            p_low_value, p_upp_value = numpy.percentile(array, [p_low, p_upp])
            if l == 0:
                b_low, b_upp = p_low_value, p_upp_value
            else:
                iqr_value = p_upp_value - p_low_value
                b_low = p_low_value - (iqr_value * l)
                b_upp = iqr_value * l + p_upp_value
            return b_low, b_upp

        y_b_low, y_b_upp = borders(array=self.y, l=y_l, p_low=y_p_low, p_upp=y_p_upp)
        e_b_low, e_b_upp = borders(array=self.errors, l=e_l, p_low=e_p_low, p_upp=e_p_upp)

        # plots

        # TODO: add assumed normal dist borders
        # TODO: improve colors

        nn = range(self.errors.shape[0])

        fig1, ax1 = pyplot.subplots(2, 2)
        fig2, ax2 = pyplot.subplots(1, 2)

        gs = ax1[0, 1].get_gridspec()
        # remove the underlying axes
        for a in ax1[0, :]:
            a.remove()
        ax_big = fig1.add_subplot(gs[0, :])

        ax_big.plot(nn, self.errors)
        ax_big.axhline(y=e_b_low)
        ax_big.axhline(y=e_b_upp)
        ax_big.set_title("Errors dynamics\n[OUT e: l={0:.2f}, p_low={1:.2f}, p_upp={2:.2f}]".format(
            e_l, e_p_low, e_p_upp))

        ax1[1, 0].hist(self.y, bins=hists_n_bins)
        ax1[1, 0].axvline(x=y_b_low)
        ax1[1, 0].axvline(x=y_b_upp)
        ax1[1, 0].set_title("Y distribution\n[OUT e: l={0:.2f}, p_low={1:.2f}, p_upp={2:.2f}]".format(
            y_l, y_p_low, y_p_upp))

        ax1[1, 1].hist(self.errors, bins=hists_n_bins)
        ax1[1, 1].axvline(x=e_b_low)
        ax1[1, 1].axvline(x=e_b_upp)
        ax1[1, 1].set_title("Errors distribution\n[OUT e: l={0:.2f}, p_low={1:.2f}, p_upp={2:.2f}]".format(
            e_l, e_p_low, e_p_upp))

        # TODO: check that influence model and dfbetas
        plot_leverage_resid2(self.model.model, ax=ax2[0])
        ax2[0].set_title("Outliers' influence plot")

        # TODO: update with trend
        # TODO: add R2 and formula
        ax2[1].axhline(y=e_b_low)
        ax2[1].axhline(y=e_b_upp)
        ax2[1].axvline(x=y_b_low)
        ax2[1].axvline(x=y_b_upp)
        ax2[1].axhline(y=0)
        ax2[1].scatter(self.y, self.errors)
        ax2[1].set_title("Errors over Y dependence:\nR2={0:.4f}\ne ~ {1:.4f} + {2:.4f} * y".format(-0, -0, -0))

    def summary(self):
        """
        Summary tables:
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
        """
        part_1 = pandas.DataFrame(data={'h0': [self.ls_h0, self.zm_h0, self.no_h0, self.ho_h0, 'Autocorrelation check'],
                                        'n_passed': [self.ls_np, self.zm_np, self.no_np, self.ho_np, self.ac_value],
                                        'thresh': [self.linear_spec_threshold,
                                                   self.zero_mean_threshold,
                                                   self.normal_distribution_threshold,
                                                   self.homoskedasticity_threshold,
                                                   [-1, 1]]},
                                  )
        part_2 = pandas.DataFrame(data={'h0': ['H0: coefficient is zero'] * self.model.params.shape[0] +
                                              ['H0: all coefficients are zero'],
                                        'factor_name': self.x_factors + ['model'],
                                        'pvalue': self.ts_pvalues.tolist() + [self.fs_pvalue]})

        print(part_1)
        print(part_2)

        return part_1, part_2

    def values(self):
        """
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
        """
        part_1 = pandas.DataFrame(data={'h0': [self.ls_h0, self.zm_h0, self.no_h0, self.ho_h0, 'Autocorrelation check'],
                                        'n_passed': [self.ls_np, self.zm_np, self.no_np, self.ho_np, self.ac_value],
                                        'thresh': [self.linear_spec_threshold,
                                                   self.zero_mean_threshold,
                                                   self.normal_distribution_threshold,
                                                   self.homoskedasticity_threshold,
                                                   [-1, 1]]},
                                  )
        part_2 = pandas.DataFrame(data={'h0': ['H0: coefficient is zero'] * self.model.params.shape[0] +
                                              ['H0: all coefficients are zero'],
                                        'factor_name': self.x_factors + ['model'],
                                        'pvalue': self.ts_pvalues.tolist() + [self.fs_pvalue]})

        return part_1, part_2, self.ls, self.zm, self.no, self.ho, self.ac_value, self.ts_pvalues, self.fs_pvalue


class M1010(M0000):
    """
    [Operative Unit level]

    Measure specific to OLS-related arms

    Measures assessed:
        > NSE [=NormalizedSquaredErrors, actually equal to R2, normalized MSE]
        > NAE [=NormalizedAbsoluteErrors, R2-style normalization applied to MAE]
        > SMAPE [=SymmetricMeanAbsoluteError]
    Also confidence intervals are estimated using bootstrapped subsamples [TBD]
    """
    def __init__(self, x, y, model, n_boots=100):
        super().__init__(x=x, y=y, model=model)

        self.n_boots = n_boots
        self.y_hat = self.model.predict(x=self.x)

        # # main measures estimation

        # nse

        def nse(y_true, y_hat):
            y_avg = y_true.mean()
            ss_res = ((y_true - y_hat) ** 2).sum()
            ss_tot = ((y_true - y_avg) ** 2).sum()
            measured = 1 - (ss_res / ss_tot)
            return measured

        self.nse_value = nse(y_true=self.y, y_hat=self.y_hat)

        # nae

        def nae(y_true, y_hat):
            y_avg = numpy.median(y_true)
            sa_res = ((y_true - y_hat) ** 2).sum()
            sa_tot = ((y_true - y_avg) ** 2).sum()
            measured = 1 - (sa_res / sa_tot)
            return measured

        self.nae_value = nae(y_true=self.y, y_hat=self.y_hat)

        # smape

        def smape(y_true, y_hat):
            err_absolutes = numpy.abs(y_true - y_hat)
            avg_absolutes = (numpy.abs(y_true) + numpy.abs(y_hat)) / 2
            measured = (err_absolutes / avg_absolutes).sum() / y_true.shape[0]
            return measured

        self.smape_value = smape(y_true=self.y, y_hat=self.y_hat)

        # TODO: implement subsamples for bootstrap confidence interval estimation of measures

    def plot(self):
        """
        Plots distribution for each measure and main estimate point on it [TBD]
        """
        # TODO: TBD
        raise NotImplemented()

    def summary(self, q_low=0.10, q_upp=0.90):
        """
        Measures and their quantiles printed [TBD]
        """
        # TODO: TBD
        summary = pandas.DataFrame(data={'{0:.2f}-q_low'.format(q_low): [-1, -1, -1],
                                         'main_value': [self.nse_value, self.nae_value, self.smape_value],
                                         '{0:.2f}-q_upp'.format(q_upp): [-1, -1, -1]})
        print(summary)

    def values(self, q_low=0.10, q_upp=0.90):
        """
        Measures and their quantiles returned [TBD]
        """
        # TODO: TBD
        summary = pandas.DataFrame(data={'{0:.2f}-q_low'.format(q_low): [-1, -1, -1],
                                         'main_value': [self.nse_value, self.nae_value, self.smape_value],
                                         '{0:.2f}-q_upp'.format(q_upp): [-1, -1, -1]})
        return summary
