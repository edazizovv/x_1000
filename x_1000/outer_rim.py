#


#


#


#
class X0000:
    """
    [Operative Unit level]

    Base class to be used for all arms

    Namespace reserved: X0000-X0009
    """
    def __init__(self, diagnose, fit, assess, measure, diagnose_kwargs, fit_kwargs, assess_kwargs, measure_kwargs):
        """
        Follows standard 4-step approach:

            1) diagnose
            2) fit
            3) assess
            4) measure

        Should utilize respective classes for each step-component
        """
        self._diagnose = diagnose
        self._fit = fit
        self._assess = assess
        self._measure = measure

        self._diagnose_kwargs = diagnose_kwargs
        self._fit_kwargs = fit_kwargs
        self._assess_kwargs = assess_kwargs
        self._measure_kwargs = measure_kwargs

    def diagnose(self, **kwargs):
        """
        Should be used to call diagnose
        """
        pass

    def fit(self, **kwargs):
        """
        Should be used to call fit
        """
        pass

    def assess(self, **kwargs):
        """
        Should be used to call assess
        """
        pass

    def measure(self, **kwargs):
        """
        Should be used to call measure
        """
        pass


class Sub:
    """
    [Operative Unit level]

    Base class for each sub 4-step component
    """

    def __init__(self, **kwargs):
        """
        Performs respective activity and stores results
        """
        pass

    def plot(self, **kwargs):
        """
        Provides plot representation for the results
        """
        pass

    def summary(self, **kwargs):
        """
        Provides tabular representation for the results
        """
        pass

    def values(self, **kwargs):
        """
        Returns values yielded on init
        """
        pass


class D0000(Sub):
    """
    [Operative Unit level]

    D for diagnose

    Specifies base class to diagnose

    Namespace should reflect specific model name
    """

    def __init__(self, x, y, **kwargs):
        super().__init__()
        self.x = x.copy()
        self.y = y.copy()


class F0000(Sub):
    """
    [Operative Unit level]

    F for fit

    Specifies base class to fit

    Namespace should reflect specific model name
    """

    def __init__(self, x, y, **kwargs):
        super().__init__()
        self.x = x.copy()
        self.y = y.copy()


class A0000(Sub):
    """
    [Operative Unit level]

    A for assess

    Specifies base class to assess

    Namespace should reflect specific model name
    """

    def __init__(self, x, y, model, **kwargs):
        super().__init__()
        self.x = x.copy()
        self.y = y.copy()
        self.model = model


class M0000(Sub):
    """
    [Operative Unit level]

    M for measure

    Specifies base class to measure

    Namespace should reflect specific model name
    """

    def __init__(self, x, y, model, **kwargs):
        super().__init__()
        self.x = x.copy()
        self.y = y.copy()
        self.model = model
