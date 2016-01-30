import datetime

import numpy as np
import pandas as pd

__author__ = 'ielemin'


class Random:
    def __init__(self):
        self.size = 0
        self.dt = None
        self.start_datetime = None
        self.idxT = None

    def generate_index(self, size, dt_minutes=5, start_datetime=datetime.datetime(2010, 1, 4)):
        """Generate an evenly spaced DatetimeIndex.

        :param size: the number of steps
        :param dt_minutes: the time interval
        :param start_datetime: the first point in the index
        """
        self.size = size
        self.dt = datetime.timedelta(minutes=dt_minutes)
        self.start_datetime = start_datetime
        self.idxT = pd.DatetimeIndex(
            pd.Series(self.dt, index=np.arange(self.size)).cumsum() + self.start_datetime - self.dt)
        # cumsum is approx 200x faster than a simple comprehension

    def generate_data_flip(self, betas):
        """Generate individual Xs and their linear combination as 'y'.

        y = sum_i[a_i(t) + beta_i * (1+b_i(t)) * s_i(t) * x_i(t)]
        where:
        x_i(t) in N(0,1)
        a_i(t), b_i(t) in N(0,10)
        s_i(t) in {+1;-1} with 'low' variance

        :param betas: a (label,beta) dictionary
        :return: a DataFrame of xs and y, a DataFrame of the weight of each x in y
        """
        series = pd.DataFrame(index=self.idxT)
        true_betas = pd.DataFrame(index=self.idxT)
        y = pd.Series(0, index=self.idxT)

        for label in betas:
            beta = betas[label]
            data = Random._generate_data_flip(beta, self.size)

            series.loc[:, label] = data['x']
            y += data['y']
            true_betas.loc[:, label] = data['true_beta']
        series.loc[:, 'y'] = y

        return series, true_betas

    @staticmethod
    def _generate_data_flip(beta, size):
        """Generate a couple (x,y) with a defined betaSR wih a sign that flips randomly

        y = sum_i[a_i(t) + beta_i * (1+b_i(t)) * s_i(t) * x_i(t)]
        where:
        x_i(t) in N(0,1)
        a_i(t), b_i(t) in N(0,10)
        s_i(t) in {+1;-1} with 'low' variance

        :param beta: the targeted beta
        :param size: the number of points
        :return: a dictionary containing the series of x, y and the rue beta
        """
        data_x = np.random.normal(0, 1, size)

        noise_a = np.random.normal(0, 10, size)
        noise_b = np.random.normal(1, 10, size)
        noise_s = np.random.normal(0, .0001, size)

        sign = (np.abs(noise_s.cumsum() % 2) > 1) * 2 - 1

        data_y = noise_a + beta * sign * noise_b * data_x

        return {'x': data_x, 'y': data_y, 'true_beta': beta * sign}

    def generate_data_lognorm(self, vols):
        """Generate time series of pseudo-prices with a lognormal distribution.

        :param vols: a {'name':yearly vol} dictionary for the instruments
        :return: a DataFrame of prices
        """
        vol_factor = np.sqrt(self.dt.total_seconds() / datetime.timedelta(days=250).total_seconds())

        return pd.DataFrame(
            {instr: Random._generate_data_lognorm(self.size, step_vol=vols[instr] * vol_factor) for instr in vols},
            index=self.idxT)

    @staticmethod
    def _generate_data_lognorm(size, step_vol, step_alpha=0):
        """Generate a singe time series of pseudo-price.

        :param size: the number of points
        :param step_vol: the stepwise volatility of returns
        :param step_alpha: the stepwise alpha
        :return: a numpy array of prices
        """
        return np.exp(np.random.normal(step_alpha, step_vol, size)).cumprod()
