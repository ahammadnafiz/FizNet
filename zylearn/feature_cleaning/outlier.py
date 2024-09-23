import pandas as pd
import numpy as np
import logging
import plotly.graph_objects as go

from ..misc.helper import Helper

class OutlierDetector:
    def __init__(self, data, log_level=logging.INFO):
        self.data = data
        self.miscs = Helper()
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.logger.addHandler(logging.StreamHandler())

    def _calculate_outliers(self, col, upper_fence, lower_fence):
        outlier_index = self.data[(self.data[col] > upper_fence) | (self.data[col] < lower_fence)].index
        num_outliers = len(outlier_index)
        prop_outliers = num_outliers / len(self.data)
        return outlier_index, num_outliers, prop_outliers

    def iqr_outlier_detection(self, col, threshold=1.5, show_graph=False):
        try:
            Q1 = np.percentile(self.data[col], 25)
            Q3 = np.percentile(self.data[col], 75)
            IQR = Q3 - Q1
            upper_fence = Q3 + 1.5 * IQR
            lower_fence = Q1 - 1.5 * IQR

            outlier_index, num_outliers, prop_outliers = self._calculate_outliers(col, upper_fence, lower_fence)

            if prop_outliers > threshold:
                raise ValueError(f"Proportion of outliers ({prop_outliers:.2%}) exceeds the threshold ({threshold:.2%})")

            if show_graph:
                self.miscs.plot_outliers(self.data, col, outlier_index)
                return

            print('Num of outliers detected:', num_outliers)
            print('Proportion of outliers detected:', prop_outliers)
            return {'outliers': outlier_index, 'upper_fence': upper_fence, 'lower_fence': lower_fence}

        except Exception as e:
            self.logger.info("An error occurred while processing your request. Please check your input data and try again.")
            self.logger.error("An error occurred while detecting outliers: %s", str(e))
            return {'outliers': [], 'upper_fence': None, 'lower_fence': None}

    def z_score_detection(self, col, threshold=3, show_graph=False):
        try:
            mean = self.data[col].mean()
            std_dev = self.data[col].std()
            upper_fence = mean + threshold * std_dev
            lower_fence = mean - threshold * std_dev

            outlier_index, num_outliers, prop_outliers = self._calculate_outliers(col, upper_fence, lower_fence)

            if show_graph:
                self.miscs.plot_outliers(self.data, col, outlier_index)
                return

            print('Num of outliers detected:', num_outliers)
            print('Proportion of outliers detected:', prop_outliers)
            return {'outliers': outlier_index, 'upper_fence': upper_fence, 'lower_fence': lower_fence}

        except Exception as e:
            self.logger.error("An error occurred while detecting outliers: %s", str(e))
            return {'outliers': [], 'upper_fence': None, 'lower_fence': None}

    def mad_outlier_detection(self, col, threshold=3.5, show_graph=False):
        try:
            median = self.data[col].median()
            median_absolute_deviation = np.median([np.abs(y - median) for y in self.data[col]])
            modified_z_scores = pd.Series([0.6745 * (y - median) / median_absolute_deviation for y in self.data[col]])

            outlier_index = self.data.index[np.abs(modified_z_scores) > threshold]
            num_outliers = len(outlier_index)
            prop_outliers = num_outliers / len(self.data)

            if show_graph:
                self.miscs.plot_outliers(self.data, col, outlier_index)
                return

            print('Num of outliers detected:', num_outliers)
            print('Proportion of outliers detected:', prop_outliers)
            return {'outliers': outlier_index}

        except Exception as e:
            self.logger.error("An error occurred while detecting outliers: %s", str(e))
            return {'outliers': [], 'upper_fence': None, 'lower_fence': None}

    def windsorization(self, col, para, strategy='both'):
        try:
            if strategy == 'both':
                self.data.loc[self.data[col] > para[0], col] = para[0]
                self.data.loc[self.data[col] < para[1], col] = para[1]
            elif strategy == 'top':
                self.data.loc[self.data[col] > para[0], col] = para[0]
            elif strategy == 'bottom':
                self.data.loc[self.data[col] < para[1], col] = para[1]
            return self.data
        except Exception as e:
            self.logger.error("An error occurred while performing windsorization: %s", str(e))
            raise

    def drop_outlier(self, outlier_index):
        try:
            self.data = self.data.loc[~self.data.index.isin(outlier_index)]
            return self.data
        except Exception as e:
            self.logger.error("An error occurred while dropping outliers: %s", str(e))
            raise