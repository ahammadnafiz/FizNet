import pandas as pd
import numpy as np
import logging
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # Required to enable IterativeImputer
from sklearn.impute import IterativeImputer
from sklearn.impute import MissingIndicator

class DataImputer:
    def __init__(self, data, logger=None, random_seed=0):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        self.logger = logger or self._setup_default_logger()

    def _setup_default_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        logger.addHandler(logging.StreamHandler())
        return logger

    def validate_columns(self, columns):
        """Validates if the columns exist in the DataFrame."""
        invalid_columns = [col for col in columns if col not in self.data.columns]
        if invalid_columns:
            raise ValueError(f"Invalid columns: {', '.join(invalid_columns)}")

    def drop_columns(self, columns_to_drop):
        try:
            self.validate_columns(columns_to_drop)
            # Drop the specified columns
            self.data.drop(columns_to_drop, axis=1, inplace=True)
            self.logger.info("Dropped the following columns: %s" % ", ".join(columns_to_drop))
            return self
        except Exception as e:
            self.logger.error("An error occurred while dropping columns: %s" % str(e))
            raise

    def check_missing(self):
        try:
            result = pd.concat([self.data.isnull().sum(), self.data.isnull().mean()], axis=1)
            result.columns = ['total missing', 'proportion']
            return result
        except Exception as e:
            self.logger.error("An error occurred while checking missing values: %s", str(e))
            raise

    def drop_missing(self, axis=0):
        try:
            original_shape = self.data.shape
            self.data.dropna(axis=axis, inplace=True)
            if self.data.shape == original_shape:
                return None
            return self
        except Exception as e:
            self.logger.error("An error occurred while dropping missing values: %s", str(e))
            raise

    def add_var_denote_na(self, columns_with_na):
        """Creates a binary column for each column to indicate if the value was missing."""
        try:
            self.validate_columns(columns_with_na)
            for column in columns_with_na:
                self.data[column + '_was_na'] = np.where(self.data[column].isnull(), 1, 0)
            return self
        except Exception as e:
            self.logger.error("An error occurred while adding NA indicator variables: %s", str(e))
            raise

    def impute_na_with_arbitrary(self, arbitrary_value, columns_with_na):
        """Imputes missing values with an arbitrary value."""
        try:
            self.validate_columns(columns_with_na)
            for column in columns_with_na:
                self.data[column].fillna(arbitrary_value, inplace=True)
            return self
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with arbitrary value: %s", str(e))
            raise

    def impute_na_with_avg(self, strategy='mean', columns_with_na=[]):
        """Imputes missing values with mean, median, or mode."""
        try:
            self.validate_columns(columns_with_na)
            for column in columns_with_na:
                if strategy == 'mean':
                    self.data[column].fillna(self.data[column].mean(), inplace=True)
                elif strategy == 'median':
                    self.data[column].fillna(self.data[column].median(), inplace=True)
                elif strategy == 'mode':
                    self.data[column].fillna(self.data[column].mode()[0], inplace=True)
            return self
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with average: %s", str(e))
            raise

    def impute_na_with_end_of_distribution(self, columns_with_na):
        """Imputes missing values with values at the end of the distribution (mean + 3*std)."""
        try:
            self.validate_columns(columns_with_na)
            for column in columns_with_na:
                self.data[column].fillna(self.data[column].mean() + 3 * self.data[column].std(), inplace=True)
            return self
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with end of distribution: %s", str(e))
            raise

    def impute_na_with_random(self, columns_with_na, random_state=0):
        """Imputes missing values with random samples from existing values."""
        try:
            self.validate_columns(columns_with_na)
            for column in columns_with_na:
                random_sample = self.data[column].dropna().sample(self.data[column].isnull().sum(), random_state=random_state)
                random_sample.index = self.data[self.data[column].isnull()].index
                self.data.loc[self.data[column].isnull(), column] = random_sample
            return self
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with random sampling: %s", str(e))
            raise

    def impute_na_with_interpolation(self, columns_with_na, method='linear', limit=None, limit_direction='forward'):
        """Imputes missing values using interpolation."""
        try:
            self.validate_columns(columns_with_na)
            for column in columns_with_na:
                self.data[column].interpolate(method=method, limit=limit, limit_direction=limit_direction, inplace=True)
            return self
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with interpolation: %s", str(e))
            raise

    def impute_na_with_knn(self, columns_with_na, n_neighbors=5):
        """Imputes missing values using K-Nearest Neighbors."""
        try:
            self.validate_columns(columns_with_na)
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            for column in columns_with_na:
                self.data[column] = knn_imputer.fit_transform(self.data[column].values.reshape(-1, 1)).ravel()
            return self
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with KNN: %s", str(e))
            raise

    def impute_na_with_simple_imputer(self, columns_with_na, strategy='mean'):
        """Imputes missing values using sklearn's SimpleImputer."""
        try:
            self.validate_columns(columns_with_na)
            imputer = SimpleImputer(strategy=strategy)
            for column in columns_with_na:
                self.data[column] = imputer.fit_transform(self.data[column].values.reshape(-1, 1)).ravel()
            return self
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with SimpleImputer: %s", str(e))
            raise

    def impute_all_columns_with_strategy(self, strategy='mean'):
        """Impute missing values for all columns using a specified strategy (mean, median, mode)."""
        try:
            for column in self.data.columns:
                if self.data[column].isnull().sum() > 0:
                    self.impute_na_with_avg(strategy=strategy, columns_with_na=[column])
            return self
        except Exception as e:
            self.logger.error("An error occurred while imputing all columns with strategy: %s", str(e))
            raise

    def impute_na_exclude_outliers(self, columns_with_na, threshold=3):
        """Imputes missing values while excluding outliers from the calculation."""
        try:
            self.validate_columns(columns_with_na)
            for column in columns_with_na:
                non_outliers = self.data[np.abs(self.data[column] - self.data[column].mean()) <= (threshold * self.data[column].std())]
                self.data[column].fillna(non_outliers[column].mean(), inplace=True)
            return self
        except Exception as e:
            self.logger.error("An error occurred while imputing NA excluding outliers: %s", str(e))
            raise

    def impute_with_custom_function(self, columns_with_na, custom_function):
        """Allows custom imputation logic provided by the user."""
        try:
            self.validate_columns(columns_with_na)
            for column in columns_with_na:
                self.data[column] = self.data[column].apply(custom_function)
            return self
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with custom function: %s", str(e))
            raise

    def impute_by_group_mean(self, group_column, columns_with_na):
        """Imputes missing values based on group mean (e.g., by category)."""
        try:
            self.validate_columns([group_column] + columns_with_na)
            for column in columns_with_na:
                self.data[column] = self.data.groupby(group_column)[column].transform(lambda x: x.fillna(x.mean()))
            return self
        except Exception as e:
            self.logger.error("An error occurred while imputing by group mean: %s", str(e))
            raise
        
    def impute_NA_with_iterative(self, NA_col=[], max_iter=10, random_state=0, tol=1e-3):
        """Impute missing values using Iterative Imputer."""
        try:
            imputer = IterativeImputer(max_iter=max_iter, random_state=random_state, tol=tol)
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    self.data[i] = imputer.fit_transform(self.data[i].values.reshape(-1, 1)).ravel()
                    self.logger.info(f"Iterative imputation applied on column: {i}")
                else:
                    self.logger.warning(f"Column {i} has no missing values.")
            return self.data
        except Exception as e:
            self.logger.error(f"An error occurred while applying Iterative Imputation: {e}")
            raise

    def add_missing_indicators(self, NA_col=[]):
        """Add binary indicators for missing values."""
        try:
            indicator = MissingIndicator()
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    indicator_data = indicator.fit_transform(self.data[[i]])
                    self.data[i + '_missing'] = indicator_data
                    self.logger.info(f"Missing indicator added for column: {i}")
                else:
                    self.logger.warning(f"Column {i} has no missing values.")
            return self.data
        except Exception as e:
            self.logger.error(f"An error occurred while adding missing indicators: {e}")
            raise
    
    def impute_NA_with_mice(self, NA_col=[], max_iter=10, random_state=0, tol=1e-3):
        """Impute missing values using Multiple Imputation by Chained Equations (MICE)."""
        try:
            imputer = IterativeImputer(max_iter=max_iter, random_state=random_state, tol=tol)
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    self.data[i] = imputer.fit_transform(self.data).ravel()
                    self.logger.info(f"MICE applied on column: {i}")
                else:
                    self.logger.warning(f"Column {i} has no missing values.")
            return self.data
        except Exception as e:
            self.logger.error(f"An error occurred while applying MICE: {e}")
            raise     