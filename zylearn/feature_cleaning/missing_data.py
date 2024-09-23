import pandas as pd
import numpy as np
import logging


class DataImputer:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        self.data = data
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        self.logger.addHandler(logging.StreamHandler())

    def drop_columns(self, columns_to_drop):
        try:
            # Check if the columns exist in the DataFrame
            non_existent_cols = [col for col in columns_to_drop if col not in self.data.columns]
            if non_existent_cols:
                self.logger.error("The following columns do not exist in the DataFrame and will be ignored: %s" % ", ".join(non_existent_cols))
                columns_to_drop = [col for col in columns_to_drop if col in self.data.columns]

            # Drop the specified columns
            self.data = self.data.drop(columns_to_drop, axis=1)
            self.logger.info("Dropped the following columns: %s" % ", ".join(columns_to_drop))
            return self.data

        except Exception as e:
            self.logger.error("An error occurred while dropping columns: %s" % str(e))
    
    def check_missing(self):
        try:
            result = pd.concat([self.data.isnull().sum(), self.data.isnull().mean()], axis=1)
            result = result.rename(index=str, columns={0: 'total missing', 1: 'proportion'})
            return result
        except Exception as e:
            self.logger.error("An error occurred while checking missing values: %s", str(e))
            raise

    def drop_missing(self, axis=0):
        try:
            original_shape = self.data.shape
            self.data = self.data.dropna(axis=axis)
            if self.data.shape == original_shape:
                return None  
            else:
                return self.data
        except Exception as e:
            self.logger.error("An error occurred while dropping missing values: %s", str(e))
            raise
            
    def add_var_denote_NA(self, NA_col=[]):
        try:
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    self.data[i] = np.where(self.data[i].isnull(), 1, 0)
                    return self.data
                else:
                    self.logger.warning("Column %s has no missing cases", i)
        except Exception as e:
            self.logger.error("An error occurred while adding variable to denote NA: %s", str(e))
            raise

    def impute_NA_with_arbitrary(self, impute_value, NA_col=[]):
        try:
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    self.data[i].fillna(impute_value, inplace=True)
                else:
                    self.logger.warning("Column %s has no missing cases", i)
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with arbitrary value: %s", str(e))
            raise

    def impute_NA_with_avg(self, strategy='mean', NA_col=[]):
        try:
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    if strategy == 'mean':
                        self.data[i].fillna(self.data[i].mean(), inplace=True)
                    elif strategy == 'median':
                        self.data[i].fillna(self.data[i].median(), inplace=True)
                    elif strategy == 'mode':
                        self.data[i].fillna(self.data[i].mode()[0], inplace=True)
                else:
                    self.logger.warning("Column %s has no missing", i)
            return self.data
        except Exception as e:
            error_msg = "An error occurred while imputing NA with average: %s" % str(e)
            self.logger.error(error_msg)

    def impute_NA_with_end_of_distribution(self, NA_col=[]):
        try:
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    self.data[i].fillna(self.data[i].mean() + 3 * self.data[i].std(), inplace=True)
                    return self.data
                else:
                    self.logger.warning("Column %s has no missing", i)
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with end of distribution: %s", str(e))
            raise

    def impute_NA_with_random(self, NA_col=[], random_state=0):
        try:
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    random_sample = self.data[i].dropna().sample(self.data[i].isnull().sum(), random_state=random_state)
                    random_sample.index = self.data[self.data[i].isnull()].index
                    self.data.loc[self.data[i].isnull(), i] = random_sample
                    return self.data
                
                else:
                    self.logger.warning("Column %s has no missing", i)
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with random sampling: %s", str(e))
            raise

    def impute_NA_with_interpolation(self, method='linear', limit=None, limit_direction='forward', NA_col=[]):
        try:
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    self.data[i] = self.data[i].interpolate(method=method, limit=limit, limit_direction=limit_direction)
                    return self.data
                else:
                    self.logger.warning("Column %s has no missing cases", i)
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with interpolation: %s", str(e))
            raise

    def impute_NA_with_knn(self, NA_col=[], n_neighbors=5):
        try:
            from sklearn.impute import KNNImputer
            knn_imputer = KNNImputer(n_neighbors=n_neighbors)
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    imputed_values = knn_imputer.fit_transform(self.data[i].values.reshape(-1, 1))
                    self.data[i] = imputed_values.ravel()
                    return self.data
                else:
                    self.logger.warning("Column %s has no missing cases", i)
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with KNN: %s", str(e))
            raise

    def impute_NA_with_simple_imputer(self, NA_col=[], strategy='mean'):
        try:
            from sklearn.impute import SimpleImputer
            
            for i in NA_col:
                if self.data[i].isnull().sum() > 0:
                    imputer = SimpleImputer(strategy=strategy)
                    imputed_data = imputer.fit_transform(self.data[i].values.reshape(-1, 1))
                    self.data[i] = imputed_data.ravel()
                else:
                    self.logger.warning("Column %s has no missing cases", i)
            return self.data
        
        except Exception as e:
            self.logger.error("An error occurred while imputing NA with SimpleImputer: %s", str(e))
            raise