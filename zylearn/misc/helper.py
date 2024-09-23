import pandas as pd
import logging
import plotly.graph_objects as go

class Helper:
    def __init__(self):
        pass
    
    def convert_to_float(self, data):

        # Convert integer columns to float
        int_cols = data.select_dtypes(include=['int']).columns
        data[int_cols] = data[int_cols].astype('float')
        
        # Convert object columns to float
        obj_cols = data.select_dtypes(include=['object']).columns
        for col in obj_cols:
            try:
                data[col] = pd.to_numeric(data[col], errors='coerce')
            except ValueError as e:
                logging.error(f"Error converting column '{col}': {e}")  # Use logging instead of print
        
        return data
    
    def plot_outliers(self, data, col, outlier_index):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data[col], mode='markers', name='Data'))
        fig.add_trace(go.Scatter(x=outlier_index, y=data.loc[outlier_index, col],
                                mode='markers', marker=dict(color='red', size=8), name='Outliers'))
        fig.update_layout(title=f'Outlier Detection: {col}', xaxis_title='Index', yaxis_title=col)
        fig.show()