import pandas as pd
from tabulate import tabulate
from IPython.display import display, HTML
import ipywidgets as widgets

class DataOverview:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        self.data = data
    
    def _display_info(self):
        """
        Displays summary information about the dataset in a notebook-friendly format.
        """
        display(HTML("<h3>Dataset Information</h3>"))
        display(HTML(f"<p>Rows: {self.data.shape[0]}</p>"))
        display(HTML(f"<p>Columns: {self.data.shape[1]}</p>"))
        display(HTML("<p>Info:</p>"))
        
        # Get the summary information of the DataFrame
        summary_info = self.data.dtypes.reset_index()
        summary_info.columns = ['Column', 'Dtype']
        summary_info['Non-Null Count'] = self.data.count().values
        
        # Display the summary information in a table format
        display(HTML(tabulate(summary_info, headers='keys', tablefmt='html')))
    
    def _describe(self):
        """
        Generates descriptive statistics for the dataset.
        """
        profile = self.data.describe(include='all')
        return profile

    def data_overview(self):
        """
        Displays an overview of the dataset, including its structure and descriptive statistics,
        using widgets for interactivity in a Jupyter Notebook.
        """
        display(HTML("<h1 style='text-align: center; font-size: 30px;'>Dataset Overview</h1>"))
        display(HTML("<hr>"))
        display(HTML("<h2 style='text-align: center; font-size: 20px;'>Dataset</h2>"))
        
        # Display the dataset
        display(self.data)
        
        # Dropdown to select overview option
        overview_option = widgets.Dropdown(
            options=['Dataset Information', 'Describe'],
            description='Select Overview Option',
            style={'description_width': 'initial'},
            layout={'width': 'max-content'}
        )
        
        # Display the dropdown
        display(overview_option)

        # Callback function to update display based on selection
        def on_option_change(change):
            if change['new'] == "Dataset Information":
                self._display_info()
            elif change['new'] == "Describe":
                display(HTML("<h3>Descriptive Statistics</h3>"))
                display(self._describe())
        
        # Trigger the callback function when the dropdown selection changes
        overview_option.observe(on_option_change, names='value')
