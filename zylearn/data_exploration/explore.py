import os
from dateutil import parser
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import logging


# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')


class DataAnalyzer:
    def __init__(self, data):
        if not isinstance(data, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")
        self.data = data
        self.numeric_data = self.data.select_dtypes(include='number')
            

    def _line_plot(self, x, y_list):
        """Creates a line plot for multiple y-axis variables against a single x-axis variable. """
        
        # Use Plotly Express to create the line plot
        fig = px.line(self.data, x=x, y=y_list, markers=True)

        # Update layout and display the plot using Streamlit
        fig.update_layout(title=f'Line Plot: {", ".join(y_list)} vs {x}', xaxis_title=x, yaxis_title="Value")
        fig.show()
    
    def _boxplot_with_outliers(self, x, y, output_path=None):
        """
        Creates a box plot with optional outlier analysis.
        """
        fig = go.Figure()
        fig.add_trace(go.Box(x=self.data[x], y=self.data[y], name=y, boxpoints='outliers'))
        
        fig.update_layout(title=f'Box Plot: {y} vs {x}')
        
        if output_path:
            output = os.path.join(output_path, f'Boxplot_with_outliers_{x}_{y}.html')
            fig.write_html(output)
            print('Figure saved at:', output)
        
        fig.show()
   
    def _pie_chart(self, category_columns, target_column):
        """
        Creates dynamic pie charts to visualize the distribution of multiple categorical variables.

        Args:
            category_columns (list of str): List of column names for the categorical variables.

        """
       # Iterate over each category column and create a pie chart for each category
        for category_column in category_columns:
            # Get unique categories in the current category column
            categories = self.data[category_column].unique()

            # Define a color palette for the pie chart slices
            color_palette = px.colors.qualitative.Pastel

            # Iterate over each category and create a pie chart
            for category in categories:
                category_data = self.data[self.data[category_column] == category]
                target_counts = category_data[target_column].value_counts()

                # Use Plotly Express to create the pie chart
                fig = px.pie(names=target_counts.index, values=target_counts.values,
                             title=f'Distribution of {target_column} in {category_column} {category}',
                             color_discrete_sequence=color_palette)

                # Customize pie chart layout
                fig.update_traces(textposition='inside', textinfo='percent+label', pull=0.05,
                                  marker=dict(line=dict(color='white', width=1)))

                # Display the pie chart using Streamlit's st.plotly_chart
                fig.show()
                
    def _pairwise_scatter_matrix(self, variables, output_path=None):
        """
        Creates a pairwise scatter plot matrix for multiple variables.
        """
        scatter_matrix_fig = px.scatter_matrix(self.data[variables], title='Pairwise Scatter Plot Matrix')
        
        if output_path:
            output = os.path.join(output_path, 'pairwise_scatter_matrix.html')
            scatter_matrix_fig.write_html(output)
            print('Pairwise scatter plot matrix saved at:', output)
        
        scatter_matrix_fig.show()
    
    def _categorical_heatmap(self, x, y, output_path=None):
        """
        Creates a heatmap for visualizing relationships between two categorical variables.
        """
        pivot_table = self.data.pivot_table(index=x, columns=y, aggfunc='size', fill_value=0)
        
        heatmap_fig = go.Figure(data=go.Heatmap(
            z=pivot_table.values,
            x=pivot_table.columns.tolist(),
            y=pivot_table.index.tolist(),
            colorscale='Viridis'
        ))
        
        heatmap_fig.update_layout(title=f'Heatmap: {x} vs {y}')
        
        if output_path:
            output = os.path.join(output_path, f'heatmap_{x}_vs_{y}.html')
            heatmap_fig.write_html(output)
            print('Heatmap saved at:', output)
        
        heatmap_fig.show()

    def _discrete_var_barplot(self, x, y, output_path=None):
        """
        Creates a bar plot for discrete variables.
        """
        
        fig = go.Figure(data=[go.Bar(x=self.data[x], y=self.data[y], marker_color="#f95738")])
        
        if output_path:
            output = os.path.join(output_path, f'Barplot_{x}.html')
            fig.write_html(output)
            print('Figure saved at:', output)
    
        fig.show()

    def _discrete_var_countplot(self, x, output_path=None):
        """
        Creates a count plot for a discrete variable.
        """
        if x not in self.data.columns:
            logging.error(f"'{x}' column does not exist in the dataset.")
            return

        counts = self.data[x].value_counts()

        if counts.empty:
            logging.error(f"No data available for '{x}'.")
            return

        bar_trace = go.Bar(x=counts.index, y=counts.values, marker_color="#f95738")
        layout = go.Layout(title=f'Count Plot: {x}')
        
        # Create figure
        fig = go.Figure(data=[bar_trace], layout=layout)
        
        if output_path:
            output = os.path.join(output_path, f'Countplot_{x}.html')
            fig.write_html(output)
            print('Figure saved at:', output)
        
        fig.show()


    def _continuous_var_distplot(self, x, output_path=None, bins=None):
        """
        Creates a distribution plot for a continuous variable.
        """
        try:
            fig = px.histogram(self.data, x=x, nbins=bins, title=f'Distribution Plot: {x}', histnorm='probability density', marginal='box')

            x_data = self.data[x].values

            # Convert x_data to numeric type if needed (e.g., from string)
            try:
                x_data = pd.to_numeric(x_data)
            except ValueError:
                # Handle non-convertible values or NaNs if necessary
                logging.warning(f"Failed to convert '{x}' column to numeric type. Plotting distribution without KDE.")
            
            kde_fig = ff.create_distplot([x_data], ['KDE'], curve_type='kde')

            # Add KDE traces to the histogram
            for trace in kde_fig['data']:
                fig.add_trace(trace)

            if output_path:
                output = os.path.join(output_path, f'Distplot_{x}.html')
                fig.write_html(output)
                print('Figure saved at:', output)
            
            # Show the Plotly figure using Streamlit
            fig.show()

        except Exception as e:
            logging.error(f"An error occurred while generating the distribution plot: {e}")

    def _scatter_plot(self, x, y, output_path=None):
        """
        Creates a scatter plot for two continuous variables.
        """
        # Create scatter trace
        scatter_trace = go.Scatter(x=self.data[x], y=self.data[y], mode='markers', marker_color="#f95738")
        
        # Create layout
        layout = go.Layout(title=f'Scatter Plot: {y} vs {x}')
        
        # Create figure
        fig = go.Figure(data=[scatter_trace], layout=layout)
        
        if output_path:
            output = os.path.join(output_path, f'Scatter_plot_{x}_{y}.html')
            fig.write_html(output)
            print('Figure saved at:', output)
        
        fig.show()

    def _scatter_3d_plot(self, x, y, z, output_path=None):
        """
        Creates a 3D scatter plot for three variables.
        Args:
            x (str): Name of the x-axis variable.
            y (str): Name of the y-axis variable.
            z (str): Name of the z-axis variable.
            output_path (str): Optional path to save the plot as an HTML file.
        """
        # Check if z-axis data is categorical (string values)
        if pd.api.types.is_string_dtype(self.data[z]):
            logging.info("The selected z-axis variable contains categorical data (string values). "
                     "Please choose a numeric variable for the z-axis to create a 3D scatter plot.")
            return

        # Convert z data to numeric if needed
        self.data[z] = pd.to_numeric(self.data[z], errors='coerce')

        # Create 3D scatter trace
        scatter_3d_trace = go.Scatter3d(
            x=self.data[x],
            y=self.data[y],
            z=self.data[z],
            mode='markers',
            marker=dict(
                size=8,
                color=self.data[z],  # Use z variable for color
                colorscale='Viridis',
                opacity=0.8,
                colorbar=dict(title=z)  # Optional colorbar title
            )
        )

        # Create layout for the 3D scatter plot
        layout = go.Layout(
            title=f'3D Scatter Plot: {x} vs {y} vs {z}',
            scene=dict(
                xaxis=dict(title=x),
                yaxis=dict(title=y),
                zaxis=dict(title=z)
            )
        )

        # Create figure
        fig = go.Figure(data=[scatter_3d_trace], layout=layout)

        # Display the 3D scatter plot using Streamlit
        fig.show()

    def _correlation_plot(self, output_path=None):
        """
        Creates a correlation plot for numerical columns.
        """
        corr_data = self.numeric_data.corr()

        # Create heatmap trace
        heatmap_trace = go.Heatmap(
                                x=corr_data.columns,
                                y=corr_data.index,
                                z=corr_data.values,
                                colorscale='Viridis'
                                )

        # Create layout
        layout = go.Layout(title='Correlation Plot')

        # Create figure
        fig = go.Figure(data=[heatmap_trace], layout=layout)

        # Add annotations with correlation coefficients
        for i in range(len(corr_data)):
            for j in range(len(corr_data)):
                fig.add_annotation(x=corr_data.columns[i], y=corr_data.index[j],
                                text=str(round(corr_data.iloc[j, i], 2)),
                                showarrow=False)

        if output_path:
            output = os.path.join(output_path, 'Corr_plot.html')
            fig.write_html(output)
            print('Figure saved at:', output)

        fig.show()


    def _time_series_plot(self, time_column, value_column, aggregation_function='mean', time_interval='D', smoothing_technique=None, output_path=None):
        """
        Creates a time series plot based on the specified time and value columns.
        """
        try:
            # Ensure time column is datetime type
            self.data[time_column] = self.data[time_column].apply(lambda x: parser.parse(str(x)))
        
        except Exception as e:
            logging.warning(f"An error occurred: {e}")
            return
        
        try:
            # Aggregate data based on time intervals
            aggregated_data = self.data.resample(time_interval, on=time_column).agg({value_column: aggregation_function}).reset_index()
        except ValueError as e:
            logging.warning(f"An error occurred: {e}")
            return
        
        # Reset the index
        aggregated_data.reset_index(drop=True, inplace=True)
        
        # Plot time series
        time_series_fig = px.line(aggregated_data, x=time_column, y=value_column, title="Time Series Plot", labels={time_column: "Time", value_column: value_column})
        

        if smoothing_technique:
            # Apply smoothing technique
            aggregated_data[value_column] = aggregated_data[value_column].rolling(window=7).mean()
        
        # Display the plot
        time_series_fig.show()
        
        # Save plot as HTML if output path is provided
        if output_path:
            ts_plot_path = os.path.join(output_path, 'time_series_plot.html')
            time_series_fig.write_html(ts_plot_path)
            print('Time series plot saved at:', ts_plot_path)

    def _distribution_comparison_plot(self, columns, output_path=None):
        """
        Creates a plot to compare the distributions of multiple columns.
        """
        if not columns:
            logging.warning("Please select at least one column for comparison.")
            return
        
        fig = make_subplots(rows=1, cols=len(columns), subplot_titles=[f"Distribution of {column}" for column in columns])
        
        # Plot distribution for each column
        for i, column in enumerate(columns):
            fig.add_trace(go.Histogram(x=self.data[column], name=column, histnorm='probability density'), row=1, col=i+1)
        fig.update_layout(title="Distribution Comparison Plot", showlegend=False)
        
        fig.show()
        if output_path:
            dist_comp_plot_path = os.path.join(output_path, 'distribution_comparison_plot.html')
            fig.write_html(dist_comp_plot_path)
            print('Distribution comparison plot saved at:', dist_comp_plot_path)

    def _cohort_analysis(self, date_column, user_id_column, value_column):
        """
        Performs cohort analysis on the dataset.
        
        Args:
            date_column (str): Name of the column containing dates.
            user_id_column (str): Name of the column containing user IDs.
            value_column (str): Name of the column containing values for analysis.
        """
        try:
            self.data_copy = self.data.copy()
            # Convert date column to datetime
            self.data_copy[date_column] = pd.to_datetime(self.data_copy[date_column])
        except Exception as e:
            logging.info("Error: You must use a valid date column for cohort analysis.")
            logging.write(f"Details: {e}")
            return

        try:
            # Create cohort groups
            self.data_copy['Cohort'] = self.data_copy.groupby(user_id_column)[date_column].transform('min').dt.to_period('M')
            self.data_copy['Cohort_Index'] = (self.data_copy[date_column].dt.to_period('M') - self.data_copy['Cohort']).apply(lambda x: x.n)

            # Create cohort table
            cohort_table = self.data_copyx.groupby(['Cohort', 'Cohort_Index'])[value_column].mean().unstack()

            # Ensure cohort sizes are properly indexed
            cohort_sizes = cohort_table.iloc[:, 0]
            cohort_sizes.index = cohort_table.index

            # Calculate retention rates
            retention_table = cohort_table.divide(cohort_sizes, axis=0)

            # Plot heatmap
            fig = px.imshow(retention_table, 
                            labels=dict(x="Months Since First Purchase", y="Cohort", color="Retention Rate"),
                            x=retention_table.columns, 
                            y=retention_table.index.astype(str))
            fig.update_layout(title="Cohort Analysis - Retention Rates")
            fig.show()
        except Exception as e:
            logging.info("An error occurred during the cohort analysis. Please ensure your data is formatted correctly and try again.")
            logging.write(f"Details: {e}")

    def _funnel_analysis(self, stages):
       """
       Performs funnel analysis on the dataset.
       
       Args:
           stages (list): List of column names representing stages in the funnel.
       """
       # Calculate conversion at each stage
       funnel_data = []
       total = len(self.data)
       for stage in stages:
           count = self.data[stage].sum()
           percentage = (count / total) * 100
           funnel_data.append({'Stage': stage, 'Count': count, 'Percentage': percentage})
       
       # Create funnel chart
       fig = go.Figure(go.Funnel(
           y=[d['Stage'] for d in funnel_data],
           x=[d['Count'] for d in funnel_data],
           textinfo="value+percent initial"))
       
       fig.update_layout(title="Funnel Analysis")
       fig.show()
       
       # Display conversion rates between stages
       for i in range(len(funnel_data) - 1):
           conversion_rate = (funnel_data[i+1]['Count'] / funnel_data[i]['Count']) * 100
           print(f"Conversion from {funnel_data[i]['Stage']} to {funnel_data[i+1]['Stage']}: {conversion_rate:.2f}%")

    def _customer_segmentation(self, features, n_clusters=3):
       """
       Performs customer segmentation using K-means clustering.
       
       Args:
           features (list): List of column names to use for clustering.
           n_clusters (int): Number of clusters to create.
       """
       from sklearn.preprocessing import StandardScaler
       from sklearn.cluster import KMeans
       
       # Prepare the data
       X = self.data[features]
       scaler = StandardScaler()
       X_scaled = scaler.fit_transform(X)
       
       # Perform clustering
       kmeans = KMeans(n_clusters=n_clusters, random_state=42)
       self.data['Cluster'] = kmeans.fit_predict(X_scaled)
       
       # Visualize the clusters (for 2D or 3D)
       if len(features) == 2:
           fig = px.scatter(self.data, x=features[0], y=features[1], color='Cluster', title="Customer Segments")
       elif len(features) == 3:
           fig = px.scatter_3d(self.data, x=features[0], y=features[1], z=features[2], color='Cluster', title="Customer Segments")
       else:
           fig = px.parallel_coordinates(self.data, dimensions=features, color='Cluster', title="Customer Segments")
       
       fig.show()
       
       # Display cluster statistics
       for cluster in range(n_clusters):
           print(f"Cluster {cluster} statistics:")
           print(self.data[self.data['Cluster'] == cluster][features].describe())