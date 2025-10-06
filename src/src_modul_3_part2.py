import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class spectral_plotter:
    """
    Class container to plot region of interest
    """
    #Initialize the class. 
    def __init__(self, sample_quality):
        """Initialize using functions from sample_quality class"""
        self.sq = sample_quality
        self.band_names = self.sq.band_names
        self.class_property = self.sq.class_property
    
    #-----------------OVERLAID HISTOGRAM--------------------------------
    def plot_histogram(self, df, bands=None, max_bands = 3, bins=30, opacity = 0.6):
        """
        Plot overlaid histograms using Plotly for better interactivity.
        All classes shown on same plot for easy comparison.

        Parameters
        ----------
        df : pandas.DataFrame
        bands : list, optional. Bands to plot. If None, take the first 'max_bands'.
        max_bands : int. Maximum number of bands to plot.
        bins : int. Number of bins for histogram
        opacity : float. Transparency of bars (0-1)
        
        Returns
        -------
        list of plotly.graph_objects.Figure
        """
        #Print error message if dataframe from sample analysis is empty
        if df.empty:
            print("No data avaliable for creating histogram")
            return []
        #print error message if bands are empty 
        if bands is None:
            bands = [b for b in self.band_names if b in df.columns][:max_bands]
        #Empty figures list for storing the result
        figures = []
        classes = sorted(df[self.class_property].unique())
        class_mapping = self.sq.class_renaming()
        #Histogram plotting function
        for band in bands:
            figs = go.Figure()
            for class_id in classes:
                class_data = df[df[self.class_property] == class_id][band]
                
                #Get display name
                if class_mapping and class_id in class_mapping:
                    display_name = f"{class_mapping[class_id]} (ID: {class_id})"
                else:
                    display_name = f"Class {class_id}"
                
                figs.add_trace(go.Histogram(
                    x=class_data,
                    name=display_name,
                    opacity=opacity,
                    nbinsx=bins,
                    hovertemplate='<b>%{fullData.name}</b><br>' +
                                  f'{band}: %{{x:.4f}}<br>' +
                                  'Count: %{y}<br>' +
                                  '<extra></extra>'
                ))
            #Modified the annotation for the final plot
            figs.update_layout(
                title = f'Distribution of {band} Reflectance by Class',
                xaxis_title = f'{band} Reflectance',
                yaxis_title = 'Frequency',
                barmode = 'overlay',
                hovermode = 'closest',
                height = 500, 
                legend = dict(
                    title = 'Land Cover Class',
                    orientation = 'v',
                    yanchor = "top",
                    y=1,
                    xanchor = "left",
                    x=1.02
                ),
                template = 'plotly_white'
            )
            figures.append(figs)
        #Return plotly object, therefore it is customizeable during the visualization
        return figures
    
    #-----------------BOX Plot--------------------------------
    def plot_boxplot(self, df, bands=None, max_bands = 5):
        """
        Plot boxplot interactively using plotly
        All classes shown on same plot for easy comparison.
        
        Parameters
        ----------
        df : pandas.DataFrame
        bands : list, optional
            Bands to plot. If None, take the first `max_bands`.
        max_bands : int
            Maximum number of bands to plot.
        bins : int
            Number of bins for histogram
        opacity : float
            Transparency of bars (0-1)
            
        Returns
        -------
        list of plotly.graph_objects.Figure
        """
        #Print error message if dataframe from sample analysis is empty
        if df.empty:
            print("No data avaliable for creating histogram")
            return []
        #print error message if bands are empty 
        if bands is None:
            bands = [b for b in self.band_names if b in df.columns][:max_bands]
        #Empty list to store the figures
        figures = []
        class_mapping = self.sq.class_renaming()
        #get the dataframe from sample_quality analysis and display them on box plot
        df_plot = df.copy()
        if class_mapping:
            df_plot ['Class_Display'] = df_plot[self.class_property].map(
                lambda x: f"{class_mapping.get(x, f'Class {x}')}(ID: {x})"
            )
        else:
            df_plot['Class_Display'] = df_plot[self.class_property].map(lambda x: f"Class {x}")
        #core function for box plot visualiazation
        for band in bands:
            fig = px.box(
                df_plot, 
                x='Class_Display', 
                y=band,
                color='Class_Display',
                title=f'Boxplot of {band} by Class',
                labels={'Class_Display': 'Class', band: f'{band} Reflectance'},
                hover_data={self.class_property: True, band: ':.4f'}
            )
            
            fig.update_layout(
                height=500,
                showlegend=False,
                xaxis_tickangle=-45,
                template='plotly_white',
                hovermode='closest'
            )
            
            fig.update_traces(
                hovertemplate='<b>%{x}</b><br>' +
                              f'{band}: %{{y:.4f}}<br>' +
                              '<extra></extra>'
            )
            
            figures.append(fig)
        
        return figures
  
    #-----------------Interactive Scatter Plot--------------------------------

    def interactive_scatter_plot(self, df, x_band=None, y_band=None, marker_size=6, 
                    opacity=0.6):
        """
        Create interactive scatter plot using Plotly.

        Parameters
        ----------
        df : pandas.DataFrame
        bands : list, optional
            Bands to plot. If None, take the first `max_bands`.
        max_bands : int
            Maximum number of bands to plot.
            
        Returns
        -------
        list of plotly.graph_objects.Figure
        """
        if df.empty:
            print("No data available for plotting")
            return None
        
        # Get available spectral bands
        available_bands = [col for col in df.columns 
                          if col != self.class_property and col in self.band_names]
        
        if len(available_bands) < 2:
            print("Need at least 2 bands for scatter plot.")
            return None
        
        # Set default bands if not provided
        if x_band is None:
            x_band = available_bands[0]
        if y_band is None:
            y_band = available_bands[1]
        
        # Check if specified bands exist
        if x_band not in available_bands or y_band not in available_bands:
            print(f"Specified bands not found. Available: {available_bands}")
            return None
        
        # Prepare data with display names
        df_plot = df.copy()
        class_mapping = self.sq.class_renaming()
        
        if class_mapping:
            df_plot['Class_Display'] = df_plot[self.class_property].map(
                lambda x: f"{class_mapping.get(x, f'Class {x}')} (ID: {x})"
            )
        else:
            df_plot['Class_Display'] = df_plot[self.class_property].map(lambda x: f"Class {x}")
        
        # Create scatter plot
        fig = px.scatter(
            df_plot,
            x=x_band,
            y=y_band,
            color='Class_Display',
            title=f'Spectral Scatter Plot: {y_band} vs {x_band}',
            labels={
                x_band: f'{x_band} Reflectance',
                y_band: f'{y_band} Reflectance',
                'Class_Display': 'Land Cover Class'
            },
            hover_data={
                self.class_property: True,
                x_band: ':.4f',
                y_band: ':.4f',
                'Class_Display': False
            },

            opacity=opacity
        )
        
        fig.update_traces(
            marker=dict(size=marker_size, line=dict(width=0.5, color='white')),
            hovertemplate='<b>%{fullData.name}</b><br>' +
                         f'{x_band}: %{{x:.4f}}<br>' +
                         f'{y_band}: %{{y:.4f}}<br>' +
                         '<extra></extra>'
        )
        
        fig.update_layout(
            height=600,
            template='plotly_white',
            hovermode='closest',
            legend=dict(
                title='Land Cover Classes',
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig    
    #-----------------Static Scatter Plot--------------------------------
    def static_scatter_plot(self, df, x_band=None, y_band=None, alpha=0.6, figsize=(10, 8), 
                         color_palette='tab10', add_legend=True, add_ellipse=False):
        """
        Plot region of interest in a feature space between two bands
        Parameters:
            df : pandas.DataFrame. The dataframe from extract_spectral_values containing spectral data
            x_band : str, optional. Band name for x-axis. If None, uses first available band
            y_band : str, optional. Band name for y-axis. If None, uses second available band
            alpha : float. Transparency of points (0-1)
            figsize : tuple. Figure size (width, height)
            color_palette : str. Color palette for different classes
            add_legend : bool. Whether to add legend
            add_ellipse : bool. Whether to add confidence ellipses for each class
        Returns:
        scatter plot figures
                """
        if df.empty:
            print("No data avaliable for plotting")
        # Get available spectral bands
        available_bands = [col for col in df.columns 
                        if col != self.class_property and col in self.band_names]
        if len(available_bands) < 2:
            print("Need at least 2 bands for scatter plot.")
            return None
        # Set default bands if not provided
        if x_band is None:
            x_band = available_bands[0]
        if y_band is None:
            y_band = available_bands[1] if len(available_bands) > 1 else available_bands[0]    
        # Check if specified bands exist
        if x_band not in available_bands:
            print(f"Band {x_band} not found. Available bands: {available_bands}")
            return None
        if y_band not in available_bands:
            print(f"Band {y_band} not found. Available bands: {available_bands}")
            return None
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        # Get unique classes and create color mapping
        classes = sorted(df[self.class_property].unique())
        colors = plt.cm.get_cmap(color_palette)(np.linspace(0, 1, len(classes)))
        # Get class names if available
        class_mapping = self.sq.class_renaming()
        # Plot each class
        for i, class_id in enumerate(classes):
            class_data = df[df[self.class_property] == class_id]
            # Get display name for class
            if class_mapping and class_id in class_mapping:
                display_name = f"{class_mapping[class_id]} (ID: {class_id})"
            else:
                display_name = f"Class {class_id}"
            # Create scatter plot
            ax.scatter(
                class_data[x_band], 
                class_data[y_band],
                c=[colors[i]], 
                alpha=alpha,
                label=display_name,
                s=20,  # Point size
                edgecolors='white',
                linewidths=0.5
            )
                # Add confidence ellipse if requested
            if add_ellipse and len(class_data) > 2:
                self.add_elipse(
                    ax, class_data[x_band], class_data[y_band], 
                    colors[i], alpha=0.2
                )            
        # Customize plot
        ax.set_xlabel(f'{x_band} Reflectance', fontsize=12)
        ax.set_ylabel(f'{y_band} Reflectance', fontsize=12)
        ax.set_title(f'Spectral Scatter Plot: {y_band} vs {x_band}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        # Add legend
        if add_legend:
            legend = ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            legend.set_title('Land Cover Classes', prop={'size': 11, 'weight': 'bold'})
        
        # Adjust layout to prevent legend cutoff
        plt.tight_layout()
        
        return fig
    #-----------------Scatter Plot Elipse--------------------------------
    def add_elipse(self, ax, x, y, color, n_std = 2, alpha=0.2):
        """
        """
        try:
            cov = np.cov(x, y)
            pearson = cov[0,1]/np.sqrt(cov[0,0] * cov[1,1])
            ell_radius_x = np.sqrt(1 + pearson)
            ell_radius_y = np.sqrt(1 - pearson)
            # Create ellipse
            ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                            facecolor=color, alpha=alpha, edgecolor=color, linewidth=1.5)
            
            # Transform ellipse to data coordinates
            scale_x = np.sqrt(cov[0, 0]) * n_std
            mean_x = np.mean(x)
            scale_y = np.sqrt(cov[1, 1]) * n_std
            mean_y = np.mean(y)  
            transf = transforms.Affine2D() \
            .scale(scale_x, scale_y) \
            .translate(mean_x, mean_y)
        
            ellipse.set_transform(transf + ax.transData)
            ax.add_patch(ellipse)
        except Exception as e:
            print(f"Warning: Could not add confidence ellipse: {e}")
    
    
    def plot_scatter_combo(self, df, band_combinations=None, max_combinations=6, 
                       marker_size=5, opacity=0.6):
        """
        Plot multiple scatter plots for different band combinations using Plotly subplots.
        
        Parameters
        ----------
        df : pandas.DataFrame
        band_combinations : list of tuples, optional. List of (x_band, y_band) tuples
        max_combinations : int. Maximum number of combinations to plot
        marker_size : int. Size of scatter points
        opacity : float. Point transparency

        Returns
        -------
        plotly.graph_objects.Figure
        """
        if df.empty:
            print("No data available for plotting.")
            return None
        
        available_bands = [col for col in df.columns 
                          if col != self.class_property and col in self.band_names]
        
        if len(available_bands) < 2:
            print("Need at least 2 bands for scatter plots.")
            return None
        
        # Create default band combinations if not provided
        if band_combinations is None:
            band_combinations = []
            common_pairs = [
                ('NIR', 'RED'), ('SWIR1', 'NIR'), ('GREEN', 'RED'),
                ('SWIR2', 'SWIR1'), ('BLUE', 'GREEN'), ('NIR', 'SWIR1')
            ]
            
            for x_band, y_band in common_pairs:
                if x_band in available_bands and y_band in available_bands:
                    band_combinations.append((x_band, y_band))
            
            if not band_combinations:
                for i in range(len(available_bands)):
                    for j in range(i+1, min(len(available_bands), i+4)):
                        band_combinations.append((available_bands[i], available_bands[j]))
        
        band_combinations = band_combinations[:max_combinations]
        
        # Calculate subplot grid
        n_plots = len(band_combinations)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Prepare data with display names
        df_plot = df.copy()
        class_mapping = self.sq.class_renaming()
        classes = sorted(df[self.class_property].unique())
        
        if class_mapping:
            df_plot['Class_Display'] = df_plot[self.class_property].map(
                lambda x: f"{class_mapping.get(x, f'Class {x}')}"
            )
        else:
            df_plot['Class_Display'] = df_plot[self.class_property].map(lambda x: f"Class {x}")
        
        # Create subplots
        subplot_titles = [f'{y_band} vs {x_band}' for x_band, y_band in band_combinations]
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=subplot_titles,
            horizontal_spacing=0.1,
            vertical_spacing=0.12
        )
        
        # Color mapping for classes
        colors = px.colors.qualitative.Plotly[:len(classes)]
        if len(classes) > len(colors):
            colors = px.colors.sample_colorscale("turbo", [n/(len(classes)-1) for n in range(len(classes))])
        
        # Plot each combination
        for idx, (x_band, y_band) in enumerate(band_combinations):
            row = idx // n_cols + 1
            col = idx % n_cols + 1
            
            for i, class_id in enumerate(classes):
                class_data = df_plot[df_plot[self.class_property] == class_id]
                
                if class_mapping and class_id in class_mapping:
                    display_name = f"{class_mapping[class_id]}"
                else:
                    display_name = f"Class {class_id}"
                
                fig.add_trace(
                    go.Scatter(
                        x=class_data[x_band],
                        y=class_data[y_band],
                        mode='markers',
                        name=display_name,
                        marker=dict(
                            size=marker_size,
                            color=colors[i],
                            opacity=opacity,
                            line=dict(width=0.3, color='white')
                        ),
                        legendgroup=display_name,
                        showlegend=(idx == 0),  # Only show legend for first subplot
                        hovertemplate=f'<b>{display_name}</b><br>' +
                                     f'{x_band}: %{{x:.4f}}<br>' +
                                     f'{y_band}: %{{y:.4f}}<br>' +
                                     '<extra></extra>'
                    ),
                    row=row,
                    col=col
                )
            
            # Update axes labels
            fig.update_xaxes(title_text=x_band, row=row, col=col)
            fig.update_yaxes(title_text=y_band, row=row, col=col)
        
        fig.update_layout(
            title_text='Spectral Scatter Plot Combinations',
            height=300 * n_rows,
            template='plotly_white',
            hovermode='closest',
            legend=dict(
                title='Land Cover Classes',
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            )
        )
        
        return fig

        

