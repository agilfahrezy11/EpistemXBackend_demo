import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms
import seaborn as sns

class spectral_plotter:
    """
    Class container to plot region of interest
    """

    def __init__(self, sample_quality):
        """Initialize using functions from sample_quality class"""
        self.sq = sample_quality
        self.band_names = self.sq.band_names
        self.class_property = self.sq.class_property
    
    #-----------------FACET HISTOGRAM--------------------------------
    def plot_facet_histograms(self, df, bands=None, max_bands=3, bins=30):
        """
        Plot faceted histograms by class for selected bands.
        """
        if df.empty:
            print("No data available for histogram plotting.")
            return
        
        if bands is None:
            bands = [b for b in self.band_names if b in df.columns][:max_bands]
        
        for band in bands:
            g = sns.displot(
                data=df, x=band, col=self.class_property,
                bins=bins, facet_kws={'sharey': False, 'sharex': True}, height=3
            )
            g.set_titles(col_template="Class {col_name}")
            plt.suptitle(f"Faceted Histograms for {band}", y=1.05, fontsize=14)
            plt.show()   

    #-----------------BOX Plot--------------------------------
    def plot_boxplots_by_band(self, df, bands=None, max_bands=5):
        """
        Plot boxplots for each band across all classes.
        
        Parameters
        ----------
        df : pandas.DataFrame
        bands : list, optional
            Bands to plot. If None, take the first `max_bands`.
        max_bands : int
            Maximum number of bands to plot.
        """
        #return error message if no data is availiable
        if df.empty:
            print("No data available for boxplot plotting.")
            return
        #used max bands if number of band is not specified
        if bands is None:
            bands = [b for b in self.band_names if b in df.columns][:max_bands]
        #create the plot
        for band in bands:
            plt.figure(figsize=(12, 6))
            sns.boxplot(x=self.class_property, y=band, data=df)
            plt.title(f"Boxplot of {band} by Class")
            plt.xticks(rotation=90)  # rotate labels for 17 classes
            plt.show()    
    #-----------------Scatter Plot--------------------------------
    def scatter_plot(self, df, x_band=None, y_band=None, alpha=0.6, figsize=(10, 8), 
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
    
    
    def plot_band_combo(self, df, band_combinations=None, max_combinations=6, 
                            figsize=(15, 10), alpha=0.6):
        """
        Plot multiple scatter plots for different band combinations in a subplot grid.
        Parameters:

        df : pandas.DataFrame. The dataframe from extract_spectral_values (must contain spectral data)
        band_combinations : list of tuples, optional. List of (x_band, y_band) tuples. If None, creates common combinations
        max_combinations : int. Maximum number of combinations to plot
        figsize : tuple. Overall figure size
        alpha : float. Point transparency

        Returns
        -------
        matplotlib.figure.Figure
        """
        #return error message if the bands is not avaliable
        if df.empty:
            print("No data available for plotting.")
            return None
        #list of avaliable bands
        available_bands = [col for col in df.columns 
                        if col != self.class_property and col in self.band_names]
        
        if len(available_bands) < 2:
            print("Need at least 2 bands for scatter plots.")
            return None
        
        # Create default band combinations if not provided
        if band_combinations is None:
            band_combinations = []
            # Common combinations for land cover analysis
            common_pairs = [
                ('NIR', 'RED'), ('SWIR1', 'NIR'), ('GREEN', 'RED'),
                ('SWIR2', 'SWIR1'), ('BLUE', 'GREEN'), ('NIR', 'SWIR1')
            ]      
            # Find available combinations
        for x_band, y_band in common_pairs:
            if x_band in available_bands and y_band in available_bands:
                band_combinations.append((x_band, y_band))
        
        # If no common pairs found, create combinations from available bands
        if not band_combinations:
            for i in range(len(available_bands)):
                for j in range(i+1, min(len(available_bands), i+4)):
                    band_combinations.append((available_bands[i], available_bands[j]))
    
        # Limit combinations
        band_combinations = band_combinations[:max_combinations]
        
        # Calculate subplot grid
        n_plots = len(band_combinations)
        n_cols = min(3, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes
        else:
            axes = axes.flatten()
        
        # Get classes and colors
        classes = sorted(df[self.class_property].unique())
        colors = plt.cm.get_cmap('tab10')(np.linspace(0, 1, len(classes)))
        class_mapping = self.sq.class_renaming()
        
        # Plot each combination
        for idx, (x_band, y_band) in enumerate(band_combinations):
            ax = axes[idx]
            
            # Plot each class
            for i, class_id in enumerate(classes):
                class_data = df[df[self.class_property] == class_id]
                
                # Get display name
                if class_mapping and class_id in class_mapping:
                    display_name = f"{class_mapping[class_id]}"
                else:
                    display_name = f"Class {class_id}"
                
                ax.scatter(
                    class_data[x_band], 
                    class_data[y_band],
                    c=[colors[i]], 
                    alpha=alpha,
                    label=display_name if idx == 0 else "",  # Only show legend on first plot
                    s=15,
                    edgecolors='white',
                    linewidths=0.3
                )
            
            # Customize subplot
            ax.set_xlabel(f'{x_band}', fontsize=10)
            ax.set_ylabel(f'{y_band}', fontsize=10)
            ax.set_title(f'{y_band} vs {x_band}', fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide empty subplots
        for idx in range(n_plots, len(axes)):
            axes[idx].set_visible(False)
        
        # Add overall legend
        if classes:
            legend_labels = []
            legend_colors = []
            for i, class_id in enumerate(classes):
                if class_mapping and class_id in class_mapping:
                    legend_labels.append(f"{class_mapping[class_id]} (ID: {class_id})")
                else:
                    legend_labels.append(f"Class {class_id}")
                legend_colors.append(colors[i])
            
            fig.legend(legend_labels, bbox_to_anchor=(1.02, 0.5), loc='center left', 
                    fontsize=9, title='Land Cover Classes', title_fontsize=10)
        
        plt.suptitle('Spectral Scatter Plot Combinations', fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        return fig  

        

