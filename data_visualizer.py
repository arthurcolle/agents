import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger("data-visualizer")

class AdvancedDataVisualizer:
    """
    Advanced data visualization capabilities for the CLI agent
    """
    def __init__(self, output_dir="visualizations"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Advanced data visualizer initialized with output directory: {output_dir}")
        
        # Set default style
        sns.set_theme(style="whitegrid")
        plt.rcParams["figure.figsize"] = (12, 8)
    
    def create_visualization(self, data: Any, viz_type: str, **kwargs) -> Dict:
        """Create a visualization based on the specified type"""
        try:
            # Convert data to DataFrame if needed
            df = self._ensure_dataframe(data)
            
            # Generate a unique filename
            import uuid
            filename = f"{viz_type}_{uuid.uuid4().hex[:8]}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Create the visualization
            if viz_type == "correlation_matrix":
                self._create_correlation_matrix(df, filepath, **kwargs)
            elif viz_type == "pairplot":
                self._create_pairplot(df, filepath, **kwargs)
            elif viz_type == "distribution":
                self._create_distribution_plot(df, filepath, **kwargs)
            elif viz_type == "boxplot":
                self._create_boxplot(df, filepath, **kwargs)
            elif viz_type == "timeseries":
                self._create_timeseries_plot(df, filepath, **kwargs)
            elif viz_type == "3d_scatter":
                self._create_3d_scatter(df, filepath, **kwargs)
            else:
                return {
                    "success": False,
                    "message": f"Unknown visualization type: {viz_type}",
                    "data": None
                }
            
            return {
                "success": True,
                "message": f"Successfully created {viz_type} visualization",
                "data": {
                    "file": filepath,
                    "type": viz_type
                }
            }
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")
            return {
                "success": False,
                "message": f"Error creating visualization: {str(e)}",
                "data": None
            }
    
    def _ensure_dataframe(self, data: Any) -> pd.DataFrame:
        """Convert various data formats to pandas DataFrame"""
        if isinstance(data, pd.DataFrame):
            return data
        elif isinstance(data, dict):
            if "head" in data and isinstance(data["head"], list):
                return pd.DataFrame(data["head"])
            elif "data" in data and isinstance(data["data"], list):
                return pd.DataFrame(data["data"])
            else:
                return pd.DataFrame(data)
        elif isinstance(data, list):
            if all(isinstance(item, dict) for item in data):
                return pd.DataFrame(data)
            else:
                return pd.DataFrame(data)
        else:
            raise ValueError(f"Cannot convert data of type {type(data)} to DataFrame")
    
    def _create_correlation_matrix(self, df: pd.DataFrame, filepath: str, **kwargs) -> None:
        """Create a correlation matrix heatmap"""
        plt.figure(figsize=(12, 10))
        
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculate correlation
        corr = numeric_df.corr()
        
        # Create heatmap
        mask = np.triu(np.ones_like(corr, dtype=bool))
        cmap = kwargs.get("cmap", "coolwarm")
        
        sns.heatmap(
            corr, 
            mask=mask, 
            cmap=cmap,
            vmax=1, 
            vmin=-1, 
            center=0,
            square=True, 
            linewidths=.5, 
            annot=True, 
            fmt=".2f"
        )
        
        plt.title(kwargs.get("title", "Correlation Matrix"))
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _create_pairplot(self, df: pd.DataFrame, filepath: str, **kwargs) -> None:
        """Create a pairplot for multivariate analysis"""
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Limit to max 5 columns to avoid excessive plots
        if numeric_df.shape[1] > 5:
            columns = kwargs.get("columns", numeric_df.columns[:5].tolist())
            numeric_df = numeric_df[columns]
        
        hue = kwargs.get("hue", None)
        if hue and hue in df.columns:
            g = sns.pairplot(df, vars=numeric_df.columns, hue=hue)
        else:
            g = sns.pairplot(df[numeric_df.columns])
            
        plt.suptitle(kwargs.get("title", "Pairwise Relationships"), y=1.02)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _create_distribution_plot(self, df: pd.DataFrame, filepath: str, **kwargs) -> None:
        """Create distribution plots for numeric columns"""
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Limit to max 4 columns
        if numeric_df.shape[1] > 4:
            columns = kwargs.get("columns", numeric_df.columns[:4].tolist())
            numeric_df = numeric_df[columns]
        else:
            columns = numeric_df.columns
        
        n_cols = len(columns)
        fig, axes = plt.subplots(n_cols, 1, figsize=(12, 4 * n_cols))
        
        if n_cols == 1:
            axes = [axes]
        
        for i, col in enumerate(columns):
            sns.histplot(df[col], kde=True, ax=axes[i])
            axes[i].set_title(f"Distribution of {col}")
            
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _create_boxplot(self, df: pd.DataFrame, filepath: str, **kwargs) -> None:
        """Create boxplots for numeric columns"""
        # Get numeric columns only
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Limit to max 6 columns
        if numeric_df.shape[1] > 6:
            columns = kwargs.get("columns", numeric_df.columns[:6].tolist())
            numeric_df = numeric_df[columns]
        else:
            columns = numeric_df.columns
        
        plt.figure(figsize=(12, 8))
        sns.boxplot(data=numeric_df)
        plt.title(kwargs.get("title", "Box Plot of Numeric Features"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _create_timeseries_plot(self, df: pd.DataFrame, filepath: str, **kwargs) -> None:
        """Create a time series plot"""
        # Check for datetime column
        date_col = kwargs.get("date_column", None)
        value_col = kwargs.get("value_column", None)
        
        if date_col is None:
            # Try to find a datetime column
            for col in df.columns:
                if pd.api.types.is_datetime64_any_dtype(df[col]) or "date" in col.lower() or "time" in col.lower():
                    date_col = col
                    break
        
        if date_col is None:
            raise ValueError("No date column specified or found")
        
        if value_col is None:
            # Use the first numeric column that's not the date
            for col in df.columns:
                if col != date_col and pd.api.types.is_numeric_dtype(df[col]):
                    value_col = col
                    break
        
        if value_col is None:
            raise ValueError("No value column specified or found")
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # Sort by date
        df = df.sort_values(by=date_col)
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(x=date_col, y=value_col, data=df)
        plt.title(kwargs.get("title", f"Time Series: {value_col} over {date_col}"))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
    
    def _create_3d_scatter(self, df: pd.DataFrame, filepath: str, **kwargs) -> None:
        """Create a 3D scatter plot"""
        from mpl_toolkits.mplot3d import Axes3D
        
        # Get columns for x, y, z
        x_col = kwargs.get("x_column")
        y_col = kwargs.get("y_column")
        z_col = kwargs.get("z_column")
        
        if not (x_col and y_col and z_col):
            # Use the first three numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) < 3:
                raise ValueError("Need at least 3 numeric columns for 3D scatter plot")
            x_col, y_col, z_col = numeric_cols[:3]
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Add color dimension if specified
        color_col = kwargs.get("color_column")
        if color_col and color_col in df.columns:
            scatter = ax.scatter(
                df[x_col], 
                df[y_col], 
                df[z_col], 
                c=df[color_col], 
                cmap=kwargs.get("cmap", "viridis"),
                alpha=0.7
            )
            plt.colorbar(scatter, label=color_col)
        else:
            ax.scatter(df[x_col], df[y_col], df[z_col], alpha=0.7)
        
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_zlabel(z_col)
        ax.set_title(kwargs.get("title", "3D Scatter Plot"))
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close()
