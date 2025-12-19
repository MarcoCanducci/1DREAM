#!/usr/bin/env python3
"""
1DREAM GUI - Interactive Toolbox for Manifold Learning

A local GUI application for the 1DREAM toolbox that allows users to:
1. Load point cloud datasets (CSV format)
2. Run any of the 5 methodologies: LAAT, MBMS, DimIndex, Crawling, SGTM
3. Follow the recommended workflow: LAAT → MBMS → DimIndex → Crawling → SGTM
4. Visualize results at each step

Copyright (C) 2024
License: GNU Affero General Public License v3.0

Tested on: Ubuntu 20.04+, Windows 10+, macOS 11+
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import sys
import platform
import threading
import queue
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
import traceback

# Matplotlib imports for visualization
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for embedding in Tkinter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import colorsys

# Determine the base directory (where this script is located)
BASE_DIR = Path(__file__).parent.resolve()

# Add module paths (using os.path for cross-platform compatibility)
sys.path.insert(0, str(BASE_DIR / "SGTM" / "Python"))
sys.path.insert(0, str(BASE_DIR / "Crawling" / "Python"))
sys.path.insert(0, str(BASE_DIR / "DimIndex" / "Python"))

# Platform detection for any OS-specific handling
IS_LINUX = platform.system() == "Linux"
IS_MACOS = platform.system() == "Darwin"
IS_WINDOWS = platform.system() == "Windows"

class TextRedirector:
    """Redirects stdout/stderr to a text widget"""
    def __init__(self, text_widget, tag="stdout"):
        self.text_widget = text_widget
        self.tag = tag

    def write(self, string):
        self.text_widget.configure(state='normal')
        self.text_widget.insert(tk.END, string, (self.tag,))
        self.text_widget.see(tk.END)
        self.text_widget.configure(state='disabled')

    def flush(self):
        pass


class CSVColumnSelector(tk.Toplevel):
    """
    Dialog for loading CSV files and interactively selecting columns.
    Allows users to preview data and assign columns to specific variables.
    """
    
    def __init__(self, parent, title="Select CSV Columns", 
                 column_assignments=None, single_column=False,
                 description=None):
        """
        Initialize the CSV column selector dialog.
        
        Args:
            parent: Parent window
            title: Dialog title
            column_assignments: List of dicts with 'name' and 'description' for each assignment
                               e.g., [{'name': 'X', 'description': 'X coordinate'},
                                      {'name': 'Y', 'description': 'Y coordinate'}]
                               If None, defaults to multi-column data selection
            single_column: If True, only allow selecting a single column (for indices/weights)
            description: Optional description text shown at top of dialog
        """
        super().__init__(parent)
        self.parent = parent
        self.title(title)
        self.geometry("900x600")
        self.minsize(700, 500)
        
        # Make dialog modal
        self.transient(parent)
        self.grab_set()
        
        # Data storage
        self.csv_data = None
        self.csv_path = None
        self.column_names = []
        self.result = None  # Will hold the selected data when OK is clicked
        self.single_column = single_column
        self.description = description
        
        # Column assignments configuration
        if column_assignments is None:
            if single_column:
                self.column_assignments = [{'name': 'Value', 'description': 'Select column'}]
            else:
                self.column_assignments = [
                    {'name': 'Dim 1', 'description': 'First dimension'},
                    {'name': 'Dim 2', 'description': 'Second dimension'},
                    {'name': 'Dim 3', 'description': 'Third dimension'}
                ]
        else:
            self.column_assignments = column_assignments
        
        self.assignment_vars = {}  # Will hold StringVars for each assignment
        
        self._create_widgets()
        
        # Center dialog on parent
        self.update_idletasks()
        x = parent.winfo_x() + (parent.winfo_width() - self.winfo_width()) // 2
        y = parent.winfo_y() + (parent.winfo_height() - self.winfo_height()) // 2
        self.geometry(f"+{x}+{y}")
        
    def _create_widgets(self):
        """Create the dialog widgets"""
        # Main container
        main_frame = ttk.Frame(self, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Description label (if provided)
        if self.description:
            desc_label = ttk.Label(main_frame, text=self.description, 
                                   wraplength=850, justify=tk.LEFT)
            desc_label.pack(fill=tk.X, pady=(0, 10))
        
        # File selection frame
        file_frame = ttk.LabelFrame(main_frame, text="CSV File", padding=5)
        file_frame.pack(fill=tk.X, pady=5)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=70)
        file_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        browse_btn = ttk.Button(file_frame, text="Browse...", command=self._browse_file)
        browse_btn.pack(side=tk.LEFT)
        
        # Options frame
        options_frame = ttk.Frame(main_frame)
        options_frame.pack(fill=tk.X, pady=5)
        
        self.has_header_var = tk.BooleanVar(value=False)
        header_check = ttk.Checkbutton(options_frame, text="First row is header",
                                        variable=self.has_header_var,
                                        command=self._reload_preview)
        header_check.pack(side=tk.LEFT)
        
        ttk.Label(options_frame, text="Delimiter:").pack(side=tk.LEFT, padx=(20, 5))
        self.delimiter_var = tk.StringVar(value=",")
        delimiter_combo = ttk.Combobox(options_frame, textvariable=self.delimiter_var,
                                        values=[",", ";", "\\t", " "], width=5)
        delimiter_combo.pack(side=tk.LEFT)
        delimiter_combo.bind("<<ComboboxSelected>>", lambda e: self._reload_preview())
        
        reload_btn = ttk.Button(options_frame, text="Reload", command=self._reload_preview)
        reload_btn.pack(side=tk.LEFT, padx=(10, 0))
        
        # OK/Cancel buttons frame - pack at BOTTOM first so it's always visible
        dialog_btn_frame = ttk.Frame(main_frame)
        dialog_btn_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=10)
        
        cancel_btn = ttk.Button(dialog_btn_frame, text="Cancel", command=self._cancel)
        cancel_btn.pack(side=tk.RIGHT, padx=5)
        
        ok_btn = ttk.Button(dialog_btn_frame, text="OK", command=self._ok)
        ok_btn.pack(side=tk.RIGHT, padx=5)
        
        # Preview frame
        preview_frame = ttk.LabelFrame(main_frame, text="Data Preview (first 10 rows)", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create treeview with scrollbars for preview
        tree_frame = ttk.Frame(preview_frame)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        
        self.preview_tree = ttk.Treeview(tree_frame, show='headings', height=8)
        
        vsb = ttk.Scrollbar(tree_frame, orient="vertical", command=self.preview_tree.yview)
        hsb = ttk.Scrollbar(tree_frame, orient="horizontal", command=self.preview_tree.xview)
        self.preview_tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        self.preview_tree.grid(row=0, column=0, sticky='nsew')
        vsb.grid(row=0, column=1, sticky='ns')
        hsb.grid(row=1, column=0, sticky='ew')
        
        tree_frame.grid_columnconfigure(0, weight=1)
        tree_frame.grid_rowconfigure(0, weight=1)
        
        # Column assignment frame
        assign_frame = ttk.LabelFrame(main_frame, text="Column Assignments", padding=5)
        assign_frame.pack(fill=tk.X, pady=5)
        
        if self.single_column:
            # Single column selection
            row_frame = ttk.Frame(assign_frame)
            row_frame.pack(fill=tk.X, pady=2)
            
            assignment = self.column_assignments[0]
            ttk.Label(row_frame, text=f"{assignment['description']}:", width=20).pack(side=tk.LEFT)
            
            var = tk.StringVar(value="")
            self.assignment_vars[assignment['name']] = var
            combo = ttk.Combobox(row_frame, textvariable=var, state='readonly', width=30)
            combo.pack(side=tk.LEFT, padx=5)
            self.assignment_combos = {assignment['name']: combo}
        else:
            # Multi-column selection with "Add Dimension" capability
            self.assignments_container = ttk.Frame(assign_frame)
            self.assignments_container.pack(fill=tk.X)
            
            self.assignment_combos = {}
            self.dimension_count = 0
            
            # Add initial dimensions
            for assignment in self.column_assignments:
                self._add_dimension_row(assignment['name'], assignment['description'])
            
            # Add dimension button (for point cloud data)
            if not self.single_column:
                dim_btn_frame = ttk.Frame(assign_frame)
                dim_btn_frame.pack(fill=tk.X, pady=5)
                
                add_dim_btn = ttk.Button(dim_btn_frame, text="+ Add Dimension", 
                                         command=self._add_dimension)
                add_dim_btn.pack(side=tk.LEFT)
                
                remove_dim_btn = ttk.Button(dim_btn_frame, text="- Remove Last", 
                                            command=self._remove_dimension)
                remove_dim_btn.pack(side=tk.LEFT, padx=5)
                
                select_all_btn = ttk.Button(dim_btn_frame, text="Select All Columns", 
                                            command=self._select_all_columns)
                select_all_btn.pack(side=tk.LEFT, padx=5)
        
        # Info label
        self.info_label = ttk.Label(main_frame, text="", foreground='gray')
        self.info_label.pack(fill=tk.X, pady=5)
        
    def _add_dimension_row(self, name, description):
        """Add a dimension assignment row"""
        row_frame = ttk.Frame(self.assignments_container)
        row_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(row_frame, text=f"{description}:", width=15).pack(side=tk.LEFT)
        
        var = tk.StringVar(value="")
        self.assignment_vars[name] = var
        combo = ttk.Combobox(row_frame, textvariable=var, state='readonly', width=30)
        combo.pack(side=tk.LEFT, padx=5)
        self.assignment_combos[name] = combo
        
        self.dimension_count += 1
        
    def _add_dimension(self):
        """Add a new dimension assignment"""
        new_dim_num = self.dimension_count + 1
        name = f'Dim {new_dim_num}'
        description = f'Dimension {new_dim_num}'
        self._add_dimension_row(name, description)
        self._update_column_options()
        
    def _remove_dimension(self):
        """Remove the last dimension assignment"""
        if self.dimension_count > 1:
            name = f'Dim {self.dimension_count}'
            if name in self.assignment_vars:
                del self.assignment_vars[name]
            if name in self.assignment_combos:
                del self.assignment_combos[name]
            
            # Remove the last row frame
            children = self.assignments_container.winfo_children()
            if children:
                children[-1].destroy()
            
            self.dimension_count -= 1
            
    def _select_all_columns(self):
        """Automatically assign all columns to dimensions"""
        if not self.column_names:
            return
            
        # First, ensure we have enough dimensions
        while self.dimension_count < len(self.column_names):
            self._add_dimension()
            
        # Assign columns in order
        for i, col_name in enumerate(self.column_names):
            dim_name = f'Dim {i + 1}'
            if dim_name in self.assignment_vars:
                self.assignment_vars[dim_name].set(col_name)
                
    def _browse_file(self):
        """Browse for CSV file"""
        filepath = filedialog.askopenfilename(
            title="Select CSV File",
            filetypes=[("CSV files", "*.csv"), ("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            self.file_path_var.set(filepath)
            self.csv_path = filepath
            self._load_csv()
            
    def _get_delimiter(self):
        """Get the actual delimiter character"""
        delim = self.delimiter_var.get()
        if delim == "\\t":
            return "\t"
        return delim
        
    def _load_csv(self):
        """Load the CSV file and update preview"""
        if not self.csv_path:
            return
            
        try:
            delimiter = self._get_delimiter()
            header = 0 if self.has_header_var.get() else None
            
            self.csv_data = pd.read_csv(self.csv_path, delimiter=delimiter, header=header)
            
            # Generate column names
            if self.has_header_var.get():
                self.column_names = list(self.csv_data.columns)
            else:
                self.column_names = [f"Column {i+1}" for i in range(len(self.csv_data.columns))]
                self.csv_data.columns = self.column_names
                
            self._update_preview()
            self._update_column_options()
            
            self.info_label.config(
                text=f"Loaded: {len(self.csv_data)} rows × {len(self.csv_data.columns)} columns",
                foreground='green'
            )
            
        except Exception as e:
            self.info_label.config(text=f"Error loading file: {str(e)}", foreground='red')
            self.csv_data = None
            self.column_names = []
            
    def _reload_preview(self):
        """Reload the CSV with current options"""
        if self.csv_path:
            self._load_csv()
            
    def _update_preview(self):
        """Update the treeview preview"""
        # Clear existing data
        self.preview_tree.delete(*self.preview_tree.get_children())
        
        if self.csv_data is None:
            return
            
        # Configure columns
        self.preview_tree['columns'] = self.column_names
        for col in self.column_names:
            self.preview_tree.heading(col, text=col)
            self.preview_tree.column(col, width=100, minwidth=50)
            
        # Add first 10 rows
        for idx, row in self.csv_data.head(10).iterrows():
            values = [str(v) for v in row.values]
            self.preview_tree.insert('', 'end', values=values)
            
    def _update_column_options(self):
        """Update the column options in assignment comboboxes"""
        for combo in self.assignment_combos.values():
            combo['values'] = self.column_names
            
    def _ok(self):
        """Handle OK button - extract selected data"""
        if self.csv_data is None:
            messagebox.showwarning("Warning", "Please load a CSV file first.")
            return
            
        try:
            if self.single_column:
                # Single column mode
                col_name = list(self.assignment_vars.values())[0].get()
                if not col_name:
                    messagebox.showwarning("Warning", "Please select a column.")
                    return
                self.result = self.csv_data[col_name].values
            else:
                # Multi-column mode
                selected_cols = []
                for name, var in self.assignment_vars.items():
                    col_name = var.get()
                    if col_name:
                        selected_cols.append(col_name)
                        
                if not selected_cols:
                    messagebox.showwarning("Warning", "Please select at least one column.")
                    return
                    
                self.result = self.csv_data[selected_cols].values
                
            self.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Error extracting data: {str(e)}")
            
    def _cancel(self):
        """Handle Cancel button"""
        self.result = None
        self.destroy()
        
    def get_result(self):
        """Get the result after dialog closes"""
        self.wait_window()
        return self.result


class VisualizationDialog(tk.Toplevel):
    """
    Base class for visualization dialogs with interactive matplotlib plots.
    Supports 2D and 3D plotting with user rotation for 3D.
    """
    
    def __init__(self, parent, title="Visualization", figsize=(10, 8)):
        super().__init__(parent)
        self.parent = parent
        self.title(title)
        self.geometry("1100x800")
        self.minsize(900, 700)
        
        # Make dialog modal
        self.transient(parent)
        
        self._create_base_widgets(figsize)
        
    def _create_base_widgets(self, figsize):
        """Create base widget structure"""
        # Main container
        self.main_frame = ttk.Frame(self, padding=5)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Control panel at top
        self.control_frame = ttk.LabelFrame(self.main_frame, text="Options", padding=5)
        self.control_frame.pack(fill=tk.X, pady=(0, 5))
        
        # Figure frame
        self.fig_frame = ttk.Frame(self.main_frame)
        self.fig_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=figsize, dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.fig_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Navigation toolbar for zoom/pan/save
        self.toolbar = NavigationToolbar2Tk(self.canvas, self.fig_frame)
        self.toolbar.update()
        
        # Close button at bottom
        btn_frame = ttk.Frame(self.main_frame)
        btn_frame.pack(fill=tk.X, pady=5)
        
        save_btn = ttk.Button(btn_frame, text="Save Figure", command=self._save_figure)
        save_btn.pack(side=tk.LEFT, padx=5)
        
        close_btn = ttk.Button(btn_frame, text="Close", command=self.destroy)
        close_btn.pack(side=tk.RIGHT, padx=5)
        
    def _save_figure(self):
        """Save the current figure"""
        filetypes = [
            ("PNG files", "*.png"),
            ("PDF files", "*.pdf"),
            ("SVG files", "*.svg"),
            ("All files", "*.*")
        ]
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=filetypes,
            title="Save Figure"
        )
        if filepath:
            self.fig.savefig(filepath, dpi=150, bbox_inches='tight')
            messagebox.showinfo("Success", f"Figure saved to:\n{filepath}")
            
    def _get_plot_dims(self, data):
        """Get dimensions for plotting (max 3)"""
        ndim = data.shape[1] if len(data.shape) > 1 else 1
        return min(ndim, 3)
        
    def _create_axis(self, is_3d=False, subplot_pos=111):
        """Create 2D or 3D axis"""
        if is_3d:
            return self.fig.add_subplot(subplot_pos, projection='3d')
        else:
            return self.fig.add_subplot(subplot_pos)


class LAATVisualizationDialog(VisualizationDialog):
    """
    LAAT visualization: Point cloud colored by pheromone percentile.
    Features: colorbar, linear/log scale, percentile threshold selection.
    """
    
    def __init__(self, parent, data, pheromone, threshold_pct=5.0):
        super().__init__(parent, title="LAAT Results Visualization")
        
        self.data = data
        self.pheromone = pheromone
        self.threshold_pct = threshold_pct
        self.colorbar = None
        
        self._create_controls()
        self._plot()
        
    def _create_controls(self):
        """Create LAAT-specific controls"""
        # Scale selection
        ttk.Label(self.control_frame, text="Color Scale:").pack(side=tk.LEFT, padx=5)
        self.scale_var = tk.StringVar(value="linear")
        scale_combo = ttk.Combobox(self.control_frame, textvariable=self.scale_var,
                                   values=["linear", "log"], width=8, state='readonly')
        scale_combo.pack(side=tk.LEFT, padx=5)
        scale_combo.bind("<<ComboboxSelected>>", lambda e: self._plot())
        
        # Colorbar toggle
        self.show_colorbar_var = tk.BooleanVar(value=True)
        colorbar_check = ttk.Checkbutton(self.control_frame, text="Show Colorbar",
                                          variable=self.show_colorbar_var,
                                          command=self._plot)
        colorbar_check.pack(side=tk.LEFT, padx=10)
        
        # Percentile threshold
        ttk.Label(self.control_frame, text="Threshold Percentile:").pack(side=tk.LEFT, padx=(20, 5))
        self.percentile_var = tk.StringVar(value=str(self.threshold_pct))
        percentile_entry = ttk.Entry(self.control_frame, textvariable=self.percentile_var, width=8)
        percentile_entry.pack(side=tk.LEFT, padx=5)
        
        apply_btn = ttk.Button(self.control_frame, text="Apply", command=self._plot)
        apply_btn.pack(side=tk.LEFT, padx=5)
        
        # Point size
        ttk.Label(self.control_frame, text="Point Size:").pack(side=tk.LEFT, padx=(20, 5))
        self.point_size_var = tk.StringVar(value="1.0")
        size_entry = ttk.Entry(self.control_frame, textvariable=self.point_size_var, width=6)
        size_entry.pack(side=tk.LEFT, padx=5)
        
    def _plot(self):
        """Plot the LAAT results"""
        self.fig.clear()
        
        ndims = self._get_plot_dims(self.data)
        is_3d = ndims >= 3
        
        ax = self._create_axis(is_3d=is_3d)
        
        # Get percentile threshold
        try:
            pct = float(self.percentile_var.get())
        except ValueError:
            pct = self.threshold_pct
            
        # Convert pheromone to percentile values (0-100)
        pheromone_percentile = np.zeros_like(self.pheromone)
        sorted_indices = np.argsort(self.pheromone)
        pheromone_percentile[sorted_indices] = np.linspace(0, 100, len(self.pheromone))
        
        # Determine colormap normalization
        scale = self.scale_var.get()
        if scale == "log":
            # Shift percentiles slightly to avoid log(0)
            pheromone_percentile = np.clip(pheromone_percentile, 0.1, 100)
            norm = LogNorm(vmin=0.1, vmax=100)
        else:
            norm = Normalize(vmin=0, vmax=100)
            
        try:
            point_size = float(self.point_size_var.get())
        except ValueError:
            point_size = 1.0
            
        # Plot
        cmap = plt.cm.viridis
        
        if is_3d:
            sc = ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2],
                           c=pheromone_percentile, cmap=cmap, norm=norm, s=point_size)
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
        elif ndims == 2:
            sc = ax.scatter(self.data[:, 0], self.data[:, 1],
                           c=pheromone_percentile, cmap=cmap, norm=norm, s=point_size)
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
        else:
            sc = ax.scatter(self.data[:, 0], np.zeros_like(self.data[:, 0]),
                           c=pheromone_percentile, cmap=cmap, norm=norm, s=point_size)
            ax.set_xlabel('Dim 1')
            
        # Add threshold line indicator
        threshold_val = 100 - pct
        ax.set_title(f"LAAT Pheromone Distribution (Percentile)\nTop {pct}% threshold at {threshold_val:.1f} percentile")
        
        # Colorbar
        if self.show_colorbar_var.get():
            self.colorbar = self.fig.colorbar(sc, ax=ax, label='Pheromone Percentile')
        
        self.canvas.draw()


class MBMSVisualizationDialog(VisualizationDialog):
    """
    MBMS visualization: Input data (blue) and MBMS output (purple).
    """
    
    def __init__(self, parent, input_data, mbms_data, original_data=None):
        super().__init__(parent, title="MBMS Results Visualization")
        
        self.input_data = input_data
        self.mbms_data = mbms_data
        self.original_data = original_data
        
        self._create_controls()
        self._plot()
        
    def _create_controls(self):
        """Create MBMS-specific controls"""
        # Background data selection
        ttk.Label(self.control_frame, text="Background:").pack(side=tk.LEFT, padx=5)
        self.bg_var = tk.StringVar(value="input")
        bg_options = ["none", "input"]
        if self.original_data is not None:
            bg_options.append("original")
        bg_combo = ttk.Combobox(self.control_frame, textvariable=self.bg_var,
                                values=bg_options, width=10, state='readonly')
        bg_combo.pack(side=tk.LEFT, padx=5)
        bg_combo.bind("<<ComboboxSelected>>", lambda e: self._plot())
        
        # Point sizes
        ttk.Label(self.control_frame, text="BG Size:").pack(side=tk.LEFT, padx=(20, 5))
        self.bg_size_var = tk.StringVar(value="0.5")
        bg_size_entry = ttk.Entry(self.control_frame, textvariable=self.bg_size_var, width=6)
        bg_size_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(self.control_frame, text="MBMS Size:").pack(side=tk.LEFT, padx=(10, 5))
        self.mbms_size_var = tk.StringVar(value="2.0")
        mbms_size_entry = ttk.Entry(self.control_frame, textvariable=self.mbms_size_var, width=6)
        mbms_size_entry.pack(side=tk.LEFT, padx=5)
        
        apply_btn = ttk.Button(self.control_frame, text="Apply", command=self._plot)
        apply_btn.pack(side=tk.LEFT, padx=10)
        
    def _plot(self):
        """Plot the MBMS results"""
        self.fig.clear()
        
        ndims = self._get_plot_dims(self.mbms_data)
        is_3d = ndims >= 3
        
        ax = self._create_axis(is_3d=is_3d)
        
        try:
            bg_size = float(self.bg_size_var.get())
            mbms_size = float(self.mbms_size_var.get())
        except ValueError:
            bg_size, mbms_size = 0.5, 2.0
            
        bg_choice = self.bg_var.get()
        
        # Plot background
        if bg_choice == "input" and self.input_data is not None:
            bg_data = self.input_data
            if is_3d and bg_data.shape[1] >= 3:
                ax.scatter(bg_data[:, 0], bg_data[:, 1], bg_data[:, 2],
                          c='blue', s=bg_size, alpha=0.5, label='Input (LAAT selected)')
            elif bg_data.shape[1] >= 2:
                ax.scatter(bg_data[:, 0], bg_data[:, 1],
                          c='blue', s=bg_size, alpha=0.5, label='Input (LAAT selected)')
        elif bg_choice == "original" and self.original_data is not None:
            bg_data = self.original_data
            if is_3d and bg_data.shape[1] >= 3:
                ax.scatter(bg_data[:, 0], bg_data[:, 1], bg_data[:, 2],
                          c='lightgray', s=bg_size*0.5, alpha=0.3, label='Original data')
            elif bg_data.shape[1] >= 2:
                ax.scatter(bg_data[:, 0], bg_data[:, 1],
                          c='lightgray', s=bg_size*0.5, alpha=0.3, label='Original data')
                          
        # Plot MBMS results
        if is_3d:
            ax.scatter(self.mbms_data[:, 0], self.mbms_data[:, 1], self.mbms_data[:, 2],
                      c='purple', s=mbms_size, label=f'MBMS output ({len(self.mbms_data)} pts)')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
        elif ndims == 2:
            ax.scatter(self.mbms_data[:, 0], self.mbms_data[:, 1],
                      c='purple', s=mbms_size, label=f'MBMS output ({len(self.mbms_data)} pts)')
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
        else:
            ax.scatter(self.mbms_data[:, 0], np.zeros_like(self.mbms_data[:, 0]),
                      c='purple', s=mbms_size, label=f'MBMS output ({len(self.mbms_data)} pts)')
            ax.set_xlabel('Dim 1')
            
        ax.set_title(f"MBMS Results\nInput: {len(self.input_data)} pts → Output: {len(self.mbms_data)} pts")
        ax.legend(loc='upper right', markerscale=3)
        
        self.canvas.draw()


class DimIndexVisualizationDialog(VisualizationDialog):
    """
    DimIndex visualization: MBMS results colored by dimensionality.
    1D = red, 2D = blue, 3D = green
    """
    
    def __init__(self, parent, mbms_data, indexes, labels, laat_data=None, original_data=None):
        super().__init__(parent, title="DimIndex Results Visualization")
        
        self.mbms_data = mbms_data
        self.indexes = indexes  # Column 0: original, Column 1: smoothed
        self.labels = labels
        self.laat_data = laat_data
        self.original_data = original_data
        
        # Filter MBMS data based on labels
        self.mbms_filtered = mbms_data[labels == 1] if labels is not None else mbms_data
        
        self._create_controls()
        self._plot()
        
    def _create_controls(self):
        """Create DimIndex-specific controls"""
        # Index type selection
        ttk.Label(self.control_frame, text="Index Type:").pack(side=tk.LEFT, padx=5)
        self.index_type_var = tk.StringVar(value="smoothed")
        index_combo = ttk.Combobox(self.control_frame, textvariable=self.index_type_var,
                                   values=["original", "smoothed"], width=10, state='readonly')
        index_combo.pack(side=tk.LEFT, padx=5)
        index_combo.bind("<<ComboboxSelected>>", lambda e: self._plot())
        
        # Background data selection
        ttk.Label(self.control_frame, text="Background:").pack(side=tk.LEFT, padx=(20, 5))
        self.bg_var = tk.StringVar(value="none")
        bg_options = ["none", "laat", "original"]
        bg_combo = ttk.Combobox(self.control_frame, textvariable=self.bg_var,
                                values=bg_options, width=10, state='readonly')
        bg_combo.pack(side=tk.LEFT, padx=5)
        bg_combo.bind("<<ComboboxSelected>>", lambda e: self._plot())
        
        # Point size
        ttk.Label(self.control_frame, text="Point Size:").pack(side=tk.LEFT, padx=(20, 5))
        self.point_size_var = tk.StringVar(value="2.0")
        size_entry = ttk.Entry(self.control_frame, textvariable=self.point_size_var, width=6)
        size_entry.pack(side=tk.LEFT, padx=5)
        
        apply_btn = ttk.Button(self.control_frame, text="Apply", command=self._plot)
        apply_btn.pack(side=tk.LEFT, padx=10)
        
    def _plot(self):
        """Plot the DimIndex results"""
        self.fig.clear()
        
        ndims = self._get_plot_dims(self.mbms_filtered)
        is_3d = ndims >= 3
        
        ax = self._create_axis(is_3d=is_3d)
        
        try:
            point_size = float(self.point_size_var.get())
        except ValueError:
            point_size = 2.0
            
        # Get index column (0=original, 1=smoothed)
        idx_col = 1 if self.index_type_var.get() == "smoothed" else 0
        dim_indices = self.indexes[:, idx_col]
        
        # Plot background
        bg_choice = self.bg_var.get()
        if bg_choice == "laat" and self.laat_data is not None:
            if is_3d and self.laat_data.shape[1] >= 3:
                ax.scatter(self.laat_data[:, 0], self.laat_data[:, 1], self.laat_data[:, 2],
                          c='lightgray', s=0.3, alpha=0.3, label='LAAT data')
            elif self.laat_data.shape[1] >= 2:
                ax.scatter(self.laat_data[:, 0], self.laat_data[:, 1],
                          c='lightgray', s=0.3, alpha=0.3, label='LAAT data')
        elif bg_choice == "original" and self.original_data is not None:
            if is_3d and self.original_data.shape[1] >= 3:
                ax.scatter(self.original_data[:, 0], self.original_data[:, 1], self.original_data[:, 2],
                          c='lightgray', s=0.1, alpha=0.2, label='Original data')
            elif self.original_data.shape[1] >= 2:
                ax.scatter(self.original_data[:, 0], self.original_data[:, 1],
                          c='lightgray', s=0.1, alpha=0.2, label='Original data')
        
        # Separate data by dimension
        dim1_mask = dim_indices == 0
        dim2_mask = dim_indices == 1
        dim3_mask = dim_indices == 2
        
        colors = {'1D': 'red', '2D': 'blue', '3D': 'green'}
        
        # Plot each dimension category
        for mask, dim_label in [(dim1_mask, '1D'), (dim2_mask, '2D'), (dim3_mask, '3D')]:
            if np.sum(mask) > 0:
                data_dim = self.mbms_filtered[mask]
                if is_3d and data_dim.shape[1] >= 3:
                    ax.scatter(data_dim[:, 0], data_dim[:, 1], data_dim[:, 2],
                              c=colors[dim_label], s=point_size, 
                              label=f'{dim_label} ({np.sum(mask)} pts)')
                elif data_dim.shape[1] >= 2:
                    ax.scatter(data_dim[:, 0], data_dim[:, 1],
                              c=colors[dim_label], s=point_size,
                              label=f'{dim_label} ({np.sum(mask)} pts)')
                              
        if is_3d:
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
        else:
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            
        idx_type = self.index_type_var.get().capitalize()
        ax.set_title(f"DimIndex Results ({idx_type})\n1D=Red, 2D=Blue, 3D=Green")
        ax.legend(loc='upper right', markerscale=3)
        
        self.canvas.draw()


class CrawlingVisualizationDialog(VisualizationDialog):
    """
    Crawling visualization: 1D point cloud and recovered structures as graphs.
    """
    
    def __init__(self, parent, graphs, subsets, spine_data=None, original_data=None):
        super().__init__(parent, title="Crawling Results Visualization")
        
        self.graphs = graphs  # List of NetworkX graphs
        self.subsets = subsets  # List of point arrays
        self.spine_data = spine_data
        self.original_data = original_data
        
        self._create_controls()
        self._plot()
        
    def _create_controls(self):
        """Create Crawling-specific controls"""
        # Background selection
        ttk.Label(self.control_frame, text="Background:").pack(side=tk.LEFT, padx=5)
        self.bg_var = tk.StringVar(value="spine")
        bg_options = ["none", "spine", "original"]
        bg_combo = ttk.Combobox(self.control_frame, textvariable=self.bg_var,
                                values=bg_options, width=10, state='readonly')
        bg_combo.pack(side=tk.LEFT, padx=5)
        bg_combo.bind("<<ComboboxSelected>>", lambda e: self._plot())
        
        # Show graphs toggle
        self.show_graphs_var = tk.BooleanVar(value=True)
        graphs_check = ttk.Checkbutton(self.control_frame, text="Show Graph Edges",
                                        variable=self.show_graphs_var,
                                        command=self._plot)
        graphs_check.pack(side=tk.LEFT, padx=10)
        
        # Show subsets toggle
        self.show_subsets_var = tk.BooleanVar(value=True)
        subsets_check = ttk.Checkbutton(self.control_frame, text="Show Subsets",
                                         variable=self.show_subsets_var,
                                         command=self._plot)
        subsets_check.pack(side=tk.LEFT, padx=10)
        
        # Node size
        ttk.Label(self.control_frame, text="Node Size:").pack(side=tk.LEFT, padx=(10, 5))
        self.node_size_var = tk.StringVar(value="5.0")
        size_entry = ttk.Entry(self.control_frame, textvariable=self.node_size_var, width=6)
        size_entry.pack(side=tk.LEFT, padx=5)
        
        apply_btn = ttk.Button(self.control_frame, text="Apply", command=self._plot)
        apply_btn.pack(side=tk.LEFT, padx=10)
        
    def _get_graph_nodes(self, G, ndim=3):
        """Extract node positions from graph"""
        from CrawlingModule import AllNodeNames
        return AllNodeNames(G, ndim)
        
    def _plot(self):
        """Plot the Crawling results"""
        self.fig.clear()
        
        # Determine dimensionality from first subset
        if self.subsets and len(self.subsets) > 0:
            ndims = min(self.subsets[0].shape[1], 3) if len(self.subsets[0].shape) > 1 else 1
        else:
            ndims = 3
        is_3d = ndims >= 3
        
        ax = self._create_axis(is_3d=is_3d)
        
        try:
            node_size = float(self.node_size_var.get())
        except ValueError:
            node_size = 5.0
            
        # Generate distinct colors for each graph
        n_graphs = len(self.graphs)
        colors = [colorsys.hsv_to_rgb(i / max(n_graphs, 1), 0.8, 0.9) for i in range(n_graphs)]
        
        # Plot background
        bg_choice = self.bg_var.get()
        if bg_choice == "spine" and self.spine_data is not None:
            if is_3d and self.spine_data.shape[1] >= 3:
                ax.scatter(self.spine_data[:, 0], self.spine_data[:, 1], self.spine_data[:, 2],
                          c='lightgray', s=0.5, alpha=0.3, label='1D spine data')
            elif self.spine_data.shape[1] >= 2:
                ax.scatter(self.spine_data[:, 0], self.spine_data[:, 1],
                          c='lightgray', s=0.5, alpha=0.3, label='1D spine data')
        elif bg_choice == "original" and self.original_data is not None:
            if is_3d and self.original_data.shape[1] >= 3:
                ax.scatter(self.original_data[:, 0], self.original_data[:, 1], self.original_data[:, 2],
                          c='lightgray', s=0.1, alpha=0.2, label='Original data')
            elif self.original_data.shape[1] >= 2:
                ax.scatter(self.original_data[:, 0], self.original_data[:, 1],
                          c='lightgray', s=0.1, alpha=0.2, label='Original data')
        
        # Plot each graph
        for i, (G, subset) in enumerate(zip(self.graphs, self.subsets)):
            color = colors[i % len(colors)]
            
            # Plot subset points
            if self.show_subsets_var.get() and subset is not None and len(subset) > 0:
                if is_3d and subset.shape[1] >= 3:
                    ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2],
                              c=[color], s=1.0, alpha=0.5)
                elif subset.shape[1] >= 2:
                    ax.scatter(subset[:, 0], subset[:, 1],
                              c=[color], s=1.0, alpha=0.5)
            
            # Plot graph nodes and edges
            if self.show_graphs_var.get():
                try:
                    pos = self._get_graph_nodes(G, ndims)
                    
                    # Plot nodes
                    if is_3d and pos.shape[1] >= 3:
                        ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2],
                                  c=[color], s=node_size, label=f'Graph {i+1}')
                    elif pos.shape[1] >= 2:
                        ax.scatter(pos[:, 0], pos[:, 1],
                                  c=[color], s=node_size, label=f'Graph {i+1}')
                    
                    # Plot edges
                    for edge in G.edges():
                        try:
                            t1 = np.frombuffer(G.nodes[edge[0]]['Name'])
                            t2 = np.frombuffer(G.nodes[edge[1]]['Name'])
                            if is_3d:
                                ax.plot([t1[0], t2[0]], [t1[1], t2[1]], [t1[2], t2[2]],
                                       c='black', linewidth=0.5, alpha=0.7)
                            else:
                                ax.plot([t1[0], t2[0]], [t1[1], t2[1]],
                                       c='black', linewidth=0.5, alpha=0.7)
                        except (IndexError, KeyError):
                            pass
                except Exception as e:
                    print(f"Warning: Could not plot graph {i+1}: {e}")
                    
        if is_3d:
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
        else:
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            
        ax.set_title(f"Crawling Results: {len(self.graphs)} Graph(s) Found")
        ax.legend(loc='upper right', markerscale=2)
        
        self.canvas.draw()


class SGTMVisualizationDialog(VisualizationDialog):
    """
    SGTM visualization with two modes:
    1. Model View: Probabilistic model with means (black squares) and covariances (red ellipsoids)
    2. Likelihood View: Points colored by likelihood/posterior to belong to models
    """
    
    def __init__(self, parent, nets, graphs, noisy_subsets, gm_dists, 
                 spine_data=None, original_data=None):
        super().__init__(parent, title="SGTM Results Visualization")
        
        self.nets = nets  # List of trained networks
        self.graphs = graphs
        self.noisy_subsets = noisy_subsets
        self.gm_dists = gm_dists
        self.spine_data = spine_data
        self.original_data = original_data
        
        # Pre-compute likelihoods for original data if available
        self.likelihoods = None
        self.posteriors = None
        self._compute_likelihoods()
        
        self._create_controls()
        self._plot()
        
    def _compute_likelihoods(self):
        """Compute likelihoods and posteriors for original data points"""
        if self.original_data is None or not self.gm_dists:
            return
            
        try:
            n_points = len(self.original_data)
            n_models = len(self.gm_dists)
            
            # Compute log-likelihood for each point under each model
            log_likelihoods = np.zeros((n_points, n_models))
            
            for i, gm in enumerate(self.gm_dists):
                if gm is not None and self.noisy_subsets[i] is not None:
                    try:
                        # Fit GMM on noisy subset and score original data
                        gm_fitted = gm.fit(self.noisy_subsets[i])
                        log_likelihoods[:, i] = gm_fitted.score_samples(self.original_data)
                    except Exception as e:
                        print(f"Warning: Could not compute likelihood for model {i+1}: {e}")
                        log_likelihoods[:, i] = -np.inf
                        
            self.likelihoods = log_likelihoods
            
            # Compute posteriors (normalized likelihoods)
            # Use log-sum-exp trick for numerical stability
            max_ll = np.max(log_likelihoods, axis=1, keepdims=True)
            exp_ll = np.exp(log_likelihoods - max_ll)
            sum_exp_ll = np.sum(exp_ll, axis=1, keepdims=True)
            self.posteriors = exp_ll / sum_exp_ll
            
        except Exception as e:
            print(f"Warning: Could not compute likelihoods: {e}")
            self.likelihoods = None
            self.posteriors = None
        
    def _create_controls(self):
        """Create SGTM-specific controls"""
        # Create a notebook for different control sets
        control_notebook = ttk.Notebook(self.control_frame)
        control_notebook.pack(fill=tk.X, expand=True)
        
        # Tab 1: Model View controls
        model_frame = ttk.Frame(control_notebook, padding=5)
        control_notebook.add(model_frame, text="Model View")
        
        # Background selection
        ttk.Label(model_frame, text="Background:").pack(side=tk.LEFT, padx=5)
        self.bg_var = tk.StringVar(value="spine")
        bg_options = ["none", "spine", "noisy_subsets", "original"]
        bg_combo = ttk.Combobox(model_frame, textvariable=self.bg_var,
                                values=bg_options, width=12, state='readonly')
        bg_combo.pack(side=tk.LEFT, padx=5)
        bg_combo.bind("<<ComboboxSelected>>", lambda e: self._plot())
        
        # Show means toggle
        self.show_means_var = tk.BooleanVar(value=True)
        means_check = ttk.Checkbutton(model_frame, text="Means (■)",
                                       variable=self.show_means_var,
                                       command=self._plot)
        means_check.pack(side=tk.LEFT, padx=5)
        
        # Show covariances toggle
        self.show_cov_var = tk.BooleanVar(value=True)
        cov_check = ttk.Checkbutton(model_frame, text="Covariances",
                                     variable=self.show_cov_var,
                                     command=self._plot)
        cov_check.pack(side=tk.LEFT, padx=5)
        
        # Show edges toggle
        self.show_edges_var = tk.BooleanVar(value=True)
        edges_check = ttk.Checkbutton(model_frame, text="Edges",
                                       variable=self.show_edges_var,
                                       command=self._plot)
        edges_check.pack(side=tk.LEFT, padx=5)
        
        # Ellipsoid scale
        ttk.Label(model_frame, text="Ellipsoid Scale:").pack(side=tk.LEFT, padx=(10, 2))
        self.ellipsoid_scale_var = tk.StringVar(value="1.0")
        scale_entry = ttk.Entry(model_frame, textvariable=self.ellipsoid_scale_var, width=5)
        scale_entry.pack(side=tk.LEFT, padx=2)
        
        apply_btn1 = ttk.Button(model_frame, text="Apply", command=self._plot)
        apply_btn1.pack(side=tk.LEFT, padx=10)
        
        # Tab 2: Likelihood View controls
        likelihood_frame = ttk.Frame(control_notebook, padding=5)
        control_notebook.add(likelihood_frame, text="Likelihood View")
        
        # View mode
        self.view_mode_var = tk.StringVar(value="model")
        
        # Color by selection
        ttk.Label(likelihood_frame, text="Color by:").pack(side=tk.LEFT, padx=5)
        self.color_by_var = tk.StringVar(value="max_posterior")
        
        # Build color options dynamically based on number of models
        color_options = ["max_posterior"]
        for i in range(len(self.nets)):
            color_options.append(f"Model {i+1}")
        
        color_combo = ttk.Combobox(likelihood_frame, textvariable=self.color_by_var,
                                   values=color_options, width=15, state='readonly')
        color_combo.pack(side=tk.LEFT, padx=5)
        color_combo.bind("<<ComboboxSelected>>", lambda e: self._plot_likelihood())
        
        # Colormap selection
        ttk.Label(likelihood_frame, text="Colormap:").pack(side=tk.LEFT, padx=(10, 2))
        self.cmap_var = tk.StringVar(value="viridis")
        cmap_combo = ttk.Combobox(likelihood_frame, textvariable=self.cmap_var,
                                  values=["viridis", "plasma", "coolwarm", "RdYlBu", "jet"],
                                  width=10, state='readonly')
        cmap_combo.pack(side=tk.LEFT, padx=2)
        cmap_combo.bind("<<ComboboxSelected>>", lambda e: self._plot_likelihood())
        
        # Point size
        ttk.Label(likelihood_frame, text="Point Size:").pack(side=tk.LEFT, padx=(10, 2))
        self.ll_point_size_var = tk.StringVar(value="1.0")
        size_entry = ttk.Entry(likelihood_frame, textvariable=self.ll_point_size_var, width=5)
        size_entry.pack(side=tk.LEFT, padx=2)
        
        apply_btn2 = ttk.Button(likelihood_frame, text="Plot Likelihoods", command=self._plot_likelihood)
        apply_btn2.pack(side=tk.LEFT, padx=10)
        
        # Bind tab change to switch view
        control_notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)
        self.control_notebook = control_notebook
        
    def _on_tab_changed(self, event):
        """Handle tab change to switch between views"""
        selected_tab = self.control_notebook.index(self.control_notebook.select())
        if selected_tab == 0:
            self.view_mode_var.set("model")
            self._plot()
        else:
            self.view_mode_var.set("likelihood")
            self._plot_likelihood()
            
    def _draw_ellipsoid_3d(self, ax, center, cov, color='red', alpha=0.2, scale=1.0, n_points=20):
        """
        Draw a 3D ellipsoid representing covariance matrix.
        Axes aligned along eigenvectors, sizes equal to eigenvalues.
        """
        try:
            # Get 3x3 submatrix of covariance
            cov_3d = cov[:3, :3] if cov.shape[0] >= 3 else cov
            center_3d = center[:3] if len(center) >= 3 else center
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_3d)
            eigenvalues = np.maximum(eigenvalues, 1e-10)  # Ensure positive
            
            # Generate sphere points
            u = np.linspace(0, 2 * np.pi, n_points)
            v = np.linspace(0, np.pi, n_points)
            x = np.outer(np.cos(u), np.sin(v))
            y = np.outer(np.sin(u), np.sin(v))
            z = np.outer(np.ones_like(u), np.cos(v))
            
            # Scale by sqrt of eigenvalues (std dev) and apply scale factor
            # Transform sphere to ellipsoid: scale by eigenvalues, rotate by eigenvectors
            for i in range(len(x)):
                for j in range(len(x[0])):
                    # Point on unit sphere
                    point = np.array([x[i, j], y[i, j], z[i, j]])
                    # Scale by sqrt(eigenvalue) * scale factor
                    point = np.sqrt(eigenvalues) * scale * point
                    # Rotate by eigenvectors
                    point = eigenvectors @ point
                    # Translate to center
                    x[i, j] = point[0] + center_3d[0]
                    y[i, j] = point[1] + center_3d[1]
                    z[i, j] = point[2] + center_3d[2]
                    
            ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)
            
        except Exception as e:
            print(f"Warning: Could not draw ellipsoid: {e}")
            
    def _draw_ellipse_2d(self, ax, center, cov, color='red', alpha=0.3, scale=1.0, n_points=50):
        """Draw a 2D ellipse representing covariance matrix."""
        try:
            from matplotlib.patches import Ellipse
            
            cov_2d = cov[:2, :2] if cov.shape[0] >= 2 else cov
            center_2d = center[:2] if len(center) >= 2 else center
            
            # Eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(cov_2d)
            eigenvalues = np.maximum(eigenvalues, 1e-10)
            
            # Calculate angle of rotation (in degrees)
            angle = np.degrees(np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0]))
            
            # Width and height are 2 * sqrt(eigenvalue) * scale
            width = 2 * np.sqrt(eigenvalues[0]) * scale
            height = 2 * np.sqrt(eigenvalues[1]) * scale
            
            ellipse = Ellipse(xy=center_2d, width=width, height=height, angle=angle,
                             facecolor=color, alpha=alpha, edgecolor='darkred', linewidth=0.5)
            ax.add_patch(ellipse)
            
        except Exception as e:
            print(f"Warning: Could not draw ellipse: {e}")
        
    def _plot(self):
        """Plot the SGTM model view (means, covariances, edges)"""
        self.fig.clear()
        
        # Determine dimensionality
        if self.spine_data is not None:
            ndims = min(self.spine_data.shape[1], 3)
        elif self.noisy_subsets and len(self.noisy_subsets) > 0:
            ndims = min(self.noisy_subsets[0].shape[1], 3)
        else:
            ndims = 3
        is_3d = ndims >= 3
        
        ax = self._create_axis(is_3d=is_3d)
        
        pastel_blue = '#ADD8E6'  # Pastel light blue
        
        try:
            ellipsoid_scale = float(self.ellipsoid_scale_var.get())
        except ValueError:
            ellipsoid_scale = 1.0
        
        # Plot background
        bg_choice = self.bg_var.get()
        if bg_choice == "spine" and self.spine_data is not None:
            if is_3d and self.spine_data.shape[1] >= 3:
                ax.scatter(self.spine_data[:, 0], self.spine_data[:, 1], self.spine_data[:, 2],
                          c='red', s=0.5, alpha=0.5, label='1D data')
            elif self.spine_data.shape[1] >= 2:
                ax.scatter(self.spine_data[:, 0], self.spine_data[:, 1],
                          c='red', s=0.5, alpha=0.5, label='1D data')
        elif bg_choice == "noisy_subsets" and self.noisy_subsets:
            for i, subset in enumerate(self.noisy_subsets):
                if subset is not None and len(subset) > 0:
                    if is_3d and subset.shape[1] >= 3:
                        ax.scatter(subset[:, 0], subset[:, 1], subset[:, 2],
                                  c=pastel_blue, s=0.3, alpha=0.4,
                                  label='Noisy subset' if i == 0 else None)
                    elif subset.shape[1] >= 2:
                        ax.scatter(subset[:, 0], subset[:, 1],
                                  c=pastel_blue, s=0.3, alpha=0.4,
                                  label='Noisy subset' if i == 0 else None)
            # Also show spine in red
            if self.spine_data is not None:
                if is_3d and self.spine_data.shape[1] >= 3:
                    ax.scatter(self.spine_data[:, 0], self.spine_data[:, 1], self.spine_data[:, 2],
                              c='red', s=1.0, alpha=0.6, label='1D data')
                elif self.spine_data.shape[1] >= 2:
                    ax.scatter(self.spine_data[:, 0], self.spine_data[:, 1],
                              c='red', s=1.0, alpha=0.6, label='1D data')
        elif bg_choice == "original" and self.original_data is not None:
            if is_3d and self.original_data.shape[1] >= 3:
                ax.scatter(self.original_data[:, 0], self.original_data[:, 1], self.original_data[:, 2],
                          c='lightgray', s=0.1, alpha=0.2, label='Original data')
            elif self.original_data.shape[1] >= 2:
                ax.scatter(self.original_data[:, 0], self.original_data[:, 1],
                          c='lightgray', s=0.1, alpha=0.2, label='Original data')
        
        # Generate distinct colors for each network
        n_nets = len(self.nets)
        net_colors = [colorsys.hsv_to_rgb(i / max(n_nets, 1), 0.8, 0.8) for i in range(n_nets)]
        
        # Plot SGTM networks
        for net_idx, net in enumerate(self.nets):
            try:
                # Get node positions (means) and covariances
                node_pos = net.gmmnet.V_  # Means (shape: m x d)
                
                # Get covariances from the trained GMM
                # In AGTMModule, Sigma has shape (d, d, m) where m is number of components
                # So Sigma[:, :, i] gives the covariance for component i
                covs = None
                n_components = len(node_pos)
                
                if hasattr(net.gmmnet, 'Sigma') and net.gmmnet.Sigma is not None:
                    sigma = net.gmmnet.Sigma
                    # Sigma is (d, d, m), convert to list of (d, d) matrices
                    if len(sigma.shape) == 3:
                        covs = [sigma[:, :, i] for i in range(sigma.shape[2])]
                        print(f"Found {len(covs)} covariance matrices of shape {covs[0].shape}")
                    else:
                        print(f"Sigma has unexpected shape: {sigma.shape}")
                        covs = None
                elif hasattr(net.gmmnet, 'covariances_'):
                    covs = net.gmmnet.covariances_
                    print(f"Found covariances_ with {len(covs)} matrices")
                else:
                    print(f"No covariance matrices found in network {net_idx+1}")
                    
                # Plot means as black squares
                if self.show_means_var.get():
                    if is_3d and node_pos.shape[1] >= 3:
                        ax.scatter(node_pos[:, 0], node_pos[:, 1], node_pos[:, 2],
                                  c='black', marker='s', s=25,
                                  label=f'Means' if net_idx == 0 else None, zorder=10)
                    elif node_pos.shape[1] >= 2:
                        ax.scatter(node_pos[:, 0], node_pos[:, 1],
                                  c='black', marker='s', s=25,
                                  label=f'Means' if net_idx == 0 else None, zorder=10)
                
                # Plot covariances as ellipsoids
                if self.show_cov_var.get() and covs is not None:
                    for j, center in enumerate(node_pos):
                        if j < len(covs):
                            cov = covs[j]
                            if is_3d:
                                self._draw_ellipsoid_3d(ax, center, cov, 
                                                       color='red', alpha=0.15, 
                                                       scale=ellipsoid_scale)
                            else:
                                self._draw_ellipse_2d(ax, center, cov,
                                                     color='red', alpha=0.2,
                                                     scale=ellipsoid_scale)
                
                # Plot edges
                if self.show_edges_var.get() and net_idx < len(self.graphs):
                    G = self.graphs[net_idx]
                    for edge in G.edges():
                        try:
                            p1 = node_pos[edge[0]]
                            p2 = node_pos[edge[1]]
                            if is_3d:
                                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                       c='darkblue', linewidth=1.5, alpha=0.8, zorder=5)
                            else:
                                ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                       c='darkblue', linewidth=1.5, alpha=0.8, zorder=5)
                        except (IndexError, KeyError):
                            pass
                        
            except Exception as e:
                print(f"Warning: Could not plot network {net_idx+1}: {e}")
                
        if is_3d:
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
        else:
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            
        ax.set_title(f"SGTM Model View: {len(self.nets)} Network(s)\nMeans=■ (black), Covariances=Ellipsoids (red)")
        ax.legend(loc='upper right', markerscale=2)
        
        self.canvas.draw()
        
    def _plot_likelihood(self):
        """Plot the likelihood/posterior view"""
        self.fig.clear()
        
        if self.original_data is None:
            ax = self._create_axis(is_3d=False)
            ax.text(0.5, 0.5, "Original data required for likelihood view",
                   ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return
            
        if self.posteriors is None:
            ax = self._create_axis(is_3d=False)
            ax.text(0.5, 0.5, "Could not compute likelihoods.\nCheck that GM models are available.",
                   ha='center', va='center', transform=ax.transAxes)
            self.canvas.draw()
            return
        
        ndims = min(self.original_data.shape[1], 3)
        is_3d = ndims >= 3
        
        ax = self._create_axis(is_3d=is_3d)
        
        try:
            point_size = float(self.ll_point_size_var.get())
        except ValueError:
            point_size = 1.0
            
        cmap = plt.cm.get_cmap(self.cmap_var.get())
        
        # Determine what to color by
        color_choice = self.color_by_var.get()
        
        if color_choice == "max_posterior":
            # Color by maximum posterior probability
            color_values = np.max(self.posteriors, axis=1)
            title_suffix = "Max Posterior Probability"
            cbar_label = "Max Posterior"
        else:
            # Color by specific model's posterior
            model_idx = int(color_choice.split()[-1]) - 1
            if model_idx < self.posteriors.shape[1]:
                color_values = self.posteriors[:, model_idx]
                title_suffix = f"Posterior for Model {model_idx + 1}"
                cbar_label = f"P(Model {model_idx + 1})"
            else:
                color_values = np.zeros(len(self.original_data))
                title_suffix = "Invalid model selection"
                cbar_label = ""
        
        # Plot
        if is_3d:
            sc = ax.scatter(self.original_data[:, 0], self.original_data[:, 1], 
                           self.original_data[:, 2],
                           c=color_values, cmap=cmap, s=point_size, alpha=0.7)
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            ax.set_zlabel('Dim 3')
        else:
            sc = ax.scatter(self.original_data[:, 0], self.original_data[:, 1],
                           c=color_values, cmap=cmap, s=point_size, alpha=0.7)
            ax.set_xlabel('Dim 1')
            ax.set_ylabel('Dim 2')
            
        # Add colorbar
        cbar = self.fig.colorbar(sc, ax=ax, shrink=0.8)
        cbar.set_label(cbar_label)
        
        ax.set_title(f"SGTM Likelihood View\n{title_suffix}")
        
        self.canvas.draw()


class DREAMToolbox(tk.Tk):
    """Main GUI Application for 1DREAM Toolbox"""
    
    def __init__(self):
        super().__init__()
        
        self.title("1DREAM Toolbox - Manifold Learning Suite")
        self.geometry("1200x900")
        self.minsize(1000, 700)
        
        # Data storage
        self.data = None
        self.data_path = None
        self.output_dir = None
        self.workflow_state = {
            'data_loaded': False,
            'laat_complete': False,
            'mbms_complete': False,
            'dimindex_complete': False,
            'crawling_complete': False,
            'sgtm_complete': False
        }
        
        # Results storage
        self.results = {
            'pheromone': None,
            'laat_selected': None,
            'mbms_data': None,
            'dimindex_data': None,
            'crawling_graphs': None,
            'crawling_subsets': None,
            'sgtm_nets': None
        }
        
        # Task queue for threading
        self.task_queue = queue.Queue()
        
        # Build the GUI
        self._create_styles()
        self._create_menu()
        self._create_main_layout()
        self._create_status_bar()
        
        # Start queue checker
        self.after(100, self._check_queue)
        
    def _create_styles(self):
        """Create custom styles for the application"""
        style = ttk.Style()
        style.configure('Header.TLabel', font=('Helvetica', 12, 'bold'))
        style.configure('Status.TLabel', font=('Helvetica', 10))
        style.configure('Method.TLabelframe.Label', font=('Helvetica', 11, 'bold'))
        style.configure('Run.TButton', font=('Helvetica', 10, 'bold'))
        
    def _create_menu(self):
        """Create the menu bar"""
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Load Point Cloud...", command=self._load_data, accelerator="Ctrl+O")
        file_menu.add_command(label="Set Output Directory...", command=self._set_output_dir)
        file_menu.add_separator()
        file_menu.add_command(label="Load Previous Results...", command=self._load_results)
        file_menu.add_command(label="Save Current Results...", command=self._save_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        # Workflow menu
        workflow_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Workflow", menu=workflow_menu)
        workflow_menu.add_command(label="Run Full Pipeline", command=self._run_full_pipeline)
        workflow_menu.add_separator()
        workflow_menu.add_command(label="Reset Workflow", command=self._reset_workflow)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self._show_about)
        help_menu.add_command(label="Workflow Guide", command=self._show_workflow_guide)
        
        # Keyboard shortcuts
        self.bind('<Control-o>', lambda e: self._load_data())
        
    def _create_main_layout(self):
        """Create the main application layout"""
        # Main container with paned windows
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel - Controls
        left_frame = ttk.Frame(main_paned, width=450)
        main_paned.add(left_frame, weight=1)
        
        # Right panel - Output/Log
        right_frame = ttk.Frame(main_paned, width=400)
        main_paned.add(right_frame, weight=1)
        
        # Build left panel content
        self._create_left_panel(left_frame)
        
        # Build right panel content
        self._create_right_panel(right_frame)
        
    def _create_left_panel(self, parent):
        """Create the left control panel"""
        # Scrollable canvas for controls
        canvas = tk.Canvas(parent)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        # Data loading section
        self._create_data_section(scrollable_frame)
        
        # Workflow indicator
        self._create_workflow_indicator(scrollable_frame)
        
        # Method sections
        self._create_laat_section(scrollable_frame)
        self._create_mbms_section(scrollable_frame)
        self._create_dimindex_section(scrollable_frame)
        self._create_crawling_section(scrollable_frame)
        self._create_sgtm_section(scrollable_frame)
        
    def _create_data_section(self, parent):
        """Create data loading section"""
        frame = ttk.LabelFrame(parent, text="📁 Data Input", style='Method.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Data file selection
        data_frame = ttk.Frame(frame)
        data_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(data_frame, text="Point Cloud:").pack(side=tk.LEFT)
        self.data_label = ttk.Label(data_frame, text="No file loaded", foreground='gray')
        self.data_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(data_frame, text="Browse...", command=self._load_data).pack(side=tk.RIGHT)
        
        # Output directory
        output_frame = ttk.Frame(frame)
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(output_frame, text="Output Dir:").pack(side=tk.LEFT)
        self.output_label = ttk.Label(output_frame, text="Same as input", foreground='gray')
        self.output_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        ttk.Button(output_frame, text="Set...", command=self._set_output_dir).pack(side=tk.RIGHT)
        
        # Data info
        self.data_info_label = ttk.Label(frame, text="", style='Status.TLabel')
        self.data_info_label.pack(fill=tk.X, padx=5, pady=5)
        
    def _create_workflow_indicator(self, parent):
        """Create workflow progress indicator"""
        frame = ttk.LabelFrame(parent, text="📊 Workflow Progress", style='Method.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Workflow steps
        steps_frame = ttk.Frame(frame)
        steps_frame.pack(fill=tk.X, padx=5, pady=10)
        
        self.workflow_labels = {}
        steps = ['LAAT', 'MBMS', 'DimIndex', 'Crawling', 'SGTM']
        
        for i, step in enumerate(steps):
            step_frame = ttk.Frame(steps_frame)
            step_frame.pack(side=tk.LEFT, padx=5, expand=True)
            
            # Status indicator
            indicator = ttk.Label(step_frame, text="○", font=('Helvetica', 16))
            indicator.pack()
            
            # Step name
            label = ttk.Label(step_frame, text=step, font=('Helvetica', 9))
            label.pack()
            
            self.workflow_labels[step.lower()] = indicator
            
            # Arrow between steps
            if i < len(steps) - 1:
                arrow = ttk.Label(steps_frame, text="→", font=('Helvetica', 14))
                arrow.pack(side=tk.LEFT, padx=2)
                
    def _create_laat_section(self, parent):
        """Create LAAT configuration section"""
        frame = ttk.LabelFrame(parent, text="1️⃣ LAAT - Ant-based Feature Selection", style='Method.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Parameters
        params_frame = ttk.Frame(frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Row 1
        row1 = ttk.Frame(params_frame)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Label(row1, text="Num Ants:", width=12).pack(side=tk.LEFT)
        self.laat_num_ants = ttk.Entry(row1, width=10)
        self.laat_num_ants.insert(0, "125")
        self.laat_num_ants.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row1, text="Iterations:", width=10).pack(side=tk.LEFT, padx=(10,0))
        self.laat_iterations = ttk.Entry(row1, width=10)
        self.laat_iterations.insert(0, "100")
        self.laat_iterations.pack(side=tk.LEFT, padx=5)
        
        # Row 2
        row2 = ttk.Frame(params_frame)
        row2.pack(fill=tk.X, pady=2)
        
        ttk.Label(row2, text="Steps:", width=12).pack(side=tk.LEFT)
        self.laat_steps = ttk.Entry(row2, width=10)
        self.laat_steps.insert(0, "2500")
        self.laat_steps.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row2, text="Threads:", width=10).pack(side=tk.LEFT, padx=(10,0))
        self.laat_threads = ttk.Entry(row2, width=10)
        self.laat_threads.insert(0, "4")
        self.laat_threads.pack(side=tk.LEFT, padx=5)
        
        # Row 3
        row3 = ttk.Frame(params_frame)
        row3.pack(fill=tk.X, pady=2)
        
        ttk.Label(row3, text="Radius:", width=12).pack(side=tk.LEFT)
        self.laat_radius = ttk.Entry(row3, width=10)
        self.laat_radius.insert(0, "3.5")
        self.laat_radius.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row3, text="Kappa:", width=10).pack(side=tk.LEFT, padx=(10,0))
        self.laat_kappa = ttk.Entry(row3, width=10)
        self.laat_kappa.insert(0, "0.8")
        self.laat_kappa.pack(side=tk.LEFT, padx=5)
        
        # Row 4 - Neighbor threshold and Gamma
        row4 = ttk.Frame(params_frame)
        row4.pack(fill=tk.X, pady=2)
        
        ttk.Label(row4, text="Neighb. Thresh:", width=12).pack(side=tk.LEFT)
        self.laat_th_neighb = ttk.Entry(row4, width=10)
        self.laat_th_neighb.insert(0, "9")
        self.laat_th_neighb.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row4, text="Gamma:", width=10).pack(side=tk.LEFT, padx=(10,0))
        self.laat_gamma = ttk.Entry(row4, width=10)
        self.laat_gamma.insert(0, "0.0")
        self.laat_gamma.pack(side=tk.LEFT, padx=5)
        
        # Row 5 - Threshold for pheromone selection
        row5 = ttk.Frame(params_frame)
        row5.pack(fill=tk.X, pady=2)
        
        ttk.Label(row5, text="Pheromone %:", width=12).pack(side=tk.LEFT)
        self.laat_threshold = ttk.Entry(row5, width=10)
        self.laat_threshold.insert(0, "50")
        self.laat_threshold.pack(side=tk.LEFT, padx=5)
        ttk.Label(row5, text="(top percentile)", foreground='gray').pack(side=tk.LEFT)
        
        # Advanced options separator
        adv_separator = ttk.Separator(frame, orient='horizontal')
        adv_separator.pack(fill=tk.X, padx=5, pady=5)
        
        adv_label = ttk.Label(frame, text="Advanced Options", font=('Helvetica', 9, 'italic'), foreground='gray')
        adv_label.pack(anchor=tk.W, padx=5)
        
        adv_frame = ttk.Frame(frame)
        adv_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Row 6 - Initialization mode
        row6 = ttk.Frame(adv_frame)
        row6.pack(fill=tk.X, pady=2)
        
        ttk.Label(row6, text="Init Mode:", width=12).pack(side=tk.LEFT)
        self.laat_init_mode = ttk.Combobox(row6, values=["Random (0)", "Custom Indices (1)"], width=18)
        self.laat_init_mode.set("Random (0)")
        self.laat_init_mode.pack(side=tk.LEFT, padx=5)
        
        # Row 7 - Custom initialization indices file
        row7 = ttk.Frame(adv_frame)
        row7.pack(fill=tk.X, pady=2)
        
        ttk.Label(row7, text="Init Indices:", width=12).pack(side=tk.LEFT)
        self.laat_init_indices_path = tk.StringVar()
        self.laat_init_indices_entry = ttk.Entry(row7, textvariable=self.laat_init_indices_path, width=25)
        self.laat_init_indices_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(row7, text="Browse...", command=self._browse_init_indices).pack(side=tk.LEFT)
        
        # Row 8 - External field/weights file
        row8 = ttk.Frame(adv_frame)
        row8.pack(fill=tk.X, pady=2)
        
        ttk.Label(row8, text="Ext. Field:", width=12).pack(side=tk.LEFT)
        self.laat_ext_field_path = tk.StringVar()
        self.laat_ext_field_entry = ttk.Entry(row8, textvariable=self.laat_ext_field_path, width=25)
        self.laat_ext_field_entry.pack(side=tk.LEFT, padx=5)
        ttk.Button(row8, text="Browse...", command=self._browse_ext_field).pack(side=tk.LEFT)
        
        # Help text for advanced options
        help_frame = ttk.Frame(adv_frame)
        help_frame.pack(fill=tk.X, pady=2)
        help_text = ttk.Label(help_frame, 
                              text="Init Indices: CSV with point indices for ant initialization\n"
                                   "Ext. Field: CSV with per-point weights (external field)",
                              foreground='gray', font=('Helvetica', 8))
        help_text.pack(anchor=tk.W)
        
        # Run button
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.laat_btn = ttk.Button(btn_frame, text="▶ Run LAAT", style='Run.TButton', 
                                   command=self._run_laat)
        self.laat_btn.pack(side=tk.RIGHT)
        self.laat_status = ttk.Label(btn_frame, text="", foreground='gray')
        self.laat_status.pack(side=tk.LEFT)
        
    def _browse_init_indices(self):
        """Browse for initialization indices CSV file with column selection"""
        description = ("Select a CSV file containing point indices for ant initialization.\n"
                       "Choose the column that contains the integer indices (0-based).")
        
        selector = CSVColumnSelector(
            self,
            title="Select Initialization Indices",
            column_assignments=[{'name': 'Indices', 'description': 'Index column'}],
            single_column=True,
            description=description
        )
        
        result = selector.get_result()
        
        if result is not None:
            # Store the indices directly instead of file path
            self.laat_custom_init_indices = result.astype(np.uint64)
            self.laat_init_indices_path.set(f"[{len(result)} indices loaded]")
            self._log(f"Loaded init indices: {len(result)} indices", 'info')
            
    def _browse_ext_field(self):
        """Browse for external field/weights CSV file with column selection"""
        description = ("Select a CSV file containing per-point external field weights.\n"
                       "Choose the column that contains the weight values (one per data point).")
        
        selector = CSVColumnSelector(
            self,
            title="Select External Field Weights",
            column_assignments=[{'name': 'Weights', 'description': 'Weight column'}],
            single_column=True,
            description=description
        )
        
        result = selector.get_result()
        
        if result is not None:
            # Store the weights directly instead of file path
            self.laat_external_weights = result.astype(np.float32)
            self.laat_ext_field_path.set(f"[{len(result)} weights loaded]")
            self._log(f"Loaded external field: {len(result)} weights", 'info')
            
    def _load_mbms_custom_input(self):
        """Load custom input for MBMS with column selection"""
        description = ("Select a CSV file containing point cloud data for MBMS.\n"
                       "This replaces the default LAAT output as input.")
        
        selector = CSVColumnSelector(
            self,
            title="Load Custom MBMS Input",
            column_assignments=[
                {'name': 'Dim 1', 'description': 'Dimension 1'},
                {'name': 'Dim 2', 'description': 'Dimension 2'},
                {'name': 'Dim 3', 'description': 'Dimension 3'}
            ],
            single_column=False,
            description=description
        )
        
        result = selector.get_result()
        
        if result is not None:
            self.mbms_custom_data = result.astype(np.float64)
            self.mbms_custom_input_label.config(text=f"[{result.shape[0]}×{result.shape[1]}]")
            self._log(f"Loaded custom MBMS input: {result.shape}", 'info')
            
    def _load_dimindex_custom_input(self):
        """Load custom input for DimIndex with column selection"""
        description = ("DimIndex requires TWO datasets:\n"
                       "1. Original/LAAT-selected data\n"
                       "2. MBMS-processed data\n\n"
                       "First, load the original/LAAT data:")
        
        # Load first dataset (original/LAAT data)
        selector1 = CSVColumnSelector(
            self,
            title="Load DimIndex Input - Original/LAAT Data",
            column_assignments=[
                {'name': 'Dim 1', 'description': 'Dimension 1'},
                {'name': 'Dim 2', 'description': 'Dimension 2'},
                {'name': 'Dim 3', 'description': 'Dimension 3'}
            ],
            single_column=False,
            description=description
        )
        
        result1 = selector1.get_result()
        
        if result1 is None:
            return
            
        # Load second dataset (MBMS data)
        description2 = "Now load the MBMS-processed data:"
        
        selector2 = CSVColumnSelector(
            self,
            title="Load DimIndex Input - MBMS Data",
            column_assignments=[
                {'name': 'Dim 1', 'description': 'Dimension 1'},
                {'name': 'Dim 2', 'description': 'Dimension 2'},
                {'name': 'Dim 3', 'description': 'Dimension 3'}
            ],
            single_column=False,
            description=description2
        )
        
        result2 = selector2.get_result()
        
        if result2 is not None:
            self.dimindex_custom_laat = result1.astype(np.float64)
            self.dimindex_custom_mbms = result2.astype(np.float64)
            self.dimindex_custom_input_label.config(
                text=f"[LAAT:{result1.shape[0]}, MBMS:{result2.shape[0]}]")
            self._log(f"Loaded custom DimIndex inputs: LAAT {result1.shape}, MBMS {result2.shape}", 'info')
            
    def _load_crawling_custom_input(self):
        """Load custom input for Crawling with column selection"""
        description = ("Select a CSV file containing spine/filtered data for Crawling.\n"
                       "This replaces the default DimIndex output as input.")
        
        selector = CSVColumnSelector(
            self,
            title="Load Custom Crawling Input",
            column_assignments=[
                {'name': 'Dim 1', 'description': 'Dimension 1'},
                {'name': 'Dim 2', 'description': 'Dimension 2'},
                {'name': 'Dim 3', 'description': 'Dimension 3'}
            ],
            single_column=False,
            description=description
        )
        
        result = selector.get_result()
        
        if result is not None:
            self.crawling_custom_data = result.astype(np.float64)
            self.crawling_custom_input_label.config(text=f"[{result.shape[0]}×{result.shape[1]}]")
            self._log(f"Loaded custom Crawling input: {result.shape}", 'info')
            
    def _load_sgtm_custom_input(self):
        """Load custom input for SGTM (requires Crawling pickle output)"""
        filepath = filedialog.askopenfilename(
            title="Select Crawling Output Pickle",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                FG, NoisyMan = pickle.load(open(filepath, 'rb'))
                self.sgtm_custom_graphs = FG
                self.sgtm_custom_subsets = NoisyMan
                self.sgtm_custom_input_label.config(text=f"[{len(FG)} graphs]")
                self._log(f"Loaded custom SGTM input: {len(FG)} graphs", 'info')
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load pickle: {str(e)}")
        
    def _create_mbms_section(self, parent):
        """Create MBMS configuration section"""
        frame = ttk.LabelFrame(parent, text="2️⃣ MBMS - Mean-Shift Denoising", style='Method.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Input source selection
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(input_frame, text="Input:", width=12).pack(side=tk.LEFT)
        self.mbms_input_source = tk.StringVar(value="previous")
        ttk.Radiobutton(input_frame, text="From LAAT output", variable=self.mbms_input_source, 
                        value="previous").pack(side=tk.LEFT)
        ttk.Radiobutton(input_frame, text="Custom file", variable=self.mbms_input_source,
                        value="custom").pack(side=tk.LEFT, padx=10)
        ttk.Button(input_frame, text="Load...", command=self._load_mbms_custom_input).pack(side=tk.LEFT)
        self.mbms_custom_input_label = ttk.Label(input_frame, text="", foreground='gray')
        self.mbms_custom_input_label.pack(side=tk.LEFT, padx=5)
        
        params_frame = ttk.Frame(frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Row 1
        row1 = ttk.Frame(params_frame)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Label(row1, text="Iterations:", width=12).pack(side=tk.LEFT)
        self.mbms_iter = ttk.Entry(row1, width=10)
        self.mbms_iter.insert(0, "2")
        self.mbms_iter.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row1, text="Radius:", width=10).pack(side=tk.LEFT, padx=(10,0))
        self.mbms_radius = ttk.Entry(row1, width=10)
        self.mbms_radius.insert(0, "3.5")
        self.mbms_radius.pack(side=tk.LEFT, padx=5)
        
        # Row 2
        row2 = ttk.Frame(params_frame)
        row2.pack(fill=tk.X, pady=2)
        
        ttk.Label(row2, text="Sigma:", width=12).pack(side=tk.LEFT)
        self.mbms_sigma = ttk.Entry(row2, width=10)
        self.mbms_sigma.insert(0, "1.5")
        self.mbms_sigma.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row2, text="K (neighbors):", width=10).pack(side=tk.LEFT, padx=(10,0))
        self.mbms_k = ttk.Entry(row2, width=10)
        self.mbms_k.insert(0, "3")
        self.mbms_k.pack(side=tk.LEFT, padx=5)
        
        # Run button
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.mbms_btn = ttk.Button(btn_frame, text="▶ Run MBMS", style='Run.TButton', 
                                   command=self._run_mbms)
        self.mbms_btn.pack(side=tk.RIGHT)
        self.mbms_status = ttk.Label(btn_frame, text="", foreground='gray')
        self.mbms_status.pack(side=tk.LEFT)
        
    def _create_dimindex_section(self, parent):
        """Create DimIndex configuration section"""
        frame = ttk.LabelFrame(parent, text="3️⃣ DimIndex - Dimensionality Estimation", style='Method.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Input source selection
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(input_frame, text="Input:", width=12).pack(side=tk.LEFT)
        self.dimindex_input_source = tk.StringVar(value="previous")
        ttk.Radiobutton(input_frame, text="From LAAT+MBMS", variable=self.dimindex_input_source, 
                        value="previous").pack(side=tk.LEFT)
        ttk.Radiobutton(input_frame, text="Custom files", variable=self.dimindex_input_source,
                        value="custom").pack(side=tk.LEFT, padx=10)
        ttk.Button(input_frame, text="Load...", command=self._load_dimindex_custom_input).pack(side=tk.LEFT)
        self.dimindex_custom_input_label = ttk.Label(input_frame, text="", foreground='gray')
        self.dimindex_custom_input_label.pack(side=tk.LEFT, padx=5)
        
        params_frame = ttk.Frame(frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Row 1
        row1 = ttk.Frame(params_frame)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Label(row1, text="Radius:", width=12).pack(side=tk.LEFT)
        self.dimindex_radius = ttk.Entry(row1, width=10)
        self.dimindex_radius.insert(0, "3.5")
        self.dimindex_radius.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row1, text="Cutoff:", width=10).pack(side=tk.LEFT, padx=(10,0))
        self.dimindex_cutoff = ttk.Entry(row1, width=10)
        self.dimindex_cutoff.insert(0, "5")
        self.dimindex_cutoff.pack(side=tk.LEFT, padx=5)
        
        # Row 2
        row2 = ttk.Frame(params_frame)
        row2.pack(fill=tk.X, pady=2)
        
        ttk.Label(row2, text="Simplex:", width=12).pack(side=tk.LEFT)
        self.dimindex_simplex = ttk.Combobox(row2, values=["Barycentric", "Original"], width=12)
        self.dimindex_simplex.set("Barycentric")
        self.dimindex_simplex.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row2, text="Smooth:", width=10).pack(side=tk.LEFT, padx=(10,0))
        self.dimindex_smooth = ttk.Combobox(row2, values=["l2", "l1", "none"], width=10)
        self.dimindex_smooth.set("l2")
        self.dimindex_smooth.pack(side=tk.LEFT, padx=5)
        
        # Row 3 - Target dimension
        row3 = ttk.Frame(params_frame)
        row3.pack(fill=tk.X, pady=2)
        
        ttk.Label(row3, text="Target Dim:", width=12).pack(side=tk.LEFT)
        self.dimindex_target = ttk.Combobox(row3, values=["1", "2"], width=10)
        self.dimindex_target.set("1")
        self.dimindex_target.pack(side=tk.LEFT, padx=5)
        ttk.Label(row3, text="(dimension to select)", foreground='gray').pack(side=tk.LEFT)
        
        # Run button
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.dimindex_btn = ttk.Button(btn_frame, text="▶ Run DimIndex", style='Run.TButton', 
                                       command=self._run_dimindex)
        self.dimindex_btn.pack(side=tk.RIGHT)
        self.dimindex_status = ttk.Label(btn_frame, text="", foreground='gray')
        self.dimindex_status.pack(side=tk.LEFT)
        
    def _create_crawling_section(self, parent):
        """Create Crawling configuration section"""
        frame = ttk.LabelFrame(parent, text="4️⃣ Crawling - Graph Construction", style='Method.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Input source selection
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(input_frame, text="Input:", width=12).pack(side=tk.LEFT)
        self.crawling_input_source = tk.StringVar(value="previous")
        ttk.Radiobutton(input_frame, text="From DimIndex", variable=self.crawling_input_source, 
                        value="previous").pack(side=tk.LEFT)
        ttk.Radiobutton(input_frame, text="Custom file", variable=self.crawling_input_source,
                        value="custom").pack(side=tk.LEFT, padx=10)
        ttk.Button(input_frame, text="Load...", command=self._load_crawling_custom_input).pack(side=tk.LEFT)
        self.crawling_custom_input_label = ttk.Label(input_frame, text="", foreground='gray')
        self.crawling_custom_input_label.pack(side=tk.LEFT, padx=5)
        
        params_frame = ttk.Frame(frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Row 1
        row1 = ttk.Frame(params_frame)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Label(row1, text="Radius:", width=12).pack(side=tk.LEFT)
        self.crawling_radius = ttk.Entry(row1, width=10)
        self.crawling_radius.insert(0, "3.5")
        self.crawling_radius.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row1, text="Latent Dim:", width=10).pack(side=tk.LEFT, padx=(10,0))
        self.crawling_ldim = ttk.Entry(row1, width=10)
        self.crawling_ldim.insert(0, "1")
        self.crawling_ldim.pack(side=tk.LEFT, padx=5)
        
        # Row 2
        row2 = ttk.Frame(params_frame)
        row2.pack(fill=tk.X, pady=2)
        
        ttk.Label(row2, text="Beta:", width=12).pack(side=tk.LEFT)
        self.crawling_beta = ttk.Entry(row2, width=10)
        self.crawling_beta.insert(0, "0.4")
        self.crawling_beta.pack(side=tk.LEFT, padx=5)
        
        # Run button
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.crawling_btn = ttk.Button(btn_frame, text="▶ Run Crawling", style='Run.TButton', 
                                       command=self._run_crawling)
        self.crawling_btn.pack(side=tk.RIGHT)
        self.crawling_status = ttk.Label(btn_frame, text="", foreground='gray')
        self.crawling_status.pack(side=tk.LEFT)
        
    def _create_sgtm_section(self, parent):
        """Create SGTM configuration section"""
        frame = ttk.LabelFrame(parent, text="5️⃣ SGTM - Generative Topographic Mapping", style='Method.TLabelframe')
        frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Input source selection (SGTM needs Crawling graphs)
        input_frame = ttk.Frame(frame)
        input_frame.pack(fill=tk.X, padx=5, pady=2)
        
        ttk.Label(input_frame, text="Input:", width=12).pack(side=tk.LEFT)
        self.sgtm_input_source = tk.StringVar(value="previous")
        ttk.Radiobutton(input_frame, text="From Crawling", variable=self.sgtm_input_source, 
                        value="previous").pack(side=tk.LEFT)
        ttk.Radiobutton(input_frame, text="Custom (pkl)", variable=self.sgtm_input_source,
                        value="custom").pack(side=tk.LEFT, padx=10)
        ttk.Button(input_frame, text="Load...", command=self._load_sgtm_custom_input).pack(side=tk.LEFT)
        self.sgtm_custom_input_label = ttk.Label(input_frame, text="", foreground='gray')
        self.sgtm_custom_input_label.pack(side=tk.LEFT, padx=5)
        
        params_frame = ttk.Frame(frame)
        params_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Row 1
        row1 = ttk.Frame(params_frame)
        row1.pack(fill=tk.X, pady=2)
        
        ttk.Label(row1, text="Int. Dim:", width=12).pack(side=tk.LEFT)
        self.sgtm_intdim = ttk.Entry(row1, width=10)
        self.sgtm_intdim.insert(0, "1")
        self.sgtm_intdim.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row1, text="Radius:", width=10).pack(side=tk.LEFT, padx=(10,0))
        self.sgtm_radius = ttk.Entry(row1, width=10)
        self.sgtm_radius.insert(0, "3.5")
        self.sgtm_radius.pack(side=tk.LEFT, padx=5)
        
        # Row 2
        row2 = ttk.Frame(params_frame)
        row2.pack(fill=tk.X, pady=2)
        
        ttk.Label(row2, text="Epsilon:", width=12).pack(side=tk.LEFT)
        self.sgtm_epsilon = ttk.Entry(row2, width=10)
        self.sgtm_epsilon.insert(0, "2")
        self.sgtm_epsilon.pack(side=tk.LEFT, padx=5)
        
        ttk.Label(row2, text="Mem Mode:", width=10).pack(side=tk.LEFT, padx=(10,0))
        self.sgtm_mem = ttk.Combobox(row2, values=["0", "1", "2"], width=10)
        self.sgtm_mem.set("2")
        self.sgtm_mem.pack(side=tk.LEFT, padx=5)
        
        # Run button
        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)
        self.sgtm_btn = ttk.Button(btn_frame, text="▶ Run SGTM", style='Run.TButton', 
                                   command=self._run_sgtm)
        self.sgtm_btn.pack(side=tk.RIGHT)
        self.sgtm_status = ttk.Label(btn_frame, text="", foreground='gray')
        self.sgtm_status.pack(side=tk.LEFT)
        
    def _create_right_panel(self, parent):
        """Create the right panel with log output"""
        # Output/Log section
        log_frame = ttk.LabelFrame(parent, text="📋 Output Log", style='Method.TLabelframe')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, 
                                                   font=('Consolas', 9), state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Configure tags for colored output
        self.log_text.tag_configure('stdout', foreground='black')
        self.log_text.tag_configure('stderr', foreground='red')
        self.log_text.tag_configure('info', foreground='blue')
        self.log_text.tag_configure('success', foreground='green')
        self.log_text.tag_configure('warning', foreground='orange')
        
        # Control buttons
        ctrl_frame = ttk.Frame(log_frame)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(ctrl_frame, text="Clear Log", command=self._clear_log).pack(side=tk.LEFT)
        ttk.Button(ctrl_frame, text="Save Log...", command=self._save_log).pack(side=tk.LEFT, padx=5)
        
        # Redirect stdout/stderr
        sys.stdout = TextRedirector(self.log_text, 'stdout')
        sys.stderr = TextRedirector(self.log_text, 'stderr')
        
    def _create_status_bar(self):
        """Create status bar at the bottom"""
        status_frame = ttk.Frame(self)
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        
        status_label = ttk.Label(status_frame, textvariable=self.status_var, 
                                  relief=tk.SUNKEN, anchor=tk.W)
        status_label.pack(fill=tk.X, side=tk.LEFT, expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate', length=150)
        self.progress.pack(side=tk.RIGHT, padx=5)
        
    def _log(self, message, tag='info'):
        """Log a message to the output panel"""
        self.log_text.configure(state='normal')
        self.log_text.insert(tk.END, f"{message}\n", (tag,))
        self.log_text.see(tk.END)
        self.log_text.configure(state='disabled')
        
    def _update_workflow_indicator(self):
        """Update the workflow progress indicators"""
        for step, label in self.workflow_labels.items():
            if self.workflow_state.get(f'{step}_complete', False):
                label.config(text="●", foreground='green')
            else:
                label.config(text="○", foreground='gray')
                
    def _load_data(self):
        """Load point cloud data from CSV file with interactive column selection"""
        description = ("Select a CSV file and choose which columns to use as point cloud dimensions.\n"
                       "You can select any subset of columns and they will be used in order.")
        
        selector = CSVColumnSelector(
            self,
            title="Load Point Cloud Data",
            column_assignments=[
                {'name': 'Dim 1', 'description': 'Dimension 1 (e.g., X)'},
                {'name': 'Dim 2', 'description': 'Dimension 2 (e.g., Y)'},
                {'name': 'Dim 3', 'description': 'Dimension 3 (e.g., Z)'}
            ],
            single_column=False,
            description=description
        )
        
        result = selector.get_result()
        
        if result is not None:
            try:
                self.data = result.astype(np.float64)
                self.data_path = Path(selector.csv_path) if selector.csv_path else None
                
                if self.output_dir is None and self.data_path:
                    self.output_dir = self.data_path.parent / "Output"
                    self.output_dir.mkdir(exist_ok=True)
                
                filename = self.data_path.name if self.data_path else "Custom data"
                self.data_label.config(text=filename, foreground='black')
                self.data_info_label.config(
                    text=f"Shape: {self.data.shape[0]} points × {self.data.shape[1]} dimensions"
                )
                self.workflow_state['data_loaded'] = True
                self._log(f"Loaded data: {filename}", 'success')
                self._log(f"  Shape: {self.data.shape}", 'info')
                self.status_var.set(f"Loaded: {filename}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to process data:\n{str(e)}")
                self._log(f"Error processing data: {str(e)}", 'stderr')
                
    def _set_output_dir(self):
        """Set custom output directory"""
        dirpath = filedialog.askdirectory(title="Select Output Directory")
        if dirpath:
            self.output_dir = Path(dirpath)
            self.output_dir.mkdir(exist_ok=True)
            self.output_label.config(text=str(self.output_dir), foreground='black')
            self._log(f"Output directory set to: {self.output_dir}", 'info')
            
    def _check_queue(self):
        """Check the task queue for results"""
        try:
            while True:
                task, result = self.task_queue.get_nowait()
                if task == 'laat_done':
                    self._on_laat_complete(result)
                elif task == 'mbms_done':
                    self._on_mbms_complete(result)
                elif task == 'dimindex_done':
                    self._on_dimindex_complete(result)
                elif task == 'crawling_done':
                    self._on_crawling_complete(result)
                elif task == 'sgtm_done':
                    self._on_sgtm_complete(result)
                elif task == 'error':
                    self._on_task_error(result)
        except queue.Empty:
            pass
        self.after(100, self._check_queue)
        
    def _run_in_thread(self, func, *args, **kwargs):
        """Run a function in a background thread"""
        def wrapper():
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                self.task_queue.put(('error', (str(e), traceback.format_exc())))
                return None
        
        thread = threading.Thread(target=wrapper, daemon=True)
        thread.start()
        
    def _start_progress(self, message):
        """Start progress indicator"""
        self.status_var.set(message)
        self.progress.start(10)
        
    def _stop_progress(self):
        """Stop progress indicator"""
        self.progress.stop()
        self.status_var.set("Ready")
        
    def _on_task_error(self, error_info):
        """Handle task error"""
        error_msg, trace = error_info
        self._log(f"Error: {error_msg}", 'stderr')
        self._log(trace, 'stderr')
        self._stop_progress()
        messagebox.showerror("Error", f"Task failed:\n{error_msg}")
        
    # ==================== LAAT ====================
    def _run_laat(self):
        """Run LAAT algorithm"""
        if not self.workflow_state['data_loaded']:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        try:
            import LAAT_MBMS
            laat_available = hasattr(LAAT_MBMS, 'LAAT')
        except ImportError:
            laat_available = False
            
        if not laat_available:
            msg = ("LAAT_MBMS module not found or LAAT function not available.\n\n"
                   "On Linux/Ubuntu, install with:\n"
                   "  cd LAAT_MBMS\n"
                   "  cp python_library_CMakeLists/CMakeLists_LAAT_AND_MBMS.txt CMakeLists.txt\n"
                   "  pip install .\n\n"
                   "On Windows, LAAT requires the pure C++ executable.\n"
                   "See LAAT_MBMS/pure_cpp_LAAT/PURE_CPP_LAAT_README.md")
            messagebox.showerror("Error", msg)
            return
            
        self._start_progress("Running LAAT...")
        self._log("=" * 50, 'info')
        self._log("Starting LAAT...", 'info')
        
        # Get advanced options from stored arrays (set by column selector dialogs)
        external_weights = getattr(self, 'laat_external_weights', np.array([], dtype=np.float32))
        custom_init_indices = getattr(self, 'laat_custom_init_indices', np.array([], dtype=np.uint64))
        
        if len(external_weights) > 0:
            self._log(f"Using external field: {len(external_weights)} values", 'info')
        if len(custom_init_indices) > 0:
            self._log(f"Using init indices: {len(custom_init_indices)} indices", 'info')
        
        def run_laat_task():
            import LAAT_MBMS
            
            num_ants = int(self.laat_num_ants.get())
            iterations = int(self.laat_iterations.get())
            steps = int(self.laat_steps.get())
            threads = int(self.laat_threads.get())
            radius = float(self.laat_radius.get())
            kappa = float(self.laat_kappa.get())
            gamma = float(self.laat_gamma.get())
            th_neighb = int(self.laat_th_neighb.get())
            threshold_pct = float(self.laat_threshold.get())
            
            # Parse initialization mode
            init_mode_str = self.laat_init_mode.get()
            init_mode = 1 if "Custom" in init_mode_str else 0
            
            print(f"Parameters: ants={num_ants}, iter={iterations}, steps={steps}")
            print(f"            radius={radius}, kappa={kappa}, gamma={gamma}, thresh={th_neighb}")
            print(f"            init_mode={init_mode}, threads={threads} (OpenMP)")
            
            if len(external_weights) > 0:
                print(f"            external_weights: {len(external_weights)} values")
            if len(custom_init_indices) > 0:
                print(f"            custom_init_indices: {len(custom_init_indices)} indices")
            
            pheromone = LAAT_MBMS.LAAT(
                self.data.astype(np.float32),
                num_ants,
                iterations,
                steps,
                30,  # pso_number_particles
                1.0,  # pso_min_radii
                radius,  # pso_max_radii
                0,  # dynamic_radius_actived
                th_neighb,
                kappa,
                gamma,
                external_weights,
                init_mode,
                custom_init_indices,
                threads
            )
            
            # Select points based on pheromone threshold
            threshold = np.percentile(pheromone, 100 - threshold_pct)
            selected_idx = pheromone >= threshold
            selected_data = self.data[selected_idx]
            
            print(f"Selected {len(selected_data)} points (top {threshold_pct}%)")
            
            self.task_queue.put(('laat_done', (pheromone, selected_data)))
            
        self._run_in_thread(run_laat_task)
        
    def _on_laat_complete(self, result):
        """Handle LAAT completion"""
        pheromone, selected_data = result
        self.results['pheromone'] = pheromone
        self.results['laat_selected'] = selected_data
        
        # Save results
        np.savetxt(self.output_dir / "LAAT_output_pheromone.csv", pheromone, delimiter=',', fmt='%5.8f')
        np.savetxt(self.output_dir / "LAAT_output_selected_data.csv", selected_data, delimiter=',', fmt='%5.8f')
        
        self.workflow_state['laat_complete'] = True
        self._update_workflow_indicator()
        self._stop_progress()
        self._log(f"LAAT complete! Selected {len(selected_data)} points", 'success')
        self.laat_status.config(text=f"✓ {len(selected_data)} pts", foreground='green')
        
        # Show visualization
        try:
            threshold_pct = float(self.laat_threshold.get())
        except ValueError:
            threshold_pct = 5.0
        LAATVisualizationDialog(self, self.data, pheromone, threshold_pct)
        
    # ==================== MBMS ====================
    def _run_mbms(self):
        """Run MBMS algorithm"""
        # Check input source
        if self.mbms_input_source.get() == "custom":
            input_data = getattr(self, 'mbms_custom_data', None)
            if input_data is None:
                messagebox.showwarning("Warning", 
                    "Custom input selected but no data loaded. Use 'Load...' to load custom data.")
                return
            self._log("Using custom input data for MBMS", 'info')
        else:
            # Use previous step output
            input_data = self.results.get('laat_selected')
            if input_data is None:
                # Try to load from file
                laat_file = self.output_dir / "LAAT_output_selected_data.csv" if self.output_dir else None
                if laat_file and laat_file.exists():
                    input_data = np.loadtxt(laat_file, delimiter=',')
                    self.results['laat_selected'] = input_data
                else:
                    messagebox.showwarning("Warning", 
                        "No LAAT output found. Please run LAAT first, load LAAT results, or use custom input.")
                    return
                
        try:
            import LAAT_MBMS
            mbms_available = hasattr(LAAT_MBMS, 'MBMS')
        except ImportError:
            mbms_available = False
            
        if not mbms_available:
            msg = ("LAAT_MBMS module not found or MBMS function not available.\n\n"
                   "Install with:\n"
                   "  cd LAAT_MBMS\n"
                   "  pip install .")
            messagebox.showerror("Error", msg)
            return
            
        self._start_progress("Running MBMS...")
        self._log("=" * 50, 'info')
        self._log("Starting MBMS...", 'info')
        self._log(f"Input shape: {input_data.shape}", 'info')
        
        def run_mbms_task():
            import LAAT_MBMS
            
            iterations = int(self.mbms_iter.get())
            radius = float(self.mbms_radius.get())
            sigma = float(self.mbms_sigma.get())
            k = int(self.mbms_k.get())
            
            print(f"Parameters: iter={iterations}, radius={radius}, sigma={sigma}, k={k}")
            
            mbms_data = LAAT_MBMS.MBMS(
                input_data.astype(np.float32),
                iter=iterations,
                radius=radius,
                sigma=sigma,
                k=k
            )
            
            print(f"MBMS output shape: {mbms_data.shape}")
            
            self.task_queue.put(('mbms_done', (mbms_data, input_data)))
            
        self._run_in_thread(run_mbms_task)
        
    def _on_mbms_complete(self, result):
        """Handle MBMS completion"""
        mbms_data, input_data = result
        self.results['mbms_data'] = mbms_data
        self.results['mbms_input'] = input_data  # Store for visualization
        
        np.savetxt(self.output_dir / "MBMS_output.csv", mbms_data, delimiter=',', fmt='%5.8f')
        
        self.workflow_state['mbms_complete'] = True
        self._update_workflow_indicator()
        self._stop_progress()
        self._log(f"MBMS complete! Output shape: {mbms_data.shape}", 'success')
        self.mbms_status.config(text=f"✓ {mbms_data.shape[0]} pts", foreground='green')
        
        # Show visualization
        MBMSVisualizationDialog(self, input_data, mbms_data, self.data)
        
    # ==================== DimIndex ====================
    def _run_dimindex(self):
        """Run DimIndex algorithm"""
        from DimIndexModule import Dim_Index, Filtering
        
        # Check input source
        if self.dimindex_input_source.get() == "custom":
            laat_data = getattr(self, 'dimindex_custom_laat', None)
            mbms_data = getattr(self, 'dimindex_custom_mbms', None)
            if laat_data is None or mbms_data is None:
                messagebox.showwarning("Warning", 
                    "Custom input selected but data not loaded. Use 'Load...' to load custom data.")
                return
            self._log("Using custom input data for DimIndex", 'info')
        else:
            # Use previous step output
            laat_data = self.results.get('laat_selected')
            mbms_data = self.results.get('mbms_data')
            
            if laat_data is None or mbms_data is None:
                # Try to load from files
                if self.output_dir:
                    laat_file = self.output_dir / "LAAT_output_selected_data.csv"
                    mbms_file = self.output_dir / "MBMS_output.csv"
                    if laat_file.exists() and mbms_file.exists():
                        laat_data = np.loadtxt(laat_file, delimiter=',')
                        mbms_data = np.loadtxt(mbms_file, delimiter=',')
                        self.results['laat_selected'] = laat_data
                        self.results['mbms_data'] = mbms_data
                    else:
                        messagebox.showwarning("Warning", 
                            "LAAT and MBMS outputs required. Please run previous steps first or use custom input.")
                        return
                else:
                    messagebox.showwarning("Warning", "Please set output directory first.")
                    return
                
        self._start_progress("Running DimIndex...")
        self._log("=" * 50, 'info')
        self._log("Starting DimIndex...", 'info')
        self._log(f"LAAT input shape: {laat_data.shape}, MBMS input shape: {mbms_data.shape}", 'info')
        
        def run_dimindex_task():
            from DimIndexModule import Dim_Index, Filtering
            
            radius = float(self.dimindex_radius.get())
            cutoff = int(self.dimindex_cutoff.get())
            simplex = self.dimindex_simplex.get()
            smooth = self.dimindex_smooth.get()
            target_dim = int(self.dimindex_target.get()) - 1  # 0-indexed
            
            print(f"Parameters: radius={radius}, cutoff={cutoff}, simplex={simplex}")
            
            # Filtering
            laat_filtered, mbms_filtered, labels, _, _ = Filtering(
                laat_data, mbms_data, radius, cutoff
            )
            
            print(f"After filtering: {mbms_filtered.shape[0]} points")
            
            # Run DimIndex
            struct, indexes = Dim_Index(mbms_filtered, radius, simplex, smooth)
            
            # Select target dimension points
            idx_dim = indexes[:, 1]  # smoothed index
            selected = mbms_filtered[idx_dim == target_dim]
            
            print(f"Selected {len(selected)} points of dimension {target_dim + 1}")
            
            self.task_queue.put(('dimindex_done', (indexes, selected, labels, mbms_filtered, laat_data)))
            
        self._run_in_thread(run_dimindex_task)
        
    def _on_dimindex_complete(self, result):
        """Handle DimIndex completion"""
        indexes, selected, labels, mbms_filtered, laat_data = result
        self.results['dimindex_data'] = selected
        self.results['dimindex_mbms_filtered'] = mbms_filtered  # Store for visualization
        self.results['dimindex_indexes'] = indexes  # Store for visualization
        self.results['dimindex_labels'] = labels  # Store for visualization
        
        np.savetxt(self.output_dir / "dimindexes.csv", indexes, delimiter=',', fmt='%d')
        np.savetxt(self.output_dir / "Labels.csv", labels, delimiter=',', fmt='%i')
        np.savetxt(self.output_dir / "Selected_data_after_DimIndex_smoothed.csv", 
                   selected, delimiter=',', fmt='%5.8f')
        
        self.workflow_state['dimindex_complete'] = True
        self._update_workflow_indicator()
        self._stop_progress()
        self._log(f"DimIndex complete! Selected {len(selected)} points", 'success')
        self.dimindex_status.config(text=f"✓ {len(selected)} pts", foreground='green')
        
        # Show visualization
        DimIndexVisualizationDialog(self, mbms_filtered, indexes, labels, 
                                    laat_data, self.data)
        
    # ==================== Crawling ====================
    def _run_crawling(self):
        """Run Crawling algorithm"""
        from CrawlingModule import MultiM, AllNodeNames
        
        # Check input source
        if self.crawling_input_source.get() == "custom":
            spine_data = getattr(self, 'crawling_custom_data', None)
            if spine_data is None:
                messagebox.showwarning("Warning", 
                    "Custom input selected but no data loaded. Use 'Load...' to load custom data.")
                return
            self._log("Using custom input data for Crawling", 'info')
        else:
            # Use previous step output
            spine_data = self.results.get('dimindex_data')
            
            if spine_data is None:
                if self.output_dir:
                    dimindex_file = self.output_dir / "Selected_data_after_DimIndex_smoothed.csv"
                    if dimindex_file.exists():
                        spine_data = np.loadtxt(dimindex_file, delimiter=',')
                        self.results['dimindex_data'] = spine_data
                    else:
                        messagebox.showwarning("Warning", 
                            "DimIndex output required. Please run DimIndex first or use custom input.")
                        return
                else:
                    messagebox.showwarning("Warning", "Please set output directory first.")
                    return
                
        if self.data is None:
            messagebox.showwarning("Warning", "Please load original data first.")
            return
            
        self._start_progress("Running Crawling...")
        self._log("=" * 50, 'info')
        self._log("Starting Crawling...", 'info')
        self._log(f"Spine input shape: {spine_data.shape}", 'info')
        
        def run_crawling_task():
            from CrawlingModule import MultiM, AllNodeNames
            
            radius = float(self.crawling_radius.get())
            ldim = int(self.crawling_ldim.get())
            beta = float(self.crawling_beta.get())
            
            print(f"Parameters: radius={radius}, ldim={ldim}, beta={beta}")
            
            FG, NoisyMan, _ = MultiM(spine_data, self.data, radius, ldim, beta)
            
            print(f"Found {len(FG)} graph(s)")
            
            self.task_queue.put(('crawling_done', (FG, NoisyMan, spine_data)))
            
        self._run_in_thread(run_crawling_task)
        
    def _on_crawling_complete(self, result):
        """Handle Crawling completion"""
        from CrawlingModule import AllNodeNames
        
        FG, NoisyMan, spine_data = result
        self.results['crawling_graphs'] = FG
        self.results['crawling_subsets'] = NoisyMan
        self.results['crawling_spine'] = spine_data  # Store for visualization
        
        # Save results
        pickle.dump((FG, NoisyMan), open(self.output_dir / "Crawling_output.pkl", 'wb'))
        
        for i, G in enumerate(FG):
            pos = AllNodeNames(G, 3)
            np.savetxt(self.output_dir / f"NodePos{i+1}.csv", pos, delimiter=',', fmt='%5.8f')
            np.savetxt(self.output_dir / f"Subset{i+1}.csv", NoisyMan[i], delimiter=',', fmt='%5.8f')
        
        self.workflow_state['crawling_complete'] = True
        self._update_workflow_indicator()
        self._stop_progress()
        self._log(f"Crawling complete! Found {len(FG)} graph(s)", 'success')
        self.crawling_status.config(text=f"✓ {len(FG)} graphs", foreground='green')
        
        # Show visualization
        CrawlingVisualizationDialog(self, FG, NoisyMan, spine_data, self.data)
        
    # ==================== SGTM ====================
    def _run_sgtm(self):
        """Run SGTM algorithm"""
        from AGTMModule import Standardized_AGTM_InitTrain
        
        # Check input source
        if self.sgtm_input_source.get() == "custom":
            FG = getattr(self, 'sgtm_custom_graphs', None)
            if FG is None:
                messagebox.showwarning("Warning", 
                    "Custom input selected but no data loaded. Use 'Load...' to load custom data.")
                return
            self._log("Using custom input data for SGTM", 'info')
        else:
            # Use previous step output
            FG = self.results.get('crawling_graphs')
            
            if FG is None:
                if self.output_dir:
                    crawling_file = self.output_dir / "Crawling_output.pkl"
                    if crawling_file.exists():
                        FG, NoisyMan = pickle.load(open(crawling_file, 'rb'))
                        self.results['crawling_graphs'] = FG
                        self.results['crawling_subsets'] = NoisyMan
                    else:
                        messagebox.showwarning("Warning", 
                            "Crawling output required. Please run Crawling first or use custom input.")
                        return
                else:
                    messagebox.showwarning("Warning", "Please set output directory first.")
                    return
                
        if self.data is None:
            messagebox.showwarning("Warning", "Please load original data first.")
            return
            
        self._start_progress("Running SGTM...")
        self._log("=" * 50, 'info')
        self._log("Starting SGTM...", 'info')
        self._log(f"Number of graphs: {len(FG)}", 'info')
        
        def run_sgtm_task():
            from AGTMModule import Standardized_AGTM_InitTrain
            
            intdim = int(self.sgtm_intdim.get())
            radius = float(self.sgtm_radius.get())
            epsilon = float(self.sgtm_epsilon.get())
            mem = int(self.sgtm_mem.get())
            
            print(f"Parameters: intdim={intdim}, radius={radius}, epsilon={epsilon}, mem={mem}")
            
            net, logL, GMDist, NoisyMan = Standardized_AGTM_InitTrain(
                FG, self.data, intdim, radius, epsilon, mem
            )
            
            print(f"Trained {len(net)} network(s)")
            
            self.task_queue.put(('sgtm_done', (net, GMDist, NoisyMan, FG)))
            
        self._run_in_thread(run_sgtm_task)
        
    def _on_sgtm_complete(self, result):
        """Handle SGTM completion"""
        net, GMDist, NoisyMan, FG = result
        self.results['sgtm_nets'] = net
        self.results['sgtm_gmdist'] = GMDist  # Store for visualization
        self.results['sgtm_noisy_subsets'] = NoisyMan  # Store for visualization
        
        # Save results
        pickle.dump((net, GMDist, NoisyMan), open(self.output_dir / "SGTM_output.pkl", 'wb'))
        
        for i in range(len(net)):
            node_pos = net[i].gmmnet.V_
            np.savetxt(self.output_dir / f"SGTM_positions_node_{i+1}.csv", 
                       node_pos, delimiter=',', fmt='%5.8f')
        
        self.workflow_state['sgtm_complete'] = True
        self._update_workflow_indicator()
        self._stop_progress()
        self._log(f"SGTM complete! Trained {len(net)} network(s)", 'success')
        self.sgtm_status.config(text=f"✓ {len(net)} nets", foreground='green')
        
        # Show visualization
        spine_data = self.results.get('dimindex_data')  # 1D data
        SGTMVisualizationDialog(self, net, FG, NoisyMan, GMDist, spine_data, self.data)
        
    # ==================== Utility Methods ====================
    def _run_full_pipeline(self):
        """Run the full pipeline sequentially"""
        if not self.workflow_state['data_loaded']:
            messagebox.showwarning("Warning", "Please load data first!")
            return
            
        messagebox.showinfo("Full Pipeline", 
            "This will run all 5 methods sequentially:\n"
            "LAAT → MBMS → DimIndex → Crawling → SGTM\n\n"
            "This may take a while depending on your data size.")
        
        # For now, just show info - sequential execution would need careful threading
        self._log("Full pipeline not yet implemented. Please run each step manually.", 'warning')
        
    def _reset_workflow(self):
        """Reset all workflow state"""
        if messagebox.askyesno("Confirm", "Reset all results and workflow state?"):
            self.workflow_state = {k: False for k in self.workflow_state}
            self.workflow_state['data_loaded'] = self.data is not None
            self.results = {k: None for k in self.results}
            self._update_workflow_indicator()
            
            # Reset status labels
            for status in [self.laat_status, self.mbms_status, self.dimindex_status,
                          self.crawling_status, self.sgtm_status]:
                status.config(text="", foreground='gray')
                
            self._log("Workflow reset", 'info')
            
    def _load_results(self):
        """Load previous results from files"""
        if self.output_dir is None:
            self._set_output_dir()
            if self.output_dir is None:
                return
                
        loaded = []
        
        # Try loading each result type
        files_to_check = [
            ("LAAT_output_selected_data.csv", 'laat_selected', 'laat'),
            ("MBMS_output.csv", 'mbms_data', 'mbms'),
            ("Selected_data_after_DimIndex_smoothed.csv", 'dimindex_data', 'dimindex'),
        ]
        
        for filename, key, step in files_to_check:
            filepath = self.output_dir / filename
            if filepath.exists():
                self.results[key] = np.loadtxt(filepath, delimiter=',')
                self.workflow_state[f'{step}_complete'] = True
                loaded.append(step.upper())
                
        # Check pickle files
        crawling_file = self.output_dir / "Crawling_output.pkl"
        if crawling_file.exists():
            FG, NoisyMan = pickle.load(open(crawling_file, 'rb'))
            self.results['crawling_graphs'] = FG
            self.results['crawling_subsets'] = NoisyMan
            self.workflow_state['crawling_complete'] = True
            loaded.append("Crawling")
            
        sgtm_file = self.output_dir / "SGTM_output.pkl"
        if sgtm_file.exists():
            net, GMDist, NoisyMan = pickle.load(open(sgtm_file, 'rb'))
            self.results['sgtm_nets'] = net
            self.workflow_state['sgtm_complete'] = True
            loaded.append("SGTM")
            
        self._update_workflow_indicator()
        
        if loaded:
            self._log(f"Loaded results for: {', '.join(loaded)}", 'success')
        else:
            self._log("No previous results found in output directory", 'warning')
            
    def _save_results(self):
        """Save all current results"""
        if self.output_dir is None:
            self._set_output_dir()
            
        self._log("Results are automatically saved after each step", 'info')
        
    def _clear_log(self):
        """Clear the log output"""
        self.log_text.configure(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.configure(state='disabled')
        
    def _save_log(self):
        """Save log to file"""
        filepath = filedialog.asksaveasfilename(
            title="Save Log",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )
        if filepath:
            with open(filepath, 'w') as f:
                f.write(self.log_text.get(1.0, tk.END))
            self._log(f"Log saved to {filepath}", 'success')
            
    def _show_about(self):
        """Show about dialog"""
        messagebox.showinfo("About 1DREAM",
            "1DREAM Toolbox\n\n"
            "A suite of algorithms for manifold learning:\n"
            "• LAAT - Ant-based feature selection\n"
            "• MBMS - Mean-shift denoising\n"
            "• DimIndex - Dimensionality estimation\n"
            "• Crawling - Graph construction\n"
            "• SGTM - Generative topographic mapping\n\n"
            "Version 1.0\n"
            "License: GNU AGPL v3.0")
            
    def _show_workflow_guide(self):
        """Show workflow guide"""
        messagebox.showinfo("Workflow Guide",
            "Recommended Workflow:\n\n"
            "1. LAAT: Select points with high manifold likelihood\n"
            "   using ant colony optimization\n\n"
            "2. MBMS: Denoise selected points using\n"
            "   manifold-based mean shift\n\n"
            "3. DimIndex: Estimate local dimensionality\n"
            "   and filter to target dimension\n\n"
            "4. Crawling: Build graph structure on the\n"
            "   filtered manifold\n\n"
            "5. SGTM: Fit generative topographic mapping\n"
            "   to learn the manifold parameterization")


def main():
    """Main entry point"""
    app = DREAMToolbox()
    app.mainloop()


if __name__ == "__main__":
    main()
