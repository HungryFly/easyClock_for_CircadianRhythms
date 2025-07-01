# fix Cosinor-Kendall Lag (acrophase) via corrected_lag;
# fix Python-JTK Lag (acrophase) via (1-asymetry);
# change amplitude calculation methods in Cosinor-Kendall and Python-JTK.

import os
import sys
import shutil
import tempfile
import subprocess
import pandas as pd
import numpy as np
import colorsys
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from scipy.stats import kendalltau
from scipy.optimize import curve_fit
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QMessageBox, QInputDialog, QDialog, QVBoxLayout, QHBoxLayout,QComboBox,QFileDialog,QDialogButtonBox,
    QLabel, QPushButton, QTextEdit, QScrollArea, QWidget, QSizePolicy, QColorDialog,QLineEdit
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Arial"

def generate_distinct_colors(n):
    hsv_colors = [(i / n, 0.5, 0.95) for i in range(n)]
    rgb_colors = [colorsys.hsv_to_rgb(*h) for h in hsv_colors]
    return ['#%02x%02x%02x' % tuple(int(c * 255) for c in rgb) for rgb in rgb_colors]


def get_resource_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)

def acrophase_to_hours(rad_phase, period=24):
    hours = (rad_phase * period) / (2 * np.pi)
    return hours % period

# ------------------------
# Python-JTK function
# ------------------------

def generate_triangle_template(length, peak_index):
    """
    Create triangle waveform of specified length and peak location (index).
    """
    template = np.zeros(length)
    if peak_index > 0:
        template[:peak_index] = np.linspace(1, peak_index, peak_index)
    if length - peak_index > 0:
        template[peak_index:] = np.linspace(length - peak_index, 1, length - peak_index)
    return template

def create_ranked_templates(n_points, period, lag_range, asymmetry=0.5):
    """
    Generate all shifted triangle templates of a given period and asymmetry within the lag_range.
    """
    peak_index = int(np.round(asymmetry * period))
    base = generate_triangle_template(period, peak_index)
    ranked = pd.Series(base).rank().values

    # Extend the template to at least cover all timepoints
    full_template = np.tile(ranked, int(np.ceil(n_points / period)) + 1)[:n_points]

    # Generate only the requested phase shifts (lags)
    templates = []
    for lag in lag_range:
        lag = int(lag % period)
        ref = np.roll(full_template, lag)
        templates.append((ref, lag))

    return templates

def generate_triangle_template_time(times, period, lag, asymmetry=0.5):
    """
    Generate a triangle template aligned to real timepoints.
    `times`: array of time values
    `period`: float, desired period in same units as time
    `lag`: phase shift in time units (e.g., hours)
    `asymmetry`: float between 0 and 1 indicating peak position in the cycle
    """
    peak_time = (lag + (1-asymmetry) * period) % period
    template = np.zeros_like(times, dtype=float)
    for i, t in enumerate(times):
        t_mod = (t - lag) % period
        if t_mod <= peak_time:
            template[i] = t_mod / peak_time if peak_time != 0 else 1.0
        else:
            template[i] = (period - t_mod) / (period - peak_time) if period != peak_time else 0.0
    return pd.Series(template).rank().values


def run_discrete_jtk(series, period_range=range(22, 27), lag_range=None, asymmetries=[0.5]):
    """
    Run JTK using triangle templates aligned to actual timepoints (non-uniform supported).
    Run triangle-based JTK_CYCLE with user-defined period and lag (acrophase) ranges.
    """
    times = series.index.to_numpy()
    y = series.rank().values
    n = len(y)

    best_p = 1.0
    best_tau = 0.0
    best_per = None
    best_lag = None
    best_asym = None

    test_results = []

    for period in period_range:
        lags = lag_range if lag_range else np.arange(0, period, 1)
        for asym in asymmetries:
            for lag in lags:
                ref = generate_triangle_template_time(times, period, lag, asym)
                tau, pval = kendalltau(y, ref)
                test_results.append((pval, tau, period, lag, asym))
                if pval < best_p:
                    best_p = pval
                    best_tau = tau
                    best_per = period
                    best_lag = lag
                    best_asym = asym

    bonf_p = min(1.0, best_p * len(test_results))
    amp = (np.percentile(series.values, 90) - np.percentile(series.values, 10)) / 2

    return {
        'ADJ.P': round(bonf_p, 6),
        'PER': round(best_per, 2),
        'LAG': round(best_lag, 2),
        'ASYM': round(best_asym, 2),
        'AMP': round(amp, 4),
        'TAU': round(best_tau, 4),
        'Method': 'Python-JTK'
    }



# ------------------------
# Cosine-Kendall function
# ------------------------

def run_Cosine_Kendall(series, period_range=[20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28], interval=1):
    y = series.rank().values
    n = len(y)
    best_p = 1.0
    best_tau = 0.0
    best_per = None
    best_lag = None

    test_results = []
    t = np.arange(n) * interval # time vector in real units
    for period in period_range:
        for lag in np.arange(0, period, 0.5):
            radians = 2 * np.pi * (t - lag) / period
            ref = np.cos(radians)
            ref_ranked = pd.Series(ref).rank().values
            tau, pval = kendalltau(y, ref_ranked)
            test_results.append((pval, tau, period, lag))
            if pval < best_p:
                best_p = pval
                best_tau = tau
                best_per = period
                best_lag = lag

    bonf_p = min(1.0, best_p * len(test_results))
    amp = (np.percentile(series.values, 90) - np.percentile(series.values, 10)) / 2
        #### using corrected_lag
    corrected_lag = (best_lag - period / 2) % best_per

    return {
        'ADJ.P': round(bonf_p, 6),
        'PER': round(best_per, 2),
        'LAG': round(corrected_lag, 2),
        'AMP': round(amp, 4),
        'Method': 'Cosine-Kendall'
    }

# ------------------------
# Cosinor analysis function
# ------------------------

def fit_group_cosinor(df, period_list=[20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28]):

    results = []

    for test in df['test'].unique():
        subset = df[df['test'] == test]
        x = subset['x'].values
        y = subset['y'].values

        best_aic = np.inf
        best_result = None

        for per in period_list:
            omega = 2 * np.pi / per
            cos_term = np.cos(omega * x)
            sin_term = np.sin(omega * x)
            X = np.column_stack([np.ones(len(x)), cos_term, sin_term])
            model = sm.OLS(y, X).fit()

            if model.aic < best_aic:
                beta_cos, beta_sin = model.params[1], model.params[2]
                amp = np.sqrt(beta_cos ** 2 + beta_sin ** 2)
                phase = np.arctan2(-beta_sin, beta_cos)

                cov = model.cov_params()
                var_amp = (beta_cos**2 * cov[2, 2] +
                           beta_sin**2 * cov[1, 1] +
                           2 * beta_cos * beta_sin * cov[1, 2]) / amp**2
                se_amp = np.sqrt(var_amp)
                ci_amp = (amp - 1.96 * se_amp, amp + 1.96 * se_amp)

                var_phase = ((beta_sin**2 * cov[1, 1] +
                              beta_cos**2 * cov[2, 2] -
                              2 * beta_cos * beta_sin * cov[1, 2]) /
                             (beta_cos**2 + beta_sin**2)**2)
                se_phase = np.sqrt(var_phase)
                ci_phase = (phase - 1.96 * se_phase, phase + 1.96 * se_phase)

                best_aic = model.aic
                best_result = {
                    'test': test,
                    'period': per,
                    'p': model.f_pvalue,
                    'amplitude': amp,
                    'p(amplitude)': model.pvalues[1],
                    'CI(amplitude)': [ci_amp[0], ci_amp[1]],
                    'acrophase': phase,
                    'p(acrophase)': model.pvalues[2],
                    'CI(acrophase)': [ci_phase[0], ci_phase[1]],
                    'acrophase[h]': acrophase_to_hours(phase, per)
                }

        if best_result:
            results.append(best_result)

    df_results = pd.DataFrame(results)
    if not df_results.empty:
        df_results['q'] = multipletests(df_results['p'], method='fdr_bh')[1]
        df_results['q(amplitude)'] = multipletests(df_results['p(amplitude)'], method='fdr_bh')[1]
        df_results['q(acrophase)'] = multipletests(df_results['p(acrophase)'], method='fdr_bh')[1]

    return df_results
    
# ------------------------
# Harmonic Cosinor function (Kendall's tau test)
# ------------------------

def fit_group_harmonic_cosinor(df, period_range=[20,20.5,21,21.5,22,22.5,23,23.5,24,24.5,25,25.5,26,26.5,27,27.5,28],
                                harmonics=2):
    x = df['x'].values
    y = df['y'].values
    y_ranked = pd.Series(y).rank().values
    best_p = 1.0
    best_tau = 0.0
    best_per = None
    best_lag = None

    test_results = []

    for period in period_range:
        for lag in np.arange(0, period, 0.5):
            ref = np.zeros(len(x))
            for h in range(1, harmonics + 1):
                ref += np.cos(2 * np.pi * h * (x - lag) / period)
            ref_ranked = pd.Series(ref).rank().values

            # Ensure same size before correlation
            if len(y_ranked) != len(ref_ranked):
                continue

            tau, pval = kendalltau(y_ranked, ref_ranked)
            test_results.append((pval, tau, period, lag))

            if pval < best_p:
                best_p = pval
                best_tau = tau
                best_per = period
                best_lag = lag

    bonf_p = min(1.0, best_p * len(test_results))  # Bonferroni correction
    amp = (np.percentile(y, 75) - np.percentile(y, 25)) / 2
    acrophase_h = acrophase_to_hours(-2 * np.pi * best_lag / best_per, best_per)

    return pd.DataFrame([{
        'ADJ.P': round(bonf_p, 6),
        'PER': round(best_per, 2),
        'AMP': round(amp, 4),
        'acrophase[h]': round(acrophase_h, 2),
        'Method': 'Harmonic-Cosinor'
    }])


class SpanDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Shaded Span")
        layout = QVBoxLayout(self)

        self.start_label = QLabel("Start Time (hr):")
        self.start_box = QtWidgets.QSpinBox()
        self.start_box.setRange(0, 999)

        self.end_label = QLabel("End Time (hr):")
        self.end_box = QtWidgets.QSpinBox()
        self.end_box.setRange(0, 999)
        
        self.color_label = QLabel("Span Color:")
        self.color_button = QPushButton("Choose Color")
        self.color = "#888888"  # default gray
        self.color_button.clicked.connect(self.choose_color)


        layout.addWidget(self.start_label)
        layout.addWidget(self.start_box)
        layout.addWidget(self.end_label)
        layout.addWidget(self.end_box)
        layout.addWidget(self.color_label)
        layout.addWidget(self.color_button)

        btn = QPushButton("Add Span")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    def choose_color(self):
        color = QColorDialog.getColor()
        if color.isValid():
            self.color = color.name()
            self.color_button.setStyleSheet(f"background-color: {self.color};")

    def get_values(self):
        if not hasattr(self, "color"):
            self.color = "#888888"
        return self.start_box.value(), self.end_box.value(), self.color


class RenameLegendDialog(QDialog):
    def __init__(self, labels, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Rename Legend Labels")
        self.layout = QVBoxLayout(self)
        self.inputs = {}
        for label in labels:
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            edit = QtWidgets.QLineEdit()
            edit.setText(label)
            self.inputs[label] = edit
            row.addWidget(edit)
            self.layout.addLayout(row)
        btn = QPushButton("Apply")
        btn.clicked.connect(self.accept)
        self.layout.addWidget(btn)

    def get_renamed(self):
        return {old: box.text() for old, box in self.inputs.items()}

class JTKParamDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("JTK Parameter Setup")
        layout = QVBoxLayout(self)

        self.period_input = QLineEdit("22,23,24,25,26")
        self.lag_input = QLineEdit("0,2,4,6,8,10,12")
        self.asym_input = QLineEdit("0.5")  # Optional

        layout.addWidget(QLabel("<<estimate Periods (comma-separated)>>\n note: more periods you select, the slower efficiency you get!"))
        layout.addWidget(self.period_input)
        layout.addWidget(QLabel("<<estimate Lags (acrophase, or peak time) (comma-separated)>>\n note: more lags you select, the slower efficiency you get!"))
        layout.addWidget(self.lag_input)
        layout.addWidget(QLabel("<<Asymmetries (comma-separated, 0-1)>>\n 0.5 = symmetric shape\n < 0.5 = asym (left peak)\n > 0.5 = asym (right peak)"))
        layout.addWidget(self.asym_input)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        layout.addWidget(buttons)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

    def get_params(self):
        periods = [int(p.strip()) for p in self.period_input.text().split(",")]
        lags = [int(l.strip()) for l in self.lag_input.text().split(",")]
        asyms = [float(a.strip()) for a in self.asym_input.text().split(",")]
        return periods, lags, asyms


LEGEND_OPTIONS = {
    "loc": "upper right",
    "fontsize": 10,
    "framealpha": 0.5
}


class CircadianApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("easyClock")
        self.resize(1000, 900)

        self.legend_settings = {
            "loc": "upper right",
            "fontsize": 10,
            "framealpha": 0.5
        }
        self.group_display_names = {}
        self.raw_data = {k: None for k in ["file_1", "file_2", "file_3"]}
        self.group_assignments = {}
        self.group_means = {k: {} for k in self.raw_data}
        self.group_sems = {k: {} for k in self.raw_data}
        self.group_colors = {}
        self.shaded_spans = {}
        self.result_table = []
        self.y_axis_limits = {
            "file_1": None,
            "file_2": None,
            "file_3": None
        }
        self.x_labels = {k: "Time" for k in self.raw_data}
        self.y_labels = {
            "file_1": "Y Label",
            "file_2": "Y Label",
            "file_3": "Y Label"
        }
        self.titles = {k: k.capitalize() for k in self.raw_data}

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        self.status = QTextEdit("Status: waiting for input")
        self.status.setReadOnly(True)
        layout.addWidget(self.status)

        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas_widget = QWidget()
        canvas_layout = QVBoxLayout(self.canvas_widget)
        canvas_layout.addWidget(self.canvas)
        self.canvas_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setWidget(self.canvas_widget)
        layout.addWidget(self.scroll)

        layout.addWidget(self._add_button("Export Plot to PDF", self.export_plot))
        layout.addWidget(self._add_button("Export Result Table", self.export_results_table))

        menu = self.menuBar()
        file_menu = menu.addMenu("File")
        file_menu.addAction("Load Data", self.load_data)

        edit_menu = menu.addMenu("Edit")
        edit_menu.addAction("Axis Labels", self.set_axis_labels)
        yaxis_action = QtWidgets.QAction("Axis-Y Limits", self)
        yaxis_action.triggered.connect(self.edit_y_axis_limits)
        edit_menu.addAction(yaxis_action)
        edit_menu.addAction("Shaded Span Adding", self.collect_shaded_spans)
        edit_menu.addAction("Shaded Span Removing", self.remove_shaded_span)
        edit_menu.addAction("Legend Format", self.set_legend_style)
        edit_menu.addAction("Legend Labels", self.rename_legend_labels)
        
        analysis_menu=menu.addMenu("Analysis")
        analysis_menu.addAction("Cosine_Kendall and Cosinor",self.run_analysis)
        analysis_menu.addAction("Python-JTK (non-parametric test)",self.run_pythonJTK_analysis)
        analysis_menu.addAction("Harmonic Cosinor (bimodal test)",self.run_fit_group_harmonic_cosinor)
        
        visualize_menu=menu.addMenu("Visualization")
        actogram_action = QtWidgets.QAction("Plot Actogram", self)
        actogram_action.triggered.connect(self.plot_actogram)
        visualize_menu.addAction(actogram_action)
        
        about_menu = menu.addMenu("Read Me")
        about_menu.addAction("Developer", self.show_developer)
        about_menu.addAction("Note", self.show_Notes)
        about_menu.addAction("Acknowledgements and Feedback", self.show_Feedback)


    def _add_button(self, label, callback):
        btn = QPushButton(label)
        btn.clicked.connect(callback)
        return btn

    def show_developer(self):
        QMessageBox.about(
            self,
            "About easyClock",
            "🕓 easyClock v1.5\n\n"
            "Developed by: Binbin Wu Ph.D.\n"
            "Ja Lab, UF Scripps Institute, University of Florida\n"
            "© 2025. All rights reserved.\n\n"
            "Please cite this paper:\neasyClock: A User-Friendly Desktop Application for Circadian Rhythm Analysis and Visualization.\n\n")

    def show_Feedback(self):
        QMessageBox.about(
            self,
            "Acknowledgments",
            "We thank the following individuals for providing valuable feedback:\n\n"
            "Dr. Yutong Xiao (Max Planck, Florida)\n"
            "Alayna Garland (Kenan Fellow)\n"
            "Dr. Qiankun He (Zhengzhou University)\n\n"
            "- - - - Feedback - - - -\n"
            "Email me < binbinwu.phd@gmail.com >")
        
    def show_Notes(self):
        QMessageBox.about(
            self,
            "About Instructions",
            "This app can input 1 to 3 files in the same time, click cancel to skip file input.\n\n"
            "Data file format (.csv):\n"
            "1st column [time series (hr)], e.g., 0...48...with any time intervals;\n\n"
            "1st row [sample names] e.g., fly_1, fly_2 ... ;\n\n"
            "This app is desigined for analyzing circadian rhythms, so at least 48 hr data is required for analysis.\n\n")
            
            
    def rename_legend_labels(self):
        if not self.group_assignments:
            QMessageBox.information(self, "No groups", "No groups available to rename.")
            return

        current = list(self.group_assignments.keys())
        dialog = RenameLegendDialog(current, self)
        if dialog.exec_() == QDialog.Accepted:
            renamed = dialog.get_renamed()

            # Update group assignments and colors
            new_assignments = {}
            new_colors = {}
            for old, new in renamed.items():
                new_assignments[new] = self.group_assignments[old]
                new_colors[new] = self.group_colors.get(old, "#000000")

            self.group_assignments = new_assignments
            self.group_colors = new_colors

            # Recalculate means and SEMs with new group names
            self.group_means = {k: {} for k in self.raw_data}
            self.group_sems = {k: {} for k in self.raw_data}
            for dtype, df in self.raw_data.items():
                if df is not None:
                    for group, flies in self.group_assignments.items():
                        valid = [f for f in flies if f in df.columns]
                        if not valid:
                            continue
                        gdf = df[valid]
                        self.group_means[dtype][group] = gdf.mean(axis=1)
                        self.group_sems[dtype][group] = gdf.sem(axis=1)

            self.plot_all()

    def set_axis_labels(self):
        dtype, ok = QInputDialog.getItem(self, "File Type", "Which plot?", ["file_1", "file_2", "file_3"], 0, False)
        if not ok:
            return
        title, ok1 = QInputDialog.getText(self, "Plot Title", "Title:", text=self.titles.get(dtype, dtype.capitalize()))
        if ok1:
            self.titles[dtype] = title
        xlabel, ok2 = QInputDialog.getText(self, "X Label", "Label:", text=self.x_labels.get(dtype, "Time"))
        if ok2:
            self.x_labels[dtype] = xlabel
        ylabel, ok3 = QInputDialog.getText(self, "Y Label", "Label:", text=self.y_labels.get(dtype, "Value"))
        if ok3:
            self.y_labels[dtype] = ylabel
        self.plot_all()

    def set_legend_style(self):
        locs = ['upper right', 'upper left', 'lower right', 'lower left',
                'upper center', 'lower center', 'center right', 'center left', 'center']
        loc, ok1 = QInputDialog.getItem(self, "Legend Location", "Select location:", locs, editable=False)
        if not ok1:
            return
        fs, ok2 = QInputDialog.getInt(self, "Font Size", "Size:", value=self.legend_settings['fontsize'], min=0, max=20)
        if not ok2:
            return
        alpha, ok3 = QInputDialog.getDouble(self, "Frame Alpha", "Transparency (0-1):", value=self.legend_settings['framealpha'], min=0.0, max=1.0, decimals=1)
        if not ok3:
            return
        self.legend_settings = {"loc": loc, "fontsize": fs, "framealpha": alpha}
        self.plot_all()

    def edit_y_axis_limits(self):
        dtype, ok = QInputDialog.getItem(self, "Select Plot", "Choose plot:", ["file_1", "file_2", "file_3"], 0, False)
        if not ok:
            return

        current_limits = self.y_axis_limits.get(dtype, (None, None))
        ymin, ok1 = QInputDialog.getDouble(self, f"{dtype.capitalize()} Y Min", "Enter Y-axis min (leave 0 for auto):", 0, decimals=3)
        if not ok1:
            return
        ymax, ok2 = QInputDialog.getDouble(self, f"{dtype.capitalize()} Y Max", "Enter Y-axis max (leave 0 for auto):", 0, decimals=3)
        if not ok2:
            return

        # Auto range if either is zero
        if ymin == 0 and ymax == 0:
            self.y_axis_limits[dtype] = None
        else:
            self.y_axis_limits[dtype] = (ymin, ymax)

        self.plot_all()

    def collect_shaded_spans(self):
        dtype, ok = QInputDialog.getItem(
            self, "File Type", "Assign shaded span to which plot?",
            list(self.group_means.keys()), 0, False
        )
        if not ok or not dtype:
            return
            
        if dtype not in self.shaded_spans:
             self.shaded_spans[dtype] = []
        
        while True:
            dialog = SpanDialog(self)
            if dialog.exec_() != QDialog.Accepted:
                break
            start, end, color = dialog.get_values()
            if end <= start:
                QMessageBox.warning(self, "Invalid Span", "End must be greater than start.")
                continue
            self.shaded_spans[dtype].append((start, end, color))
            self.plot_all()


    def load_data(self):
        self.group_assignments.clear()
        self.group_means = {k: {} for k in self.raw_data}
        self.group_sems = {k: {} for k in self.raw_data}
        self.group_colors.clear()

        for dtype in self.raw_data:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(self, f"Load {dtype} CSV", "", "CSV Files (*.csv)")
            if path:
                try:
                    df = pd.read_csv(path, index_col="Time")
                    self.raw_data[dtype] = df
                    self.status.append(f"Loaded {dtype} from {os.path.basename(path)}")
                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load {dtype}: {e}")
                    self.raw_data[dtype] = None

        fly_ids = []
        for df in self.raw_data.values():
            if df is not None:
                fly_ids = list(df.columns)
                break
        if not fly_ids:
            QMessageBox.warning(self, "No data", "No valid columns found in any dataset.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Group Assignment")
        vbox = QVBoxLayout(dlg)
        label = QLabel("Grouping data")
        vbox.addWidget(label)

        group_map = {}
        dropdowns = {}
        groups = [f"Group_{chr(65+i)}" for i in range(24)]
        scroll = QScrollArea()
        inner = QWidget()
        form = QVBoxLayout(inner)
        for fly in fly_ids:
            row = QHBoxLayout()
            row.addWidget(QLabel(fly))
            drop = QComboBox()
            drop.addItems(groups)
            dropdowns[fly] = drop
            row.addWidget(drop)
            form.addLayout(row)
        scroll.setWidget(inner)
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(300)
        vbox.addWidget(scroll)

        btn = QPushButton("Confirm")
        btn.clicked.connect(dlg.accept)
        vbox.addWidget(btn)

        if dlg.exec_() == QDialog.Accepted:
            for fly, drop in dropdowns.items():
                grp = drop.currentText()
                group_map.setdefault(grp, []).append(fly)
            self.group_assignments = group_map

        color_choice = QMessageBox.question(
            self, "Color Assignment for Groups",
            "Color Assignment for Groups\n\nYes----Default colors\nNo-----Manual selection",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes # default selected
        )
        if color_choice == QMessageBox.Yes:
            default_palette = generate_distinct_colors(len(self.group_assignments))
            for i, group in enumerate(self.group_assignments):
                self.group_colors[group] = default_palette[i]
        else:
            for group in self.group_assignments:
                QMessageBox.information(self, "Color Group", f"Choose color for {group}")
                color = QColorDialog.getColor(title=f"Choose color for {group}")
                if color.isValid():
                    self.group_colors[group] = color.name()

        for dtype, df in self.raw_data.items():
            if df is not None:
                for group, flies in self.group_assignments.items():
                    valid = [f for f in flies if f in df.columns]
                    if valid:
                        sub = df[valid]
                        self.group_means[dtype][group] = sub.mean(axis=1)
                        self.group_sems[dtype][group] = sub.sem(axis=1)

        self.status.append("Groups and colors assigned.")
        self.plot_all()

    def plot_all(self):
        self.figure.clf()
        self.canvas.draw_idle() # Schedule redraw
        
        data_types = [k for k in self.group_means if self.group_means[k]]

        dpi = self.figure.get_dpi()
        height_per_plot = 1.5
        total_height = height_per_plot * len(data_types)
        height_in_pixels = int(dpi * total_height)
        self.canvas_widget.setMinimumHeight(height_in_pixels)
        
        for i, dtype in enumerate(data_types):
            ax = self.figure.add_subplot(len(data_types), 1, i+1)
            for group in self.group_means[dtype]:
                m = self.group_means[dtype][group]
                s = self.group_sems[dtype][group]
                c = self.group_colors.get(group, None)
                ax.errorbar(m.index, m, yerr=s, label=group, color=c)
            for span in self.shaded_spans.get(dtype, []):
                s, e, color = span
                ax.axvspan(s, e, color=color, alpha=0.3)

            ax.set_title(self.titles.get(dtype, dtype.capitalize()))
            ax.set_xlabel(self.x_labels.get(dtype, "Time"))
            ax.set_ylabel(self.y_labels.get(dtype, "Value"))
            ax.legend(loc=self.legend_settings.get("loc", "upper right"),
                      fontsize=self.legend_settings.get("fontsize", 10),
                      framealpha=self.legend_settings.get("framealpha", 0.5))

            # Apply Y-axis limits if set
            y_limits = self.y_axis_limits.get(dtype)
            if y_limits:
                ax.set_ylim(y_limits)

        self.figure.tight_layout()
        #refresh
        self.canvas.draw()



    def remove_shaded_span(self):
        dtype, ok = QInputDialog.getItem(
            self, "Select Plot", "Remove span from which Plot?",
            list(self.shaded_spans.keys()), 0, False
        )
        if not ok or dtype not in self.shaded_spans or not self.shaded_spans[dtype]:
            QMessageBox.information(self, "No Spans", "No shaded spans found for selected plot.")
            return

    # Format span labels for selection
        span_labels = [f"plot#{i+1}: Time({s}-{e}): Color({color})" for i, (s, e, color) in enumerate(self.shaded_spans[dtype])]

        item, ok = QInputDialog.getItem(
            self, "Select Span", "Which span to remove?", span_labels, 0, False
        )
        if not ok:
            return

        index = span_labels.index(item)
        del self.shaded_spans[dtype][index]

    # Redraw updated plot
        self.plot_all()
        QMessageBox.information(self, "Removed", "Shaded span removed successfully.")




    def plot_actogram(self):
        try:
            dtype, ok1 = QInputDialog.getItem(self, "Choose Dataset", "File type:", list(self.group_means.keys()), 0, False)
            if not ok1:
                return
            
            if not self.group_means[dtype]:
                QMessageBox.warning(self, "Error", f"No {dtype} data loaded.")
                return
            group, ok2 = QInputDialog.getItem(self, "Choose Group", "Select group:", list(self.group_means[dtype].keys()), 0, False)
            if not ok2:
                return
                
                
            series = self.group_means[dtype][group]
            values = series.values
            times = series.index.to_numpy()
            

                # Determine time resolution and reshape
            dt = times[1] - times[0]
            points_per_day = int(round(24 / dt))
            total_points = len(values)
            n_days = total_points // points_per_day
            fig, ax = plt.subplots(figsize=(10, 6))
                
            for day in range(n_days):
                start = day * points_per_day
                mid= start + points_per_day
                end = mid + points_per_day
                
                left = values[start:mid]  # left side plot ends with on a repeated last day.
                
                y_offset = day * (np.max(values) * 1.2)
                ax.bar(np.arange(0, 24, dt), left, width=dt*0.8, bottom=y_offset, color='black')
                
                if end > total_points:
                    break
                right = values[mid:end] # right side plot ends with on the last day.
                ax.bar(np.arange(24, 48, dt), right, width=dt*0.8, bottom=y_offset, color='black')
                
                
            ax.set_xlabel("Time (hr)")
            ax.set_ylabel("Day")
            ax.set_yticks([i * np.max(values) * 1.2 for i in range(n_days)])
            ax.set_yticklabels([f"{i+1}" for i in range(n_days)])
            ax.invert_yaxis()
            ax.set_xlim(0, 48)
            ax.set_xticks(np.arange(0, 49, 6))
            plt.tight_layout()

            # Ask for PDF export
            pdf_path, _ = QFileDialog.getSaveFileName(self, "Save PDF", f"{group}_{dtype}_actogram.pdf", "PDF Files (*.pdf)")
            if pdf_path:
                fig.savefig(pdf_path)
                QMessageBox.information(self, "Saved", f"Actogram saved to:\n{pdf_path}")
            plt.close(fig)
        except Exception as e:
            QMessageBox.critical(self, "Actogram Error", str(e))
                
                

    def run_analysis(self):
        try:
            dtype, ok = QInputDialog.getItem(
                self, "Select Dataset", "Which File?", ["file_1", "file_2", "file_3"], 0, False)
            if not ok:
                return

            if not self.group_means[dtype]:
                QMessageBox.warning(self, "Error", f"No {dtype} data loaded.")
                return

            df = self.group_means.get(dtype)
            if not df:
                QMessageBox.warning(self, "Missing Data", "Please load data first.")
                return


            time_index = df[next(iter(df))].index
            min_time = float(time_index.min())
            max_time = float(time_index.max())

            start, ok2 = QInputDialog.getDouble(
                self, "Start Time", f"Start time (from Time index {min_time}–{max_time}):", 
                min_time, min_time, max_time,decimals=1)
            if not ok2:
                return
            end, ok3 = QInputDialog.getDouble(
                self, "End Time", f"End time (from Time index {min_time}–{max_time}):",
                min_time, min_time, max_time,decimals=1)
            if not ok3 or end <= start:
                QMessageBox.warning(self, "Invalid Range", "End time must be greater than start time.")
                return

            self.result_table = []
            for group, series in df.items():
                # Use .loc to slice by actual time index values
                sliced = series.loc[start:end]
                time = sliced.index.to_numpy()
                
                if len(time) < 2:
                    continue
                
                interval = np.diff(time).mean()
                duration = interval * len(time)
                
                if duration < 48:
                    QMessageBox.warning(self, "Too Short", f"{group} has less than 48h of data.")
                    continue
                
                # --- Cosine-Kendall ---
                jtk_res = run_Cosine_Kendall(sliced, interval=interval)

                # --- Cosinor ---
                cos_df = pd.DataFrame({
                    'test': [group] * len(sliced),
                    'x': time,
                    'y': sliced.values
                })
                cos_res_df = fit_group_cosinor(cos_df)
                if not cos_res_df.empty:
                    row = cos_res_df.iloc[0]
                    cos_res = {
                        'ADJ.P': round(row['q'], 4),
                        'PER': round(row['period'],2),
                        'LAG': round(row['acrophase[h]'], 2),
                        'AMP': round(row['amplitude'], 4),
                        'Method': 'Cosinor analysis'
                    }
                    # Append both results
                    self.result_table.append({
                        'Group': group,
                        'Method': 'Cosine-Kendall',
                        **jtk_res
                    })
                    self.result_table.append({
                        'Group': group,
                        'Method': 'Cosinor analysis',
                        **cos_res
                    })

            if self.result_table:
                self.status.setText(pd.DataFrame(self.result_table).to_string(index=False, col_space=20))
            else:
                QMessageBox.information(self, "No Results", "No groups met the criteria.")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))
            
            
            
            
    def run_pythonJTK_analysis(self):
        try:
            dtype, ok = QInputDialog.getItem(
                self, "Select Dataset", "Which File?",
                ["file_1", "file_2", "file_3"], 0, False)
            if not ok:
                return

            df = self.group_means.get(dtype)
            if not df:
                QMessageBox.warning(self, "Missing Data", "Please load data first.")
                return

            time_index = df[next(iter(df))].index
            min_time = float(time_index.min())
            max_time = float(time_index.max())
            
            
            start, ok1 = QInputDialog.getDouble(
                self, "Start Time", f"Start time index ({min_time}–{max_time}):",
                min_time, min_time, max_time, decimals=1)
            if not ok1:
                return
            end, ok2 = QInputDialog.getDouble(
                self, "End Time", f"End time index({min_time}–{max_time}):",
                start + 1, start + 1, max_time, decimals=1)
            if not ok2 or end <= start:
                return

            # Ask for period and lag range
            dlg = JTKParamDialog(self)
            if dlg.exec_() != QDialog.Accepted:
                return
            period_range, lag_range, asymmetries = dlg.get_params()

            self.result_table = []
            for group, series in df.items():
                sliced = series.loc[(series.index >= start) & (series.index <= end)]
                jtk_res = run_discrete_jtk(
                    sliced, period_range=period_range, lag_range=lag_range, asymmetries=asymmetries)

                self.result_table.append({
                    'Group': group,
                    'Method': 'Python-JTK',
                    **jtk_res
                })

            df_out = pd.DataFrame(self.result_table)
            self.status.setText(df_out.to_string(index=False, col_space=20))

        except Exception as e:
            QMessageBox.critical(self, "Analysis Error", str(e))
            




    def run_fit_group_harmonic_cosinor(self):
        try:
            dtype, ok = QInputDialog.getItem(
                self, "Select Dataset", "Which File?", ["file_1", "file_2", "file_3"], 0, False)
            if not ok:
                return

            if not self.group_means[dtype]:
                QMessageBox.warning(self, "Error", f"No {dtype} data loaded.")
                return

            df = self.group_means.get(dtype)
            if not df:
                QMessageBox.warning(self, "Missing Data", "Please load data first.")
                return


            time_index = df[next(iter(df))].index
            min_time = float(time_index.min())
            max_time = float(time_index.max())

            start, ok2 = QInputDialog.getDouble(
                self, "Start Time", f"Start time (from Time index {min_time}–{max_time}):", 
                min_time, min_time, max_time,decimals=1)
            if not ok2:
                return
            end, ok3 = QInputDialog.getDouble(
                self, "End Time", f"End time (from Time index {min_time}–{max_time}):",
                min_time, min_time, max_time,decimals=1)
            if not ok3 or end <= start:
                QMessageBox.warning(self, "Invalid Range", "End time must be greater than start time.")
                return

            self.result_table = []
            for group, series in df.items():
                # Use .loc to slice by actual time index values
                sliced = series.loc[start:end]
                time = sliced.index.to_numpy()
                
                if len(time) < 2:
                    continue
                
                interval = np.diff(time).mean()
                duration = interval * len(time)
                
                if duration < 48:
                    QMessageBox.warning(self, "Too Short", f"{group} has less than 48h of data.")
                    continue


                harmonic_df = pd.DataFrame({
                    'test': [group] * len(sliced),
                    'x': time,
                    'y': sliced.values
                })

                harmonic_res_df = fit_group_harmonic_cosinor(harmonic_df)
                    # Append results
                if not harmonic_res_df.empty:
                    row = harmonic_res_df.iloc[0].to_dict()
                    self.result_table.append({
                        'Group': group,
                        'Method': 'Harmonic-Cosinor',
                        **row
                    })

            if self.result_table:
                self.status.setText(pd.DataFrame(self.result_table).to_string(index=False, col_space=30))
            else:
                QMessageBox.information(self, "No Results", "No groups met the criteria.")

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))



    def export_plot(self):
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save PDF", "", "PDF Files (*.pdf)")
        if fname:
            self.figure.savefig(fname)
            QMessageBox.information(self, "Export", f"Plot saved to {fname}")

    def export_results_table(self):
        if not self.result_table:
            QMessageBox.warning(self, "No Results", "Please run analysis first.")
            return
        fname, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Save CSV", "", "CSV Files (*.csv)")
        if fname:
            pd.DataFrame(self.result_table).to_csv(fname, index=False)
            QMessageBox.information(self, "Export", f"Results saved to {fname}")
            
            
if __name__ == '__main__':
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    app = QtWidgets.QApplication(sys.argv)
    win = CircadianApp()
    win.show()
    sys.exit(app.exec_())
