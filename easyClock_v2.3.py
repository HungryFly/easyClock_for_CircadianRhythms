# restore asym display in Python-JTK.
# fix a bug in manual grouping.


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
from matplotlib.patches import Rectangle
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
    peak_time = asymmetry * period
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
    acrophase = (
        (best_lag + best_asym * best_per+ best_per / 2) % best_per if best_tau < 0
        else (best_lag + best_asym * best_per) % best_per
    )


    return {
        'ADJ.P': round(bonf_p, 6),
        'PER': round(best_per, 2),
        'AMP': round(amp, 4),
        'Acrophase': round(acrophase, 2),
        'ASYM': round(best_asym, 2),
        'TAU':round(best_tau, 2),
        'LAG':round(best_lag, 2),
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
        #### using corrected lag when tau < 0 because of Kendall's tau method.
    corrected_lag = (best_lag - best_per / 2) % best_per if best_tau < 0 else best_lag

    
    return {
        'ADJ.P': round(bonf_p, 6),
        'PER': round(best_per, 2),
        'AMP': round(amp, 4),
        'LAG': round(best_lag, 2),
        'Acrophase': round(corrected_lag, 2),
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
                    'mesor': model.params[0],
                    'amplitude': amp,
                    'p(amplitude)': model.pvalues[1],
                    'CI(amplitude)': [ci_amp[0], ci_amp[1]],
                    'acrophase': phase,
                    'p(acrophase)': model.pvalues[2],
                    'CI(acrophase)': [ci_phase[0], ci_phase[1]],
                    'Acrophase': acrophase_to_hours(phase, per)
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
    amp_est = (np.percentile(y, 90) - np.percentile(y, 10)) / 2
    
    #Generate model for parameter estimation (1 cycle)
    t_grid = np.linspace(0, best_per,1000)
    model_wave = np.zeros_like(t_grid)
    for h in range(1, harmonics + 1):
        model_wave += np.cos(2 * np.pi * h * (t_grid - best_lag) / best_per)
    
    # For plotting (full time)
    t_grid_full = np.linspace(x.min(), x.max(), 1000)
    model_wave_full = np.zeros_like(t_grid_full)
    for h in range(1, harmonics + 1):
        model_wave_full += np.cos(2 * np.pi * h * (t_grid_full - best_lag) / best_per)
    
    
    #Find peaks (local maxima)
    from scipy.signal import find_peaks
    peaks, _ = find_peaks(model_wave, distance=200)  # ensures apart two peaks from 200/1000 period.
    
    # Fallback if not enough peaks detected
    if len(peaks) < 2:
        # Pick top 2 highest points (by value), sorted by time
        peak_indices = np.argsort(model_wave)[-2:]
    else:
        # Get the 2 highest peaks among detected peaks
        peak_indices = peaks[np.argsort(model_wave[peaks])[-2:]]

    # Sort indices by time
    peak_indices = sorted(peak_indices)
    
    # Calculate acrophase and amplitude per peak
    acrophases = [acrophase_to_hours(t_grid[i] / best_per * 2 * np.pi, best_per) for i in peak_indices]
    if best_tau < 0:
        acrophases = [(a + best_per / 2) % best_per for a in acrophases]

    # Normalize the model wave to [0, 1]
    norm_model_wave = (model_wave - np.min(model_wave)) / np.ptp(model_wave)

    # Get real amplitude estimates based on normalized peak height
    peak_amps = [round(norm_model_wave[i] * amp_est * 2, 4) for i in peak_indices]

    # # Pair acrophases with amps, sort by acrophase time
    acrophase_amp_pairs = sorted(zip(acrophases, peak_amps), key=lambda x: x[0])
    acrophases = [round(a, 2) for a, _ in acrophase_amp_pairs]
    peak_amps = [round(a, 4) for _, a in acrophase_amp_pairs]

    # return the model wave parameters for plotting the best fit
    fit_model = {
        't_grid': t_grid,
        'model_wave': amp_est * norm_model_wave,
        't_grid_full': t_grid_full,
        'model_wave_full': amp_est * (model_wave_full - np.min(model_wave)) / np.ptp(model_wave),
        'params': {
            'period': best_per,
            'lag': best_lag,
            'acrophases': acrophases,
            'amplitudes': peak_amps,
            'bonferroni_p': bonf_p
        }
    }


    return pd.DataFrame([{
        'ADJ.P': round(bonf_p, 6),
        'PER': round(best_per, 2),
        'AMP1': round(peak_amps[0], 4),
        'AMP2': round(peak_amps[1], 4),
        'Acrophase1': round(acrophases[0], 2),
        'Acrophase2': round(acrophases[1], 2),
        'Method': 'Harmonic-Cosinor'
    }]), fit_model


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
        self.lag_input = QLineEdit("0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23")
        self.asym_input = QLineEdit("0.5")  # Optional

        layout.addWidget(QLabel("<<estimate Periods (comma-separated)>>\n note: more periods you select, the slower efficiency you get!"))
        layout.addWidget(self.period_input)
        layout.addWidget(QLabel("<<estimate Lags (when the waveform starts rising) (comma-separated)>>\n note: please use default values if you are unsure!"))
        layout.addWidget(self.lag_input)
        layout.addWidget(QLabel("<<estimate Asymmetries (range: 0-1)>>\n = 0.5 → symmetric (equal rise and fall time)\n < 0.5 → left-skewed (rises quickly, falls slowly)\n > 0.5 → right-skewed (rises slowly, falls quickly)\nExample: 0.8,0.9"))
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
        analysis_menu.addAction("Cosine-Kendall and Cosinor",self.run_analysis)
        analysis_menu.addAction("Python-JTK (non-parametric test)",self.run_pythonJTK_analysis)
        analysis_menu.addAction("Harmonic Cosinor (bimodal test)",self.run_fit_group_harmonic_cosinor)
        
        visualize_menu=menu.addMenu("Visualization")
        
        actogram_action = QtWidgets.QAction("Plot Actogram", self)
        actogram_action.triggered.connect(self.plot_actogram)
        visualize_menu.addAction(actogram_action)
        
        cosinor_plot_action = QtWidgets.QAction("Plot Cosinor Fitting", self)
        cosinor_plot_action.triggered.connect(lambda:self.plot_cosinor_fitting_model())
        visualize_menu.addAction(cosinor_plot_action)
        
        cosinor_kendall_plot_action = QtWidgets.QAction("Plot Cosinor-Kendall Fitting", self)
        cosinor_kendall_plot_action.triggered.connect(lambda:self.plot_cosinor_kendall_fitting_model())
        visualize_menu.addAction(cosinor_kendall_plot_action)

        python_jtk_plot_action = QtWidgets.QAction("Plot Python-JTK Fitting", self)
        python_jtk_plot_action.triggered.connect(lambda:self.plot_python_jtk_fitting_model())
        visualize_menu.addAction(python_jtk_plot_action)
        
        harmonic_cosinor_plot_action = QtWidgets.QAction("Plot Harmonic-Cosinor Fitting", self)
        harmonic_cosinor_plot_action.triggered.connect(lambda:self.plot_harmonic_cosinor_fitting_model())
        visualize_menu.addAction(harmonic_cosinor_plot_action)


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
            "🕓 easyClock v2.3\n\n"
            "Developed by: Binbin Wu Ph.D.\n"
            "Ja Lab, UF Scripps Institute, University of Florida\n"
            "© 2025. All rights reserved.\n\n"
            "Please cite:\neasyClock: A User-Friendly Desktop Application for Circadian Rhythm Analysis and Visualization.\n\n")

    def show_Notes(self):
        QMessageBox.about(
            self,
            "About Instructions",
            "This app can input up to 3 files in the same time, click cancel to skip 1 or 2 file input.\n\n"
            "This app is desigined for analyzing circadian rhythms, so at least 48 hr data is required for analysis.\n\n"
            "Read Figure 2 of the following paper for understading the correct data format:\n\n"
            "easyClock: A User-Friendly Desktop Application for Circadian Rhythm Analysis and Visualization.")

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
        import csv

        self.group_assignments.clear()
        self.group_means = {k: {} for k in self.raw_data}
        self.group_sems = {k: {} for k in self.raw_data}
        self.group_colors.clear()

        fly_ids = []  # original labels (may have duplicates)
        unique_ids = []       # unique column ids
        self.column_label_map = {}  # maps unique_id → original label


        for dtype in self.raw_data:
            path, _ = QtWidgets.QFileDialog.getOpenFileName(
                self, f"Load {dtype} CSV", "", "CSV Files (*.csv)"
            )
            if path:
                try:
                    # Read raw header with utf-8-sig to support BOM and preserve duplicate labels
                    with open(path, "r", encoding="utf-8-sig", newline="") as f:
                        reader = csv.reader(f)
                        header = next(reader)

                    if "Time" not in header:
                        raise ValueError("'Time' column missing in header.")

                    time_index = header.index("Time")
                    raw_cols = header[:time_index] + header[time_index + 1:]

                    # Clean headers: preserve duplicates(original labels), label unnamed ones
                    cleaned_labels = []
                    unnamed_counter = 1
                    for label in raw_cols:
                        clean = str(label).strip()
                        if clean == "" or clean.lower().startswith("unnamed"):
                            clean = f"Unnamed_{unnamed_counter}"
                            unnamed_counter += 1
                        cleaned_labels.append(clean)
                        
                    # --- Ensure unique IDs for DataFrame columns ---
                    seen = {}
                    unique_labels = []
                    for label in cleaned_labels:
                        if label not in seen:
                            seen[label] = 1
                            unique_labels.append(label)
                        else:
                            seen[label] += 1
                            unique_labels.append(f"{label}_{seen[label]}")

                    # Load data with proper encoding and label assignment
                    df = pd.read_csv(path, index_col="Time", encoding="utf-8-sig")
                    df.columns = unique_labels
                    self.raw_data[dtype] = df
                    
                    # Save mapping (unique → original)
                    self.column_label_map.update(dict(zip(unique_labels, cleaned_labels)))
                    # Save for group dialog (keep both lists aligned by index)
                    fly_ids = cleaned_labels
                    unique_ids = unique_labels
                    
                    self.status.append(f"Loaded {dtype} from {os.path.basename(path)}")

                except Exception as e:
                    QMessageBox.warning(self, "Error", f"Failed to load {dtype}: {e}")
                    self.raw_data[dtype] = None

        if not fly_ids:
            QMessageBox.warning(self, "No data", "No valid columns found in any dataset.")
            return

        # ----- Group Assignment Dialog -----
        dlg = QDialog(self)
        dlg.setWindowTitle("Group Assignment")
        vbox = QVBoxLayout(dlg)
        vbox.addWidget(QLabel("Assign groups to each column:"))

        # Pre-fill logic: same original label → same group
        label_to_group = {}
        dropdown_defaults = []
        group_counter = 1

        for label in fly_ids:
            if label not in label_to_group:
                label_to_group[label] = f"Group_{group_counter}"
                group_counter += 1
            dropdown_defaults.append(label_to_group[label])

        # Make group choices = number of columns
        max_groups = len(fly_ids)
        group_choices = [f"Group_{i+1}" for i in range(max_groups)]

        # Create dropdowns
        dropdowns = {}
        scroll = QScrollArea()
        inner = QWidget()
        form = QVBoxLayout(inner)

        for i, label in enumerate(fly_ids):
            row = QHBoxLayout()
            row.addWidget(QLabel(label))
            drop = QComboBox()
            drop.addItems(group_choices)
            drop.setCurrentText(dropdown_defaults[i])
            dropdowns[i] = drop
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
            new_assignments = {}
            for i, drop in dropdowns.items():
                unique_label = unique_ids[i]
                group = drop.currentText()
                new_assignments.setdefault(group, []).append(unique_label)
            self.group_assignments = new_assignments

        # Color Assignment
        color_choice = QMessageBox.question(
            self, "Color Assignment for Groups",
            "Color Assignment for Groups\n\nYes----Default colors\nNo-----Manual selection",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.Yes
        )

        if color_choice == QMessageBox.Yes:
            palette = generate_distinct_colors(len(self.group_assignments))
            for i, group in enumerate(self.group_assignments):
                self.group_colors[group] = palette[i]
        else:
            for group in self.group_assignments:
                QMessageBox.information(self, "Color Group", f"Choose color for {group}")
                color = QColorDialog.getColor(title=f"Choose color for {group}")
                if color.isValid():
                    self.group_colors[group] = color.name()

        # Compute group means/SEMs
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
            self, "Select Plot", "Remove span from which plot?",
            list(self.shaded_spans.keys()), 0, False
        )
        if not ok or dtype not in self.shaded_spans or not self.shaded_spans[dtype]:
            QMessageBox.information(self, "No Spans", "No shaded spans found for selected plot.")
            return

    # Format span labels for selection
        span_labels = [f"#{i+1}: Time({s}-{e}): Color({color})" for i, (s, e, color) in enumerate(self.shaded_spans[dtype])]

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
            # Select dataset type
            dtype, ok1 = QInputDialog.getItem(self, "Choose Dataset", "File type:", list(self.group_means.keys()), 0, False)
            if not ok1:
                return
            
            if not self.group_means[dtype]:
                QMessageBox.warning(self, "Error", f"No {dtype} data loaded.")
                return
            
            # Select group
            group, ok2 = QInputDialog.getItem(self, "Choose Group", "Select group:", list(self.group_means[dtype].keys()), 0, False)
            if not ok2:
                return
            
            # Get data
            series = self.group_means[dtype][group]
            values_all = series.values
            times_all = series.index.to_numpy(dtype=float)

            # Determine overall time range
            min_time, max_time = float(times_all[0]), float(times_all[-1])
            
            
            # Input shaded span(s)
            span_text, ok3 = QInputDialog.getText(self, "Shaded Spans", f"Enter time ranges (from Time index {min_time:.1f}–{max_time:.1f}) for multiple shaded spans (e.g., 12-24,36-96):\nLeave it blank for no shaded spans:")
            if not ok3:
                return
            
            span_list = []
            if span_text.strip():
                try:
                    for part in span_text.split(','):
                        start_str, end_str = part.strip().split('-')
                        start, end = float(start_str), float(end_str)
                        if start >= end:
                            raise ValueError("Start must be less than end.")
                        span_list.append((start, end))
                except Exception:
                    QMessageBox.warning(self, "Input Error", "Please check your input special symbol like - or ,")
                    return
            


            # Let user choose time window
            start_time, ok4 = QInputDialog.getDouble(
                self, "Start Time for Actogram", f"Start time (from Time index {min_time:.1f}–{max_time:.1f})\nPlease start from ZT0/CT0:", 
                min_time, min_time, max_time, decimals=1)
            if not ok4:
                return

            end_time, ok5 = QInputDialog.getDouble(
                self, "End Time for Actogram", f"End time (from Time index {min_time:.1f}–{max_time:.1f}):",
                max_time, min_time, max_time, decimals=1)
            if not ok5 or end_time <= start_time:
                QMessageBox.warning(self, "Invalid Range", "End time must be greater than start time.")
                return

            # Apply filter to time range
            mask = (times_all >= start_time) & (times_all <= end_time)
            times = times_all[mask]
            values = values_all[mask]

            # Check enough data
            if len(values) < 2:
                QMessageBox.warning(self, "Too Short", "Not enough data in selected time range.")
                return

            # Estimate dt using median of diffs to handle irregular intervals
            time_diffs = np.diff(times)
            dt = np.median(time_diffs)
            if dt <= 0:
                QMessageBox.warning(self, "Invalid Time", "Time steps are invalid.")
                return

            points_per_day = int(round(24 / dt))
            total_points = len(values)
            n_days = total_points // points_per_day

            if n_days < 1:
                QMessageBox.warning(self, "Too Short", "Not enough data to plot even one full actogram day.")
                return

            # Start plotting
            fig, ax = plt.subplots(figsize=(10, 6))
            daily_max = np.max(values) * 1.2

            for s, e in span_list:
                for day in range(n_days):
                    y_offset = day * daily_max
                    start = day * points_per_day
                    mid = start + points_per_day
                    end = mid + points_per_day
                    if end > len(values):
                        continue

                    left_hours = times[start:mid]
                    right_hours = times[mid:end]
                    x_left = np.arange(0, 24, dt)
                    x_right = np.arange(24, 48, dt)

                    for i, real_hour in enumerate(left_hours):
                        if i >= len(x_left):
                            break
                        if s <= real_hour < e:
                            rect = Rectangle((x_left[i], y_offset), dt, daily_max, facecolor='gray', alpha=0.3, edgecolor=None)
                            ax.add_patch(rect)

                    for i, real_hour in enumerate(right_hours):
                        if i >= len(x_right):
                            break
                        if s <= real_hour < e:
                            rect = Rectangle((x_right[i], y_offset), dt, daily_max, facecolor='gray', alpha=0.3, edgecolor=None)
                            ax.add_patch(rect)

            # Draw actogram bars
            for day in range(n_days):
                start = day * points_per_day
                mid = start + points_per_day
                end = mid + points_per_day
                y_offset = day * daily_max
                if end > len(values):
                    break

                left = values[start:mid]
                x_left = np.arange(0, 24, dt)
                if len(left) > len(x_left):
                    left = left[:len(x_left)]
                ax.bar(x_left, left, width=dt * 0.8, bottom=y_offset, color='black')

                right = values[mid:end]
                x_right = np.arange(24, 48, dt)
                if len(right) > len(x_right):
                    right = right[:len(x_right)]
                ax.bar(x_right, right, width=dt * 0.8, bottom=y_offset, color='black')

            ax.set_xlabel("Time (hr)",fontsize=14)
            ax.set_ylabel("Day",fontsize=14)
            ax.set_yticks([i * daily_max for i in range(n_days)])
            ax.set_yticklabels([f"{i+1}" for i in range(n_days)])
            ax.invert_yaxis()
            ax.set_xlim(0, 48)
            ax.set_xticks(np.arange(0, 49, 6))
            plt.tight_layout()

            # Ask for PDF save
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

            self.last_analyzed_dtype = dtype
            self.last_analyzed_timerange = (start, end)
            self.result_table = []
            if not hasattr(self, 'best_cosinor_fits_by_dtype'):
                self.best_cosinor_fits_by_dtype = {}
            self.best_cosinor_fits_by_dtype[dtype] = {}  # Store best fitting cosinor template


             # Store best fitting cosinor_kendall template
            if not hasattr(self, 'kendall_fits_by_dtype'):
                self.kendall_fits_by_dtype = {}
            self.kendall_fits_by_dtype[dtype] = {}


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
                output_fields = {
                    'ADJ.P': 'ADJ.P',
                    'PER': 'PER',
                    'AMP': 'AMP',
                    'Acrophase': 'Acrophase'
                }
                
                
                jtk_res = run_Cosine_Kendall(sliced, interval=interval)

                selected = {
                    'Group': group,
                    'Method': 'Cosine-Kendall',
                    **{output_fields[k]: jtk_res[k] for k in output_fields if k in jtk_res}
                }


                # Save Kendall results
                self.kendall_fits_by_dtype[dtype][group] = {
                    'PER': jtk_res['PER'],
                    'LAG': jtk_res['Acrophase'],
                    'AMP': jtk_res['AMP']
                }


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
                        'AMP': round(row['amplitude'], 4),
                        'Acrophase': round(row['Acrophase'], 2),
                        'Method': 'Cosinor analysis'
                    }

                    # Save for plotting of regression model
                    self.best_cosinor_fits_by_dtype[dtype][group] = {
                        'period': row['period'],
                        'acrophase': row['Acrophase'],
                        'amplitude': row['amplitude'],
                        'mesor': row['mesor']
                    }

                    # Append both results
                    self.result_table.append({
                        'Group': group,
                        'Method': 'Cosine-Kendall',
                        **selected
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

            self.last_analyzed_dtype = dtype
            self.last_analyzed_timerange = (start, end)
            self.result_table = []

             # Store best fitting python_jtk template
            if not hasattr(self, 'python_jtk_fits_by_dtype'):
                self.python_jtk_fits_by_dtype = {}
            self.python_jtk_fits_by_dtype[dtype] = {}


            output_fields = {
                'ADJ.P': 'ADJ.P',
                'PER': 'PER',
                'AMP': 'AMP',
                'Acrophase': 'Acrophase',
                'ASYM': 'ASYM'
            }

            for group, series in df.items():
                sliced = series.loc[(series.index >= start) & (series.index <= end)]
                jtk_res = run_discrete_jtk(
                    sliced, period_range=period_range, lag_range=lag_range, asymmetries=asymmetries)

                if jtk_res:
                    selected = {
                        'Group': group,
                        'Method': 'Python-JTK',
                        **{output_fields[k]: jtk_res[k] for k in output_fields if k in jtk_res}
                    }

                    self.result_table.append(selected)

                # save fit curve for plotting
                try:
                    triangle_rank = generate_triangle_template_time(
                        sliced.index.to_numpy(), jtk_res['PER'], jtk_res['LAG'], jtk_res['ASYM'])
                        
                    # Flip if tau < 0
                    tau_sign = np.sign(jtk_res.get('TAU', 1.0))  # default to 1 if not available
                    triangle_rank *= tau_sign
                    
                    # Normalize triangle wave to [-1, 1] and rescale
                    triangle_norm = (triangle_rank - np.mean(triangle_rank)) / np.ptp(triangle_rank)
                    mesor = sliced.mean()
                    amp = jtk_res['AMP']
                    fit_curve = mesor + amp * triangle_norm
                    
                    self.python_jtk_fits_by_dtype[dtype][group] = {
                        'Time': sliced.index.to_numpy(),
                        'Fit': fit_curve,
                        'Raw': sliced.values,
                    }

                except Exception as e:
                    print(f"Fit generation failed for group {group}: {e}")


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
            if not hasattr(self, 'harmonic_fits_by_dtype'):
                self.harmonic_fits_by_dtype = {}
            self.harmonic_fits_by_dtype[dtype] = {} # for plotting best fit
            
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

                harmonic_res_df, fit_model = fit_group_harmonic_cosinor(harmonic_df)
                # Append results
                if not harmonic_res_df.empty:
                    row = harmonic_res_df.iloc[0].to_dict()
                    self.result_table.append({
                        'Group': group,
                        'Method': 'Harmonic-Cosinor',
                        **row
                    })
                    self.harmonic_fits_by_dtype[dtype][group] = fit_model  #save best fit for plotting
                    fit_model["time_range"] = (start, end)


            if self.result_table:
                self.status.setText(pd.DataFrame(self.result_table).to_string(index=False, col_space=20))
            else:
                QMessageBox.information(self, "No Results", "No groups met the criteria.")
                
            # Save last-used dataset and time window for plotting
            self.last_analyzed_dtype = dtype
            self.last_analyzed_timerange = (start, end)

        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))




 # plot_cosinor_fitting_model under visualization functionality.
    def plot_cosinor_fitting_model(self, dtype=None, groups_to_plot=None, time_range=None):
        if dtype is None:
            dtype = getattr(self, "last_analyzed_dtype", None)
        if time_range is None:
            time_range = getattr(self, "last_analyzed_timerange", None)

        if dtype not in self.group_means:
            QMessageBox.warning(self, "Missing Data", f"Please use Cosinor analysis first.")
            return

        if not hasattr(self, 'best_cosinor_fits_by_dtype') or dtype not in self.best_cosinor_fits_by_dtype:
            QMessageBox.warning(self, "Missing Fits", "Run analysis for this file first to calculate fits.")
            return

        fits = self.best_cosinor_fits_by_dtype[dtype]
        data = self.group_means[dtype]

        # === Let user select group(s) to plot ===
        if not groups_to_plot:
            group_list = list(fits.keys())
            selected, ok = QInputDialog.getItem(
                self, "Select Group", "Which group to plot?", group_list, 0, False)
            if not ok:
                return
            groups_to_plot = [selected]

        fig, ax = plt.subplots(figsize=(10, 6))
        for group in groups_to_plot:
            if group not in data or group not in fits:
                continue

            series = data[group]
            if time_range:
                series = series.loc[(series.index >= time_range[0]) & (series.index <= time_range[1])]
            times = series.index.to_numpy()
            values = series.values

            fit = fits[group]
            period = fit['period']
            acrophase = fit['acrophase']
            amp = fit['amplitude']
            mesor = fit['mesor']

            time_eval = np.linspace(times.min(), times.max(), 300)
            phase_rad = 2 * np.pi * (acrophase / period)
            fit_curve = mesor + amp * np.cos((2 * np.pi * time_eval / period) + phase_rad)

            #color = self.group_colors.get(group, None)
            ax.plot(time_eval, fit_curve, label="Cosinor Fit", color='black')
            ax.scatter(times, values, label="Mean of Data", color='gray', marker='o', alpha=0.5)

        ax.set_xlabel("Time (hr)",fontsize=14)
        ax.set_ylabel("Value",fontsize=14)
        ax.set_title(f"{dtype}:{group}",fontsize=14)

        # set the legend always upper right.
        all_y = np.concatenate([values, fit_curve])
        ymin, ymax = np.min(all_y), np.max(all_y)
        yrange = ymax - ymin
        padding = 0.20 * yrange  # 20% headroom
        ax.set_ylim(ymin, ymax + padding)
        ax.legend(loc="upper right",fontsize=14)

        ax.grid(False)
        plt.tight_layout()
        plt.show()



 # plot_cosinor_kendall_fitting_model within visualization functionality.
    def plot_cosinor_kendall_fitting_model(self, dtype=None, groups_to_plot=None, time_range=None):
        if dtype is None:
            dtype = getattr(self, "last_analyzed_dtype", None)
        if time_range is None:
            time_range = getattr(self, "last_analyzed_timerange", None)

        if dtype not in self.group_means:
            QMessageBox.warning(self, "Missing Data", f"Please use Cosinor-Kendall analysis first.")
            return

        if not hasattr(self, 'kendall_fits_by_dtype') or dtype not in self.kendall_fits_by_dtype:
            QMessageBox.warning(self, "Missing Fits", "Run analysis first for this file to calculate Kendall fits.")
            return

        fits = self.kendall_fits_by_dtype[dtype]
        data = self.group_means[dtype]

        # === Let user select group(s) to plot ===
        if not groups_to_plot:
            group_list = list(fits.keys())
            selected, ok = QInputDialog.getItem(
                self, "Select Group", "Which group to plot?", group_list, 0, False)
            if not ok:
                return
            groups_to_plot = [selected]


        fig, ax = plt.subplots(figsize=(10, 6))
        for group in groups_to_plot:
            if group not in data or group not in fits:
                continue

            series = data[group]
            if time_range:
                series = series.loc[(series.index >= time_range[0]) & (series.index <= time_range[1])]
            times = series.index.to_numpy()
            values = series.values

            fit = fits[group]
            period = fit['PER']
            lag = fit['LAG']
            amp = fit['AMP']
            mesor = np.mean(values)

            # Cosine model using lag in hours
            time_eval = np.linspace(times.min(), times.max(), 300)
            phase_rad = 2 * np.pi * (lag / period)
            radians = (2 * np.pi * time_eval / period) + phase_rad
            fit_curve = mesor + amp * np.cos(radians)

            ax.plot(time_eval, fit_curve, label="Cosine-Kendall Fit", color='black')
            ax.scatter(times, values, label="Mean of Data", color='gray', marker='o', alpha=0.5)

        ax.set_xlabel("Time (hr)",fontsize=14)
        ax.set_ylabel("Value",fontsize=14)
        ax.set_title(f"{dtype}:{group}",fontsize=14)

        # set the legend always upper right.
        all_y = np.concatenate([values, fit_curve])
        ymin, ymax = np.min(all_y), np.max(all_y)
        yrange = ymax - ymin
        padding = 0.20 * yrange  # 20% headroom
        ax.set_ylim(ymin, ymax + padding)
        ax.legend(loc="upper right",fontsize=14)

        ax.grid(False)
        plt.tight_layout()
        plt.show()




 # plot_python_jtk_fitting_model within visualization functionality.
    def plot_python_jtk_fitting_model(self, dtype=None, groups_to_plot=None, time_range=None):
        if dtype is None:
            dtype = getattr(self, "last_analyzed_dtype", None)
        if time_range is None:
            time_range = getattr(self, "last_analyzed_timerange", None)

        if not hasattr(self, 'python_jtk_fits_by_dtype') or dtype not in self.python_jtk_fits_by_dtype:
            QMessageBox.warning(self, "Missing Fits", "Run Python-JTK analysis first.")
            return

        fits = self.python_jtk_fits_by_dtype[dtype]

        # === Let user select group(s) to plot ===
        if not groups_to_plot:
            group_list = list(fits.keys())
            selected, ok = QInputDialog.getItem(
                self, "Select Group", "Which group to plot?", group_list, 0, False)
            if not ok:
                return
            groups_to_plot = [selected]

        # === Plot ===

        #plt.figure(figsize=(8, 4 * len(groups_to_plot)))
        fig, ax = plt.subplots(figsize=(10, 6))
        for i, group in enumerate(groups_to_plot):
            if group not in fits:
                continue

            fit_data = fits[group]
            time = fit_data.get("Time")
            fit_curve = fit_data.get("Fit")
            raw_data = fit_data.get("Raw")

            if time is None or fit_curve is None or raw_data is None:
                continue

            plt.subplot(len(groups_to_plot), 1, i + 1)
            plt.plot(time, fit_curve, '-', label="Python-JTK Fit", color="black")
            plt.plot(time, raw_data, 'o', label="Mean of Data", alpha=0.5, color='gray')
            
            plt.title(f"{dtype}:{group}",fontsize=14)
            plt.xlabel("Time (hr)",fontsize=14)
            plt.ylabel("Value",fontsize=14)
            
                    # set the legend always upper right.
            all_y = np.concatenate([raw_data, fit_curve])
            ymin, ymax = np.min(all_y), np.max(all_y)
            yrange = ymax - ymin
            padding = 0.20 * yrange  # 20% headroom
            plt.ylim(ymin, ymax + padding)
            plt.legend(loc="upper right",fontsize=14)
            
            plt.grid(False)

        plt.tight_layout()
        plt.show()



# plot_harmonic_cosinor_fitting_model within visualization functionality.
    def plot_harmonic_cosinor_fitting_model(self, dtype=None, groups_to_plot=None, time_range=None):

        if dtype is None:
            dtype = getattr(self, "last_analyzed_dtype", None)
        if time_range is None:
            time_range = getattr(self, "last_analyzed_timerange", None)

        if not hasattr(self, 'harmonic_fits_by_dtype') or dtype not in self.harmonic_fits_by_dtype:
            QMessageBox.warning(self, "Missing Fits", "Run Harmonic-Cosinor analysis first.")
            return

        fits = self.harmonic_fits_by_dtype[dtype]
        data = self.group_means[dtype]
        

        if not groups_to_plot:
            group_list = list(fits.keys())
            selected, ok = QInputDialog.getItem(
                self, "Select Group", "Which group to plot?", group_list, 0, False)
            if not ok:
                return
            groups_to_plot = [selected]

        fig, ax = plt.subplots(figsize=(10, 6))

        for group in groups_to_plot:
            if group not in data or group not in fits:
                continue

            series = data[group]
            if time_range:
                series = series.loc[(series.index >= time_range[0]) & (series.index <= time_range[1])]
            times = series.index.to_numpy()
            values = series.values
            mesor = np.mean(values)

            fit = fits[group]  # should contain t_grid and model_wave
            t_grid_full = fit.get('t_grid_full')
            model_wave_full = mesor + fit.get('model_wave_full')
            if t_grid_full is None or model_wave_full is None:
                continue


            ax.plot(t_grid_full, model_wave_full, label="Harmonic-Cosinor Fit", color='black', lw=2)
            ax.scatter(times, values, label="Mean of Data", color='gray', alpha=0.5)

        ax.set_xlabel("Time (hr)",fontsize=14)
        ax.set_ylabel("Value",fontsize=14)
        ax.set_title(f"{dtype}:{group}",fontsize=14)
        
        # set the legend always upper right.
        all_y = np.concatenate([values, model_wave_full])
        ymin, ymax = np.min(all_y), np.max(all_y)
        yrange = ymax - ymin
        padding = 0.20 * yrange  # 20% headroom
        ax.set_ylim(ymin, ymax + padding)
        ax.legend(loc="upper right",fontsize=14)
        
        ax.grid(False)
        plt.tight_layout()
        plt.show()
        


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
