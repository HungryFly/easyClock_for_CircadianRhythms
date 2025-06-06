import os
import sys
import shutil
import tempfile
import subprocess
import pandas as pd
import numpy as np
from scipy.stats import t
from scipy.stats import kendalltau
from scipy.optimize import curve_fit
from PyQt5 import QtWidgets, QtCore
from PyQt5.QtWidgets import (
    QMessageBox, QInputDialog, QDialog, QVBoxLayout, QHBoxLayout,QComboBox,QFileDialog,
    QLabel, QPushButton, QTextEdit, QScrollArea, QWidget, QSizePolicy, QColorDialog
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


def get_resource_path(filename):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, filename)
    return os.path.join(os.path.abspath("."), filename)

def acrophase_to_hours(rad_phase, period=24):
    hours = (rad_phase * period) / (2 * np.pi)
    return hours % period

# ------------------------
# Cosinor-Kendall function
# ------------------------

def run_Cosinor_Kendall(series, period_range=range(20, 28), interval=1):
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
    amp = (np.percentile(series.values, 75) - np.percentile(series.values, 25)) / 2
    return {
        'ADJ.P': round(bonf_p, 6),
        'PER': round(best_per, 2),
        'LAG': round(best_lag, 2),
        'AMP': round(amp, 4),
        'Method': 'Cosinor-Kendall'
    }

# ------------------------
# Cosinor analysis function
# ------------------------


def fit_group_cosinor(df, period=24):
    import statsmodels.api as sm
    from statsmodels.stats.multitest import multipletests

    results = []

    if isinstance(period, (int, float)):
        period = [period]

    for per in period:
        omega = 2 * np.pi / per
        for test in df['test'].unique():
            subset = df[df['test'] == test]
            x = subset['x'].values
            y = subset['y'].values

            cos_term = np.cos(omega * x)
            sin_term = np.sin(omega * x)
            X = np.column_stack([np.ones(len(x)), cos_term, sin_term])
            model = sm.OLS(y, X).fit()

            mesor = model.params[0]
            beta_cos = model.params[1]
            beta_sin = model.params[2]

            amp = np.sqrt(beta_cos**2 + beta_sin**2)
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

            result = {
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

            results.append(result)

    df_results = pd.DataFrame(results)
    df_results['q'] = multipletests(df_results['p'], method='fdr_bh')[1]
    df_results['q(amplitude)'] = multipletests(df_results['p(amplitude)'], method='fdr_bh')[1]
    df_results['q(acrophase)'] = multipletests(df_results['p(acrophase)'], method='fdr_bh')[1]

    return df_results



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

        layout.addWidget(self.start_label)
        layout.addWidget(self.start_box)
        layout.addWidget(self.end_label)
        layout.addWidget(self.end_box)

        btn = QPushButton("Add Span")
        btn.clicked.connect(self.accept)
        layout.addWidget(btn)

    def get_values(self):
        return self.start_box.value(), self.end_box.value()


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
        self.raw_data = {k: None for k in ["Behavior_1", "Behavior_2", "Behavior_3"]}
        self.group_assignments = {}
        self.group_means = {k: {} for k in self.raw_data}
        self.group_sems = {k: {} for k in self.raw_data}
        self.group_colors = {}
        self.shaded_spans = []
        self.result_table = []
        self.y_axis_limits = {
            "Behavior_1": None,
            "Behavior_2": None,
            "Behavior_3": None
        }
        self.x_labels = {k: "Time" for k in self.raw_data}
        self.y_labels = {
            "Behavior_1": "Y Label",
            "Behavior_2": "Y Label",
            "Behavior_3": "Y Label"
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
        edit_menu.addAction("Shaded Span setting", self.collect_shaded_spans)
        edit_menu.addAction("Legend Format", self.set_legend_style)
        edit_menu.addAction("Legend Labels", self.rename_legend_labels)
        
        analysis_menu=menu.addMenu("Analysis")
        analysis_menu.addAction("Cosinor_Kendall and Cosinor",self.run_analysis)
        
        visualize_menu=menu.addMenu("Visualize")
        actogram_action = QtWidgets.QAction("Plot Actogram", self)
        actogram_action.triggered.connect(self.plot_actogram)
        visualize_menu.addAction(actogram_action)
        
        about_menu = menu.addMenu("Read Me")
        about_menu.addAction("Acknowlegements", self.show_Acknowlegements)
        about_menu.addAction("Note", self.show_Notes)


    def _add_button(self, label, callback):
        btn = QPushButton(label)
        btn.clicked.connect(callback)
        return btn

    def show_Acknowlegements(self):
        QMessageBox.about(
            self,
            "About easyClock",
            "🕓 easyClock v1.0\n\n"
            "Developed by: Binbin Wu Ph.D.\n"
            "binbinwu.phd@gmail.com\n"
            "Ja Lab, Fl Scripps Institute, University of Florida"
            "© 2025. All rights reserved.\n\n"
            "Acknowlege the developer's contribution once you use it for publication, presentation or other public purpose.\n\n")

    def show_Notes(self):
        QMessageBox.about(
            self,
            "About Instructions",
            "This app can input up to 3 files in the same time, click cancel to skip 1 or 2 file input.\n\n"
            "Data file format (.csv):\n"
            "Fill the time series (hr)into the 1st column, e.g., 0...48...with any time interval;\n\n"
            "Fill the sample name series (e.g., fly_1, fly_2 ..) into the 1st row;\n\n"
            "This version is desigined for analyzing circadian rhythm, so at least 48 hr data is required for analysis.\n\n"
            "- - - - Feedback and Concerns - - - -\n"
            "Eamil me < binbinwu.phd@gmail.com > with your feedback and concerns.")
            
            
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
        dtype, ok = QInputDialog.getItem(self, "Behavior Type", "Which plot?", ["Behavior_1", "Behavior_2", "Behavior_3"], 0, False)
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
        dtype, ok = QInputDialog.getItem(self, "Select Plot", "Choose behavior:", ["Behavior_1", "Behavior_2", "Behavior_3"], 0, False)
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
        spans = []
        while True:
            dialog = SpanDialog(self)
            if dialog.exec_() != QDialog.Accepted:
                break
            start, end = dialog.get_values()
            if end <= start:
                QMessageBox.warning(self, "Invalid Span", "End must be greater than start.")
                continue
            spans.append((start, end))
        self.shaded_spans = spans
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
            QMessageBox.warning(self, "No Flies", "No valid fly columns found in any dataset.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Group Assignment")
        vbox = QVBoxLayout(dlg)
        label = QLabel("Assign flies to groups")
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
        self.figure.clf()  # fully clear all axes
        self.canvas.draw()
        self.canvas.flush_events()
        
        data_types = [k for k in self.group_means if self.group_means[k]]
        height_per_plot = 3
        h = height_per_plot * len(data_types)
        self.figure.set_size_inches(8, h)
        self.canvas_widget.setMinimumHeight(h * 100)
        for i, dtype in enumerate(data_types):
            ax = self.figure.add_subplot(len(data_types), 1, i+1)
            for group in self.group_means[dtype]:
                m = self.group_means[dtype][group]
                s = self.group_sems[dtype][group]
                c = self.group_colors.get(group, None)
                ax.errorbar(m.index, m, yerr=s, label=group, color=c)
            for s, e in self.shaded_spans:
                ax.axvspan(s, e, color='gray', alpha=0.3)
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
        self.canvas.draw()
        self.canvas.flush_events()  # force full redraw

    def plot_actogram(self):
        try:
            dtype, ok1 = QInputDialog.getItem(self, "Choose Dataset", "Behavior type:", list(self.group_means.keys()), 0, False)
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
                self, "Select Dataset", "Which behavior?", ["Behavior_1", "Behavior_2", "Behavior_3"], 0, False)
            if not ok:
                return

            df = self.group_means.get(dtype)
            if not df:
                QMessageBox.warning(self, "Missing Data", "Please load data first.")
                return

            # Use Time-based values from index for start and end
            any_series = next(iter(df.values()))
            all_times = any_series.index.to_list()
            min_time, max_time = min(all_times), max(all_times)


            start, ok1 = QInputDialog.getInt(
                self, "Start Time", f"Start time (from Time index {min_time}–{max_time}):", 
                min_time, min_time, max_time)
            if not ok1:
                return
            end, ok2 = QInputDialog.getInt(
                self, "End Time", f"End time (from Time index {min_time}–{max_time}):",
                min_time, min_time, max_time)
            if not ok2 or end <= start:
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
                
                # --- Cosinor-Kendall ---
                jtk_res = run_Cosinor_Kendall(sliced, interval=interval)

                # --- Cosinor ---
                cos_df = pd.DataFrame({
                    'test': [group] * len(sliced),
                    'x': time,
                    'y': sliced.values
                })
                cos_res_df = fit_group_cosinor(cos_df,period=24)
                if not cos_res_df.empty:
                    row = cos_res_df.iloc[0]
                    cos_res = {
                        'ADJ.P': round(row['q'], 4),
                        'PER': round(row['period'],2),
                        'LAG': round(row['acrophase[h]'], 2),
                        'AMP': round(row['amplitude'], 4),
                        'Method': 'Cosinor'
                    }
                    # Append both results
                    self.result_table.append({
                        'Group': group,
                        'Method': 'Cosinor-Kendall',
                        **jtk_res
                    })
                    self.result_table.append({
                        'Group': group,
                        'Method': 'Cosinor',
                        **cos_res
                    })

            if self.result_table:
                self.status.setText(pd.DataFrame(self.result_table).to_string(index=False))
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
