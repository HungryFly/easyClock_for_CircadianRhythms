"""Presentation-only tools for easyClock; never transforms analysis data."""
from pathlib import Path

import matplotlib as mpl
from matplotlib.colors import is_color_like
from matplotlib.collections import PathCollection, PolyCollection, LineCollection
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from PyQt5 import QtWidgets as W
from PyQt5.QtCore import Qt

PALETTE = ['#2878A0', '#D57836', '#3A9276', '#A56395', '#C3A238', '#6374B7']


def style_figure(figure):
    """Apply typography and spacing without changing data or axis bounds."""
    figure.set_facecolor('white')
    for ax in figure.axes:
        if not ax.axison:
            continue
        colorbar = hasattr(ax, '_colorbar')
        ax.set_facecolor('#FAFBFC')
        for name in ('top', 'right'):
            ax.spines[name].set_visible(False)
        for name in ('bottom', 'left'):
            ax.spines[name].set_color('#CDD5DF')
        ax.tick_params(labelsize=10, colors='#465364', length=3)
        ax.title.set(fontsize=14, fontweight='semibold', color='#203047')
        ax.xaxis.label.set(fontsize=11, color='#344256')
        ax.yaxis.label.set(fontsize=11, color='#344256')
        for text in ax.texts:
            text.set_fontsize(11)
        if not ax.images and not colorbar:
            ax.set_axisbelow(True)
            ax.grid(axis='y', color='#DFE5EC', linewidth=.6, alpha=.7)
        for line in ax.lines:
            line.set_linewidth(1.8)
        legend = ax.get_legend()
        if legend:
            for text in legend.get_texts():
                text.set_fontsize(10)
            legend.get_frame().set_edgecolor('none')
            legend.get_frame().set_facecolor('white')
            legend.get_frame().set_alpha(.9)
            legend.set_draggable(True)
    if any(ax.images for ax in figure.axes):
        figure.set_layout_engine('constrained')
    else:
        figure.set_layout_engine('tight', pad=1.6)


def export_figure(parent, figure):
    filename, selected = W.QFileDialog.getSaveFileName(
        parent, 'Export figure', 'easyClock.svg',
        'SVG vector (*.svg);;PDF vector (*.pdf);;PNG image (*.png)')
    if not filename:
        return
    extension = { 'SVG': '.svg', 'PDF': '.pdf', 'PNG': '.png' }[selected[:3]]
    if Path(filename).suffix.lower() not in ('.svg', '.pdf', '.png'):
        filename += extension
    try:
        # SVG text remains editable; PDF embeds TrueType fonts.
        with mpl.rc_context({'svg.fonttype': 'none', 'pdf.fonttype': 42}):
            figure.savefig(filename, dpi=300, facecolor=figure.get_facecolor())
    except Exception as exc:
        W.QMessageBox.critical(parent, 'Export failed', str(exc))


class PlotEditor(W.QDialog):
    """Edit existing artists in place, including scatter, SEM and heatmaps."""
    def __init__(self, figure, parent=None):
        super().__init__(parent)
        self.figure = figure
        self.setWindowTitle('Edit plot · presentation only')
        self.resize(560, 760)
        layout = W.QVBoxLayout(self)
        self.axes = [a for a in figure.axes if a.axison]
        self.selector = W.QComboBox()
        self.selector.addItems([a.get_title() or f'Panel {i + 1}' for i, a in enumerate(self.axes)])
        layout.addWidget(self.selector)
        form = W.QFormLayout()
        layout.addLayout(form)
        self.fields = {}
        for key, label in [('title', 'Title'), ('xlabel', 'X label'), ('ylabel', 'Y label'),
                           ('xmin', 'X start'), ('xmax', 'X end'), ('ymin', 'Y start'), ('ymax', 'Y end')]:
            self.fields[key] = W.QLineEdit()
            form.addRow(label, self.fields[key])
        self.font_sizes = {}
        for key, label in [('title', 'Title size'), ('xlabel', 'X label size'),
                           ('ylabel', 'Y label size'), ('xticks', 'X tick number size'),
                           ('yticks', 'Y tick number size'), ('legend', 'Legend label size')]:
            spin = W.QDoubleSpinBox()
            spin.setRange(1, 144)
            spin.setDecimals(1)
            spin.setSuffix(' pt')
            self.font_sizes[key] = spin
            form.addRow(label, spin)
        self.grid = W.QCheckBox('Show horizontal grid')
        form.addRow(self.grid)
        self.legend = W.QCheckBox('Show legend (drag to reposition)')
        form.addRow(self.legend)
        self.cmap = W.QComboBox()
        self.cmap.addItems(['viridis', 'cividis', 'magma', 'inferno', 'plasma', 'turbo', 'gray'])
        form.addRow('Heatmap palette', self.cmap)
        layout.addWidget(W.QLabel('Series and shading: color accepts a name or #RRGGBB.'))
        self.table = W.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(['Label', 'Color', 'Opacity', 'Visible'])
        self.table.horizontalHeader().setSectionResizeMode(W.QHeaderView.Stretch)
        layout.addWidget(self.table)
        buttons = W.QDialogButtonBox(W.QDialogButtonBox.Apply | W.QDialogButtonBox.Close)
        buttons.button(W.QDialogButtonBox.Apply).clicked.connect(self.apply)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.selector.currentIndexChanged.connect(self.load)
        if self.axes:
            self.load(0)

    def load(self, index):
        ax = self.axes[index]
        vals = dict(title=ax.get_title(), xlabel=ax.get_xlabel(), ylabel=ax.get_ylabel(),
                    xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1])
        for key, value in vals.items():
            self.fields[key].setText(str(value))
        def tick_size(axis):
            ticks = axis.get_major_ticks()
            return ticks[0].label1.get_fontsize() if ticks else axis.get_offset_text().get_fontsize()

        sizes = dict(title=ax.title.get_fontsize(), xlabel=ax.xaxis.label.get_fontsize(),
                     ylabel=ax.yaxis.label.get_fontsize(),
                     xticks=tick_size(ax.xaxis), yticks=tick_size(ax.yaxis))
        legend = ax.get_legend()
        sizes['legend'] = (legend.get_texts()[0].get_fontsize()
                           if legend and legend.get_texts() else 10)
        for key, value in sizes.items():
            self.font_sizes[key].setValue(value)
        self.grid.setChecked(any(line.get_visible() for line in ax.get_ygridlines()))
        self.legend.setChecked(ax.get_legend() is not None and ax.get_legend().get_visible())
        self.cmap.setEnabled(bool(ax.images))
        if ax.images:
            name = ax.images[0].get_cmap().name
            if self.cmap.findText(name) < 0:
                self.cmap.addItem(name)
            self.cmap.setCurrentText(name)
        self.artists = list(ax.lines) + [c for c in ax.collections if isinstance(c, (PathCollection, PolyCollection, LineCollection))] + list(ax.patches)
        self.table.setRowCount(len(self.artists))
        for row, artist in enumerate(self.artists):
            color = artist.get_color() if hasattr(artist, 'get_color') else artist.get_facecolor()
            try:
                color = mpl.colors.to_hex(color)
            except (ValueError, TypeError):
                color = mpl.colors.to_hex(color[0]) if len(color) else '#2878A0'
            for col, value in enumerate([artist.get_label() or '', color, str(artist.get_alpha() if artist.get_alpha() is not None else 1)]):
                self.table.setItem(row, col, W.QTableWidgetItem(value))
            item = W.QTableWidgetItem()
            item.setCheckState(Qt.Checked if artist.get_visible() else Qt.Unchecked)
            self.table.setItem(row, 3, item)

    def apply(self):
        if not self.axes:
            return
        import math
        try:
            limits = [float(self.fields[k].text()) for k in ('xmin', 'xmax', 'ymin', 'ymax')]
            if not all(math.isfinite(v) for v in limits) or limits[0] == limits[1] or limits[2] == limits[3]:
                raise ValueError('Axis bounds must be finite and different. Reversed axes are supported.')
            styles = []
            for row in range(len(self.artists)):
                label, color, alpha = [self.table.item(row, col).text() for col in range(3)]
                alpha = float(alpha)
                if not is_color_like(color) or not 0 <= alpha <= 1:
                    raise ValueError('Use a valid color and opacity between 0 and 1.')
                styles.append((label, color, alpha, self.table.item(row, 3).checkState() == Qt.Checked))
        except ValueError as exc:
            W.QMessageBox.warning(self, 'Invalid plot setting', str(exc))
            return
        ax = self.axes[self.selector.currentIndex()]
        ax.set(title=self.fields['title'].text(), xlabel=self.fields['xlabel'].text(), ylabel=self.fields['ylabel'].text(), xlim=limits[:2], ylim=limits[2:])
        ax.title.set_fontsize(self.font_sizes['title'].value())
        ax.xaxis.label.set_fontsize(self.font_sizes['xlabel'].value())
        ax.yaxis.label.set_fontsize(self.font_sizes['ylabel'].value())
        for direction, axis in [('x', ax.xaxis), ('y', ax.yaxis)]:
            size = self.font_sizes[direction + 'ticks'].value()
            ax.tick_params(axis=direction, which='both', labelsize=size)
            axis.get_offset_text().set_fontsize(size)
        ax.grid(self.grid.isChecked(), axis='y')
        for artist, (label, color, alpha, visible) in zip(self.artists, styles):
            artist.set_label(label)
            if hasattr(artist, 'set_color'):
                artist.set_color(color)
            else:
                artist.set_facecolor(color)
            artist.set_alpha(alpha)
            artist.set_visible(visible)
        for im in ax.images:
            im.set_cmap(self.cmap.currentText())
        if self.legend.isChecked():
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(handles, labels, fontsize=self.font_sizes['legend'].value(),
                          framealpha=.9).set_draggable(True)
        elif ax.get_legend():
            ax.get_legend().set_visible(False)
        self.figure.canvas.draw_idle()


class PlotToolbar(NavigationToolbar2QT):
    def __init__(self, canvas, parent):
        super().__init__(canvas, parent)
        self.addAction('Edit plot', self.edit_plot)
        self.addAction('Export SVG / PDF / PNG', lambda: export_figure(parent, canvas.figure))

    def edit_plot(self):
        PlotEditor(self.canvas.figure, self).exec_()


class PlotWindow(W.QMainWindow):
    def __init__(self, figure, title, parent):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.setWindowTitle(title)
        self.resize(1050, 780)
        canvas = FigureCanvasQTAgg(figure)
        self.setCentralWidget(canvas)
        self.addToolBar(PlotToolbar(canvas, self))
        self.statusBar().showMessage('Pan and zoom with the toolbar · Edit plot to customize · Export editable SVG or PDF')
        canvas.draw_idle()
