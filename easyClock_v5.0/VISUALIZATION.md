# easyClock visualization

Run `python3 easyClock_v5.0.py` with `easyclock_visualization.py` in the same directory. Existing scientific and Qt dependencies are unchanged.

Every preview and analysis plot now has a navigation toolbar. Use **Edit plot** to select a panel and change its title, axis labels and bounds, separate title, X/Y label and X/Y tick-number and legend-label font sizes (1–144 pt), grid, legend, series labels, colors, opacity and visibility. Heatmaps have selectable palettes. Use the standard toolbar's figure options for line styles and marker settings. Legends can be dragged within a plot.

Use **Export SVG / PDF / PNG** to save the current edited figure. SVG preserves text as text for editing in vector tools; PDF embeds TrueType fonts; PNG exports at 300 DPI. Actograms open in a plot window before export.

Edits belong to the current figure and remain while it is open. Regenerating a plot restores its plotting defaults; figure settings are not saved as a reusable project. Existing main-window Edit menu settings continue to control regenerated previews. Save your edited figure before regenerating it.

Presentation changes include consistent typography, restrained grid lines, coordinated fit/observation/SEM colors, taller preview panels, corrected JTK subplot construction, and y-axis scaling across all displayed groups and SEM. Analysis functions, fitting formulas, aggregation, statistical tests, and wavelet calculations are unchanged.

Validation with synthetic data:

