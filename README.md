# Changepoint Detection Toolbox

This repository now contains a Python-based graphical toolbox for detecting change
points in time-series data. The application combines several classical
hydro-meteorological tests with Bayesian and cumulative anomaly approaches.

## Features

- Load two-column Excel files (time, value) for analysis.
- Perform Mann-Kendall, Pettitt, sliding t-test, Cramer, Buishand, Bayesian and
  cumulative anomaly (CAP) detection in a single run.
- Adjust confidence level, sliding window parameters and Bayesian prior
  strength directly in the GUI with real-time updates.
- View detailed textual explanations and JSON-formatted statistics for every
  method.
- Visualise detected change points with static plots and an optional animated
  playback.
- Export publication-ready figures and CSV summaries of the results.

## Getting Started

1. Install the dependencies:

   ```bash
   pip install numpy pandas matplotlib seaborn scipy openpyxl
   ```

2. Launch the toolbox:

   ```bash
   python -m toolbox.gui
   ```

3. Load your Excel file (time in the first column, values in the second) and
   adjust the parameters as needed. The interface will update the detections and
   highlight the inferred change points automatically.

## License

No license has been specified for this project. 请按需要自行决定使用方式。
