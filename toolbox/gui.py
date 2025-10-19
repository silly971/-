"""Tkinter-based GUI for the changepoint detection toolbox."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.animation as animation
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .algorithms import DETECTORS, DetectionResult

sns.set_theme(style="whitegrid")


class ChangepointToolbox(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Changepoint Detection Toolbox")
        self.geometry("1200x800")

        self.data: Optional[pd.DataFrame] = None
        self.results: List[DetectionResult] = []
        self.animation: Optional[animation.FuncAnimation] = None
        self.animation_running = False

        self._build_ui()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        notebook = ttk.Notebook(self)
        notebook.grid(row=0, column=0, sticky="nsew")

        self.data_frame = ttk.Frame(notebook)
        self.results_frame = ttk.Frame(notebook)
        notebook.add(self.data_frame, text="Data & Parameters")
        notebook.add(self.results_frame, text="Results")

        self._build_data_tab()
        self._build_results_tab()

    def _build_data_tab(self) -> None:
        frame = self.data_frame
        frame.columnconfigure(1, weight=1)

        load_btn = ttk.Button(frame, text="Load Excel", command=self._load_excel)
        load_btn.grid(row=0, column=0, padx=10, pady=10, sticky="w")

        ttk.Label(frame, text="Alpha (confidence level):").grid(
            row=1, column=0, padx=10, pady=5, sticky="w"
        )
        self.alpha_var = tk.DoubleVar(value=0.05)
        self.alpha_scale = ttk.Scale(
            frame,
            from_=0.001,
            to=0.2,
            orient="horizontal",
            variable=self.alpha_var,
            command=lambda *_: self._update_alpha_label(),
        )
        self.alpha_scale.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        self.alpha_label = ttk.Label(frame, text="0.05")
        self.alpha_label.grid(row=1, column=2, padx=10, pady=5)

        ttk.Label(frame, text="Sliding window size:").grid(
            row=2, column=0, padx=10, pady=5, sticky="w"
        )
        self.window_var = tk.IntVar(value=10)
        ttk.Spinbox(frame, from_=2, to=200, textvariable=self.window_var).grid(
            row=2, column=1, padx=10, pady=5, sticky="w"
        )

        ttk.Label(frame, text="Sliding step size:").grid(
            row=3, column=0, padx=10, pady=5, sticky="w"
        )
        self.step_var = tk.IntVar(value=1)
        ttk.Spinbox(frame, from_=1, to=50, textvariable=self.step_var).grid(
            row=3, column=1, padx=10, pady=5, sticky="w"
        )

        ttk.Label(frame, text="Bayesian prior strength:").grid(
            row=4, column=0, padx=10, pady=5, sticky="w"
        )
        self.prior_var = tk.DoubleVar(value=0.1)
        ttk.Spinbox(frame, from_=0.01, to=10.0, increment=0.01, textvariable=self.prior_var).grid(
            row=4, column=1, padx=10, pady=5, sticky="w"
        )

        button_frame = ttk.Frame(frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=15, sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)
        button_frame.columnconfigure(2, weight=1)

        ttk.Button(button_frame, text="Run Analysis", command=self._run_analysis).grid(
            row=0, column=0, padx=10, sticky="ew"
        )
        ttk.Button(button_frame, text="Animate", command=self._toggle_animation).grid(
            row=0, column=1, padx=10, sticky="ew"
        )
        ttk.Button(button_frame, text="Export Results", command=self._export_results).grid(
            row=0, column=2, padx=10, sticky="ew"
        )

        self.canvas_frame = ttk.Frame(frame)
        self.canvas_frame.grid(row=6, column=0, columnspan=3, sticky="nsew")
        frame.rowconfigure(6, weight=1)

        self.figure = Figure(figsize=(8, 4), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title("Data preview")
        self.ax.set_xlabel("Index")
        self.ax.set_ylabel("Value")

        self.canvas = FigureCanvasTkAgg(self.figure, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar = NavigationToolbar2Tk(self.canvas, self.canvas_frame)
        toolbar.update()
        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._load_default_data()

    def _build_results_tab(self) -> None:
        frame = self.results_frame
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(1, weight=1)

        ttk.Label(frame, text="Detection summaries:").grid(
            row=0, column=0, sticky="w", padx=10, pady=5
        )

        self.summary_text = tk.Text(frame, wrap="word", height=12)
        self.summary_text.grid(row=1, column=0, sticky="nsew", padx=10, pady=5)

        ttk.Label(frame, text="Detailed JSON results:").grid(
            row=2, column=0, sticky="w", padx=10, pady=5
        )
        self.details_text = tk.Text(frame, wrap="word", height=12)
        self.details_text.grid(row=3, column=0, sticky="nsew", padx=10, pady=5)

    # ------------------------------------------------------------------ Data loading
    def _load_default_data(self) -> None:
        rng = np.random.default_rng(42)
        time = pd.date_range("2000-01-01", periods=100, freq="M")
        data = np.concatenate([
            rng.normal(0, 1, size=50),
            rng.normal(2, 1, size=50),
        ])
        self.data = pd.DataFrame({"Time": time, "Value": data}).set_index("Time")
        self._update_plot()

    def _load_excel(self) -> None:
        path = filedialog.askopenfilename(
            title="Select Excel file",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            df = pd.read_excel(path)
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Failed to read Excel file: {exc}")
            return

        if df.shape[1] < 2:
            messagebox.showerror(
                "Invalid file", "Excel file must have at least two columns (time, value)."
            )
            return
        df.columns = ["Time", "Value", *df.columns[2:]]
        df = df[["Time", "Value"]]
        df = df.dropna()
        try:
            df["Time"] = pd.to_datetime(df["Time"])
            df = df.set_index("Time")
        except Exception:  # noqa: BLE001
            df = df.set_index(df["Time"])
            df = df.drop(columns=["Time"])
        self.data = df
        self._update_plot()
        self._run_analysis()

    # ------------------------------------------------------------------ Analysis
    def _get_series(self) -> pd.Series:
        if self.data is None:
            raise RuntimeError("No data loaded")
        return self.data.iloc[:, 0]

    def _update_plot(self, highlight: Optional[List[int]] = None) -> None:
        series = self._get_series()
        self.ax.clear()
        self.ax.plot(series.index, series.values, label="Value", color="#1f77b4")
        self.ax.scatter(series.index, series.values, color="#1f77b4", s=15)
        self.ax.set_title("Data preview with detected change points")
        self.ax.set_xlabel("Time")
        self.ax.set_ylabel("Value")

        if highlight:
            for idx in highlight:
                if 0 <= idx < len(series):
                    self.ax.axvline(series.index[idx], color="crimson", linestyle="--")

        self.ax.legend(loc="best")
        self.canvas.draw_idle()

    def _run_analysis(self) -> None:
        if self.data is None:
            messagebox.showwarning("No data", "Please load a dataset first.")
            return

        alpha = float(self.alpha_var.get())
        window = int(self.window_var.get())
        step = int(self.step_var.get())
        prior = float(self.prior_var.get())

        series = self._get_series()
        self.results = []
        change_points: List[int] = []

        for name, func in DETECTORS.items():
            try:
                if name == "Sliding t-test":
                    result = func(series, window=window, step=step, alpha=alpha)
                elif name == "Mann-Kendall":
                    result = func(series, alpha=alpha)
                elif name == "Pettitt":
                    result = func(series, alpha=alpha)
                elif name == "Cramer":
                    result = func(series, alpha=alpha)
                elif name == "Buishand":
                    result = func(series, alpha=alpha)
                elif name == "Bayesian":
                    result = func(series, prior_strength=prior)
                else:
                    result = func(series)
                self.results.append(result)
                change_points.extend(result.change_points)
            except Exception as exc:  # noqa: BLE001
                self.results.append(
                    DetectionResult(
                        name=name,
                        statistic=float("nan"),
                        pvalue=None,
                        change_points=[],
                        details={},
                        summary=f"{name} failed: {exc}",
                    )
                )

        unique_points = sorted(set(change_points))
        self._update_plot(unique_points)
        self._update_results_text()

    def _update_alpha_label(self) -> None:
        value = float(self.alpha_var.get())
        self.alpha_label.configure(text=f"{value:.3f}")
        if self.data is not None:
            self.after(200, self._run_analysis)

    def _update_results_text(self) -> None:
        self.summary_text.delete("1.0", tk.END)
        self.details_text.delete("1.0", tk.END)

        summary_lines = []
        details: Dict[str, Dict[str, float]] = {}
        for result in self.results:
            summary_lines.append(f"{result.name}: {result.summary}")
            details[result.name] = {
                **result.details,
                "statistic": result.statistic,
            }
            if result.pvalue is not None:
                details[result.name]["pvalue"] = result.pvalue
            if result.change_points:
                cp_labels = [self._index_label(idx) for idx in result.change_points]
                summary_lines.append(f"  Change points: {', '.join(cp_labels)}")

        self.summary_text.insert("1.0", "\n".join(summary_lines))
        self.details_text.insert("1.0", json.dumps(details, indent=2))

    def _index_label(self, idx: int) -> str:
        series = self._get_series()
        if 0 <= idx < len(series.index):
            return f"{idx} ({series.index[idx]})"
        return str(idx)

    # ------------------------------------------------------------------ Animation
    def _toggle_animation(self) -> None:
        if self.animation_running:
            self.animation.event_source.stop()
            self.animation_running = False
            return

        if self.data is None:
            messagebox.showwarning("No data", "Load data before animating")
            return

        series = self._get_series()
        fig = Figure(figsize=(8, 4), dpi=100)
        ax = fig.add_subplot(111)
        line, = ax.plot([], [], lw=2)
        ax.set_xlim(series.index.min(), series.index.max())
        ax.set_ylim(series.min(), series.max())
        ax.set_title("Animated detection process")
        ax.set_xlabel("Time")
        ax.set_ylabel("Value")

        highlight_lines = []
        for cp in sorted({cp for r in self.results for cp in r.change_points}):
            if 0 <= cp < len(series.index):
                hl = ax.axvline(series.index[cp], color="crimson", linestyle="--", alpha=0.0)
                highlight_lines.append((cp, hl))

        x = series.index
        y = series.values

        def init():
            line.set_data([], [])
            for _, hl in highlight_lines:
                hl.set_alpha(0.0)
            return (line, *[hl for _, hl in highlight_lines])

        def animate(i):
            upto = min(i + 1, len(x))
            line.set_data(x[:upto], y[:upto])
            for cp, hl in highlight_lines:
                if cp < upto:
                    hl.set_alpha(0.8)
            return (line, *[hl for _, hl in highlight_lines])

        self.animation = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(x),
            interval=100,
            blit=True,
            repeat=False,
        )

        anim_window = tk.Toplevel(self)
        anim_window.title("Animation")
        canvas = FigureCanvasTkAgg(fig, master=anim_window)
        canvas.draw()
        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(canvas, anim_window)
        self.animation_running = True

        def on_close() -> None:
            if self.animation is not None:
                self.animation.event_source.stop()
            anim_window.destroy()
            self.animation_running = False

        anim_window.protocol("WM_DELETE_WINDOW", on_close)

    # ------------------------------------------------------------------ Exporting
    def _export_results(self) -> None:
        if not self.results:
            messagebox.showwarning("No results", "Run the analysis before exporting.")
            return
        directory = filedialog.askdirectory(title="Select export directory")
        if not directory:
            return

        export_path = Path(directory)
        series = self._get_series()

        fig_path = export_path / "changepoint_plot.png"
        self.figure.savefig(fig_path, dpi=300, bbox_inches="tight")

        table_rows = []
        for result in self.results:
            row = {
                "method": result.name,
                "statistic": result.statistic,
                "pvalue": result.pvalue,
                "change_points": ", ".join(self._index_label(idx) for idx in result.change_points),
                "summary": result.summary,
            }
            row.update(result.details)
            table_rows.append(row)

        df = pd.DataFrame(table_rows)
        csv_path = export_path / "changepoint_results.csv"
        df.to_csv(csv_path, index=False)

        messagebox.showinfo(
            "Export complete",
            f"Saved figure to {fig_path}\nSaved table to {csv_path}",
        )


def main() -> None:
    app = ChangepointToolbox()
    app.mainloop()


if __name__ == "__main__":
    main()
