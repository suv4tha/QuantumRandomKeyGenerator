# qrng_full_gui.py
"""
Quantum Random Key Generator - Full GUI with live animation and RNG tests.

Save as qrng_full_gui.py and run:
    python qrng_full_gui.py

Dependencies:
    pip install numpy matplotlib
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
import numpy as np
from pathlib import Path
import math

# Matplotlib embed
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# ---------- Utility functions for RNG tests ----------
def bits_to_hexkey(bits):
    bits = np.asarray(bits, dtype=np.uint8)
    n = bits.size
    pad = (-n) % 8
    if pad:
        bits_padded = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
    else:
        bits_padded = bits
    bytes_arr = np.packbits(bits_padded)
    hex_key = ''.join(f'{b:02x}' for b in bytes_arr)
    return hex_key

def shannon_entropy(bits):
    # bits: array of 0/1
    p0 = np.mean(bits == 0)
    p1 = 1.0 - p0
    ent = 0.0
    for p in (p0, p1):
        if p > 0:
            ent -= p * math.log2(p)
    return ent  # bits per symbol (max 1.0 for fair coin)

def chi_square_test(bits):
    # simple chi-square for counts vs expected 50/50
    n = len(bits)
    counts = np.bincount(bits, minlength=2)
    expected = n / 2.0
    chi2 = np.sum((counts - expected)**2 / expected)
    # degrees of freedom = 1, p-value approximation using survival function of chi2
    # we can approximate p-value using math.exp for df=1: p = exp(-chi2/2) * ... but simpler:
    # use the relation for df=1: p = erfc(sqrt(chi2/2))
    try:
        from math import erfc, sqrt
        pval = erfc(math.sqrt(chi2/2.0))
    except Exception:
        pval = None
    return chi2, pval, counts

def run_length_distribution(bits, max_len=20):
    # returns dict: length -> count (for runs of both 0 and 1)
    runs = []
    current = bits[0]
    length = 1
    for b in bits[1:]:
        if b == current:
            length += 1
        else:
            runs.append(length)
            current = b
            length = 1
    runs.append(length)
    # tally up to max_len (longer grouped into >max_len)
    dist = {i: 0 for i in range(1, max_len+1)}
    dist['>%d' % max_len] = 0
    for r in runs:
        if r <= max_len:
            dist[r] += 1
        else:
            dist['>%d' % max_len] += 1
    return dist

def autocorrelation(bits, max_lag=100):
    # compute autocorrelation for lags 1..max_lag (bits converted to +/-1)
    x = np.array(bits, dtype=float)
    if x.size == 0:
        return np.zeros(max_lag)
    x = 2 * x - 1.0  # map {0,1} -> {-1,+1}
    n = len(x)
    mean = np.mean(x)
    denom = np.sum((x - mean)**2)
    if denom == 0:
        return np.zeros(max_lag)
    ac = []
    for lag in range(1, max_lag+1):
        num = np.sum((x[:n-lag] - mean) * (x[lag:] - mean))
        ac.append(num / denom)
    return np.array(ac)

# ---------- GUI application ----------
class QRNGFullApp:
    def __init__(self, root):
        self.root = root
        root.title("Quantum RNG — Full Simulator & Analyzer")
        root.geometry("1100x750")

        self._make_controls()
        self._make_plots()
        self._make_report_area()

        self.bits = None
        self.hex_key = None
        self._stop_requested = False

    def _make_controls(self):
        frm = ttk.Frame(self.root, padding=8)
        frm.pack(side="top", fill="x")

        ttk.Label(frm, text="Number of bits:").grid(row=0, column=0, sticky="w")
        self.n_bits_var = tk.IntVar(value=4096)
        ttk.Entry(frm, textvariable=self.n_bits_var, width=10).grid(row=0, column=1, padx=(2,12))

        ttk.Label(frm, text="Seed (empty=random):").grid(row=0, column=2, sticky="w")
        self.seed_var = tk.StringVar(value="12345")
        ttk.Entry(frm, textvariable=self.seed_var, width=12).grid(row=0, column=3, padx=(2,12))

        ttk.Label(frm, text="Bias (p1 - 0.5):").grid(row=0, column=4, sticky="w")
        self.bias_var = tk.DoubleVar(value=0.0)
        ttk.Spinbox(frm, from_=-0.2, to=0.2, increment=0.01, textvariable=self.bias_var, width=6).grid(row=0, column=5, padx=(2,12))

        ttk.Label(frm, text="Chunk size (live update):").grid(row=1, column=0, sticky="w", pady=(6,0))
        self.chunk_var = tk.IntVar(value=128)
        ttk.Entry(frm, textvariable=self.chunk_var, width=8).grid(row=1, column=1, padx=(2,12), pady=(6,0))

        ttk.Label(frm, text="Delay ms per chunk:").grid(row=1, column=2, sticky="w", pady=(6,0))
        self.delay_var = tk.IntVar(value=40)
        ttk.Entry(frm, textvariable=self.delay_var, width=8).grid(row=1, column=3, padx=(2,12), pady=(6,0))

        self.generate_btn = ttk.Button(frm, text="Generate & Analyze", command=self.on_generate)
        self.generate_btn.grid(row=0, column=6, rowspan=2, padx=8)

        self.stop_btn = ttk.Button(frm, text="Stop", command=self.on_stop, state="disabled")
        self.stop_btn.grid(row=0, column=7, rowspan=2, padx=8)

        self.save_key_btn = ttk.Button(frm, text="Save Key", command=self.on_save, state="disabled")
        self.save_key_btn.grid(row=0, column=8, padx=6)

        self.save_plots_btn = ttk.Button(frm, text="Save Plots", command=self.on_save_plots, state="disabled")
        self.save_plots_btn.grid(row=0, column=9, padx=6)

        ttk.Label(frm, text="Status:").grid(row=2, column=0, sticky="w", pady=(6,0))
        self.status_lbl = ttk.Label(frm, text="Idle")
        self.status_lbl.grid(row=2, column=1, columnspan=6, sticky="w", pady=(6,0))

    def _make_plots(self):
        plot_frame = ttk.Frame(self.root)
        plot_frame.pack(side="top", fill="both", expand=True, padx=8, pady=6)

        # create matplotlib figure with four subplots (2x2)
        self.fig = Figure(figsize=(10,6))
        self.ax_running = self.fig.add_subplot(2,2,1)
        self.ax_counts = self.fig.add_subplot(2,2,2)
        self.ax_runlen = self.fig.add_subplot(2,2,3)
        self.ax_autocorr = self.fig.add_subplot(2,2,4)

        self.fig.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def _make_report_area(self):
        report_frame = ttk.Frame(self.root, padding=8)
        report_frame.pack(side="bottom", fill="x")
        ttk.Label(report_frame, text="Analysis report & metrics:").pack(anchor="w")
        self.report_text = tk.Text(report_frame, height=9)
        self.report_text.pack(fill="x")

    def on_generate(self):
        try:
            n = int(self.n_bits_var.get())
            if n <= 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Input error", "Number of bits must be a positive integer.")
            return

        seed_str = self.seed_var.get().strip()
        seed = None
        if seed_str != "":
            try:
                seed = int(seed_str)
            except:
                seed = abs(hash(seed_str)) % (2**32)

        chunk = max(1, int(self.chunk_var.get()))
        delay_ms = max(0, int(self.delay_var.get()))
        bias = float(self.bias_var.get())

        self.generate_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.save_key_btn.config(state="disabled")
        self.save_plots_btn.config(state="disabled")
        self.status_lbl.config(text="Starting generation...")
        self.report_text.delete("1.0", tk.END)
        self._stop_requested = False

        thread = threading.Thread(target=self._generate_worker, args=(n, seed, bias, chunk, delay_ms), daemon=True)
        thread.start()

    def on_stop(self):
        self._stop_requested = True
        self.status_lbl.config(text="Stop requested — finishing current chunk...")

    def _generate_worker(self, n_bits, seed, bias, chunk_size, delay_ms):
        rng = np.random.default_rng(seed)
        p1 = 0.5 + float(bias)
        p1 = min(max(p1, 0.0), 1.0)
        generated = np.zeros(0, dtype=int)

        # prepare empty plots
        self.root.after(0, lambda: self._clear_plots_initial())

        counts = np.array([0,0], dtype=int)
        running_ones = []
        # generate in chunks and update live
        produced = 0
        while produced < n_bits and not self._stop_requested:
            to_gen = min(chunk_size, n_bits - produced)
            # generate to_gen bits with p= p1
            chunk = (rng.random(to_gen) < p1).astype(int)
            generated = np.concatenate([generated, chunk])
            produced += to_gen

            counts = np.bincount(generated, minlength=2)
            running = np.cumsum(generated) / (np.arange(1, len(generated)+1))

            # schedule UI update
            self.root.after(0, lambda g=generated.copy(), r=running.copy(), c=counts.copy(): self._update_live_plots(g, r, c))

            # small delay so user sees animation
            time.sleep(delay_ms / 1000.0)

        if self._stop_requested:
            self.status_lbl.config(text="Stopped by user.")
        # final analysis
        if generated.size > 0:
            hex_key = bits_to_hexkey(generated)
            ent = shannon_entropy(generated)
            chi2, pval, cnts = chi_square_test(generated)
            run_dist = run_length_distribution(generated, max_len=30)
            ac = autocorrelation(generated, max_lag=200)

            # update final plots & report on main thread
            self.root.after(0, lambda: self._finalize_and_report(generated, hex_key, ent, chi2, pval, cnts, run_dist, ac))

        else:
            self.root.after(0, lambda: self._no_data_finish())

    def _clear_plots_initial(self):
        self.ax_running.clear()
        self.ax_running.set_title("Running proportion of 1s")
        self.ax_running.set_xlabel("n")
        self.ax_running.set_ylabel("Proportion of 1s")
        self.ax_running.grid(True)

        self.ax_counts.clear()
        self.ax_counts.set_title("Counts (0 vs 1)")
        self.ax_counts.set_xlabel("Bit value")
        self.ax_counts.set_ylabel("Count")
        self.ax_counts.grid(True)

        self.ax_runlen.clear()
        self.ax_runlen.set_title("Run-length distribution (will display after finish)")
        self.ax_runlen.set_xlabel("Run length")
        self.ax_runlen.set_ylabel("Count")
        self.ax_runlen.grid(True)

        self.ax_autocorr.clear()
        self.ax_autocorr.set_title("Autocorrelation (will display after finish)")
        self.ax_autocorr.set_xlabel("Lag")
        self.ax_autocorr.set_ylabel("Autocorr")
        self.ax_autocorr.grid(True)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw_idle()

    def _update_live_plots(self, bits, running, counts):
        # running: array of running proportion
        self.ax_running.clear()
        self.ax_running.plot(running)
        self.ax_running.set_title("Running proportion of 1s (live)")
        self.ax_running.set_xlabel("n")
        self.ax_running.set_ylabel("Proportion of 1s")
        self.ax_running.grid(True)

        self.ax_counts.clear()
        self.ax_counts.bar([0,1], counts)
        self.ax_counts.set_xticks([0,1])
        self.ax_counts.set_title(f"Counts (live) — total {len(bits)}")
        self.ax_counts.set_xlabel("Bit value")
        self.ax_counts.set_ylabel("Count")
        self.ax_counts.grid(True)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw_idle()
        self.status_lbl.config(text=f"Generating... {len(bits)} bits produced")

    def _finalize_and_report(self, bits, hex_key, ent, chi2, pval, counts, run_dist, ac):
        # fill run-length plot
        self.ax_runlen.clear()
        keys = [k for k in run_dist.keys()]
        # numeric keys first
        numeric_keys = [k for k in keys if isinstance(k, int)]
        numeric_keys.sort()
        labels = [str(k) for k in numeric_keys] + [list(run_dist.keys())[-1]]
        values = [run_dist[k] for k in numeric_keys] + [run_dist[list(run_dist.keys())[-1]]]
        self.ax_runlen.bar(range(len(values)), values)
        self.ax_runlen.set_xticks(range(len(values)))
        self.ax_runlen.set_xticklabels(labels, rotation=45, fontsize=8)
        self.ax_runlen.set_title("Run-length distribution")

        # autocorrelation
        self.ax_autocorr.clear()
        lags = np.arange(1, len(ac)+1)
        self.ax_autocorr.plot(lags, ac)
        self.ax_autocorr.set_title("Autocorrelation (lags)")
        self.ax_autocorr.set_xlabel("Lag")
        self.ax_autocorr.set_ylabel("Autocorr")
        self.ax_autocorr.grid(True)

        # redraw live plots one more time (ensure running/counts reflect final)
        final_running = np.cumsum(bits) / (np.arange(1, len(bits)+1))
        final_counts = np.bincount(bits, minlength=2)

        self.ax_running.clear()
        self.ax_running.plot(final_running)
        self.ax_running.set_title("Running proportion of 1s (final)")
        self.ax_running.set_xlabel("n")
        self.ax_running.set_ylabel("Proportion of 1s")
        self.ax_running.grid(True)

        self.ax_counts.clear()
        self.ax_counts.bar([0,1], final_counts)
        self.ax_counts.set_xticks([0,1])
        self.ax_counts.set_title(f"Counts (final) — total {len(bits)}")
        self.ax_counts.set_xlabel("Bit value")
        self.ax_counts.set_ylabel("Count")
        self.ax_counts.grid(True)

        self.fig.tight_layout(pad=3.0)
        self.canvas.draw_idle()

        # prepare report text
        report_lines = []
        report_lines.append(f"Generated bits: {len(bits)}")
        report_lines.append(f"Hex key length (chars): {len(hex_key)}")
        report_lines.append(f"Shannon entropy (bits/symbol): {ent:.6f} (max 1.0)")
        report_lines.append(f"Chi-square (df=1): {chi2:.4f}, p-value approx: {pval}")
        report_lines.append(f"Counts: 0 -> {counts[0]}, 1 -> {counts[1]}")
        report_lines.append("")
        report_lines.append("Run-length distribution (length: count) — first 12 shown:")
        all_rl = list(run_dist.items())
        for i, (k, v) in enumerate(all_rl[:12]):
            report_lines.append(f"  {k}: {v}")
        report_lines.append("")
        # autocorr summary
        max_abs_ac = np.max(np.abs(ac)) if ac.size>0 else 0.0
        report_lines.append(f"Autocorrelation (lags 1..{len(ac)}): max abs value = {max_abs_ac:.6f}")

        self.report_text.delete("1.0", tk.END)
        self.report_text.insert(tk.END, "\n".join(report_lines))

        # show hex preview in the GUI area if small; otherwise first 512 chars
        preview = hex_key[:512] + ("..." if len(hex_key) > 512 else "")
        self.report_text.insert(tk.END, "\n\nHex key preview:\n")
        self.report_text.insert(tk.END, preview)

        # store
        self.bits = bits
        self.hex_key = hex_key

        self.status_lbl.config(text="Generation & analysis complete.")
        self.generate_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.save_key_btn.config(state="normal")
        self.save_plots_btn.config(state="normal")

    def _no_data_finish(self):
        self.status_lbl.config(text="No data generated.")
        self.generate_btn.config(state="normal")
        self.stop_btn.config(state="disabled")

    def on_save(self):
        if not self.hex_key:
            messagebox.showinfo("No key", "No key to save. Generate first.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Text files","*.txt"),("All files","*.*")])
        if not path:
            return
        try:
            Path(path).write_text(self.hex_key)
            messagebox.showinfo("Saved", f"Key saved to: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save key: {e}")

    def on_save_plots(self):
        path = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG image","*.png"),("All files","*.*")])
        if not path:
            return
        try:
            self.fig.savefig(path, dpi=200)
            messagebox.showinfo("Saved", f"Plots saved to: {path}")
        except Exception as e:
            messagebox.showerror("Error", f"Could not save plots: {e}")

def main():
    root = tk.Tk()
    app = QRNGFullApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
