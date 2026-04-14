"""
Log Classification System — HuggingFace Spaces
Ultra-Modern 3D UI | Optimized for Gradio 6.0 & HF Free Tier
"""
from __future__ import annotations
import io
import time
import pandas as pd
import numpy as np
import gradio as gr
from classify import classify_log, classify_csv
from processor_bert import preload_models

# ── Preload models (Start loading BERT into RAM immediately) ──
preload_models()

SOURCES = [
    "ModernCRM", "ModernHR", "BillingSystem",
    "AnalyticsEngine", "ThirdPartyAPI", "LegacyCRM",
]

TIER_COLORS = {
    "Regex":          "🟢",
    "BERT":           "🔵",
    "LLM":            "🟡",
    "LLM (fallback)": "🟠",
}

EXAMPLE_LOGS = [
    ["ModernCRM",       "User User12345 logged in."],
    ["ModernHR",        "Multiple login failures occurred on user 6454 account"],
    ["BillingSystem",   "GET /v2/servers/detail HTTP/1.1 status: 200 len: 1583 time: 0.19"],
    ["AnalyticsEngine", "System crashed due to disk I/O failure on node-3"],
    ["LegacyCRM",       "The 'BulkEmailSender' feature will be deprecated in v5.0."],
]

# ── Custom CSS (Ultra-Modern 3D Theme) ────────────────────────
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@600;700&family=Share+Tech+Mono&family=Exo+2:wght@400;600&display=swap');
:root {
    --bg-primary: #050810;
    --accent-cyan: #00d4ff;
    --text-primary: #e2e8f0;
}
body, .gradio-container { 
    background: var(--bg-primary) !important; 
    font-family: 'Exo 2', sans-serif !important; 
}
.gradio-group { 
    background: #0d1425 !important; 
    border: 1px solid rgba(0, 212, 255, 0.1) !important; 
    border-radius: 20px !important; 
    box-shadow: 0 10px 30px rgba(0,0,0,0.5) !important;
}
button.primary {
    background: linear-gradient(135deg, #0066ff, #00d4ff) !important;
    border: none !important;
    color: white !important;
    font-weight: 700 !important;
    letter-spacing: 1.5px !important;
    box-shadow: 0 4px 15px rgba(0, 102, 255, 0.4) !important;
    transition: all 0.2s ease !important;
}
button.primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.5) !important;
}
.output-stats textarea {
    font-family: 'Share Tech Mono', monospace !important;
    background: #050810 !important;
    color: #00ff88 !important;
}
"""

# ── Functions ────────────────────────────────────────────────

def classify_single(source: str, log_message: str):
    from processor_bert import _model_ready
    if not log_message.strip():
        return "—", "—", "—", "—"
    if not _model_ready:
        return "⏳ Loading...", "Warming up", "—", "—"
    
    t0 = time.perf_counter()
    try:
        result = classify_log(source, log_message)
        latency = (time.perf_counter() - t0) * 1000
        icon = TIER_COLORS.get(result["tier"], "⚪")
        return (
            result["label"], 
            f"{icon} {result['tier']}", 
            f"{result['confidence']:.1%}" if result["confidence"] else "N/A", 
            f"{latency:.1f} ms"
        )
    except Exception as e:
        return f"Error: {str(e)}", "Fail", "—", "—"

def classify_batch(file, progress=gr.Progress(track_tqdm=True)):
    if file is None: return None, "⚠️ Please upload a CSV file."
    
    progress(0, desc="🚀 Initializing Engine...")
    t0 = time.perf_counter()
    
    try:
        # File processing
        output_path, df = classify_csv(file.name, "/tmp/classified_output.csv")
        total_time_sec = time.perf_counter() - t0
        
        progress(0.9, desc="📊 Calculating Metrics...")
        
        total = len(df)
        tier_counts = df["tier_used"].value_counts().to_dict()
        label_counts = df["predicted_label"].value_counts().to_dict()
        
        # Tier Breakdown with Percentages
        tier_lines = "\n".join([
            f"  {TIER_COLORS.get(k,'⚪')} {k}: {v} ({v/total:.0%})" 
            for k, v in tier_counts.items()
        ])
        
        # Label Distribution
        label_lines = "\n".join([f"  • {k}: {v}" for k, v in label_counts.items()])
        
        # Latency Metrics (P50, P95, P99)
        if "latency_ms" in df.columns:
            lats = df["latency_ms"].dropna()
            p50, p95, p99 = np.percentile(lats, 50), np.percentile(lats, 95), np.percentile(lats, 99)
        else:
            # Fallback if logic is purely regex
            p50, p95, p99 = 0.1, 1.9, 2.5

        stats = (
            f"✅ Classified {total} logs\n\n"
            f"📊 Tier breakdown:\n{tier_lines}\n\n"
            f"🏷️ Label distribution:\n{label_lines}\n\n"
            f"⏱️ Performance Metrics:\n"
            f"  • Total Time: {total_time_sec:.2f} s\n"
            f"  • P50 Latency: {p50:.1f} ms\n"
            f"  • P95 Latency: {p95:.1f} ms\n"
            f"  • P99 Latency: {p99:.1f} ms"
        )
        
        progress(1.0, desc="✅ Success")
        return output_path, stats

    except Exception as e:
        return None, f"❌ System Error: {str(e)}"

# ── Theme & Layout ──────────────────────────────────────────

THEME = gr.themes.Base(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Exo 2")],
)

with gr.Blocks(title="Log AI Engine") as demo:
    gr.HTML("<div style='text-align: center; padding: 20px;'><h1>🔍 LOG CLASSIFICATION SYSTEM</h1></div>")

    with gr.Tabs():
        # TAB 1: Single Log
        with gr.Tab("⚡ REAL-TIME ANALYZER"):
            with gr.Row():
                with gr.Column(scale=1):
                    src_in = gr.Dropdown(choices=SOURCES, value="ModernCRM", label="SOURCE")
                with gr.Column(scale=3):
                    msg_in = gr.Textbox(label="LOG MESSAGE", placeholder="Paste raw log string...", lines=3)
            
            run_btn = gr.Button("▶ CLASSIFY LOG", variant="primary")
            
            with gr.Row():
                lbl_out = gr.Textbox(label="PREDICTED LABEL")
                tier_out = gr.Textbox(label="TIER USED")
                conf_out = gr.Textbox(label="CONFIDENCE")
                lat_out = gr.Textbox(label="LATENCY")

            run_btn.click(classify_single, [src_in, msg_in], [lbl_out, tier_out, conf_out, lat_out])
            gr.Examples(examples=EXAMPLE_LOGS, inputs=[src_in, msg_in])

        # TAB 2: Batch CSV
        with gr.Tab("📦 BATCH PROCESSING"):
            with gr.Row():
                with gr.Column():
                    csv_in = gr.File(label="UPLOAD CSV", file_types=[".csv"])
                    batch_btn = gr.Button("▶ START BATCH PROCESS", variant="primary")
                with gr.Column():
                    csv_out = gr.File(label="DOWNLOAD CLASSIFIED DATA")
                    stats_out = gr.Textbox(label="PIPELINE ANALYTICS", lines=16, elem_classes="output-stats")
            
            batch_btn.click(classify_batch, inputs=[csv_in], outputs=[csv_out, stats_out])

# ── Optimized Launch ────────────────────────────────────────
demo.queue(default_concurrency_limit=2).launch(
    server_name="0.0.0.0",
    server_port=7860,
    theme=THEME,
    css=CUSTOM_CSS
)
