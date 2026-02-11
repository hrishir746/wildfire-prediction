#!/usr/bin/env python3
"""
Live Dashboard: FU-NetCastV2 Training Monitor

Real-time display of:
- Epoch progress
- Loss curves
- Validation metrics (IoU, Dice)
- GPU memory usage
- Automatic alerts on milestones
"""
import sys
from pathlib import Path
import time
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import psutil

# Page config
st.set_page_config(
    page_title="FU-NetCastV2 Training",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🔥 FU-NetCastV2 Training Dashboard")
st.markdown("**Real-time monitoring of multi-modal fire spread prediction model**")

# Sidebar config
with st.sidebar:
    st.header("⚙️ Settings")
    refresh_interval = st.slider("Refresh interval (seconds)", 5, 60, 10)
    show_gpu = st.checkbox("Show GPU metrics", value=True)
    show_curves = st.checkbox("Show training curves", value=True)

# Metric file paths
METRICS_FILE = ROOT / "outputs" / "firms_focal_training_metrics.json"
LIVE_PROGRESS_FILE = ROOT / "outputs" / "training_focal_progress.json"

def load_metrics():
    """Load training metrics from JSON."""
    if METRICS_FILE.exists():
        with open(METRICS_FILE) as f:
            return json.load(f)
    return None

def load_live_progress():
    """Load live progress file (updated during training)."""
    if LIVE_PROGRESS_FILE.exists():
        with open(LIVE_PROGRESS_FILE) as f:
            return json.load(f)
    return None

def get_gpu_memory():
    """Get GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            reserved = torch.cuda.memory_reserved() / 1e9
            return allocated, reserved
    except:
        pass
    return 0, 0

# ============================================================================
# MAIN DISPLAY
# ============================================================================

# Auto-refresh
st.markdown(f"<small>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Refreshing every {refresh_interval}s</small>", unsafe_allow_html=True)

metrics = load_metrics()
live_progress = load_live_progress()

if metrics is None and live_progress is None:
    st.warning("⏳ No training data found yet. Training may not have started.")
    st.stop()

# ============================================================================
# STATUS CARDS
# ============================================================================

col1, col2, col3, col4 = st.columns(4)

if live_progress:
    current_epoch = live_progress.get("current_epoch", 0)
    total_epochs = live_progress.get("total_epochs", 50)
    status = live_progress.get("status", "training")
    
    with col1:
        st.metric("Current Epoch", f"{current_epoch}/{total_epochs}")
    
    with col2:
        progress_pct = (current_epoch / total_epochs * 100) if total_epochs > 0 else 0
        st.metric("Progress", f"{progress_pct:.0f}%")
    
    with col3:
        eta_min = live_progress.get("eta_minutes", 0)
        st.metric("ETA", f"{eta_min:.1f} min" if eta_min > 0 else "N/A")
    
    with col4:
        status_emoji = {"training": "🟢", "completed": "✅", "paused": "⏸️", "error": "❌"}.get(status, "❓")
        st.metric("Status", f"{status_emoji} {status.upper()}")

if live_progress and ("train_loss" in live_progress or "val_iou" in live_progress):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        train_loss = live_progress.get("train_loss", None)
        if train_loss is not None:
            st.metric("Train Loss", f"{train_loss:.4f}")
    
    with col2:
        val_loss = live_progress.get("val_loss", None)
        if val_loss is not None:
            st.metric("Val Loss", f"{val_loss:.4f}")
    
    with col3:
        val_iou = live_progress.get("val_iou", None)
        if val_iou is not None:
            delta = f"+{val_iou*100:.1f}%" if val_iou > 0 else "0%"
            st.metric("Val IoU", f"{val_iou*100:.1f}%", delta, delta_color="off")
    
    with col4:
        val_dice = live_progress.get("val_dice", None)
        if val_dice is not None:
            st.metric("Val Dice", f"{val_dice*100:.1f}%", delta_color="off")

# ============================================================================
# ALERTS
# ============================================================================

st.markdown("---")

alerts = []

if live_progress:
    val_iou = live_progress.get("val_iou", 0)
    
    if val_iou > 0.85:
        alerts.append(("success", "🎉 EXCELLENT IoU >85%! Model is crushing the 1% baseline!"))
    elif val_iou > 0.70:
        alerts.append(("success", "✅ GREAT IoU >70%! Significant improvement over synthetic."))
    elif val_iou > 0.50:
        alerts.append(("info", "ℹ️ Good progress - 50% IoU shows real fire patterns being learned."))
    
    status = live_progress.get("status", "")
    if status == "completed":
        alerts.append(("success", "✅ TRAINING COMPLETE! Check outputs/fu_unet_training_metrics.json"))

if metrics and metrics.get("best_val_iou", 0) > 0.70:
    alerts.append(("success", f"⭐ Best recorded IoU: {metrics['best_val_iou']*100:.1f}%"))

for alert_type, message in alerts:
    if alert_type == "success":
        st.success(message)
    elif alert_type == "info":
        st.info(message)
    elif alert_type == "warning":
        st.warning(message)

# ============================================================================
# TRAINING CURVES
# ============================================================================

if show_curves and metrics and "history" in metrics:
    st.markdown("---")
    st.subheader("📈 Training Curves")
    
    history = metrics["history"]
    epochs = list(range(1, len(history.get("train_loss", [])) + 1))
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Loss", "IoU", "Dice Coefficient", "Learning Progress"),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Loss
    if history.get("train_loss"):
        fig.add_trace(
            go.Scatter(x=epochs, y=history["train_loss"], name="Train Loss", line=dict(color="blue")),
            row=1, col=1
        )
    if history.get("val_loss"):
        fig.add_trace(
            go.Scatter(x=epochs, y=history["val_loss"], name="Val Loss", line=dict(color="red", dash="dash")),
            row=1, col=1
        )
    
    # IoU
    if history.get("val_iou"):
        fig.add_trace(
            go.Scatter(x=epochs, y=[v*100 for v in history["val_iou"]], name="Val IoU", line=dict(color="green")),
            row=1, col=2
        )
    
    # Dice
    if history.get("val_dice"):
        fig.add_trace(
            go.Scatter(x=epochs, y=[v*100 for v in history["val_dice"]], name="Val Dice", line=dict(color="purple")),
            row=2, col=1
        )
    
    # Learning progress (combined metric)
    if history.get("val_iou"):
        val_iou_pct = [v*100 for v in history["val_iou"]]
        fig.add_trace(
            go.Scatter(x=epochs, y=val_iou_pct, name="Progress", fill="tozeroy", line=dict(color="orange")),
            row=2, col=2
        )
    
    fig.update_xaxes(title_text="Epoch", row=1, col=1)
    fig.update_xaxes(title_text="Epoch", row=1, col=2)
    fig.update_xaxes(title_text="Epoch", row=2, col=1)
    fig.update_xaxes(title_text="Epoch", row=2, col=2)
    
    fig.update_yaxes(title_text="Loss", row=1, col=1)
    fig.update_yaxes(title_text="IoU (%)", row=1, col=2)
    fig.update_yaxes(title_text="Dice (%)", row=2, col=1)
    fig.update_yaxes(title_text="Progress (%)", row=2, col=2)
    
    fig.update_layout(height=600, showlegend=True, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================================
# GPU METRICS
# ============================================================================

if show_gpu:
    st.markdown("---")
    st.subheader("💻 GPU Metrics")
    
    col1, col2, col3 = st.columns(3)
    
    allocated, reserved = get_gpu_memory()
    
    with col1:
        st.metric("GPU Memory Allocated", f"{allocated:.2f} GB")
    
    with col2:
        st.metric("GPU Memory Reserved", f"{reserved:.2f} GB")
    
    with col3:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_name = torch.cuda.get_device_name(0)
                st.metric("GPU Device", gpu_name[:30])
        except:
            st.metric("GPU Device", "N/A")

# ============================================================================
# SUMMARY
# ============================================================================

st.markdown("---")
st.subheader("📊 Summary")

if metrics:
    best_iou = metrics.get("best_val_iou", 0)
    training_time = metrics.get("training_time_minutes", 0)
    epochs_trained = len(metrics.get("history", {}).get("train_loss", []))
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Best IoU", f"{best_iou*100:.1f}%")
    
    with col2:
        st.metric("Epochs Complete", epochs_trained)
    
    with col3:
        st.metric("Time Elapsed", f"{training_time:.1f} min")
    
    # Interpretation
    st.markdown("---")
    st.subheader("🔍 Interpretation")
    
    if best_iou > 0.85:
        st.success("""
        ✅ **OUTSTANDING RESULT!**
        
        Your FU-NetCastV2 model achieved **85%+ accuracy** on real fire progression data.
        This is a **dramatic improvement** from the synthetic baseline (1% → 85%).
        
        **What this means:**
        - The multi-modal approach (satellite + weather + terrain) works
        - Real fire patterns can be learned from FIRMS satellite detections
        - Domain gap has been solved by training on real data
        
        **Ready for:**
        - Validation on held-out MTBS fires
        - Production deployment
        - Publication/presentation
        """)
    elif best_iou > 0.70:
        st.success("""
        ✅ **EXCELLENT RESULT!**
        
        Your model achieved **70-85% accuracy** on real data.
        This is a **70-85x improvement** from synthetic (1%).
        
        **What this means:**
        - Multi-modal data successfully addresses domain gap
        - Model learned realistic fire spread patterns
        - Ready for practical applications
        """)
    elif best_iou > 0.50:
        st.info("""
        ℹ️ **GOOD PROGRESS**
        
        Model achieved **50-70% accuracy**. Still learning, but heading in right direction.
        
        **Next steps:**
        - Add more fire pairs (if available)
        - Include real Landsat data (current is synthetic)
        - Increase weather timeseries granularity
        """)
    else:
        st.warning(f"""
        ⚠️ **TRAINING IN PROGRESS**
        
        Current IoU: {best_iou*100:.1f}%. Keep training to see convergence.
        """)

# Auto-refresh
if live_progress and live_progress.get("status") == "training":
    st.markdown(f"<small>Auto-refreshing in {refresh_interval} seconds...</small>", unsafe_allow_html=True)
    time.sleep(refresh_interval)
    st.rerun()

st.markdown("---")
st.markdown(f"<small>Dashboard by: Copilot | Model: FU-NetCastV2 | Updated: {datetime.now().isoformat()}</small>", unsafe_allow_html=True)
