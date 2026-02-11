#!/usr/bin/env python3
"""
Live Training Dashboard - Option B FIRMS Progressions

Monitors train_firms_progressions.py with live metrics and error alerts.
"""
import sys
from pathlib import Path
import json
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st

st.set_page_config(page_title="🔥 FIRMS Training", page_icon="🔥", layout="wide")

st.title("🔥 FIRMS Progressions Training Monitor")
st.markdown("**Real-time monitoring of 8-channel U-Net on real fire data**")

PROGRESS_FILE = ROOT / "outputs" / "training_firms_progress.json"
ERROR_FILE = ROOT / "outputs" / "training_errors.json"

# Auto-refresh
st.markdown("_Refreshing in real-time..._")

# Load progress
progress = None
if PROGRESS_FILE.exists():
    try:
        with open(PROGRESS_FILE) as f:
            progress = json.load(f)
    except:
        pass

# Load errors
error = None
if ERROR_FILE.exists():
    try:
        with open(ERROR_FILE) as f:
            error = json.load(f)
    except:
        pass

# ============================================================================
# ERROR ALERT (TOP PRIORITY)
# ============================================================================

if error:
    st.error(f"⚠️ **TRAINING ERROR**\n\n{error.get('error', 'Unknown error')}")
    if "timestamp" in error:
        st.caption(f"Error at: {error['timestamp']}")

# ============================================================================
# STATUS CARDS
# ============================================================================

if progress:
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        epoch = progress.get("current_epoch", 0)
        total = progress.get("total_epochs", 40)
        st.metric("Epoch", f"{epoch}/{total}")
    
    with col2:
        pct = (epoch / total * 100) if total > 0 else 0
        st.metric("Progress", f"{pct:.0f}%")
    
    with col3:
        eta = progress.get("eta_minutes", 0)
        st.metric("ETA", f"{eta:.1f} min" if eta > 0 else "~0 min")
    
    with col4:
        status = progress.get("status", "unknown")
        status_emoji = {"training": "🟢", "completed": "✅", "error": "❌", "paused": "⏸️"}.get(status, "❓")
        st.metric("Status", f"{status_emoji} {status.upper()}")

    # Metrics
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Train Loss", f"{progress.get('train_loss', 0):.4f}")
    with col2:
        st.metric("Val Loss", f"{progress.get('val_loss', 0):.4f}")
    with col3:
        iou = progress.get("val_iou", 0) * 100
        st.metric("Val IoU", f"{iou:.1f}%")
    with col4:
        dice = progress.get("val_dice", 0) * 100
        st.metric("Val Dice", f"{dice:.1f}%")

    # Progress bar
    st.markdown("---")
    st.progress(epoch / total if total > 0 else 0)

    # Alerts
    iou = progress.get("val_iou", 0)
    if iou > 0.70:
        st.success("✅ **EXCELLENT** - IoU >70%! Beating the 1% baseline!")
    elif iou > 0.50:
        st.info("ℹ️ **Good progress** - IoU >50%. Fire patterns being learned.")
    elif iou > 0.20:
        st.warning("⚠️ **Early stage** - Keep training, IoU will improve.")

    if progress.get("status") == "completed":
        st.balloons()
        st.success("✅ Training finished! Check outputs/ for model.")

else:
    st.warning("⏳ Waiting for training to start...")

# Footer
st.markdown("---")
st.markdown(f"<small>Updated: {datetime.now().strftime('%H:%M:%S')} | Model: FIRMS Progressions | Dataset: 300 real fire pairs</small>", unsafe_allow_html=True)

# Auto-rerun
if not error and progress and progress.get("status") == "training":
    import time
    time.sleep(10)
    st.rerun()
