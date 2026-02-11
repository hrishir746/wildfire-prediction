#!/usr/bin/env python3
"""
Background monitoring script - runs every 5 min and creates alerts for dashboard.
"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import json
import time
from datetime import datetime
import subprocess

alert_file = ROOT / "outputs" / "pipeline_alerts.json"

def load_alerts():
    if alert_file.exists():
        with open(alert_file) as f:
            return json.load(f)
    return []

def save_alert(alert_type, message):
    alerts = load_alerts()
    alerts.append({
        'type': alert_type,
        'time': datetime.now().strftime("%H:%M:%S"),
        'message': message
    })
    # Keep only last 10 alerts
    alerts = alerts[-10:]
    with open(alert_file, 'w') as f:
        json.dump(alerts, f, indent=2)
    print(f"[{alert_type.upper()}] {message}")

# Check for errors
try:
    longterm_dataset = ROOT / "data" / "longterm_fires" / "longterm_dataset.npz"
    longterm_model = ROOT / "outputs" / "checkpoints" / "unet_longterm_v1.pt"
    
    # Check dataset generation
    if not longterm_dataset.exists():
        # Check if process exists
        result = subprocess.run(['powershell', 'Get-Process python* -ErrorAction SilentlyContinue'], 
                              capture_output=True, text=True)
        if not result.stdout.strip():
            save_alert('error', 'No Python processes running! Dataset generation may have crashed.')
    else:
        # Dataset complete - check file size
        file_size_mb = longterm_dataset.stat().st_size / 1e6
        if file_size_mb < 10:  # Suspiciously small
            save_alert('warning', f'Dataset file is only {file_size_mb:.1f} MB - may be incomplete')
        else:
            # Check if training started
            if not longterm_model.exists():
                save_alert('success', f'Dataset complete ({file_size_mb:.1f} MB)! Ready for training.')

except Exception as e:
    save_alert('error', f'Monitor error: {str(e)[:100]}')

print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitor check complete")
