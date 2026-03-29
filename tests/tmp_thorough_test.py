import numpy as np
import pandas as pd
from src.ai_analyst import AIAnalyst

a = AIAnalyst(window_size=8, std_threshold=1e-9, contamination=0.05, random_state=42, n_estimators=50)

print("--- THOROUGH TEST START ---")

# 1) pandas.Series input
s = pd.Series(np.r_[np.linspace(0, 1, 20), np.ones(10) * 3.0, np.linspace(1, 2, 20)])
sm = a.detect_stuck_at_faults(s)
em = a.detect_seu_anomalies(s)
rep = a.generate_ai_report(sm, em)
print("series_input_ok", sm.shape == s.shape == em.shape and sm.dtype == np.bool_ and em.dtype == np.bool_ and isinstance(rep, str))

# 2) n < window_size
short = np.array([1.0, 1.0, 1.0])
short_mask = a.detect_stuck_at_faults(short)
print("short_n_lt_window_ok", short_mask.shape == short.shape and short_mask.sum() == 0)

# 3) constant signal should be detected as stuck
const = np.ones(32) * 7.0
const_mask = a.detect_stuck_at_faults(const)
print("constant_signal_stuck_detected", bool(const_mask.any()))

# 4) normal ramp should not be falsely stuck for strict threshold
normal = np.linspace(0, 1, 64)
normal_mask = a.detect_stuck_at_faults(normal)
print("normal_signal_low_false_positive", int(normal_mask.sum()) == 0)

# 5) contamination sensitivity
b = AIAnalyst(window_size=8, std_threshold=1e-9, contamination=0.01, random_state=42, n_estimators=50)
c = AIAnalyst(window_size=8, std_threshold=1e-9, contamination=0.10, random_state=42, n_estimators=50)
x = np.r_[np.linspace(0, 1, 60), 9.0, -8.0, np.linspace(1, 2, 60)]
m1 = b.detect_seu_anomalies(x)
m2 = c.detect_seu_anomalies(x)
print("contamination_sensitivity_ok", int(m2.sum()) >= int(m1.sum()))

# 6) error paths
try:
    a.detect_stuck_at_faults(np.array([]))
    print("empty_input_raises", False)
except ValueError:
    print("empty_input_raises", True)

try:
    a.detect_stuck_at_faults(np.array([1.0, np.nan, 2.0]))
    print("nan_input_raises", False)
except ValueError:
    print("nan_input_raises", True)

try:
    a.detect_stuck_at_faults(np.array([1.0, np.inf, 2.0]))
    print("inf_input_raises", False)
except ValueError:
    print("inf_input_raises", True)

try:
    a.detect_stuck_at_faults(np.array([[1.0, 2.0], [3.0, 4.0]]))
    print("non1d_input_raises", False)
except ValueError:
    print("non1d_input_raises", True)

print("--- THOROUGH TEST END ---")
