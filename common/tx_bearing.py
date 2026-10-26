"""Commanded-pose bearing model used by the pre-flight R analysis.

Optical principal point is distinct from the M3 laser pixel. No encoder,
distortion correction or camera/pivot offset compensation is implied.
"""
import math

BEARING_MODEL = dict(
    version='tx-commanded-bearing-v1', width=1296, height=972,
    fx_px=2857.1428571428573, fy_px=2857.1428571428573,
    cx_px=648, cy_px=486, rotation='Ry(pan) Rx(tilt)',
    focal_length_mm=8, native_pixel_pitch_um=1.4, image_scale=0.5,
    axes='Tx: right X, down Y, forward Z; alpha right, beta up',
    pose_source='commanded, not measured', distortion_corrected=False,
    pivot_offset_corrected=False)


def commanded_bearing(u, v, pan_deg, tilt_deg):
    """Return (azimuth, elevation) radians in the fixed Tx command frame."""
    if not all(math.isfinite(value) for value in (u, v, pan_deg, tilt_deg)):
        raise ValueError('Bearing inputs must be finite')
    m = BEARING_MODEL
    x, y = (u-m['cx_px'])/m['fx_px'], (v-m['cy_px'])/m['fy_px']
    p, t = math.radians(pan_deg), math.radians(tilt_deg)
    xt = math.cos(p)*x + math.sin(p)*(math.sin(t)*y+math.cos(t))
    yt = math.cos(t)*y-math.sin(t)
    zt = -math.sin(p)*x + math.cos(p)*(math.sin(t)*y+math.cos(t))
    return math.atan2(xt, zt), math.atan2(-yt, math.hypot(xt, zt))
