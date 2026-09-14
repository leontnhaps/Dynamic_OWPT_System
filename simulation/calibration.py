"""수동 클릭 자료의 보정 이력과 축별 국소 배율 추정. 오른쪽 22.5 cm는 제외."""
import json
import numpy as np

# 원본 simulation_data.zip은 수정하지 않는다. 좌표 단위 m, 영상 좌표 단위 px.
SAMPLES = [
    dict(id="968edabae99d499882330695219fb2fc", x=0., y=0., u=654.974, v=387.552, use=True),
    dict(id="03e6a4399c804296b034606c7d952195", x=.225, y=0., u=783.528, v=392.038, use=False),
    dict(id="dc306e20c9c44b2397725e6b94e9b596", x=.45, y=0., u=1010.739, v=393.534, use=True),
    dict(id="a514865114f246b194c12fc3125cb26c", x=-.225, y=0., u=492.040, v=386.057, use=True),
    dict(id="0999179c6d4e4d339a703515e640ad3f", x=-.45, y=0., u=299.209, v=386.057, use=True),
    dict(id="5cc23d35635b491db6734dbb99f8cd7b", x=0., y=-.27, u=671.417, v=602.888, use=True),
    dict(id="58c852c68ee24403a7ab25875c781b79", x=0., y=.05, u=669.922, v=345.682, use=True),
]


def fit_scale():
    """기준 사진에 고정한 최소제곱 Δu=kx·X, Δv=-ky·Y; 회전 실측은 아니다."""
    origin = SAMPLES[0]
    fits = []
    residuals = []
    for axis, pixel, sign in (("x", "u", 1), ("y", "v", -1)):
        rows = [r for r in SAMPLES if r["use"] and r[axis] != 0]
        distances = np.array([r[axis] for r in rows])
        shifts = np.array([r[pixel] - origin[pixel] for r in rows])
        k = float(sign * np.dot(distances, shifts) / np.dot(distances, distances))
        fits.append(k)
        residuals.extend(dict(id=r["id"], axis=axis,
                              residual_px=float(s - sign*k*d))
                         for r, d, s in zip(rows, distances, shifts))
    return np.array(fits), residuals


def report():
    scale, residuals = fit_scale()
    focal = scale * 5.0
    return dict(distance_m=5.0, resolution=[1296, 972],
                corrections=["Y +0.32 -> +0.05 m", "X +/-0.275 -> +/-0.225 m",
                             "X +0.225 m excluded at user's request"],
                samples=SAMPLES, pixels_per_cm=(scale/100).tolist(),
                effective_focal_px=focal.tolist(),
                central_pixels_per_degree=(focal*np.pi/180).tolist(),
                residuals=residuals,
                status="Local approximation from target translation, not measured servo response")


if __name__ == "__main__":
    print(json.dumps(report(), indent=2, ensure_ascii=False))
