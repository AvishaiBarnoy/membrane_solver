import sys
from pathlib import Path

import numpy as np
from scipy.special import i1, k0, k1

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools.diagnostics.free_disk_profile_fits import _fit_i1, _fit_k0, _fit_k1


def test_fit_k1_recovers_lambda() -> None:
    r = np.linspace(1.0, 3.0, 60)
    R = 1.0
    theta_r = 0.2
    lam = 5.0
    y = theta_r * k1(lam * r) / k1(lam * R)
    theta_hat, lam_hat, rmse = _fit_k1(r, y, R)
    assert rmse < 1e-6
    assert abs(theta_hat - theta_r) / theta_r < 0.05
    assert abs(lam_hat - lam) / lam < 0.05


def test_fit_i1_recovers_lambda() -> None:
    r = np.linspace(0.1, 1.0, 60)
    R = 1.0
    theta_r = 0.18
    lam = 7.0
    y = theta_r * i1(lam * r) / i1(lam * R)
    theta_hat, lam_hat, rmse = _fit_i1(r, y, R)
    assert rmse < 1e-6
    assert abs(theta_hat - theta_r) / theta_r < 0.05
    assert abs(lam_hat - lam) / lam < 0.05


def test_fit_k0_recovers_decay() -> None:
    r = np.linspace(1.0, 3.0, 60)
    amp = 0.3
    psi = 2.5
    y = amp * k0(psi * r)
    amp_hat, psi_hat, rmse = _fit_k0(r, y)
    assert rmse < 1e-6
    assert abs(amp_hat - amp) / amp < 0.05
    assert abs(psi_hat - psi) / psi < 0.05
