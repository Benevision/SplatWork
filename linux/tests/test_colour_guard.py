"""ColourGuard: per-photo colour correction (bilateral grid) must not end up supplying the scene's colour."""
from types import SimpleNamespace

import pytest

from splatworker.config import TrainingConfig
from splatworker.trainer import ColourGuard


def test_views_are_evenly_spaced():
    assert ColourGuard.views(5) == [0, 1, 2, 3, 4]
    v = ColourGuard.views(100)
    assert len(v) == 8 and v[0] == 0 and v[-1] == 99
    assert all(b > a for a, b in zip(v, v[1:]))


def test_waits_for_the_start_step_and_stops_once_off():
    g = ColourGuard(4000)
    assert not g.due(2000) and g.due(4000)
    g.off = True
    assert not g.due(10000)


def test_healthy_run_does_nothing():
    g = ColourGuard()
    for k, step in enumerate(range(4000, 40000, 2000)):
        assert g.update(step, 18 + 0.1 * k - (1.5 if k % 3 == 0 else 0), 0.84) is None   # dips under 3 dB
        assert g.update(step, None, None) is None
    assert g.resets == 0 and not g.off


def test_falling_psnr_strikes_twice_then_resets():
    g = ColourGuard()
    assert g.update(4000, 14.9, 0.9) is None
    assert g.update(6000, 11.5, 0.9) is None and g.strikes == 1     # 3.4 dB below the best
    assert g.update(8000, 14.0, 0.9) is None and g.strikes == 0     # a good eval forgives
    assert g.update(10000, 11.0, 0.88) is None and g.strikes == 1
    kind, line = g.update(12000, 9.3, 0.84)
    assert kind == "reset" and g.best is None and g.strikes == 0
    assert line == ("Per-photo colour correction reset at step 12000: it was taking over the scene's colour "
                    "(without it the photos score 5.6 dB below their best)")


def test_dark_renders_act_at_once_then_turn_off():
    g = ColourGuard()
    kind, line = g.update(6000, 15.0, 0.31)
    assert kind == "reset" and not g.off
    assert "31% of the photos' brightness" in line
    assert g.update(8000, 12.0, 0.9) is None          # recovered after the reset (and the best was forgotten)
    kind, line = g.update(10000, 12.0, 0.4)
    assert kind == "off" and g.off
    assert "turned off at step 10000" in line and "check the camera solve" in line


def test_state_round_trips_for_resume():
    g = ColourGuard()
    g.update(4000, 15.0, 0.9)
    g.update(6000, 11.0, 0.9)
    h = ColourGuard()
    h.load(g.state())
    assert h.best == 15.0 and h.strikes == 1 and h.resets == 0
    g.update(8000, 10.0, 0.2)
    g.update(10000, 10.0, 0.2)
    h.load(g.state())
    assert h.off and h.resets == 1 and not h.due(12000)
    h.load(None)                                 # a state saved before the guard existed
    assert h.off


def test_config_field():
    assert TrainingConfig.from_dict({}).colourGuardStart == 4000
    assert TrainingConfig.from_dict({"colourGuardStart": 100}).colourGuardStart == 100


def test_score_over_kept_pixels():
    torch = pytest.importorskip("torch")
    from splatworker.trainer import uncorrected_score

    gt = torch.full((4, 4, 3), 0.5)
    pred = torch.full((4, 4, 3), 0.4)
    keep = torch.ones(4, 4, 1)
    keep[:, :2] = 0
    pred[:, :2] = 5.0                            # masked out (and would clamp to 1): ignored
    psnr, bright = uncorrected_score(torch, [(pred, gt, keep)])
    assert psnr == pytest.approx(20.0, abs=1e-3) and bright == pytest.approx(0.8)
    assert uncorrected_score(torch, [(gt, torch.zeros(4, 4, 3), None)])[1] is None
    assert uncorrected_score(torch, []) == (None, None)


def test_bilateral_grid_reset():
    torch = pytest.importorskip("torch")
    from splatworker.trainer import BilateralGrid

    bg = BilateralGrid(torch, SimpleNamespace(bilateralGridLR=2e-2, bilateralGridTV=0.0), 2, "cpu")
    rgb = torch.rand(8, 8, 3)
    (bg.apply(0, rgb) - rgb * 0.2).abs().mean().backward()
    bg.step(0)
    assert not torch.allclose(bg.apply(0, rgb), rgb, atol=1e-4)
    bg.reset()
    assert torch.allclose(bg.apply(0, rgb), rgb, atol=1e-5)
    assert not bg.opts[0].state                  # fresh Adam moments
    assert bg.opts[0].param_groups[0]["lr"] == 2e-2
