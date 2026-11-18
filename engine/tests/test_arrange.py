import numpy as np

from glore_engine import arrange


def _uzi_like():
    """Bar levels shaped like the original: light intro, 2-bar drum cut, drop, breakdown, drop."""
    drums = [-35] * 14 + [-75] * 2 + [-27] * 16 + [-68] * 8 + [-27] * 10
    bass = [-76] * 16 + [-10] * 34
    music = [-18] * 14 + [-40] * 2 + [-22] * 16 + [-38] * 8 + [-22] * 10
    return [np.array(x, float) for x in (drums, bass, music)]


def test_section_map_finds_intro_cut_drop_breakdown():
    bars = arrange.section_map(*_uzi_like())
    assert len(bars) == 50
    assert bars[0].drums == "light" and bars[0].bass == "off"
    assert bars[14].drums == "off" and bars[16].drums == "full" and bars[16].bass == "on"
    assert bars[35].drums == "off" and bars[35].music == "low"
    labels = [s["label"] for s in arrange.sections(bars)]
    assert labels[:2] == ["intro", "drum cut"]
    assert "breakdown" in labels and labels[-1] == "full"


def test_bar_levels_per_bar():
    sr = 1000
    x = np.concatenate([np.ones(2000) * 0.1, np.zeros(2000)])
    lv = arrange.bar_levels(x, sr, 0.0, 2.0)
    assert len(lv) == 2 and abs(lv[0] - 20 * np.log10(0.1)) < 0.01 and lv[1] < -90


def test_arrange_stems_mutes_drums_in_breakdown_and_opens_filter():
    sr = 8000
    n = sr * 8
    rng = np.random.default_rng(0)
    noise = lambda: (rng.standard_normal((2, n)) * 0.1).astype(np.float32)
    stems = {"drums": noise(), "bass": noise(), "other": noise()}
    bars = [arrange.Bar("full", "on", "full"), arrange.Bar("off", "on", "low"),
            arrange.Bar("off", "on", "low"), arrange.Bar("full", "on", "full")]
    out = arrange.arrange_stems({"drums": stems["drums"], "bass": np.zeros((2, n), np.float32),
                                 "other": np.zeros((2, n), np.float32)}, sr, bars, [0, 2, 4, 6])
    rms = lambda a, b: float(np.sqrt(np.mean(out[:, a * sr:b * sr] ** 2)))
    assert rms(0, 2) > 0.05 and rms(2, 6) < 0.005 and rms(6, 8) > 0.05
    # music only: low-passed (quieter, darker) in the breakdown, full again after the drop
    out = arrange.arrange_stems({"drums": np.zeros((2, n), np.float32), "bass": np.zeros((2, n), np.float32),
                                 "other": stems["other"]}, sr, bars, [0, 2, 4, 6])
    assert rms(2, 4) < 0.6 * rms(0, 2) and abs(rms(6, 8) - rms(0, 2)) < 0.01
