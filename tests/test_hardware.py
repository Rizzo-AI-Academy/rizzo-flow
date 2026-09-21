"""Accelerator probing: pure functions over an injected DRM tree, no GPU required."""

from rizzo_flow import hardware


def write_vendor(drm, card, value):
    vendor = drm / card / "device" / "vendor"
    vendor.parent.mkdir(parents=True, exist_ok=True)
    vendor.write_text(value + "\n")


def test_drm_vendors_maps_pci_ids(tmp_path):
    write_vendor(tmp_path, "card0", "0x1002")
    write_vendor(tmp_path, "card1", "0x10de")
    write_vendor(tmp_path, "card2", "0x8086")
    assert hardware.drm_vendors(tmp_path) == {"amd", "nvidia", "intel"}


def test_drm_vendors_ignores_unknown_and_missing(tmp_path):
    write_vendor(tmp_path, "card0", "0x1234")
    (tmp_path / "card1" / "device").mkdir(parents=True)  # no vendor file
    (tmp_path / "card2").mkdir()  # no device directory at all
    assert hardware.drm_vendors(tmp_path) == set()


def test_drm_vendors_survives_a_missing_tree(tmp_path):
    assert hardware.drm_vendors(tmp_path / "absent") == set()


def test_probe_returns_best_first_and_always_cpu(monkeypatch, tmp_path):
    monkeypatch.setattr(hardware, "drm_vendors", lambda drm=None: {"amd"})
    monkeypatch.setattr(hardware, "apple_platform", lambda: False)
    monkeypatch.setattr(hardware, "nvidia_driver", lambda: False)
    assert hardware.probe() == ["amd", "cpu"]


def test_probe_prefers_apple_over_a_discrete_gpu(monkeypatch):
    monkeypatch.setattr(hardware, "drm_vendors", lambda drm=None: set())
    monkeypatch.setattr(hardware, "apple_platform", lambda: True)
    monkeypatch.setattr(hardware, "nvidia_driver", lambda: True)
    assert hardware.probe() == ["apple", "cpu"]


def test_probe_reports_hybrid_nvidia_and_amd(monkeypatch):
    monkeypatch.setattr(hardware, "drm_vendors", lambda drm=None: {"amd"})
    monkeypatch.setattr(hardware, "apple_platform", lambda: False)
    monkeypatch.setattr(hardware, "nvidia_driver", lambda: True)
    assert hardware.probe() == ["nvidia", "amd", "cpu"]


def test_probe_on_a_cpu_only_box(monkeypatch):
    monkeypatch.setattr(hardware, "drm_vendors", lambda drm=None: set())
    monkeypatch.setattr(hardware, "apple_platform", lambda: False)
    monkeypatch.setattr(hardware, "nvidia_driver", lambda: False)
    assert hardware.probe() == ["cpu"]


def test_probe_on_this_machine_is_a_subset_of_the_vocabulary():
    """No injection: the real probe must not raise and must stay inside the known names."""
    reported = hardware.probe()
    assert reported and reported[-1] == "cpu"
    assert set(reported) <= set(hardware.ACCELERATORS)
