# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Legacy and A5 DCMI topology use physical visibility and PCIe-local CPUs."""
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.offloader import npu_cpu_affinity as affinity

pytestmark = [pytest.mark.diffusion, pytest.mark.cpu, pytest.mark.core_model]


class Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


@pytest.mark.parametrize("v2", [False, True])
@pytest.mark.parametrize("visible", [None, "12,8,15,9"])
def test_dcmi_version_and_visible_device_mapping(monkeypatch, v2, visible):
    calls = []

    def integer(name, value):
        def call(index, output):
            calls.append((name, index))
            output._obj.value = value
            return 0
        return Function(call)

    def pcie(*args):
        calls.append(("pcie", *args[:-1]))
        info = args[-1]._obj
        info.domain, info.bus, info.slot, info.function = 0, 0x9F, 0, 0
        return 0

    def card(card, device, logical):
        calls.append(("card", logical.value))
        card._obj.value, device._obj.value = 4, 0
        return 0

    lib = SimpleNamespace(
        dcmi_init=Function(lambda: -8255 if v2 else 0),
        dcmiv2_init=Function(lambda: 0),
        dcmi_get_device_phyid_from_logicid=integer("legacy_phy", 8),
        dcmi_get_device_logicid_from_phyid=integer("legacy_logic", 18),
        dcmi_get_card_id_device_id_from_logicid=Function(card),
        dcmi_get_device_pcie_info_v2=Function(pcie),
        dcmiv2_get_chip_phy_id_by_dev_id=integer("v2_phy", 8),
        dcmiv2_get_dev_id_by_chip_phy_id=integer("v2_logic", 18),
        dcmiv2_get_device_pcie_info=Function(pcie),
    )
    monkeypatch.setattr(affinity.ctypes, "CDLL", lambda path: lib)
    monkeypatch.delenv("ASCEND_VISIBLE_DEVICES", raising=False)
    if visible is None:
        monkeypatch.delenv("ASCEND_RT_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible)
    values = {"vendor": "0x19e5", "numa_node": "1", "local_cpulist": "64-65,192-193"}

    def read(path, *args, **kwargs):
        assert str(path.parent) == "/sys/bus/pci/devices/0000:9f:00.0"
        return values[path.name]

    monkeypatch.setattr(affinity.Path, "read_text", read)
    assert affinity._npu_affinity(1) == (8, 1, {64, 65, 192, 193})
    prefix = "v2" if v2 else "legacy"
    assert (prefix + "_logic", 8) in calls
    assert ((prefix + "_phy", 1) in calls) is (visible is None)
    assert not any(name.startswith("legacy" if v2 else "v2") for name, *_ in calls)


@pytest.mark.parametrize("has_v2", [False, True])
def test_failed_dcmi_versions_do_not_guess_topology(monkeypatch, has_v2):
    lib = SimpleNamespace(dcmi_init=Function(lambda: -8255))
    if has_v2:
        lib.dcmiv2_init = Function(lambda: -1)
    monkeypatch.setattr(affinity.ctypes, "CDLL", lambda path: lib)
    with pytest.raises(RuntimeError, match="DCMI initialization failed"):
        affinity._npu_affinity(0)
