from framework.integrations import arc_gen_adapter


def test_arc_gen_adapter_is_available_flag():
    # This test does not assume ARC-GEN is actually usable; it just
    # ensures that the adapter module imports and exposes the flag.
    available = arc_gen_adapter.is_available()
    assert isinstance(available, bool)

