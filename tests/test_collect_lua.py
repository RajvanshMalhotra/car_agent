from collect.lua import STATE_GLOBAL, drain_source, install_source, uninstall_source
from collect.schema import MEASURED


def capture_block(source: str) -> str:
    """Just the record literal. `S.t` also appears in the setup, so scope it."""
    return source.split("local function capture()")[1].split("S.capture")[0]


def test_install_captures_every_measured_channel_in_schema_order():
    block = capture_block(install_source(0.01))
    positions = [block.index(c.lua) for c in MEASURED]
    assert positions == sorted(positions)


def test_install_captures_every_measured_channel_exactly_once():
    block = capture_block(install_source(0.01))
    assert block.count("\n    ") == len(MEASURED)


def test_install_sets_the_requested_interval():
    assert "0.01" in install_source(0.01)
    assert "0.05" in install_source(0.05)


def test_install_binds_every_local_the_expressions_use():
    source = install_source(0.01)
    for local in ("local p =", "local vel =", "local d =", "local e =",
                  "local eng =", "local w ="):
        assert local in source


def test_install_builds_the_wheel_index_map():
    source = install_source(0.01)
    assert "wheelCount" in source and "S.wi" in source


def test_install_computes_mass_before_the_hook_not_inside_it():
    # 783 nodes is far too expensive per sample.
    before_hook = install_source(0.01).split("function onPhysicsStep")[0]
    assert "getNodeMass" in before_hook


def test_install_keeps_the_hook_it_displaces():
    # `onPhysicsStep` is already a global in the vehicle VM, so defining one
    # replaces whatever was there. That is what stopped the AI driving.
    source = install_source(0.01)
    assert "rawget(_G, 'onPhysicsStep')" in source
    assert "pcall(S.previousHook, dt)" in source


def test_a_reinstall_does_not_chain_to_its_own_hook():
    # Chaining to ourselves would recurse until the VM died.
    source = install_source(0.01)
    assert "previousHook = old.previousHook" in source


def test_the_displaced_hook_runs_before_any_sampling():
    body = install_source(0.01).split("function onPhysicsStep")[1]
    assert body.index("previousHook") < body.index("S.t = S.t + dt")


def test_uninstall_puts_the_original_hook_back():
    source = uninstall_source()
    assert "_G.onPhysicsStep = previousHook" in source


def test_uninstall_leaves_the_hook_enabled_when_something_else_wanted_it():
    source = uninstall_source()
    assert "if previousHook == nil then" in source


def test_install_enables_the_physics_hook():
    assert "enablePhysicsStepHook(true)" in install_source(0.01)


def test_install_self_tests_so_a_bad_field_name_fails_now():
    source = install_source(0.01)
    assert "pcall(capture)" in source
    assert "ok = false" in source


def test_install_caps_the_buffer_and_counts_drops():
    assert "S.dropped" in install_source(0.01)


def test_drain_empties_the_buffer_and_encodes_json():
    source = drain_source()
    assert STATE_GLOBAL in source and "jsonEncode" in source
    assert "S.n = 0" in source


def test_drain_reports_when_the_sampler_is_not_installed():
    assert "not_installed" in drain_source()


def test_uninstall_clears_the_sampler_state():
    source = uninstall_source()
    assert "enablePhysicsStepHook(false)" in source
    assert f"_G.{STATE_GLOBAL} = nil" in source
