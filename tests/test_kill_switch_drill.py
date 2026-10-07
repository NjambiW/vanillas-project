from kill_switch_drill import run


def test_the_kill_switch_drill_passes_with_the_current_settings(capsys):
    assert run(write=False) == 0