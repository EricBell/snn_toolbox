# coding=utf-8

"""Test snntoolbox GUI."""

import os
import shutil
import time
import pytest


@pytest.mark.skipif(not os.environ.get('DISPLAY'), reason="No DISPLAY available")
def test_gui(_config):
    from snntoolbox.bin.gui.gui import tk, SNNToolboxGUI

    root = tk.Tk()
    app = SNNToolboxGUI(root, _config)
    root.update_idletasks()
    root.update()
    time.sleep(0.1)
    app.quit_toolbox()
    shutil.rmtree(app.default_path_to_pref)

    assert True
