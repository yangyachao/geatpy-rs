import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.parity.fuzz import fz


def test_random_samples_with_fixed_nan_and_large_mean():
    samples = iter([0, 1, 3])
    record = fz.run(lambda: np.array([np.nan, np.inf, 1e12 + next(samples)]), 2)
    assert record['varied']
    stat = record['stat']['']
    assert stat['special'] == [[2, 0, 0], [0, 2, 0], [0, 0, 0]]
    assert stat['mean'][2] == 1e12 + 2
    assert stat['var'][2] == 1


def test_cmp_reports_infinity_and_error_changes(tmp_path):
    files = []
    for side, value, error in [('original', 1., 'ValueError: bad'), ('rust', np.inf, 'TypeError: bad')]:
        path = tmp_path / side
        rows = [dict(name='numeric', status='ok', value=dict(t='float', v=value)),
                dict(name='input', status='error', error=error)]
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        files.append(str(path))
    cmp_script = Path(fz.__file__).with_name('cmp.py')
    report = subprocess.check_output([sys.executable, str(cmp_script)] + files).decode()
    assert 'status (1)' in report and 'value (1)' in report
