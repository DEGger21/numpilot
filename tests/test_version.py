import pytest

import numpilot
import numpilot.__main__ as cli


def test_cli_and_package_report_the_same_release_version(capsys):
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])

    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == "numpilot " + numpilot.__version__
