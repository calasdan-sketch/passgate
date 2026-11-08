# Publish checklist for passgate (for Dan; the upload commands below were never run)

## Version to release

**0.3.0.** PyPI had only 0.1.1 when last checked (2026-11-08). Version 0.2.0 was never published, so skip straight to 0.3.0. It is a normal "minor" bump: new commands, plus a few small behaviour changes listed under "Upgrading from 0.1.1" in the CHANGELOG (for example `compare` can now say `baseline-better`). The version number is already 0.3.0 everywhere; do not change it.

## Check first (about 10 minutes)

1. On GitHub, open the repository and make sure the `ci` workflow is green on Windows and Linux. It has never run on Windows. If it is red, do not publish; send me the log.
2. The author and copyright holder are now "Calas Automations Inc." (round 5). Check the first lines of `LICENSE` and `pyproject.toml` once. This becomes public and is hard to take back. No e-mail address is included; add one only if you want it shown on PyPI.
3. Confirm the repository is public (I could not check from here) (the README install line and the PyPI links point to `github.com/calasdan-sketch/passgate`).
4. Skim `CHANGELOG.md` and the top of `README.md` once.

## Commands (run on your own machine, in the repository folder)

```bash
git status                                   # should say: nothing to commit
python -m pip install --upgrade build twine
python -m build                              # makes dist/passgate-0.3.0*
python -m twine check --strict dist/*        # both lines should say PASSED
python scripts/smoke_wheel.py "dist/*.whl"   # last line: smoke: all checks passed
unzip -p dist/*.whl '*/METADATA' | grep -E '^(Author|Maintainer|Version)'   # Calas Automations Inc. and 0.3.0
```

Then upload. This is the step that costs nothing but cannot be undone (a version number can never be reused):

```bash
python -m twine upload dist/*                # asks for your PyPI token
```

After it goes up:

1. In a new empty folder, make a new virtual environment and run `pip install passgate`, then `passgate --version` (should say 0.3.0) and `passgate example rounds -o rounds.json` followed by `passgate gate rounds.json --threshold 0.9 --min-rounds 3` (should end with `verdict: FAIL`, exit code 1, as in the README).
2. Delete the grey "Release status" note near the top of `README.md` (it says 0.3.0 is not on PyPI yet), commit, and tag `v0.3.0`.

## Cost

PyPI is free. Nothing in this package calls a paid service.
