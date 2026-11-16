# Vendored / pinned dependencies

This repository vendors several upstream checkouts. Their original `.git`
directories have been renamed to `.git.vendored` locally (restore with
`mv .git.vendored .git`), so the files below are ordinary files in this repo.

## Included in this repo

| Path | Upstream | Base commit | Notes |
| --- | --- | --- | --- |
| `openpi/` | https://github.com/Physical-Intelligence/openpi | `215abfb217dbac7d5f1273282331b9b1866c0479` | clean base; local run scripts are untracked additions |
| `ContextFlow/` | https://github.com/dingjiansw101/ContextFlow | `c6dee891f0afd520aec5a8869891e9b06b99f856` | contains local modifications (e.g. `src/openpi/training/*`, `examples/libero/*`, `run_*.sh`) |
| `bpp/src/deps/icrt/` | no remote recorded | `e0c95887d577c95ef9d0eef5ee23960043d6cb5e` | vendored as-is |

## Excluded from this repo (re-clone to restore)

These are large public upstream checkouts (~400 MB each, mostly sim assets)
and are gitignored. Re-clone at the recorded commit to restore.

| Path | Upstream | Base commit |
| --- | --- | --- |
| `bpp/src/deps/LIBERO/` | https://github.com/Lifelong-Robot-Learning/LIBERO | `8f1084e3132a39270c3a13ebe37270a43ece2a01` |
| `ContextFlow/third_party/libero/` | https://github.com/Lifelong-Robot-Learning/LIBERO | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` (ContextFlow submodule) |
