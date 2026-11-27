set -euo pipefail
D=$(mktemp -d /tmp/r23e-bundle-check.XXXXXX)
trap 'rm -rf "$D"' EXIT
BASE=https://raw.githubusercontent.com/kaicreator-mm/mt-pruning-platform
fetch() { curl -fsSL --retry 1 --max-time 30 "$BASE/$1" -o "$D/$2"; test "$(git hash-object "$D/$2")" = "$3"; }
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23e-restored-serial-propagation-launch.sh launch.sh bef95fcf65b4d1399513ce882092ca9f5ec3f91e
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23e-nonuniform-propagation-batch.py propagate.py 87b12c30ebf801d2e4571b6f0f8348edbb61ba13
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23e-propagated-state-witness.py prop.py 88468e8c2a5905628365da52e1575d5251516512
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23e-nonuniform-propagation-exact-replay.py prop-audit.py e9a7c03c65403ef19c268a5818b0673b6092d04d
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23e-nonuniform-propagation-plan.json prop-plan.json cc1bbf0bb6cf19be7c7c9c95d0b5d867dd54dcde
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23f-expanded-calibration-plan.json expanded.json a1047d1a7c5624ed0e3a868bd5d76eef4c754915
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23f-stable-uniform96-recovery.py stable.py e44f99773a01252bc4813cb1a7f27645f1bc90f5
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23f-stable96-proxy-batch-admission.py admit.py 2ae771db3406d8daf634aec8b3f2d11fd1742ce9
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23f-stable96-proxy-driver.py proxy.py 248737901db40b50efb0a26a9d5ef76d1928538b
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23-bounded-calibration-batch-driver-v2.py batch.py 08abbbfbd930e532d41602dd29a5c70f2bc474e3
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23-capture-receipt-verifier-v2.py verifier.py e56b25b3fd92ccbd91455628606925a02af49450
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23-real-pilot-driver.py driver.py 0735238e15f13fd75e3f6939fc31aa834d8b5854
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23-engine-observer-integration.py collector.py 02daf65bb7fe0526b2a700d17013054f7b0eea46
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23f-power-interruption-prefix.json witness.json 5c96c7644d623f73bc9a3590542e76ffa1b9474c
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23b-fit-score-masks.py masks.py 34a06701d47ce0e3ab888882a5b20c75745204d4
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23-real-sparse-pilot-driver.py pilot.py 19c8c851b1647cc0f495d6a7f1122219c42a0cdb
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23-sparse-counterfactual-witness.py sparse.py 1c38f6e928ff653fbbef584b298ff65e194900fe
fetch 286ba3286c68142be933eb7c5b70492464be448b/docs/research/r23-output-sensitive-local-witness.py witness.py b972e153a299b7fb39e580dfacfc08aaa8fe942c
fetch 3212303e1e02994a5124db8c85195b91325c7a7c/evidence/d01b-r19/selection.json original.json d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed
bash -n "$D/launch.sh"
python3 - "$D" <<'PY'
import pathlib,ast,sys
d=pathlib.Path(sys.argv[1])
for p in d.glob("*.py"): ast.parse(p.read_text(), filename=str(p))
print("EXACT_BUNDLE_SYNTAX_OK",len(list(d.iterdir())))
PY
python3 -B "$D/propagate.py" --selftest
python3 -B "$D/admit.py" --selftest
python3 -B "$D/prop-audit.py" --batch "$D/propagate.py" --selftest
echo BUNDLE_TEST_COMPLETE_NO_MODEL_LOAD