"""R23H: 16 one-expert swaps around the frozen all8 96-expert mask.
Bidirectional fixed-Fit Prefill screen only; no Decode, Fresh, Holdout or model build."""
import copy, fcntl, hashlib, importlib.util, json, math, os, pathlib, sys, time, traceback

P = pathlib.Path
sys.path.insert(0, str(P.home() / "mt-pruning-platform"))
R = P.home() / ".cache/mt-pruning"
RR = R / "r23-results"
REPO = P.home() / "mt-pruning-platform"
USB_PRIMARY = P("/mnt/mt-storage/mt-pruning/evidence/r23g-recovery-38ed4798")
USB_FALLBACK = R / "r23g-recovery-ssd-38ed4798"
USB = USB_PRIMARY if (USB_PRIMARY / "expanded.json").is_file() else USB_FALLBACK
PREVIOUS = RR / "r23h-factorial-prefill-v5"
PREVIOUS_RUN = R / "r23h-factorial-run-v5"
TEACHER = "0d1fb958cefa104dfc2f187cbb74da2ed33735437160248f7bc05261b94bbc94"
PREVIOUS_PLAN_SHA = "580abfa0bbd3cf4fa9d88f9381e61db7615c8a739d90f840bc0e7de9fd8423a2"
PREVIOUS_REPORT_SHA = "e906a1e16b38902ea3282afc44b9afcc190b1673b3ffb9cdfc0d01d456b77fa1"
EXPANDED_BLOB = "a1047d1a7c5624ed0e3a868bd5d76eef4c754915"
CANDIDATE_BLOB = "e60510be3de1fef61f8f455b451990f6a8f028a7ae1e65e9a559f96aa18650c1"
CORPUS_SHA = "9caa788d82dea867889bdac6e62c0f1a3ff1b6d4b3ccd7cad19ba74c5a98a0c4"
ORIGINAL_BLOB = "d8bcafe0fe1d8c6bacd0a425dfde57890b18f1ed"
ORIGINAL_MASK_SHA = "23d615b5f6e33af73c008729a34b3a0dbe9713508a5fd299b72ee9559180e8e7"
R23G_MASK_SHA = "77f97b8990835dc48738680b3ec78bdce7d5088a0d1b4a41d2aef886891d42aa"
ALL8_MASK_SHA = "1f7402ae0fb540c86ace21914b45ea0c42a113e5f38072a2d3f86795ee8ad70b"
INVENTORY_BLOB = "65e57a3add8cff36979b06928af22bcdcefe3ce6"
REVERT10_BLOB = "faa4d6146a3df4192e913e2702a926afa3fb095d"
REVERT15_BLOB = "044a6f395d779489462df00a664cf7876732ab06"
PINS = {
    "driver": "0735238e15f13fd75e3f6939fc31aa834d8b5854",
    "collector": "02daf65bb7fe0526b2a700d17013054f7b0eea46",
}
LAYERS = (10, 15, 40, 42)
EXPECTED_CHANGES = {
    10: ([42, 95], [78, 100]),
    15: ([6, 64], [4, 71]),
    40: ([13, 84], [58, 62]),
    42: ([47, 120], [63, 126]),
}
BASELINES = ["original96", "all8", "r23g"]
CANDIDATE_ORDER = [
    f"l{layer}_remove{remove}_add{restore}"
    for layer in LAYERS
    for remove in EXPECTED_CHANGES[layer][1]
    for restore in EXPECTED_CHANGES[layer][0]
]
ALL_NAMES = BASELINES + CANDIDATE_ORDER


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def blob(raw):
    return hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def pin(path, expected, git=False):
    need(path.is_file() and not path.is_symlink(), "regular " + str(path))
    raw = path.read_bytes()
    need((blob if git else sha)(raw) == expected, "pin " + str(path))
    return raw


def load(directory, name):
    path = directory / (name + ".py")
    pin(path, PINS[name], True)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def publish(path, value):
    raw = canon(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return sha(raw)


def event(status, **fields):
    print(json.dumps(dict(status=status, **fields), sort_keys=True), flush=True)


def check_mask(mask, label):
    need(set(mask) == {str(i) for i in range(1, 48)}, label + " layer inventory")
    for layer, experts in mask.items():
        need(type(experts) is list and len(experts) == 96, label + " expert count " + layer)
        need(experts == sorted(set(experts))
             and all(type(x) is int and 0 <= x < 128 for x in experts),
             label + " expert order/range " + layer)


def metric(teacher, masked):
    a = teacher.double()
    b = masked.double()
    p = a.log_softmax(-1)
    q = b.log_softmax(-1)
    kl = float((p.exp() * (p - q)).sum(dim=-1).mean())
    nmse = float((a - b).square().sum() / a.square().sum())
    need(math.isfinite(kl) and kl >= -1e-12 and math.isfinite(nmse), "finite metric")
    return dict(kl=max(0.0, kl), nmse=nmse,
                argmax_mismatch=int(a[-1].argmax() != b[-1].argmax()))


def aggregate(group, name):
    need(len(group) in (8, 16), "aggregate sample count " + name)
    return {
        "kl": math.fsum(x["metrics"][name]["kl"] for x in group) / len(group),
        "nmse": math.fsum(x["metrics"][name]["nmse"] for x in group) / len(group),
        "argmax_mismatch": sum(x["metrics"][name]["argmax_mismatch"] for x in group),
    }


def build_masks(original, revert10, revert15, inventory):
    need(revert10["base_mask_sha256"] == ALL8_MASK_SHA
         and revert15["base_mask_sha256"] == ALL8_MASK_SHA, "all8 source base pins")
    need(revert10["changed_layers"] == [15, 40, 42]
         and revert15["changed_layers"] == [10, 40, 42], "all8 reversion source scopes")
    all8 = copy.deepcopy(revert10["mask"])
    all8["10"] = copy.deepcopy(revert15["mask"]["10"])
    check_mask(original, "original96")
    check_mask(all8, "all8")
    need(sha(canon(all8)) == ALL8_MASK_SHA, "reconstructed all8 mask hash")
    changed = [int(layer) for layer in all8 if all8[layer] != original[layer]]
    need(changed == list(LAYERS), "all8 changed-layer inventory")
    candidates = {}
    for layer in LAYERS:
        old = set(original[str(layer)])
        new = set(all8[str(layer)])
        removed_from_original = sorted(old - new)
        added_to_all8 = sorted(new - old)
        expected_removed, expected_added = EXPECTED_CHANGES[layer]
        need(removed_from_original == expected_removed
             and added_to_all8 == expected_added, "exact all8 layer delta " + str(layer))
        for remove in added_to_all8:
            for restore in removed_from_original:
                name = f"l{layer}_remove{remove}_add{restore}"
                mask = copy.deepcopy(all8)
                mask[str(layer)] = sorted((set(mask[str(layer)]) - {remove}) | {restore})
                check_mask(mask, name)
                need(mask[str(layer)] == sorted(new - {remove} | {restore}),
                     name + " exact layer swap")
                changed_layers = [i for i in range(1, 48)
                                  if mask[str(i)] != all8[str(i)]]
                need(changed_layers == [layer], name + " one-layer scope")
                need(len(set(mask[str(layer)]) - set(all8[str(layer)])) == 1
                     and len(set(all8[str(layer)]) - set(mask[str(layer)])) == 1,
                     name + " exact one-expert swap")
                candidates[name] = dict(
                    name=name, layer=layer, removed_expert=remove,
                    restored_expert=restore, mask_sha256=sha(canon(mask)), kept=mask
                )
    need(list(candidates) == CANDIDATE_ORDER and len(candidates) == 16,
         "complete 16-candidate inventory")
    need(inventory["status"] == "R23H_ALL8_ONE_SWAP_CANDIDATE_INVENTORY_FROZEN"
         and inventory["base_mask_sha256"] == ALL8_MASK_SHA
         and inventory["original96_mask_sha256"] == ORIGINAL_MASK_SHA
         and inventory["layers"] == list(LAYERS), "frozen candidate inventory header")
    frozen = inventory["candidates"]
    need(len(frozen) == 16, "frozen candidate inventory count")
    for expected, actual in zip(frozen, candidates.values()):
        need(expected["name"] == actual["name"]
             and expected["layer"] == actual["layer"]
             and expected["removed_expert"] == actual["removed_expert"]
             and expected["restored_expert"] == actual["restored_expert"]
             and expected["mask_sha256"] == actual["mask_sha256"],
             "frozen candidate inventory mismatch " + actual["name"])
    return all8, candidates


def prepare(directory):
    expanded = json.loads(pin(USB / "expanded.json", EXPANDED_BLOB, True))
    rows = [row for row in expanded["fit"] if row["direction"] in ("zh-my", "my-zh")]
    need(len(rows) == 16 and all(
        sum(row["direction"] == direction for row in rows) == 8
        for direction in ("zh-my", "my-zh")), "balanced 16-row Fit set")
    original_doc = json.loads(pin(directory / "original.json", ORIGINAL_BLOB, True))
    original = original_doc["candidates"]["96"]["kept"]
    r23g = json.loads(pin(USB / "candidate.json", CANDIDATE_BLOB))["kept"]
    need(sha(canon(original)) == ORIGINAL_MASK_SHA
         and sha(canon(r23g)) == R23G_MASK_SHA, "original/R23G mask pins")
    revert10 = json.loads(pin(directory / "all8_revert10.json",
                              REVERT10_BLOB, True))
    revert15 = json.loads(pin(directory / "all8_revert15.json",
                              REVERT15_BLOB, True))
    inventory = json.loads(pin(directory / "candidate-inventory.json", INVENTORY_BLOB, True))
    all8, candidates = build_masks(original, revert10, revert15, inventory)
    check_mask(r23g, "r23g")

    previous_plan = json.loads(pin(PREVIOUS / "plan.json", PREVIOUS_PLAN_SHA))
    previous_report = json.loads(pin(PREVIOUS / "report.json", PREVIOUS_REPORT_SHA))
    need(previous_plan["teacher"] == TEACHER
         and previous_plan["status"] == "R23H_FACTORIAL_FIT_PREFILL_SCREEN_ONLY",
         "previous factorial plan")
    need(previous_report["status"] == "R23H_FACTORIAL_BIDIRECTIONAL_FIT_PREFILL_SCREEN_COMPLETE_NOT_FRESH_ADMISSION",
         "previous factorial report")
    need(previous_plan["ids"] == [row["id"] for row in rows], "previous Fit row identity")
    previous_events = [json.loads(line) for line in
                       (PREVIOUS_RUN / "run.log").read_bytes().splitlines()
                       if line.startswith(b"{")]
    captures = [item for item in previous_events
                if item.get("status") == "R23H_FACTORIAL_PREFILL_CAPTURED"]
    need(len(captures) == 16 and [item["index"] for item in captures] == list(range(16)),
         "previous receipt event inventory")
    previous = []
    for index, event_row in enumerate(captures):
        raw = pin(PREVIOUS / f"{index:02d}.json", event_row["sha256"])
        item = json.loads(raw)
        need(item["id"] == rows[index]["id"], "previous receipt ID")
        need(item["teacher_sha256"] and len(item["teacher_sha256"]) == 64,
             "previous teacher logits hash")
        previous.append(item)

    corpus = pin(R / "run-d01b-r19/corpus/corpus.jsonl", CORPUS_SHA)
    context = {
        item["id"]: item for item in map(json.loads, corpus.splitlines())
        if item["split"] == "calibration" and item["id"] in {row["id"] for row in rows}
    }
    need(len(context) == 16, "calibration corpus coverage")
    for row in rows:
        need(sha(canon(context[row["id"]]["prompt_ids"])) == row["prompt_ids_sha256"],
             "prompt identity " + row["id"])

    baselines = {
        "original96": original,
        "all8": all8,
        "r23g": r23g,
    }
    plan = dict(
        status="R23H_ALL8_ONE_SWAP_BIDIRECTIONAL_PREFILL_SCREEN_ONLY",
        teacher=TEACHER,
        source_previous_plan_sha256=PREVIOUS_PLAN_SHA,
        source_previous_report_sha256=PREVIOUS_REPORT_SHA,
        source_original96_mask_sha256=ORIGINAL_MASK_SHA,
        source_r23g_mask_sha256=R23G_MASK_SHA,
        source_all8_mask_sha256=ALL8_MASK_SHA,
        source_candidate_inventory_blob=INVENTORY_BLOB,
        source_all8_revert10_blob=REVERT10_BLOB,
        source_all8_revert15_blob=REVERT15_BLOB,
        ids=[row["id"] for row in rows],
        directions={"zh-my": 8, "my-zh": 8},
        changed_layers=list(LAYERS),
        baselines=[
            {"name": name, "mask_sha256": sha(canon(mask))}
            for name, mask in baselines.items()
        ],
        candidates=[candidates[name] for name in CANDIDATE_ORDER],
        holdout="FORBIDDEN",
        gate={
            "both_directions": "candidate KL, NMSE and argmax mismatch must be no worse than both original96 and all8",
            "advance": "at least one metric strictly improves over all8; otherwise stop before Decode",
        },
    )
    return rows, context, previous, baselines, candidates, plan


def run(directory):
    rows, context, previous, baselines, candidates, plan = prepare(directory)
    lock_path = R / "r23-admissions/batch.lock"
    with lock_path.open("a+") as lock:
        deadline = time.monotonic() + 7200
        while True:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                need(time.monotonic() < deadline, "exclusive batch lock wait expired")
                time.sleep(10)
        event("R23H_ALL8_ONE_SWAP_LOCK_ACQUIRED")
        destination = RR / "r23h-all8-one-swap-prefill-v1"
        destination.mkdir(mode=0o700)
        plan_sha = publish(destination / "plan.json", plan)
        event("R23H_ALL8_ONE_SWAP_PLAN_FROZEN", sha256=plan_sha,
              candidate_count=len(CANDIDATE_ORDER), changed_layers=list(LAYERS))

        driver = load(directory, "driver")
        collector = load(directory, "collector")
        collector.verify_code()
        preflight = driver.preflight()
        need(not preflight["blockers"], str(preflight))
        event("R23H_ALL8_ONE_SWAP_RESOURCE_PREFLIGHT",
              preflight_status=preflight["status"], blockers=preflight["blockers"],
              gpu_pids=preflight["gpu_pids"], model_processes=preflight["model_processes"],
              memory=preflight["memory"])
        collector.torch.set_num_threads(8)
        collector.torch.set_num_interop_threads(1)
        identity = driver.checkpoint_identity(driver.SOURCE, cache_dir=None,
                                              fresh=True, exact_inventory=True)
        need(identity["teacher"] == TEACHER and identity["num_experts"] == 128,
             "exact teacher identity")
        event("R23H_ALL8_ONE_SWAP_TEACHER_IDENTITY", teacher=identity["teacher"],
              num_experts=identity["num_experts"], identity=identity)
        engine = collector.engine_module.Engine(
            collector.engine_module.Checkpoint(driver.SOURCE), threads=8)
        event("R23H_ALL8_ONE_SWAP_TEACHER_READY", teacher=TEACHER)

        masks = dict(baselines)
        masks.update({name: candidates[name]["kept"] for name in CANDIDATE_ORDER})

        def prefill(prompt, mask):
            try:
                engine.set_mask(None if mask is None else
                                {int(layer): experts for layer, experts in mask.items()})
                with collector.torch.inference_mode():
                    logits = engine.forward(prompt, engine.new_cache(len(prompt) + 9), "prefill")
                return collector.logits_sha(logits), logits[
                    [len(prompt) // 2 - 1, len(prompt) - 1]
                ].clone()
            finally:
                engine.set_mask(None)

        captures = []
        for index, row in enumerate(rows):
            need(not driver.run(["nvidia-smi", "--query-compute-apps=pid",
                                 "--format=csv,noheader,nounits"]), "GPU busy")
            prompt = context[row["id"]]["prompt_ids"]
            old = previous[index]
            teacher_hash, teacher_logits = prefill(prompt, None)
            need(teacher_hash == old["teacher_sha256"], "historical teacher hash " + row["id"])
            event("R23H_ALL8_ONE_SWAP_TEACHER_HASH_CHECK", index=index, id=row["id"],
                  actual=teacher_hash, expected=old["teacher_sha256"], match=True)
            receipt = dict(id=row["id"], direction=row["direction"], index=index,
                           plan_sha256=plan_sha, teacher_sha256=teacher_hash, metrics={})
            for name, mask in masks.items():
                logits_hash, masked_logits = prefill(prompt, mask)
                measured = metric(teacher_logits, masked_logits)
                if name in ("original96", "r23g"):
                    historical = old["metrics"][name]
                    need(logits_hash == historical["logits_sha256"]
                         and abs(measured["kl"] - historical["kl"]) < 1e-12
                         and abs(measured["nmse"] - historical["nmse"]) < 1e-12
                         and measured["argmax_mismatch"] == historical["argmax_mismatch"],
                         "historical exact baseline replay " + name + "/" + row["id"])
                receipt["metrics"][name] = dict(
                    measured, logits_sha256=logits_hash, mask_sha256=sha(canon(mask)))
            restored_hash, _ = prefill(prompt, None)
            need(restored_hash == teacher_hash, "teacher restoration " + row["id"])
            receipt_hash = publish(destination / f"{index:02d}.json", receipt)
            captures.append(receipt)
            event("R23H_ALL8_ONE_SWAP_PREFILL_CAPTURED", index=index, id=row["id"],
                  sha256=receipt_hash,
                  kl={name: values["kl"] for name, values in receipt["metrics"].items()})

        overall = {name: aggregate(captures, name) for name in ALL_NAMES}
        by_direction = {
            direction: {
                name: aggregate([item for item in captures if item["direction"] == direction], name)
                for name in ALL_NAMES
            }
            for direction in ("zh-my", "my-zh")
        }
        eligible = []
        for name in CANDIDATE_ORDER:
            strict = False
            passes = True
            for direction in ("zh-my", "my-zh"):
                candidate = by_direction[direction][name]
                original = by_direction[direction]["original96"]
                parent = by_direction[direction]["all8"]
                for field in ("kl", "nmse", "argmax_mismatch"):
                    passes = passes and candidate[field] <= original[field] + 1e-12
                    passes = passes and candidate[field] <= parent[field] + 1e-12
                    strict = strict or candidate[field] < parent[field] - 1e-12
            if passes and strict:
                eligible.append(name)

        report = dict(
            status="R23H_ALL8_ONE_SWAP_BIDIRECTIONAL_PREFILL_COMPLETE_NOT_FRESH_ADMISSION",
            plan_sha256=plan_sha,
            samples=16,
            metrics=overall,
            by_direction=by_direction,
            eligible_for_primary_fit_only=eligible,
            selected=eligible[0] if eligible else None,
            limits="Fixed Fit Prefill only for zh-my/my-zh; no Decode, Fresh, Holdout, free-running translation, quality admission, or physical model build.",
        )
        report_sha = publish(destination / "report.json", report)
        event(report["status"], report_sha256=report_sha,
              **{key: value for key, value in report.items() if key != "status"})


if __name__ == "__main__":
    try:
        run(P(sys.argv[1]))
    except Exception as exc:
        event("R23H_ALL8_ONE_SWAP_FATAL", error_type=type(exc).__name__,
              error=str(exc), traceback=traceback.format_exc())
        raise
