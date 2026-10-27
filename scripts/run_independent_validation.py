"""Run the frozen paired grid in isolated CUDA subprocesses with auditable resume."""

import argparse
import datetime
import os
import platform
import shutil
import subprocess
import sys
import traceback
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scenedit.common import read, save, sha, snapshot
from scenedit.infer import verify_inputs, gpu_guard
from scenedit.preservation import prepare_variant, evaluate_fixed
from scenedit.benchmark import planned_jobs, paired_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs/independent-validation.json"
    )
    args = parser.parse_args()
    config = read(args.config)
    jobs = planned_jobs(config)
    gpu_guard()
    out = Path(config["output"])
    out.mkdir(parents=True, exist_ok=True)
    protocol = out / "protocol.json"
    # A changed source tree or input protocol requires a new experiment directory.
    source_files = [
        *ROOT.glob("src/scenedit/*.py"),
        *ROOT.glob("scripts/*.py"),
        *ROOT.glob("configs/*.json"),
    ]
    source_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in source_files}
    identity = dict(
        config=config,
        code_sha256=source_hashes,
        model_identity_sha256=sha(Path(config["model_dir"]) / "identity.json"),
    )
    if protocol.exists():
        if read(protocol) != identity:
            raise ValueError("Protocol/source/model changed; use a new run directory")
    else:
        save(protocol, identity)
        snapshot(out)
        for source in ROOT.glob("scripts/*.py"):
            target = out / "sources/scripts" / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        save(
            out / "environment.json",
            dict(
                python=sys.version,
                torch=torch.__version__,
                cuda=torch.version.cuda,
                platform=platform.platform(),
                gpu=torch.cuda.get_device_name(0),
                source_revision=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                ).strip(),
            ),
        )
    lock = out / "active.lock"
    # Exclusive creation prevents accidentally starting the same queue twice.
    with lock.open("x", encoding="utf-8") as handle:
        handle.write(str(os.getpid()))
    state = dict(
        status="running",
        started=datetime.datetime.now().astimezone().isoformat(),
        pid=os.getpid(),
        planned_runs=len(jobs),
        completed_runs=0,
    )
    save(out / "status.json", state)
    try:
        rows = []
        for job in jobs:
            base = (
                Path(config["data_root"]) / job["case"] / ("prepared-" + job["scene"])
            )
            original = verify_inputs(base)
            if original["job"]["source_group"] != job["case"]:
                raise ValueError("Wrong source group")
            prepared = out / "prepared" / job["name"]
            if not prepared.exists():
                meta = prepare_variant(base, prepared, job["radius"], config["scale"])
                meta["profile"]["seed"] = job["seed"]
                save(prepared / "prepared.json", meta)
            meta = verify_inputs(prepared)
            if (
                meta["profile"]["seed"] != job["seed"]
                or meta["variant"]["protection_radius_px"] != job["radius"]
                or meta["variant"]["base_prepared_sha256"]
                != sha(base / "prepared.json")
                or meta["prompt"] != original["prompt"]
            ):
                raise ValueError("Cached condition differs from frozen task")
            state.update(stage=job["name"])
            save(out / "status.json", state)
            # Finished attempts are revalidated; failed attempts remain available.
            attempts = out / "attempts" / job["name"]
            attempts.mkdir(parents=True, exist_ok=True)
            completed = [
                p
                for p in sorted(attempts.iterdir())
                if (p / "status.json").exists()
                and read(p / "status.json").get("status") == "complete"
            ]
            if len(completed) > 1:
                raise ValueError("Multiple successful attempts would permit selection")
            if completed:
                run = completed[0]
            else:
                run = attempts / (
                    "attempt-" + str(len(list(attempts.iterdir())) + 1).zfill(3)
                )
                command = [
                    sys.executable,
                    "-u",
                    "-X",
                    "utf8",
                    "-m",
                    "scenedit.infer",
                    "--prepared",
                    str(prepared),
                    "--model-dir",
                    config["model_dir"],
                    "--output-dir",
                    str(run),
                ]
                with (out / (job["name"] + "-" + run.name + ".log")).open(
                    "w", encoding="utf-8"
                ) as log:
                    subprocess.run(
                        command,
                        cwd=ROOT,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            status = read(run / "status.json")
            if (
                status["profile"] != meta["profile"]
                or not status["use_reference"]
                or status["prepared"] != str(prepared)
                or status["model_identity_sha256"] != identity["model_identity_sha256"]
            ):
                raise ValueError("Completed inference metadata mismatch")
            fixed = evaluate_fixed(base, run)
            save(out / (job["name"] + "-fixed.json"), fixed)
            rows.append(
                dict(
                    job,
                    prepared=str(base),
                    run=str(run),
                    metrics=fixed["mean"],
                    source_sha256=original["source_sha256"],
                    reference_sha256=sha(base / "reference.png"),
                    prompt=meta["prompt"],
                    output_sha256=sha(run / "edited.mp4"),
                    seconds=status["seconds"],
                    peak_allocated_gib=status["peak_allocated_gib"],
                )
            )
            save(out / "results.json", rows)
            state["completed_runs"] = len(rows)
            save(out / "status.json", state)
            print("INDEPENDENT_RUN_COMPLETE", job["name"], fixed["mean"], flush=True)
        save(out / "paired-summary.json", paired_summary(rows, config))
        state.update(
            status="complete",
            human_review="pending",
            finished=datetime.datetime.now().astimezone().isoformat(),
        )
        save(out / "status.json", state)
        print("INDEPENDENT_VALIDATION_COMPLETE", out, flush=True)
    except Exception as error:
        state.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save(out / "status.json", state)
        raise
    finally:
        lock.unlink()


if __name__ == "__main__":
    main()
