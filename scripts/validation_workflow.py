"""Saved Windows workflow: prepare, CUDA paired validation, then two blind packages."""

import datetime
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from scenedit.common import read, save, sha
from scenedit.benchmark import paired_summary


def main():
    config = ROOT / "configs/independent-validation.json"
    out = Path(read(config)["output"])
    out.mkdir(parents=True, exist_ok=True)
    if (out / "active.lock").exists():
        raise RuntimeError(
            "An experiment lock exists. Check its PID/status before starting another queue."
        )
    if (out / "status.json").exists() and read(out / "status.json")[
        "status"
    ] == "complete":
        if read(out / "protocol.json")["config"] != read(config):
            raise ValueError(
                "Existing results use a different protocol; choose a new output directory"
            )
        rows = read(out / "results.json")
        paired_summary(rows, read(config))
        for row in rows:
            if sha(Path(row["run"]) / "edited.mp4") != row["output_sha256"]:
                raise ValueError("Existing generated video changed")
        for reviewer in (1, 2):
            subprocess.run(
                [
                    sys.executable,
                    "-X",
                    "utf8",
                    "scripts/check_blind_review.py",
                    str(out / f"reviewer-{reviewer}"),
                ],
                cwd=ROOT,
                check=True,
            )
        print("EXISTING_VALIDATION_VERIFIED_NO_GENERATION", out, flush=True)
        return
    state = dict(
        status="running", started=datetime.datetime.now().astimezone().isoformat()
    )
    save(out / "workflow-status.json", state)
    try:
        commands = [
            ["scripts/prepare_independent_validation.py", "--config", str(config)],
            ["scripts/run_independent_validation.py", "--config", str(config)],
        ]
        for command in commands:
            state["stage"] = command[0]
            save(out / "workflow-status.json", state)
            subprocess.run(
                [sys.executable, "-u", "-X", "utf8", *command], cwd=ROOT, check=True
            )
        for reviewer, seed in ((1, 1701), (2, 2903)):
            target = out / f"reviewer-{reviewer}"
            if target.exists():
                raise FileExistsError(
                    "Existing review package; verify before resuming: " + str(target)
                )
            subprocess.run(
                [
                    sys.executable,
                    "-u",
                    "-X",
                    "utf8",
                    "scripts/build_blind_review.py",
                    "--results",
                    str(out / "results.json"),
                    "--output",
                    str(target),
                    "--seed",
                    str(seed),
                ],
                cwd=ROOT,
                check=True,
            )
        state.update(
            status="awaiting_human_review",
            finished=datetime.datetime.now().astimezone().isoformat(),
        )
        save(out / "workflow-status.json", state)
        print("WORKFLOW_READY_FOR_HUMAN_REVIEW", out, flush=True)
    except Exception as error:
        state.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        save(out / "workflow-status.json", state)
        raise


if __name__ == "__main__":
    main()
