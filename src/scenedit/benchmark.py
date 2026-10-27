"""Validation contracts for paired background editing, independent of the GPU runner."""

import math


def planned_jobs(config):
    """Expand the whole frozen grid; reject aliases that could overwrite a task."""
    cases, scenes, seeds = config["cases"], config["scenes"], config["seeds"]
    if not cases or len(set(cases)) != len(cases) or not scenes:
        raise ValueError("Empty or duplicate cases/scenes")
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Empty or duplicate seeds")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("Seeds must be nonnegative integers")
    if config["radii"] != [0, 3] or config["scale"] != 1:
        raise ValueError("Frozen comparison requires radii 0/3 and scale 1")
    for label in [*cases, *scenes]:
        if not label or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789-" for c in label
        ):
            raise ValueError("Unsafe task label")
    if any(
        not isinstance(prompt, str) or not prompt.strip() for prompt in scenes.values()
    ):
        raise ValueError("Empty scene prompt")
    return [
        dict(
            case=case,
            scene=scene,
            seed=seed,
            radius=radius,
            name=f"{case}-{scene}-seed{seed}-radius{radius}",
        )
        for case in cases
        for scene in scenes
        for seed in seeds
        for radius in config["radii"]
    ]


def paired_summary(records, config):
    """Do not silently drop missing pairs or count frames as independent videos."""
    jobs = planned_jobs(config)
    names = [row["name"] for row in records]
    if len(names) != len(set(names)) or set(names) != {j["name"] for j in jobs}:
        raise ValueError("Missing, duplicated or unexpected outputs")
    by_name = {row["name"]: row for row in records}
    pairs = []
    for job in jobs[::2]:
        root = job["name"].rsplit("-radius", 1)[0]
        base, candidate = (by_name[root + "-radius" + str(r)] for r in (0, 3))
        for field in ("case", "scene", "seed"):
            if base[field] != job[field] or candidate[field] != job[field]:
                raise ValueError("Task metadata differs from frozen grid")
        for field in ("source_sha256", "reference_sha256", "prompt"):
            if base[field] != candidate[field]:
                raise ValueError("Unmatched paired inputs: " + field)
        values = {}
        for key in (
            "foreground_pixel_mae",
            "boundary_pixel_mae",
            "far_background_pixel_change",
        ):
            a, b = base["metrics"][key], candidate["metrics"][key]
            if any(not math.isfinite(v) or not 0 <= v <= 1 for v in (a, b)):
                raise ValueError("Invalid metric")
            values[key] = dict(
                baseline=a,
                candidate=b,
                delta=b - a,
                reduction_percent=100 * (a - b) / a if a else None,
            )
        pairs.append(
            dict(case=job["case"], scene=job["scene"], seed=job["seed"], metrics=values)
        )
    return dict(
        pairs=pairs,
        source_sequences=len(config["cases"]),
        paired_tasks=len(pairs),
        generated_videos=len(records),
        interpretation="Pixel diagnostics only. Background change is not instruction adherence. No frame-wise significance test; no perceptual-quality claim without human review.",
    )
