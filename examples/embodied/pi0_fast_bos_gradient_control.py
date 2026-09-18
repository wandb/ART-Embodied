"""Match both seed populations across the sampler/SFT BOS gradient comparison."""

import json
import pickle


def aligned_plan(plan, source):
    verification = json.loads((source / "remote-verification.json").read_text())
    if not verification.get("history_verified") or not verification.get(
        "artifact_verified"
    ):
        raise ValueError("Unverified BOS rollout source")
    original = json.loads((source / "plan.json").read_text())
    if len(original["groups"]) != len(plan["groups"]):
        raise ValueError("BOS reset-group population changed")
    updated = []
    for row, reference in zip(plan["groups"], original["groups"], strict=True):
        for key in (
            "group_index",
            "initial_observation_sha256",
            "policy_seeds",
            "new_policy_seeds",
            "source",
        ):
            if row[key] != reference[key]:
                raise ValueError(f"BOS baseline differs: {key}")
        gid = row["group_index"]
        with (source / f"changed-{gid}.pkl").open("rb") as stream:
            trajectories = pickle.load(stream)
        if [t.metadata["policy_seed"] for t in trajectories] != reference[
            "new_policy_seeds"
        ]:
            raise ValueError("Retained BOS seeds differ")
        if set(row["policy_seeds"]) & set(reference["new_policy_seeds"]):
            raise ValueError("BOS pair does not use independent seeds")
        # Retained aligned trajectories used native-side B seeds. Generate the
        # missing aligned A seeds, not a third unrelated population.
        updated.append(
            row
            | {
                "bos_retained_policy_seeds": reference["new_policy_seeds"],
                "new_policy_seeds": row["policy_seeds"],
            }
        )
    return plan | {
        "groups": updated,
        "bos_source": str(source.resolve()),
        "scope": "Independent same-reset GRPO gradients under SFT BOS boundary. Matches both original/repeated seed populations of the native BOS diagnostic, in reverse order. No optimizer updates or sealed test.",
        "generation_and_scoring_bos_boundary": "sft",
    }
