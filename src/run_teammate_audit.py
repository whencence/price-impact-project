"""Run teammate processed-data and notebook audit."""

from pathlib import Path

import pandas as pd

from src.teammate_audit import find_project_root, run_full_teammate_audit


def main() -> None:
    """Run the full teammate audit and print key findings."""

    root = find_project_root(Path.cwd())
    results = run_full_teammate_audit(root)
    out_dir = root / "outputs" / "teammate_audit"
    inventory = results["inventory"]
    model_audit = results["model_audit"]
    compatibility = results["compatibility"]

    print(f"Saved audit outputs under: {out_dir}")
    print(f"Final report: {out_dir / 'final_teammate_audit_report.md'}")
    print("File counts by suffix:")
    print(inventory["suffix"].value_counts().to_string() if isinstance(inventory, pd.DataFrame) and not inventory.empty else "none")
    print("\nModels detected:")
    print(model_audit[["model_name", "implemented", "evidence_files", "concerns"]].to_string(index=False))
    print("\nIntegration blockers / partial compatibility:")
    blockers = compatibility[compatibility["compatible"].isin(["partial", "no"])]
    print(blockers[["my_module", "compatible", "concerns", "action_needed"]].to_string(index=False))


if __name__ == "__main__":
    main()
