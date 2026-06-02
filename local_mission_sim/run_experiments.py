from pathlib import Path

from src.msar_sim.experiments import run_batch, run_heterogeneity_sweep, run_scalability_sweep


def main() -> None:
    root = Path(__file__).resolve().parent / "outputs"
    configs = [
        ("standard", 60, "standard", False),
        ("stress", 40, "stress", False),
        ("ablation", 40, "standard", True),
    ]
    for name, count, regime, include_ablations in configs:
        output_dir = root / name
        summary = run_batch(
            num_scenarios=count,
            output_dir=output_dir,
            regime=regime,
            include_ablations=include_ablations,
        )
        print(f"=== {name.upper()} Summary ===")
        for method, metrics in summary.items():
            print(method)
            for key, value in metrics.items():
                print(f"  {key}: {value:.4f}")

    heterogeneity_dir = root / "heterogeneity"
    heterogeneity_summary = run_heterogeneity_sweep(output_dir=heterogeneity_dir)
    print("=== HETEROGENEITY Summary ===")
    for level, method_data in heterogeneity_summary.items():
        print(f"scale={level}")
        for method, metrics in method_data.items():
            print(f"  {method}: mission_time={metrics['mission_time']:.4f}, rescue_success={metrics['rescue_success']:.4f}")

    scalability_dir = root / "scalability"
    scalability_summary = run_scalability_sweep(output_dir=scalability_dir)
    print("=== SCALABILITY Summary ===")
    for level, method_data in scalability_summary.items():
        print(f"scale={level}")
        for method, metrics in method_data.items():
            print(
                f"  {method}: mission_time={metrics['mission_time']:.4f}, "
                f"communication_load={metrics['communication_load']:.4f}"
            )


if __name__ == "__main__":
    main()
