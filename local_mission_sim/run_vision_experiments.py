from pathlib import Path

from src.msar_sim.vision import run_vision_explainability


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "outputs" / "vision"
    summary = run_vision_explainability(output_dir=output_dir)
    print("=== Vision Explainability Summary ===")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
