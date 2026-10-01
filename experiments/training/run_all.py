"""Run G0-G2 by default; G3 is an explicit frontier experiment."""
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-g3", action="store_true")
    args = parser.parse_args()
    names = ["g0_math_check", "g1_cross_credit", "g2_event_flow_time"]
    if args.include_g3:
        names.append("g3_async_bootstrap")
    for name in names:
        print(f"\n=== {name} ===", flush=True)
        subprocess.run([sys.executable, "-m", f"experiments.training.{name}"], check=True)

if __name__ == "__main__":
    main()
