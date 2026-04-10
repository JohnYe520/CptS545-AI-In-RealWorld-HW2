import json

with open("results/results.json", "r") as f:
    results = json.load(f)

print("=== Accuracy / ECE / NLL ===")
for method, res in results.items():
    if "accuracy" in res:
        print(f"{method}:")
        print(f"  Accuracy = {res['accuracy']:.4f}")
        print(f"  ECE      = {res['ece']:.4f}")
        print(f"  NLL      = {res['nll']:.4f}")
        print()

print("=== Conformal ===")
cp = results["conformal_prediction"]
print(f"Set Coverage = {cp['set_coverage']:.4f}")
print(f"Avg Set Size = {cp['avg_set_size']:.4f}")