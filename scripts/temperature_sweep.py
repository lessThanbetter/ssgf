"""Temperature (rule sharpness) sweep with detection, from the recorded run."""
import json

TAU = r"D:\CSF-main\ssgf\_remote_jobs\results\j5_tau_det.json"
OUT = r"D:\CSF-main\ssgf\_restore_tau.json"

d = json.load(open(TAU, encoding="utf-8"))
rows = []
for ev in ("vis", "agn"):
    for rule in ("activity_only", "tophat", "probe"):
        for tau in ("3.0", "1.0", "0.3", "0.1"):
            key = "%s@%s" % (rule, tau)
            if key in d[ev]:
                c = d[ev][key]
                rows.append({"evaluator": ev, "rule": rule, "tau": float(tau),
                             "mAP50": c["mAP50"], "mAP50_95": c["mAP50_95"],
                             "w_dev": c.get("w_dev")})
base = {ev: d[ev]["mean"]["mAP50"] for ev in ("vis", "agn")}
base95 = {ev: d[ev]["mean"]["mAP50_95"] for ev in ("vis", "agn")}
out = {"mean_fusion": {"vis": base["vis"], "agn": base["agn"]}, "rows": rows}
json.dump(out, open(OUT, "w"), indent=2)

print("mean fusion mAP50: vis %.4f  agn %.4f" % (base["vis"], base["agn"]))
print("%-16s %-6s %8s %8s %8s" % ("rule", "tau", "vis", "agn", "w_dev"))
for r in rows:
    print("%-16s %-6.1f %8.4f %8.4f %8.4f"
          % (r["rule"], r["tau"], r["mAP50"], 0.0, r["w_dev"] or 0.0))
