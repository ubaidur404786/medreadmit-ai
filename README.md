# medreadmit-ai


1. Optuna tuning gave +0.0012 test AUROC over the baseline LightGBM. Report it honestly. Don't quote the val number as the headline.
2. The published ceiling on UCI Diabetes with structured features is ~0.69 test AUROC. Strack 2014, Shang 2021, others converge there. You're at 0.663 — competitive, not lucky, not embarrassing.
3. The main gain on this project will come from Module 2 (clinical notes), not from squeezing Module 1. This is the real takeaway and it's exactly what published work shows: NYUTron-style architectures gain 4–6 AUROC points from adding text, not 1–2 from tuning structured models harder.