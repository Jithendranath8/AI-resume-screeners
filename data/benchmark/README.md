# Resume Prompt-Injection Benchmark (synthetic)

- Total documents: **104**
- Injected: **72**  (4 resumes x 9 techniques x 2 payload families)
- Clean: **32**  (4 resumes x 8 benign variants)

Reconstructed from the description in Barach (2026), *ResumeShield* (arXiv:2609.20188). Fully synthetic; no real personal data. Each document JSON carries its `segments`, the naive `raw_extract`, and the human `visible_render`. See `manifest.csv` for a flat index.
