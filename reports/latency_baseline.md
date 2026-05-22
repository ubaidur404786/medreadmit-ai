# API Latency Baseline — 2026-05-18T17:58:12Z

Hardware: Windows-11-10.0.26200-SP0, Python 3.12.4

## Single prediction (/predict)
| Metric | Value (ms) |
|---|---|
| p50 | 59.74 |
| p95 | 67.96 |
| p99 | 74.08 |
| mean | 61.34 |
| stdev | 11.30 |

## Batch prediction (/predict/batch)
| Batch size | Mean per-record (ms) | Total batch (ms) |
|---|---|---|
| 1 | 52.83 | 59.08 |
| 10 | 6.01 | 65.01 |
| 50 | 1.52 | 85.70 |
| 100 | 0.87 | 102.71 |

## Notes
- N=200 single requests, 10 warmup
- Sequential, single client, localhost
- Includes SHAP top-5 explanation
