# Metrics and Evidence

## Metrics

| Metric | Definition | Measurement |
|--------|-----------|-------------|
| Coverage probability | Percentage of target area achieving minimum signal threshold | Simulated coverage maps with threshold analysis |
| Outage rate | Frequency and duration of service unavailability | Time-series analysis of simulated service states |
| Latency distribution | Statistical characterization of end-to-end delay | Percentile analysis (p50, p95, p99) across scenarios |
| Throughput under contention | Achievable data rate with competing demand | Per-user and aggregate throughput in multi-user simulation |
| Reliability | Proportion of time service meets quality threshold | Availability percentage over simulation duration |
| Spectrum efficiency | Data throughput per unit of spectrum resource | Bits/second/Hz across allocation strategies |
| Cost/performance ratio | Resource expenditure per unit of delivered service | Normalized cost metric relative to baseline performance |

## Evidence

### AI-RAN Controller

Demonstrates policy decisions including:

- Dynamic spectrum allocation under varying conditions
- Adaptive scheduling responding to demand changes
- Interference-aware channel management
- Priority enforcement across service classes

### Digital Twin Adapter

Enables scenario evaluation including:

- Gary-specific geography and density modeling
- Multi-site coverage analysis
- Contention scenario construction
- Comparative policy evaluation across configurations

### Conference Paper

Documents methodology including:

- Problem formulation and system model
- Algorithm design and complexity analysis
- Experimental setup and evaluation protocol
- Results interpretation and limitations
