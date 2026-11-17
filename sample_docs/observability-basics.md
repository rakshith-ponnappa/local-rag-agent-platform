# Observability Basics

Observability rests on three signals: metrics, logs, and traces.

Metrics are numeric time series such as request rate, error rate, and latency. Prometheus scrapes metrics over HTTP and stores them for alerting and dashboards.

Logs are timestamped event records. Structured JSON logs are easier to search and correlate than free text.

Traces follow a single request across services. Each unit of work is a span, and spans that share a trace ID form the end-to-end trace. OpenTelemetry provides vendor-neutral SDKs and a collector for all three signals.

Service level objectives (SLOs) define a reliability target, for example 99.9 percent of requests succeed. The error budget is the allowed amount of failure before the team must slow down releases.
