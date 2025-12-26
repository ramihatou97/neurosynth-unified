"""
Metrics collection for NeuroSynth pipeline.

Supports:
- Prometheus text format export (for push gateway or scrape endpoint)
- JSON file export (for debugging and local analysis)
- In-memory aggregation with histograms

Usage:
    from src.core.metrics import get_metrics_collector

    metrics = get_metrics_collector()
    metrics.counter("documents_processed_total")
    metrics.histogram("pipeline_duration_seconds", 12.5)

    # Export
    print(metrics.export_prometheus())
    metrics.export_json(Path("metrics.json"))
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, TYPE_CHECKING
from enum import Enum
import time
import json
import logging
from pathlib import Path

if TYPE_CHECKING:
    from src.ingest.embeddings import BaseTextEmbedder

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Type of metric for Prometheus compatibility."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class Metric:
    """Single metric value with metadata."""
    name: str
    type: MetricType
    value: float
    labels: Dict[str, str] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    help_text: str = ""


@dataclass
class HistogramBuckets:
    """Histogram with configurable buckets."""
    observations: List[float] = field(default_factory=list)
    bucket_bounds: List[float] = field(default_factory=lambda: [
        0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, float('inf')
    ])

    def observe(self, value: float):
        """Record an observation."""
        self.observations.append(value)

    def get_buckets(self) -> Dict[float, int]:
        """Get bucket counts."""
        buckets = {bound: 0 for bound in self.bucket_bounds}
        for obs in self.observations:
            for bound in self.bucket_bounds:
                if obs <= bound:
                    buckets[bound] += 1
        return buckets

    def sum(self) -> float:
        """Sum of all observations."""
        return sum(self.observations)

    def count(self) -> int:
        """Count of observations."""
        return len(self.observations)


class MetricsCollector:
    """Central metrics collector for all pipeline components.

    Collects counters, gauges, and histograms with optional labels.
    Exports to Prometheus text format or JSON.

    Example:
        collector = MetricsCollector(prefix="neurosynth")
        collector.counter("requests_total", labels={"endpoint": "embed"})
        collector.gauge("active_connections", 5)
        collector.histogram("request_duration_seconds", 0.123)

        print(collector.export_prometheus())
    """

    # Standard metric definitions with help text
    METRIC_DEFINITIONS = {
        "documents_processed_total": ("counter", "Total documents processed"),
        "chunks_created_total": ("counter", "Total chunks created"),
        "images_extracted_total": ("counter", "Total images extracted"),
        "entities_extracted_total": ("counter", "Total entities extracted"),
        "links_created_total": ("counter", "Total image-chunk links created"),
        "embedding_requests_total": ("counter", "Total embedding API requests"),
        "embedding_success_total": ("counter", "Successful embedding requests"),
        "embedding_failures_total": ("counter", "Failed embedding requests"),
        "embedding_retries_total": ("counter", "Embedding retry attempts"),
        "vlm_captions_total": ("counter", "Total VLM caption requests"),
        "umls_extractions_total": ("counter", "Total UMLS extractions"),
        "chunks_per_document": ("gauge", "Chunks in most recent document"),
        "images_per_document": ("gauge", "Images in most recent document"),
        "entities_per_document": ("gauge", "Entities in most recent document"),
        "embedding_tokens_total": ("gauge", "Total tokens embedded"),
        "pipeline_duration_seconds": ("histogram", "Pipeline processing time"),
        "embedding_duration_seconds": ("histogram", "Embedding request duration"),
        "vlm_duration_seconds": ("histogram", "VLM captioning duration"),
    }

    def __init__(self, prefix: str = "neurosynth"):
        """Initialize collector with metric prefix.

        Args:
            prefix: Prefix for all metric names (e.g., "neurosynth" -> "neurosynth_requests_total")
        """
        self.prefix = prefix
        self._metrics: Dict[str, Metric] = {}
        self._histograms: Dict[str, HistogramBuckets] = {}
        self._start_time = time.time()

    def _key(self, name: str, labels: Optional[Dict[str, str]]) -> str:
        """Generate unique key for metric with labels."""
        if not labels:
            return f"{self.prefix}_{name}"
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{self.prefix}_{name}{{{label_str}}}"

    def _get_help(self, name: str) -> str:
        """Get help text for a metric."""
        if name in self.METRIC_DEFINITIONS:
            return self.METRIC_DEFINITIONS[name][1]
        return ""

    def counter(self, name: str, value: float = 1.0, labels: Optional[Dict[str, str]] = None):
        """Increment a counter metric.

        Args:
            name: Metric name (without prefix)
            value: Amount to increment (default 1.0)
            labels: Optional label key-value pairs
        """
        key = self._key(name, labels)
        if key in self._metrics:
            self._metrics[key].value += value
            self._metrics[key].timestamp = time.time()
        else:
            self._metrics[key] = Metric(
                name=name,
                type=MetricType.COUNTER,
                value=value,
                labels=labels or {},
                help_text=self._get_help(name)
            )

    def gauge(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Set a gauge metric to a specific value.

        Args:
            name: Metric name (without prefix)
            value: Current value
            labels: Optional label key-value pairs
        """
        key = self._key(name, labels)
        self._metrics[key] = Metric(
            name=name,
            type=MetricType.GAUGE,
            value=value,
            labels=labels or {},
            help_text=self._get_help(name)
        )

    def histogram(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a histogram observation.

        Args:
            name: Metric name (without prefix)
            value: Observed value
            labels: Optional label key-value pairs
        """
        key = self._key(name, labels)
        if key not in self._histograms:
            self._histograms[key] = HistogramBuckets()
        self._histograms[key].observe(value)

    def collect_from_embedder(self, embedder: "BaseTextEmbedder", provider: str):
        """Collect metrics from an embedder's stats.

        Args:
            embedder: Embedder instance with stats attribute
            provider: Provider name for labels (e.g., "voyage", "openai")
        """
        if not hasattr(embedder, 'stats'):
            logger.debug(f"Embedder {provider} has no stats attribute")
            return

        stats = embedder.stats
        labels = {"provider": provider}

        self.counter("embedding_requests_total", stats.total_requests, labels)
        self.counter("embedding_success_total", stats.successful_requests, labels)
        self.counter("embedding_failures_total", stats.failed_requests, labels)
        self.counter("embedding_retries_total", stats.retry_count, labels)
        self.gauge("embedding_tokens_total", stats.total_tokens, labels)

        if stats.total_time_ms > 0:
            self.histogram("embedding_duration_seconds", stats.total_time_ms / 1000.0, labels)

    def collect_from_pipeline(self, result: Any):
        """Collect metrics from a PipelineResult.

        Args:
            result: PipelineResult with chunk_count, image_count, entities, etc.
        """
        self.counter("documents_processed_total")

        if hasattr(result, 'chunk_count'):
            self.counter("chunks_created_total", result.chunk_count)
            self.gauge("chunks_per_document", result.chunk_count)

        if hasattr(result, 'image_count'):
            self.counter("images_extracted_total", result.image_count)
            self.gauge("images_per_document", result.image_count)

        if hasattr(result, 'entities'):
            entity_count = len(result.entities) if result.entities else 0
            self.counter("entities_extracted_total", entity_count)
            self.gauge("entities_per_document", entity_count)

        if hasattr(result, 'links'):
            link_count = len(result.links) if result.links else 0
            self.counter("links_created_total", link_count)

        if hasattr(result, 'duration_seconds') and result.duration_seconds:
            self.histogram("pipeline_duration_seconds", result.duration_seconds)

    def collect_from_vlm(self, stats: Any):
        """Collect metrics from VLM captioner stats.

        Args:
            stats: VLMCaptioner stats with total, successful, failed, skipped
        """
        if hasattr(stats, 'total'):
            self.counter("vlm_captions_total", stats.total, {"status": "total"})
        if hasattr(stats, 'successful'):
            self.counter("vlm_captions_total", stats.successful, {"status": "success"})
        if hasattr(stats, 'failed'):
            self.counter("vlm_captions_total", stats.failed, {"status": "failed"})
        if hasattr(stats, 'skipped'):
            self.counter("vlm_captions_total", stats.skipped, {"status": "skipped"})

    def export_prometheus(self) -> str:
        """Export metrics in Prometheus text format.

        Returns:
            String in Prometheus exposition format
        """
        lines = []
        seen_metrics = set()

        # Export counters and gauges
        for key, metric in sorted(self._metrics.items()):
            metric_name = f"{self.prefix}_{metric.name}"

            # Add HELP and TYPE only once per metric name
            if metric_name not in seen_metrics:
                if metric.help_text:
                    lines.append(f"# HELP {metric_name} {metric.help_text}")
                lines.append(f"# TYPE {metric_name} {metric.type.value}")
                seen_metrics.add(metric_name)

            lines.append(f"{key} {metric.value}")

        # Export histograms
        for key, histogram in sorted(self._histograms.items()):
            # Extract metric name from key
            base_name = key.split("{")[0] if "{" in key else key
            labels_part = key.split("{")[1].rstrip("}") if "{" in key else ""

            if base_name not in seen_metrics:
                lines.append(f"# TYPE {base_name} histogram")
                seen_metrics.add(base_name)

            buckets = histogram.get_buckets()
            for bound, count in buckets.items():
                if bound == float('inf'):
                    bucket_label = '+Inf'
                else:
                    bucket_label = str(bound)

                if labels_part:
                    lines.append(f'{base_name}_bucket{{le="{bucket_label}",{labels_part}}} {count}')
                else:
                    lines.append(f'{base_name}_bucket{{le="{bucket_label}"}} {count}')

            if labels_part:
                lines.append(f"{base_name}_sum{{{labels_part}}} {histogram.sum()}")
                lines.append(f"{base_name}_count{{{labels_part}}} {histogram.count()}")
            else:
                lines.append(f"{base_name}_sum {histogram.sum()}")
                lines.append(f"{base_name}_count {histogram.count()}")

        return "\n".join(lines)

    def export_json(self, path: Path):
        """Export metrics to JSON file.

        Args:
            path: Output file path
        """
        data = {
            "timestamp": time.time(),
            "uptime_seconds": time.time() - self._start_time,
            "prefix": self.prefix,
            "metrics": {},
            "histograms": {}
        }

        for key, metric in self._metrics.items():
            data["metrics"][key] = {
                "name": metric.name,
                "type": metric.type.value,
                "value": metric.value,
                "labels": metric.labels,
                "timestamp": metric.timestamp,
                "help": metric.help_text
            }

        for key, histogram in self._histograms.items():
            data["histograms"][key] = {
                "observations": len(histogram.observations),
                "sum": histogram.sum(),
                "count": histogram.count(),
                "buckets": {str(k): v for k, v in histogram.get_buckets().items()}
            }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Exported metrics to {path}")

    def reset(self):
        """Reset all metrics to zero."""
        self._metrics.clear()
        self._histograms.clear()
        self._start_time = time.time()

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of current metrics.

        Returns:
            Dictionary with metric counts and key values
        """
        return {
            "counters": sum(1 for m in self._metrics.values() if m.type == MetricType.COUNTER),
            "gauges": sum(1 for m in self._metrics.values() if m.type == MetricType.GAUGE),
            "histograms": len(self._histograms),
            "uptime_seconds": time.time() - self._start_time,
        }


# Global collector instance
_collector: Optional[MetricsCollector] = None


def get_metrics_collector(prefix: str = "neurosynth") -> MetricsCollector:
    """Get or create the global metrics collector.

    Args:
        prefix: Metric prefix (only used if creating new collector)

    Returns:
        Global MetricsCollector instance
    """
    global _collector
    if _collector is None:
        _collector = MetricsCollector(prefix=prefix)
    return _collector


def reset_metrics_collector():
    """Reset the global metrics collector."""
    global _collector
    if _collector is not None:
        _collector.reset()
