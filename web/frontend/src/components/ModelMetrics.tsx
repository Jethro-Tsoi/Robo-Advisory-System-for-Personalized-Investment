import { useEffect, useState } from "react";
import axios from "axios";

interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

interface ModelMetricsData {
  gamma3: Metrics;
  finbert: Metrics;
}

export default function ModelMetrics() {
  const [metrics, setMetrics] = useState<ModelMetricsData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await axios.get("/api/metrics");
        setMetrics(response.data);
        setLoading(false);
      } catch (err) {
        setError("Failed to load metrics");
        setLoading(false);
      }
    };

    fetchMetrics();
  }, []);

  if (loading) {
    return (
      <div className="flex h-48 items-center justify-center">
        <div className="h-8 w-8 animate-spin rounded-full border-4 border-primary border-t-transparent"></div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="flex h-48 items-center justify-center text-red-500">
        {error}
      </div>
    );
  }

  if (!metrics) return null;

  const metricsDisplay = [
    { name: "Accuracy", gamma3: metrics.gamma3.accuracy, finbert: metrics.finbert.accuracy },
    { name: "Precision", gamma3: metrics.gamma3.precision, finbert: metrics.finbert.precision },
    { name: "Recall", gamma3: metrics.gamma3.recall, finbert: metrics.finbert.recall },
    { name: "F1 Score", gamma3: metrics.gamma3.f1, finbert: metrics.finbert.f1 },
  ];

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left">
        <thead>
          <tr>
            <th className="pb-4 text-sm font-medium text-gray-500">Metric</th>
            <th className="pb-4 text-sm font-medium text-gray-500">Gamma 3</th>
            <th className="pb-4 text-sm font-medium text-gray-500">FinBERT</th>
          </tr>
        </thead>
        <tbody>
          {metricsDisplay.map((metric) => (
            <tr key={metric.name} className="border-t">
              <td className="py-4 text-sm font-medium text-gray-900">
                {metric.name}
              </td>
              <td className="py-4 text-sm text-gray-500">
                {(metric.gamma3 * 100).toFixed(2)}%
              </td>
              <td className="py-4 text-sm text-gray-500">
                {(metric.finbert * 100).toFixed(2)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
