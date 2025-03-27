import { useEffect, useState } from "react";
import axios from "axios";

interface Prediction {
  text: string;
  gamma_prediction: string;
  gamma_confidence: number;
  finbert_prediction: string;
  finbert_confidence: number;
}

export default function SamplePredictions() {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPredictions = async () => {
      try {
        const response = await axios.get("/api/sample_predictions");
        setPredictions(response.data);
        setLoading(false);
      } catch (err) {
        setError("Failed to load predictions");
        setLoading(false);
      }
    };

    fetchPredictions();
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

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment.toLowerCase()) {
      case "strongly_positive":
        return "text-green-600";
      case "positive":
        return "text-green-500";
      case "neutral":
        return "text-gray-500";
      case "negative":
        return "text-red-500";
      case "strongly_negative":
        return "text-red-600";
      case "uncertain":
        return "text-yellow-500";
      default:
        return "text-gray-700";
    }
  };

  return (
    <div className="space-y-6">
      {predictions.map((prediction, index) => (
        <div
          key={index}
          className="rounded-lg border bg-white p-4 shadow-sm transition-all hover:shadow-md"
        >
          <p className="mb-3 text-gray-900">{prediction.text}</p>
          <div className="grid gap-4 sm:grid-cols-2">
            <div>
              <h3 className="mb-2 font-medium text-gray-700">Gamma 3</h3>
              <div className="flex items-center justify-between">
                <span className={getSentimentColor(prediction.gamma_prediction)}>
                  {prediction.gamma_prediction}
                </span>
                <span className="text-sm text-gray-500">
                  {(prediction.gamma_confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
              <div className="mt-2 h-2 w-full rounded-full bg-gray-200">
                <div
                  className="h-2 rounded-full bg-blue-500"
                  style={{
                    width: `${prediction.gamma_confidence * 100}%`,
                  }}
                ></div>
              </div>
            </div>
            <div>
              <h3 className="mb-2 font-medium text-gray-700">FinBERT</h3>
              <div className="flex items-center justify-between">
                <span className={getSentimentColor(prediction.finbert_prediction)}>
                  {prediction.finbert_prediction}
                </span>
                <span className="text-sm text-gray-500">
                  {(prediction.finbert_confidence * 100).toFixed(1)}% confidence
                </span>
              </div>
              <div className="mt-2 h-2 w-full rounded-full bg-gray-200">
                <div
                  className="h-2 rounded-full bg-pink-500"
                  style={{
                    width: `${prediction.finbert_confidence * 100}%`,
                  }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      ))}
    </div>
  );
}
