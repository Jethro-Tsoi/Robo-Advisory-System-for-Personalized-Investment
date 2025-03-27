export interface Metrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
}

export interface ModelMetricsData {
  gamma3: Metrics;
  finbert: Metrics;
}

export interface Prediction {
  text: string;
  gamma_prediction: string;
  gamma_confidence: number;
  finbert_prediction: string;
  finbert_confidence: number;
}

export type SentimentLabel = 
  | "STRONGLY_POSITIVE"
  | "POSITIVE"
  | "NEUTRAL"
  | "NEGATIVE"
  | "STRONGLY_NEGATIVE"
  | "NOT_RELATED"
  | "UNCERTAIN";
