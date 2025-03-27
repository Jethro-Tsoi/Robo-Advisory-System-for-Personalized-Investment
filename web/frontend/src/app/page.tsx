import ComparisonChart from "@/components/ComparisonChart";
import ModelMetrics from "@/components/ModelMetrics";
import SamplePredictions from "@/components/SamplePredictions";

export default function Home() {
  return (
    <main className="min-h-screen p-4 md:p-8">
      <div className="mx-auto max-w-7xl space-y-8">
        <header className="text-center">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900 sm:text-6xl">
            Financial Sentiment Analysis
          </h1>
          <p className="mt-4 text-lg text-gray-600">
            Model Comparison Dashboard: Gamma 3 vs FinBERT
          </p>
        </header>

        <div className="grid gap-8 md:grid-cols-2">
          {/* Model Performance Metrics */}
          <div className="rounded-lg border bg-white p-6 shadow-sm">
            <h2 className="mb-4 text-2xl font-semibold">Performance Metrics</h2>
            <ModelMetrics />
          </div>

          {/* Comparison Chart */}
          <div className="rounded-lg border bg-white p-6 shadow-sm">
            <h2 className="mb-4 text-2xl font-semibold">Model Comparison</h2>
            <ComparisonChart />
          </div>
        </div>

        {/* Sample Predictions */}
        <div className="rounded-lg border bg-white p-6 shadow-sm">
          <h2 className="mb-4 text-2xl font-semibold">Sample Predictions</h2>
          <SamplePredictions />
        </div>
      </div>
    </main>
  );
}
