import React, { useState, useEffect } from 'react';
import { Container, Grid, Paper, Typography } from '@mui/material';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
} from 'chart.js';
import { Bar } from 'react-chartjs-2';
import axios from 'axios';

// Register ChartJS components
ChartJS.register(
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend
);

const API_URL = 'http://localhost:8000';

function App() {
  const [metrics, setMetrics] = useState(null);
  const [confusionMatrices, setConfusionMatrices] = useState(null);
  const [samplePredictions, setSamplePredictions] = useState([]);

  useEffect(() => {
    // Fetch data when component mounts
    const fetchData = async () => {
      try {
        const [metricsRes, matricesRes, predictionsRes] = await Promise.all([
          axios.get(`${API_URL}/metrics`),
          axios.get(`${API_URL}/confusion_matrices`),
          axios.get(`${API_URL}/sample_predictions`)
        ]);

        setMetrics(metricsRes.data);
        setConfusionMatrices(matricesRes.data);
        setSamplePredictions(predictionsRes.data);
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };

    fetchData();
  }, []);

  const metricsChartData = {
    labels: ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    datasets: metrics ? [
      {
        label: 'Gamma 3',
        data: [
          metrics.gamma3.accuracy,
          metrics.gamma3.precision,
          metrics.gamma3.recall,
          metrics.gamma3.f1
        ],
        backgroundColor: 'rgba(54, 162, 235, 0.5)',
      },
      {
        label: 'FinBERT',
        data: [
          metrics.finbert.accuracy,
          metrics.finbert.precision,
          metrics.finbert.recall,
          metrics.finbert.f1
        ],
        backgroundColor: 'rgba(255, 99, 132, 0.5)',
      }
    ] : []
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h3" component="h1" gutterBottom>
        Financial Sentiment Analysis Dashboard
      </Typography>
      
      <Grid container spacing={3}>
        {/* Model Performance Metrics */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Model Performance Comparison
            </Typography>
            {metrics && (
              <Bar
                data={metricsChartData}
                options={{
                  responsive: true,
                  scales: {
                    y: {
                      beginAtZero: true,
                      max: 1,
                    },
                  },
                  plugins: {
                    legend: {
                      position: 'top',
                    },
                    title: {
                      display: true,
                      text: 'Model Metrics Comparison'
                    }
                  }
                }}
              />
            )}
          </Paper>
        </Grid>

        {/* Sample Predictions */}
        <Grid item xs={12}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h6" gutterBottom>
              Sample Predictions
            </Typography>
            <Grid container spacing={2}>
              {samplePredictions.map((pred, index) => (
                <Grid item xs={12} key={index}>
                  <Paper elevation={2} sx={{ p: 2 }}>
                    <Typography variant="body1" gutterBottom>
                      Text: {pred.text}
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={6}>
                        <Typography variant="body2">
                          Gamma 3: {pred.gamma_prediction} ({(pred.gamma_confidence * 100).toFixed(2)}%)
                        </Typography>
                      </Grid>
                      <Grid item xs={6}>
                        <Typography variant="body2">
                          FinBERT: {pred.finbert_prediction} ({(pred.finbert_confidence * 100).toFixed(2)}%)
                        </Typography>
                      </Grid>
                    </Grid>
                  </Paper>
                </Grid>
              ))}
            </Grid>
          </Paper>
        </Grid>
      </Grid>
    </Container>
  );
}

export default App;
