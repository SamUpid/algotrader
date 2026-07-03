import axios from 'axios';

// Use the Render backend URL
const API_BASE_URL = 'https://algotrader-api-7y55.onrender.com';

export const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const API = {
  portfolio: {
    getPositions: () => apiClient.get('/portfolio/positions'),
    getEquity: (days?: number) => apiClient.get('/portfolio/equity', { params: { days } }),
    getSummary: () => apiClient.get('/portfolio/summary'),
    getRisk: () => apiClient.get('/portfolio/risk'),
  },
  signals: {
    getMetrics: () => apiClient.get('/signals/metrics'),
    getMetricsForStrategy: (strategy: string) => 
      apiClient.get(`/signals/metrics/${strategy}`),
    getTextMetrics: () => apiClient.get('/signals/metrics/text'),
    getMonthlyReturns: () => apiClient.get('/signals/metrics/monthly-returns'),
  },
};
