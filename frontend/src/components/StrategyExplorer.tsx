import { useState, useEffect } from 'react';
import { API } from '../api/client';

const SIGNAL_OPTIONS = [
  { id: 'momentum', name: 'Momentum', description: 'Cross-sectional momentum signal' },
  { id: 'rsi', name: 'RSI Mean-Reversion', description: 'Oversold/Overbought signal' },
  { id: 'ml', name: 'ML Alpha', description: 'LightGBM probability signal' },
  { id: 'composite', name: 'Composite (Ensemble)', description: 'Equal-weighted combination' },
];

export const StrategyExplorer = () => {
  const [selectedStrategy, setSelectedStrategy] = useState('composite');
  const [metrics, setMetrics] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      setLoading(true);
      try {
        const response = await API.signals.getMetricsForStrategy(selectedStrategy);
        if (response.data.status === 'success') {
          setMetrics(response.data.metrics);
        }
      } catch (error) {
        console.error('Error fetching metrics:', error);
        // Fallback to default metrics
        try {
          const defaultResponse = await API.signals.getMetrics();
          if (defaultResponse.data.status === 'success') {
            setMetrics(defaultResponse.data.metrics);
          }
        } catch (e) {
          console.error('Error fetching default metrics:', e);
        }
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, [selectedStrategy]);

  if (loading) {
    return (
      <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-[#1e2d45] rounded w-1/3"></div>
          <div className="h-4 bg-[#1e2d45] rounded"></div>
          <div className="h-4 bg-[#1e2d45] rounded"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Strategy Explorer</h3>
        <select
          value={selectedStrategy}
          onChange={(e) => setSelectedStrategy(e.target.value)}
          className="bg-[#0a0e17] border border-[#1e2d45] rounded-lg px-4 py-2 text-white text-sm focus:outline-none focus:border-[#00d4aa]"
        >
          {SIGNAL_OPTIONS.map((option) => (
            <option key={option.id} value={option.id}>
              {option.name}
            </option>
          ))}
        </select>
      </div>

      <p className="text-sm text-gray-400 mb-4">
        {SIGNAL_OPTIONS.find((s) => s.id === selectedStrategy)?.description}
      </p>

      {metrics ? (
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-[#0a0e17] rounded-lg p-3">
            <div className="text-xs text-gray-400">Sharpe</div>
            <div className="text-lg font-bold text-[#00d4aa]">
              {metrics.sharpe_ratio?.toFixed(3) || 'N/A'}
            </div>
          </div>
          <div className="bg-[#0a0e17] rounded-lg p-3">
            <div className="text-xs text-gray-400">Sortino</div>
            <div className="text-lg font-bold text-[#00d4aa]">
              {metrics.sortino_ratio?.toFixed(3) || 'N/A'}
            </div>
          </div>
          <div className="bg-[#0a0e17] rounded-lg p-3">
            <div className="text-xs text-gray-400">Calmar</div>
            <div className="text-lg font-bold text-[#00d4aa]">
              {metrics.calmar_ratio?.toFixed(3) || 'N/A'}
            </div>
          </div>
          <div className="bg-[#0a0e17] rounded-lg p-3">
            <div className="text-xs text-gray-400">Hit Rate</div>
            <div className="text-lg font-bold text-[#00d4aa]">
              {metrics.hit_rate != null ? `${(metrics.hit_rate * 100).toFixed(1)}%` : 'N/A'}
            </div>
          </div>
          <div className="bg-[#0a0e17] rounded-lg p-3 col-span-2">
            <div className="text-xs text-gray-400">Annual Return</div>
            <div className="text-lg font-bold text-[#00d4aa]">
              {metrics.annual_return != null ? `${(metrics.annual_return * 100).toFixed(1)}%` : 'N/A'}
            </div>
          </div>
        </div>
      ) : (
        <p className="text-gray-400 text-center py-4">No metrics available</p>
      )}
    </div>
  );
};