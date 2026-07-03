import { useEffect, useState } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from 'recharts';
import { API } from '../api/client';
import { MonthlyHeatmap } from './MonthlyHeatmap';

interface TearsheetData {
  equity: { date: string; value: number }[];
  drawdown: { date: string; value: number }[];
  metrics: any;
}

export const Tearsheet = () => {
  const [data, setData] = useState<TearsheetData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const metricsRes = await API.signals.getMetrics();
        const metrics = metricsRes.data.metrics;

        const equityRes = await API.portfolio.getEquity(252);
        const equityData = equityRes.data;

        const drawdownData = generateDrawdown(equityData.equity, equityData.dates);

        setData({
          equity: equityData.dates.map((d: string, i: number) => ({
            date: d,
            value: equityData.equity[i],
          })),
          drawdown: drawdownData,
          metrics: metrics,
        });
      } catch (error) {
        console.error('Error fetching tearsheet data:', error);
        const sampleData = generateSampleData();
        setData(sampleData);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const generateDrawdown = (equity: number[], dates: string[]) => {
    let peak = equity[0] || 1000000;
    return dates.map((date, i) => {
      const val = equity[i] || 1000000;
      if (val > peak) peak = val;
      const drawdown = ((val - peak) / peak) * 100;
      return { date, value: drawdown };
    });
  };

  const generateSampleData = () => {
    const dates = [];
    const equity = [];
    let e = 1000000;
    const today = new Date();
    for (let i = 252; i >= 0; i--) {
      const d = new Date(today);
      d.setDate(d.getDate() - i);
      dates.push(d.toISOString().split('T')[0]);
      e = e * (1 + (Math.random() - 0.48) * 0.015);
      equity.push(e);
    }
    const drawdown = generateDrawdown(equity, dates);
    return {
      equity: dates.map((d, i) => ({ date: d, value: equity[i] })),
      drawdown: drawdown,
      metrics: {
        sharpe_ratio: 1.2,
        sortino_ratio: 1.8,
        calmar_ratio: 0.9,
        hit_rate: 0.52,
        annual_return: 0.15,
      }
    };
  };

  if (loading) {
    return (
      <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-[#1e2d45] rounded w-1/3"></div>
          <div className="h-64 bg-[#1e2d45] rounded"></div>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
        <p className="text-gray-400 text-center py-8">No data available</p>
      </div>
    );
  }

  const { equity, drawdown, metrics } = data;
  const lastEquity = equity.length > 0 ? equity[equity.length - 1]?.value : 0;
  const minEquity = Math.min(...equity.map(d => d.value));
  const maxEquity = Math.max(...equity.map(d => d.value));
  const padding = (maxEquity - minEquity) * 0.1 || 10000;

  return (
    <div className="space-y-6">
      {/* Metrics Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-4">
          <div className="text-xs text-gray-400">Sharpe Ratio</div>
          <div className="text-xl font-bold text-[#00d4aa]">
            {metrics?.sharpe_ratio?.toFixed(3) || 'N/A'}
          </div>
        </div>
        <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-4">
          <div className="text-xs text-gray-400">Sortino Ratio</div>
          <div className="text-xl font-bold text-[#00d4aa]">
            {metrics?.sortino_ratio?.toFixed(3) || 'N/A'}
          </div>
        </div>
        <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-4">
          <div className="text-xs text-gray-400">Calmar Ratio</div>
          <div className="text-xl font-bold text-[#00d4aa]">
            {metrics?.calmar_ratio?.toFixed(3) || 'N/A'}
          </div>
        </div>
        <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-4">
          <div className="text-xs text-gray-400">Hit Rate</div>
          <div className="text-xl font-bold text-[#00d4aa]">
            {metrics?.hit_rate != null ? `${(metrics.hit_rate * 100).toFixed(1)}%` : 'N/A'}
          </div>
        </div>
      </div>

      {/* Drawdown */}
      <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Drawdown</h3>
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={drawdown}>
              <defs>
                <linearGradient id="ddGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ff6b6b" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#ff6b6b" stopOpacity={0.1} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2d45" />
              <XAxis
                dataKey="date"
                stroke="#6b7280"
                tick={{ fontSize: 10 }}
                tickLine={false}
                interval="preserveStartEnd"
              />
              <YAxis
                stroke="#6b7280"
                tick={{ fontSize: 10 }}
                tickLine={false}
                domain={[-20, 2]}
                tickFormatter={(value) => `${value}%`}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#141c2b',
                  border: '1px solid #1e2d45',
                  borderRadius: '8px',
                }}
                formatter={(value) => [`${Number(value ?? 0).toFixed(2)}%`, 'Drawdown']}
                labelStyle={{ color: '#e8edf5' }}
              />
              <ReferenceLine y={0} stroke="#00d4aa" strokeDasharray="5 5" />
              <ReferenceLine y={-5} stroke="#ff6b6b" strokeDasharray="5 5" />
              <Area
                type="monotone"
                dataKey="value"
                stroke="#ff6b6b"
                strokeWidth={2}
                fill="url(#ddGradient)"
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Monthly Returns Heatmap */}
      <MonthlyHeatmap />
    </div>
  );
};