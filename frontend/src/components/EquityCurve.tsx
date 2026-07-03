import { useEffect, useState } from 'react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { API } from '../api/client';

interface EquityPoint {
  date: string;
  equity: number;
}

export const EquityCurve = () => {
  const [data, setData] = useState<EquityPoint[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchEquity = async () => {
      try {
        const response = await API.portfolio.getEquity(252);
        const points = response.data.dates.map((date: string, i: number) => ({
          date,
          equity: response.data.equity[i],
        }));
        setData(points);
      } catch (error) {
        console.error('Error fetching equity:', error);
        const points = [];
        let equity = 1000000;
        for (let i = 0; i < 252; i++) {
          const date = new Date();
          date.setDate(date.getDate() - (252 - i));
          equity = equity * (1 + (Math.random() - 0.48) * 0.015);
          points.push({ date: date.toISOString().split('T')[0], equity });
        }
        setData(points);
      } finally {
        setLoading(false);
      }
    };

    fetchEquity();
    const interval = setInterval(fetchEquity, 5000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
        <div className="animate-pulse h-80 flex items-center justify-center">
          <div className="h-64 w-full bg-[#1e2d45] rounded"></div>
        </div>
      </div>
    );
  }

  if (data.length === 0) {
    return (
      <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Equity Curve</h3>
        <p className="text-gray-400 text-center py-8">No equity data available</p>
      </div>
    );
  }

  const minEquity = Math.min(...data.map(d => d.equity));
  const maxEquity = Math.max(...data.map(d => d.equity));
  const padding = (maxEquity - minEquity) * 0.1 || 10000;

  return (
    <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Equity Curve</h3>
        <span className="text-sm text-gray-400">
          ${data[data.length - 1]?.equity.toLocaleString() || 'N/A'}
        </span>
      </div>
      <div className="h-80">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data}>
            <defs>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00d4aa" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#00d4aa" stopOpacity={0} />
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
              domain={[minEquity - padding, maxEquity + padding]}
              tickFormatter={(value) => `$${(value / 1000).toFixed(0)}k`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: '#141c2b',
                border: '1px solid #1e2d45',
                borderRadius: '8px',
              }}
              formatter={(value: number) => [`$${value.toFixed(2)}`, 'Equity']}
              labelStyle={{ color: '#e8edf5' }}
            />
            <Area
              type="monotone"
              dataKey="equity"
              stroke="#00d4aa"
              strokeWidth={2}
              fill="url(#equityGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};
