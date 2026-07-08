import { useEffect, useState } from 'react';
import { API } from '../api/client';

interface SummaryData {
  nav: number;
  unrealized_pnl: number;
  realized_pnl: number;
  total_pnl: number;
  positions_count: number;
  last_updated: string;
}

export const MetricStrip = () => {
  const [summary, setSummary] = useState<SummaryData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchSummary = async () => {
      try {
        const response = await API.portfolio.getSummary();
        setSummary(response.data);
      } catch (error) {
        console.error('Error fetching summary:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchSummary();
    const interval = setInterval(fetchSummary, 10000);
    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        {[...Array(4)].map((_, i) => (
          <div key={i} className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-4 animate-pulse">
            <div className="h-4 bg-[#1e2d45] rounded w-1/2 mb-2"></div>
            <div className="h-6 bg-[#1e2d45] rounded w-3/4"></div>
          </div>
        ))}
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-4">
          <div className="text-xs text-gray-400">Loading...</div>
        </div>
      </div>
    );
  }

  const totalReturn = ((summary.nav - 1000000) / 1000000 * 100);
  const isPositive = totalReturn >= 0;

  const metrics = [
    {
      label: 'Portfolio NAV',
      value: `$${summary.nav.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      color: 'text-white',
    },
    {
      label: "Today's P&L",
      value: `${summary.total_pnl >= 0 ? '+' : ''}$${summary.total_pnl.toFixed(2)}`,
      color: summary.total_pnl >= 0 ? 'text-[#00d4aa]' : 'text-[#ff6b6b]',
    },
    {
      label: 'Total Return',
      value: `${isPositive ? '+' : ''}${totalReturn.toFixed(1)}%`,
      color: isPositive ? 'text-[#00d4aa]' : 'text-[#ff6b6b]',
    },
    {
      label: 'Active Positions',
      value: summary.positions_count.toString(),
      color: 'text-white',
    },
  ];

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
      {metrics.map((metric, index) => (
        <div
          key={index}
          className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-4 hover:border-[#00d4aa]/30 transition-all"
        >
          <div className="text-xs text-gray-400 uppercase tracking-wider">{metric.label}</div>
          <div className={`text-xl font-bold ${metric.color}`}>
            {metric.value}
          </div>
        </div>
      ))}
    </div>
  );
};