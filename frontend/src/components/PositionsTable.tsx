import { useEffect, useState } from 'react';
import { API } from '../api/client';

interface Position {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  weight_percent: number;
}

export const PositionsTable = () => {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchPositions = async () => {
      try {
        const response = await API.portfolio.getPositions();
        setPositions(response.data);
      } catch (error) {
        console.error('Error fetching positions:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchPositions();
    const interval = setInterval(fetchPositions, 5000);
    return () => clearInterval(interval);
  }, []);

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

  if (positions.length === 0) {
    return (
      <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Positions</h3>
        <p className="text-gray-400 text-center py-8">No open positions</p>
      </div>
    );
  }

  return (
    <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
      <h3 className="text-lg font-semibold text-white mb-4">Positions</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-gray-400 border-b border-[#1e2d45]">
              <th className="text-left py-2 px-3">Symbol</th>
              <th className="text-right py-2 px-3">Qty</th>
              <th className="text-right py-2 px-3">Entry</th>
              <th className="text-right py-2 px-3">Current</th>
              <th className="text-right py-2 px-3">P&L</th>
              <th className="text-right py-2 px-3">Weight</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((pos) => (
              <tr key={pos.symbol} className="border-b border-[#1e2d45]/50 hover:bg-[#1e2d45]/30 transition-colors">
                <td className="py-2 px-3 font-medium text-white">{pos.symbol}</td>
                <td className="text-right py-2 px-3 text-gray-300">{pos.quantity}</td>
                <td className="text-right py-2 px-3 text-gray-300">${pos.entry_price.toFixed(2)}</td>
                <td className="text-right py-2 px-3 text-gray-300">${pos.current_price.toFixed(2)}</td>
                <td className={`text-right py-2 px-3 font-medium ${pos.unrealized_pnl >= 0 ? 'text-[#00d4aa]' : 'text-[#ff6b6b]'}`}>
                  ${pos.unrealized_pnl.toFixed(2)}
                </td>
                <td className="text-right py-2 px-3 text-gray-300">{pos.weight_percent.toFixed(1)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
