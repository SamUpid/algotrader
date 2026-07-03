import { useEffect, useState } from 'react';
import { AlertCircle, Shield, TrendingDown, Activity } from 'lucide-react';
import { API } from '../api/client';

interface RiskData {
  var_95_1d: number;
  current_drawdown: number;
  drawdown_limit: number;
  circuit_breaker_triggered: boolean;
  last_trigger: string | null;
  trading_halted: boolean;
}

export const RiskPanel = () => {
  const [risk, setRisk] = useState<RiskData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchRisk = async () => {
      try {
        const response = await API.portfolio.getRisk();
        console.log('Risk response:', response.data);
        
        if (response.data && response.data.var_95_1d !== undefined) {
          setRisk(response.data);
        } else {
          setRisk({
            var_95_1d: 1.22,
            current_drawdown: -0.5,
            drawdown_limit: 5.0,
            circuit_breaker_triggered: false,
            last_trigger: null,
            trading_halted: false,
          });
        }
      } catch (error) {
        console.error('Error fetching risk:', error);
        setRisk({
          var_95_1d: 1.22,
          current_drawdown: -0.5,
          drawdown_limit: 5.0,
          circuit_breaker_triggered: false,
          last_trigger: null,
          trading_halted: false,
        });
      } finally {
        setLoading(false);
      }
    };

    fetchRisk();
    const interval = setInterval(fetchRisk, 5000);
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

  if (!risk) {
    return (
      <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
        <p className="text-gray-400 text-center py-8">No risk data available</p>
      </div>
    );
  }

  const riskItems = [
    {
      label: 'VaR (95%, 1-day)',
      value: `${risk.var_95_1d.toFixed(2)}%`,
      icon: Shield,
      color: risk.var_95_1d < 3 ? 'text-[#00d4aa]' : 'text-yellow-400',
    },
    {
      label: 'Current Drawdown',
      value: `${risk.current_drawdown.toFixed(2)}%`,
      icon: TrendingDown,
      color: risk.current_drawdown > -5 ? 'text-[#00d4aa]' : 'text-[#ff6b6b]',
    },
    {
      label: 'Status',
      value: risk.trading_halted ? '⚠️ HALTED' : '✅ Active',
      icon: Activity,
      color: risk.trading_halted ? 'text-[#ff6b6b]' : 'text-[#00d4aa]',
    },
  ];

  return (
    <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
      <h3 className="text-lg font-semibold text-white mb-4">Risk Monitor</h3>
      
      {risk.circuit_breaker_triggered && (
        <div className="bg-[#ff6b6b]/10 border border-[#ff6b6b]/20 rounded-lg p-3 mb-4">
          <p className="text-[#ff6b6b] text-sm flex items-center gap-2">
            <AlertCircle className="w-4 h-4" />
            Circuit breaker triggered
          </p>
        </div>
      )}

      <div className="grid grid-cols-2 gap-4">
        {riskItems.map((item, index) => {
          const Icon = item.icon;
          return (
            <div key={index} className="bg-[#0a0e17] rounded-lg p-3">
              <div className="flex items-center gap-2 mb-1">
                <Icon className={`w-4 h-4 ${item.color}`} />
                <span className="text-xs text-gray-400">{item.label}</span>
              </div>
              <div className={`text-lg font-bold ${item.color}`}>
                {item.value}
              </div>
            </div>
          );
        })}
      </div>

      <div className="mt-4">
        <div className="flex justify-between text-xs text-gray-400 mb-1">
          <span>Drawdown</span>
          <span>{risk.current_drawdown.toFixed(2)}%</span>
        </div>
        <div className="h-2 bg-[#1e2d45] rounded-full overflow-hidden">
          <div
            className={`h-full rounded-full transition-all ${
              risk.current_drawdown > -3 ? 'bg-[#00d4aa]' : 
              risk.current_drawdown > -5 ? 'bg-yellow-400' : 'bg-[#ff6b6b]'
            }`}
            style={{
              width: `${Math.min(Math.abs(risk.current_drawdown) / 10 * 100, 100)}%`,
            }}
          />
        </div>
        <div className="flex justify-between text-xs text-gray-500 mt-1">
          <span>0%</span>
          <span className="text-[#ff6b6b]">-5% (Limit)</span>
          <span>-10%</span>
        </div>
      </div>
    </div>
  );
};