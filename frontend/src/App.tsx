import { PositionsTable } from './components/PositionsTable';
import { EquityCurve } from './components/EquityCurve';
import { RiskPanel } from './components/RiskPanel';
import { Tearsheet } from './components/Tearsheet';
import { StrategyExplorer } from './components/StrategyExplorer';
import { MetricStrip } from './components/MetricStrip';

function App() {
  return (
    <div className="min-h-screen bg-[#0a0e17]">
      <div className="p-6">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h1 className="text-2xl font-bold text-white">AlgoTrader</h1>
            <p className="text-gray-400">Quant Trading Dashboard</p>
          </div>
          <div className="flex items-center gap-2">
            <div className="flex items-center gap-2 px-3 py-1 bg-[#00d4aa]/10 border border-[#00d4aa]/20 rounded-full">
              <span className="w-2 h-2 bg-[#00d4aa] rounded-full animate-pulse"></span>
              <span className="text-xs text-[#00d4aa]">Live</span>
            </div>
          </div>
        </div>

        {/* ⭐ NEW: Metric Strip */}
        <MetricStrip />

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2">
            <EquityCurve />
          </div>
          <div className="lg:col-span-1">
            <RiskPanel />
          </div>
        </div>

        {/* Positions */}
        <div className="mt-6">
          <PositionsTable />
        </div>

        {/* Tearsheet + Strategy Explorer */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mt-6">
          <div className="lg:col-span-2">
            <Tearsheet />
          </div>
          <div className="lg:col-span-1">
            <StrategyExplorer />
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;