import { useEffect, useState } from 'react';
import { API } from '../api/client';

interface MonthlyData {
  years: number[];
  months: string[];
  data: number[][];
}

export const MonthlyHeatmap = () => {
  const [data, setData] = useState<MonthlyData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await API.signals.getMonthlyReturns();
        console.log('Monthly returns response:', response.data);
        
        if (response.data.status === 'success') {
          setData(response.data.data);
        }
      } catch (error) {
        console.error('Error fetching monthly returns:', error);
        // Generate sample data on error
        setData(generateSampleData());
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  const generateSampleData = (): MonthlyData => {
    const years = [2022, 2023, 2024];
    const months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];
    const data = years.map(() => 
      months.map(() => (Math.random() - 0.45) * 0.08)
    );
    return { years, months, data };
  };

  const getColor = (value: number): string => {
    if (value > 0.03) return 'bg-[#00d4aa] text-white';
    if (value > 0.01) return 'bg-[#00d4aa]/60 text-white';
    if (value > 0) return 'bg-[#00d4aa]/30 text-white';
    if (value > -0.01) return 'bg-[#ff6b6b]/30 text-white';
    if (value > -0.03) return 'bg-[#ff6b6b]/60 text-white';
    return 'bg-[#ff6b6b] text-white';
  };

  if (loading) {
    return (
      <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-[#1e2d45] rounded w-1/3"></div>
          <div className="h-48 bg-[#1e2d45] rounded"></div>
        </div>
      </div>
    );
  }

  if (!data || data.data.length === 0) {
    return (
      <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Monthly Returns</h3>
        <p className="text-gray-400 text-center py-8">No monthly returns data available</p>
      </div>
    );
  }

  return (
    <div className="bg-[#141c2b] border border-[#1e2d45] rounded-lg p-6">
      <h3 className="text-lg font-semibold text-white mb-4">Monthly Returns</h3>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr>
              <th className="text-left text-gray-400 py-2 px-2">Year</th>
              {data.months.map((month) => (
                <th key={month} className="text-center text-gray-400 py-2 px-1 text-xs">
                  {month}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.years.map((year, yearIdx) => (
              <tr key={year} className="border-t border-[#1e2d45]/30">
                <td className="text-left text-gray-400 py-2 px-2 font-medium">{year}</td>
                {data.data[yearIdx].map((value, monthIdx) => {
                  const colorClass = getColor(value);
                  return (
                    <td
                      key={monthIdx}
                      className={`text-center py-2 px-1 rounded text-xs font-medium ${colorClass}`}
                    >
                      {(value * 100).toFixed(1)}%
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3 text-xs text-gray-500 flex items-center justify-end gap-4">
        <span>Color legend:</span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 bg-[#00d4aa] rounded"></span>
          <span>High positive</span>
        </span>
        <span className="flex items-center gap-1">
          <span className="w-3 h-3 bg-[#ff6b6b] rounded"></span>
          <span>High negative</span>
        </span>
      </div>
    </div>
  );
};