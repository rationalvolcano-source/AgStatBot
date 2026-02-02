import React from 'react';
import { FileSpreadsheet, Hash, Type } from 'lucide-react';

export default function DataPreview({ data }) {
  if (!data) return null;

  const columns = data.columns || [];
  const preview = data.preview || [];
  const columnInfo = data.column_info || [];

  return (
    <div className="space-y-6">
      {/* Data Info */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <FileSpreadsheet className="w-5 h-5 text-[#00F0FF]" />
          <span className="font-['Chivo'] font-bold text-[#F8FAFC]">{data.filename}</span>
        </div>
        <span className="px-2 py-1 text-xs border border-[#1E293B] text-[#94A3B8] rounded">
          {data.rows} rows
        </span>
        <span className="px-2 py-1 text-xs border border-[#1E293B] text-[#94A3B8] rounded">
          {columns.length} columns
        </span>
      </div>

      {/* Column Info */}
      {columnInfo.length > 0 && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
          {columnInfo.map((col, idx) => (
            <div 
              key={idx}
              data-testid={`column-info-${col.name}`}
              className="p-3 bg-[#0B1121] border border-[#1E293B] rounded-sm"
            >
              <div className="flex items-center gap-2 mb-1">
                {col.type === 'numeric' ? (
                  <Hash className="w-3 h-3 text-[#00F0FF]" />
                ) : (
                  <Type className="w-3 h-3 text-[#F59E0B]" />
                )}
                <span className="text-sm font-medium text-[#F8FAFC] truncate">{col.name}</span>
              </div>
              <div className="flex items-center gap-2 text-xs text-[#64748B]">
                <span className={col.type === 'numeric' ? 'text-[#00F0FF]' : 'text-[#F59E0B]'}>
                  {col.type}
                </span>
                <span>• {col.unique_values} unique</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Data Preview Table */}
      <div className="border border-[#1E293B] rounded-sm overflow-hidden">
        <div className="px-4 py-2 bg-[#1E293B]/50 border-b border-[#1E293B]">
          <span className="text-sm font-medium text-[#94A3B8]">Preview (first 10 rows)</span>
        </div>
        <div className="max-h-[400px] overflow-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-[#1E293B]">
                {columns.map((col, idx) => (
                  <th 
                    key={idx}
                    className="px-4 py-2 text-left text-[#00F0FF] font-semibold bg-[#0B1121] sticky top-0 text-sm"
                  >
                    {col}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {preview.map((row, rowIdx) => (
                <tr 
                  key={rowIdx}
                  className="border-b border-[#1E293B] hover:bg-[#1E293B]/30"
                >
                  {columns.map((col, colIdx) => (
                    <td 
                      key={colIdx}
                      className="px-4 py-2 text-[#F8FAFC] font-mono text-sm"
                    >
                      {row[col] !== null && row[col] !== undefined 
                        ? String(row[col])
                        : <span className="text-[#64748B]">—</span>
                      }
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
