import React from 'react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { FileSpreadsheet, Hash, Type } from 'lucide-react';

export default function DataPreview({ data }) {
  if (!data) return null;

  return (
    <div className="space-y-6">
      {/* Data Info */}
      <div className="flex flex-wrap items-center gap-4">
        <div className="flex items-center gap-2">
          <FileSpreadsheet className="w-5 h-5 text-[#00F0FF]" />
          <span className="font-['Chivo'] font-bold text-[#F8FAFC]">{data.filename}</span>
        </div>
        <Badge variant="outline" className="border-[#1E293B] text-[#94A3B8]">
          {data.rows} rows
        </Badge>
        <Badge variant="outline" className="border-[#1E293B] text-[#94A3B8]">
          {data.columns?.length || 0} columns
        </Badge>
      </div>

      {/* Column Info */}
      {data.column_info && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
          {data.column_info.map((col, index) => (
            <div 
              key={index}
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
                {col.missing > 0 && (
                  <span className="text-[#EF4444]">• {col.missing} missing</span>
                )}
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
        <ScrollArea className="max-h-[400px]">
          <Table>
            <TableHeader>
              <TableRow className="border-[#1E293B] hover:bg-transparent">
                {data.columns?.map((col, index) => (
                  <TableHead 
                    key={index}
                    className="text-[#00F0FF] font-semibold bg-[#0B1121] sticky top-0"
                  >
                    {col}
                  </TableHead>
                ))}
              </TableRow>
            </TableHeader>
            <TableBody>
              {data.preview?.map((row, rowIndex) => (
                <TableRow 
                  key={rowIndex}
                  className="border-[#1E293B] hover:bg-[#1E293B]/30"
                >
                  {data.columns?.map((col, colIndex) => (
                    <TableCell 
                      key={colIndex}
                      className="text-[#F8FAFC] font-mono text-sm"
                    >
                      {row[col] !== null && row[col] !== undefined 
                        ? typeof row[col] === 'number' 
                          ? row[col].toFixed(row[col] % 1 === 0 ? 0 : 2)
                          : String(row[col])
                        : <span className="text-[#64748B]">—</span>
                      }
                    </TableCell>
                  ))}
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </ScrollArea>
      </div>
    </div>
  );
}
