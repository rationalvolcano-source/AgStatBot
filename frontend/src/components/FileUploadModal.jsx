import React, { useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { X, Upload, FileSpreadsheet, AlertCircle } from 'lucide-react';
import { Button } from '@/components/ui/button';
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { useDropzone } from 'react-dropzone';

// Simple dropzone implementation without the library
function useSimpleDropzone({ onDrop, accept, maxSize }) {
  const [isDragActive, setIsDragActive] = React.useState(false);

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    
    const files = Array.from(e.dataTransfer.files);
    const validFiles = files.filter(file => {
      const ext = file.name.split('.').pop().toLowerCase();
      return ['csv', 'xlsx', 'xls'].includes(ext) && file.size <= maxSize;
    });
    
    if (validFiles.length > 0) {
      onDrop(validFiles);
    }
  };

  const handleClick = () => {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = '.csv,.xlsx,.xls';
    input.onchange = (e) => {
      const files = Array.from(e.target.files);
      if (files.length > 0) {
        onDrop(files);
      }
    };
    input.click();
  };

  return {
    getRootProps: () => ({
      onDragEnter: handleDragEnter,
      onDragLeave: handleDragLeave,
      onDragOver: handleDragOver,
      onDrop: handleDrop,
      onClick: handleClick,
    }),
    getInputProps: () => ({}),
    isDragActive,
  };
}

export default function FileUploadModal({ isOpen, onClose, onUpload, isLoading }) {
  const [error, setError] = React.useState(null);

  const onDrop = useCallback((acceptedFiles) => {
    setError(null);
    if (acceptedFiles.length > 0) {
      const file = acceptedFiles[0];
      const ext = file.name.split('.').pop().toLowerCase();
      
      if (!['csv', 'xlsx', 'xls'].includes(ext)) {
        setError('Please upload a CSV or Excel file (.csv, .xlsx, .xls)');
        return;
      }
      
      if (file.size > 10 * 1024 * 1024) {
        setError('File size must be less than 10MB');
        return;
      }
      
      onUpload(file);
    }
  }, [onUpload]);

  const { getRootProps, getInputProps, isDragActive } = useSimpleDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv'],
      'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet': ['.xlsx'],
      'application/vnd.ms-excel': ['.xls']
    },
    maxSize: 10 * 1024 * 1024
  });

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="bg-[#0B1121] border-[#1E293B] max-w-md">
        <DialogHeader>
          <DialogTitle className="font-['Chivo'] font-bold text-xl text-[#F8FAFC]">
            Upload Data
          </DialogTitle>
        </DialogHeader>

        <div className="space-y-4">
          {/* Drop Zone */}
          <div
            {...getRootProps()}
            data-testid="file-dropzone"
            className={`
              relative p-8 rounded-sm border-2 border-dashed cursor-pointer
              transition-all duration-200
              ${isDragActive 
                ? 'border-[#00F0FF] bg-[#00F0FF]/5' 
                : 'border-[#1E293B] hover:border-[#00F0FF]/50 hover:bg-[#1E293B]/30'
              }
              ${isLoading ? 'pointer-events-none opacity-50' : ''}
            `}
          >
            <input {...getInputProps()} />
            
            <div className="flex flex-col items-center text-center">
              <div className={`
                w-16 h-16 rounded-full flex items-center justify-center mb-4
                ${isDragActive ? 'bg-[#00F0FF]/20' : 'bg-[#1E293B]'}
              `}>
                {isLoading ? (
                  <div className="spinner" />
                ) : (
                  <Upload className={`w-8 h-8 ${isDragActive ? 'text-[#00F0FF]' : 'text-[#94A3B8]'}`} />
                )}
              </div>
              
              {isLoading ? (
                <p className="text-[#94A3B8]">Uploading...</p>
              ) : isDragActive ? (
                <p className="text-[#00F0FF] font-medium">Drop your file here</p>
              ) : (
                <>
                  <p className="text-[#F8FAFC] font-medium mb-1">
                    Drag & drop your file here
                  </p>
                  <p className="text-sm text-[#64748B]">
                    or click to browse
                  </p>
                </>
              )}
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <motion.div
              initial={{ opacity: 0, y: -10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-2 p-3 bg-[#EF4444]/10 border border-[#EF4444]/30 rounded-sm"
            >
              <AlertCircle className="w-4 h-4 text-[#EF4444] flex-shrink-0" />
              <p className="text-sm text-[#EF4444]">{error}</p>
            </motion.div>
          )}

          {/* Supported Formats */}
          <div className="flex items-center justify-center gap-4 pt-2">
            <div className="flex items-center gap-2 text-sm text-[#64748B]">
              <FileSpreadsheet className="w-4 h-4" />
              <span>CSV</span>
            </div>
            <div className="flex items-center gap-2 text-sm text-[#64748B]">
              <FileSpreadsheet className="w-4 h-4" />
              <span>Excel (.xlsx, .xls)</span>
            </div>
          </div>

          <p className="text-xs text-center text-[#64748B]">
            Maximum file size: 10MB
          </p>
        </div>
      </DialogContent>
    </Dialog>
  );
}
