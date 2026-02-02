import React from 'react';
import { motion } from 'framer-motion';
import { X, Download, Copy, Check } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { atomDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import { toast } from 'sonner';

export default function CodePanel({ code, onClose, onDownload }) {
  const [copied, setCopied] = React.useState(false);

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(code);
      setCopied(true);
      toast.success('Code copied to clipboard!');
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      toast.error('Failed to copy code');
    }
  };

  const customStyle = {
    ...atomDark,
    'pre[class*="language-"]': {
      ...atomDark['pre[class*="language-"]'],
      background: '#0B1121',
      margin: 0,
      padding: '1rem',
    },
    'code[class*="language-"]': {
      ...atomDark['code[class*="language-"]'],
      background: '#0B1121',
    }
  };

  return (
    <motion.div
      initial={{ height: 0, opacity: 0 }}
      animate={{ height: 'auto', opacity: 1 }}
      exit={{ height: 0, opacity: 0 }}
      transition={{ duration: 0.3 }}
      className="border-t border-[#1E293B] bg-[#0B1121]"
    >
      {/* Header */}
      <div className="flex items-center justify-between px-4 py-2 border-b border-[#1E293B]">
        <div className="flex items-center gap-2">
          <span className="font-['Chivo'] font-bold text-sm text-[#00F0FF]">Generated Code</span>
          <span className="text-xs text-[#64748B]">Python</span>
        </div>
        <div className="flex items-center gap-2">
          <Button
            data-testid="copy-code-btn"
            onClick={handleCopy}
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-[#94A3B8] hover:text-[#00F0FF] hover:bg-[#1E293B]"
          >
            {copied ? (
              <Check className="w-4 h-4 text-[#10B981]" />
            ) : (
              <Copy className="w-4 h-4" />
            )}
          </Button>
          <Button
            data-testid="download-code-panel-btn"
            onClick={onDownload}
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-[#94A3B8] hover:text-[#00F0FF] hover:bg-[#1E293B]"
          >
            <Download className="w-4 h-4" />
          </Button>
          <Button
            data-testid="close-code-panel-btn"
            onClick={onClose}
            variant="ghost"
            size="sm"
            className="h-8 px-2 text-[#94A3B8] hover:text-[#EF4444] hover:bg-[#1E293B]"
          >
            <X className="w-4 h-4" />
          </Button>
        </div>
      </div>

      {/* Code Content */}
      <ScrollArea className="max-h-[250px]">
        <SyntaxHighlighter
          language="python"
          style={customStyle}
          showLineNumbers
          lineNumberStyle={{ color: '#64748B', fontSize: '12px' }}
          customStyle={{
            background: '#0B1121',
            fontSize: '13px',
            fontFamily: "'JetBrains Mono', monospace",
          }}
        >
          {code}
        </SyntaxHighlighter>
      </ScrollArea>
    </motion.div>
  );
}
