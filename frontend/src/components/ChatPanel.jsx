import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Bot, User, Loader2, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

const API_URL = process.env.REACT_APP_BACKEND_URL;

export default function ChatPanel({ sessionId, dataId, onAnalysisRequest }) {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const scrollRef = useRef(null);
  const inputRef = useRef(null);

  // Initial welcome message
  useEffect(() => {
    if (messages.length === 0) {
      setMessages([{
        id: 'welcome',
        role: 'assistant',
        content: `**Welcome to AGstat Lite!** 🧬

Congratulations on surviving field trials, Excel disasters, and questionable CSV exports. You've earned this.

I'm your PhD-level biostatistics assistant. Here's what I can help with:

- **ANOVA** (RCBD, split-plot, factorial)
- **PCA & Clustering** (biplots, K-means, dendrograms)
- **Assumption Tests** (Shapiro-Wilk, Levene's)
- **Correlation & Regression**
- **Descriptive Statistics**

**To get started:**
1. Upload your data (CSV/Excel)
2. Ask me to analyze it, or use the quick buttons above
3. I'll run the analysis, show results, and give you the code

What would you like to analyze today?`
      }]);
    }
  }, []);

  // Scroll to bottom on new messages
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Load chat history
  useEffect(() => {
    const loadHistory = async () => {
      try {
        const response = await fetch(`${API_URL}/api/chat/history/${sessionId}`);
        if (response.ok) {
          const data = await response.json();
          if (data.messages && data.messages.length > 0) {
            setMessages(prev => [...prev, ...data.messages]);
          }
        }
      } catch (error) {
        console.error('Failed to load chat history:', error);
      }
    };

    if (sessionId) {
      loadHistory();
    }
  }, [sessionId]);

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      id: `user_${Date.now()}`,
      role: 'user',
      content: input.trim()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          session_id: sessionId,
          message: input.trim(),
          data_id: dataId
        }),
      });

      if (!response.ok) {
        throw new Error('Chat request failed');
      }

      const data = await response.json();
      setMessages(prev => [...prev, {
        id: data.id,
        role: 'assistant',
        content: data.content,
        code: data.code,
        plot_url: data.plot_url
      }]);
    } catch (error) {
      setMessages(prev => [...prev, {
        id: `error_${Date.now()}`,
        role: 'assistant',
        content: `Sorry, I encountered an error: ${error.message}. Please try again.`
      }]);
    } finally {
      setIsLoading(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-full bg-[#0B1121]/80">
      {/* Header */}
      <div className="flex-shrink-0 px-4 py-3 border-b border-[#1E293B]">
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded bg-[#00F0FF]/20 flex items-center justify-center">
            <Sparkles className="w-4 h-4 text-[#00F0FF]" />
          </div>
          <div>
            <h3 className="font-['Chivo'] font-bold text-sm text-[#F8FAFC]">AI Assistant</h3>
            <p className="text-xs text-[#64748B]">GPT-5.2 powered</p>
          </div>
        </div>
      </div>

      {/* Messages */}
      <ScrollArea className="flex-1 p-4" ref={scrollRef}>
        <div className="space-y-4">
          <AnimatePresence initial={false}>
            {messages.map((message) => (
              <motion.div
                key={message.id}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}
              >
                {/* Avatar */}
                <div className={`
                  flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center
                  ${message.role === 'user' ? 'bg-[#3B82F6]' : 'bg-[#00F0FF]/20'}
                `}>
                  {message.role === 'user' ? (
                    <User className="w-4 h-4 text-white" />
                  ) : (
                    <Bot className="w-4 h-4 text-[#00F0FF]" />
                  )}
                </div>

                {/* Message Content */}
                <div className={`
                  max-w-[85%] p-3 rounded-lg text-sm
                  ${message.role === 'user' 
                    ? 'bg-[#1E293B] text-[#F8FAFC] rounded-tr-sm' 
                    : 'bg-[#00F0FF]/5 border border-[#00F0FF]/20 text-[#F8FAFC] rounded-tl-sm'}
                `}>
                  <div className="prose prose-invert prose-sm max-w-none">
                    <ReactMarkdown 
                      remarkPlugins={[remarkGfm]}
                      components={{
                        code: ({ node, inline, className, children, ...props }) => {
                          if (inline) {
                            return (
                              <code className="bg-[#1E293B] px-1.5 py-0.5 rounded text-[#00F0FF] font-mono text-xs" {...props}>
                                {children}
                              </code>
                            );
                          }
                          return (
                            <pre className="bg-[#0B1121] border border-[#1E293B] rounded p-3 overflow-x-auto">
                              <code className="font-mono text-xs text-[#94A3B8]" {...props}>
                                {children}
                              </code>
                            </pre>
                          );
                        },
                        p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                        ul: ({ children }) => <ul className="list-disc pl-4 mb-2 space-y-1">{children}</ul>,
                        ol: ({ children }) => <ol className="list-decimal pl-4 mb-2 space-y-1">{children}</ol>,
                        li: ({ children }) => <li className="text-[#94A3B8]">{children}</li>,
                        strong: ({ children }) => <strong className="text-[#00F0FF] font-semibold">{children}</strong>,
                        h1: ({ children }) => <h1 className="text-lg font-bold text-[#F8FAFC] mb-2">{children}</h1>,
                        h2: ({ children }) => <h2 className="text-base font-bold text-[#F8FAFC] mb-2">{children}</h2>,
                        h3: ({ children }) => <h3 className="text-sm font-bold text-[#F8FAFC] mb-1">{children}</h3>,
                      }}
                    >
                      {message.content}
                    </ReactMarkdown>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {/* Loading indicator */}
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex gap-3"
            >
              <div className="w-8 h-8 rounded-full bg-[#00F0FF]/20 flex items-center justify-center">
                <Bot className="w-4 h-4 text-[#00F0FF]" />
              </div>
              <div className="bg-[#00F0FF]/5 border border-[#00F0FF]/20 rounded-lg rounded-tl-sm p-3">
                <div className="flex items-center gap-2">
                  <Loader2 className="w-4 h-4 text-[#00F0FF] animate-spin" />
                  <span className="text-sm text-[#94A3B8]">Analyzing...</span>
                </div>
              </div>
            </motion.div>
          )}
        </div>
      </ScrollArea>

      {/* Input */}
      <div className="flex-shrink-0 p-4 border-t border-[#1E293B]">
        <div className="flex gap-2">
          <Input
            ref={inputRef}
            data-testid="chat-input"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Ask about your data..."
            disabled={isLoading}
            className="flex-1 bg-[#0B1121] border-[#1E293B] text-[#F8FAFC] placeholder:text-[#64748B] focus:border-[#00F0FF] focus:ring-1 focus:ring-[#00F0FF]"
          />
          <Button
            data-testid="chat-send-btn"
            onClick={sendMessage}
            disabled={!input.trim() || isLoading}
            className="bg-[#00F0FF] text-black hover:bg-[#00CCFF] disabled:opacity-50"
          >
            {isLoading ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </Button>
        </div>
        <p className="text-xs text-[#64748B] mt-2">
          Pro tip: Describe your experiment design for better suggestions
        </p>
      </div>
    </div>
  );
}
