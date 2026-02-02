import React, { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { toast } from 'sonner';
import { 
  Send, 
  Upload, 
  BarChart3, 
  PieChart, 
  ScatterChart,
  Activity,
  FileSpreadsheet,
  Code,
  Download,
  Trash2,
  ChevronDown,
  ChevronUp,
  X,
  Home,
  Menu,
  Loader2
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu';
import { useNavigate } from 'react-router-dom';
import ChatPanel from '@/components/ChatPanel';
import DataPreview from '@/components/DataPreview';
import AnalysisPanel from '@/components/AnalysisPanel';
import CodePanel from '@/components/CodePanel';
import FileUploadModal from '@/components/FileUploadModal';

const API_URL = process.env.REACT_APP_BACKEND_URL;

export default function Dashboard() {
  const navigate = useNavigate();
  const [sessionId] = useState(() => `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`);
  
  // State
  const [uploadedData, setUploadedData] = useState(null);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [generatedCode, setGeneratedCode] = useState('');
  const [isCodePanelOpen, setIsCodePanelOpen] = useState(false);
  const [isUploadModalOpen, setIsUploadModalOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  // Analysis methods
  const analysisTypes = [
    { id: 'anova', name: 'ANOVA', icon: BarChart3, description: 'Analysis of Variance with post-hoc tests' },
    { id: 'pca', name: 'PCA', icon: ScatterChart, description: 'Principal Component Analysis' },
    { id: 'clustering', name: 'Clustering', icon: PieChart, description: 'K-Means & Hierarchical clustering' },
    { id: 'descriptive', name: 'Descriptive', icon: Activity, description: 'Summary statistics & distributions' },
    { id: 'correlation', name: 'Correlation', icon: Activity, description: 'Correlation matrix analysis' },
    { id: 'normality', name: 'Normality', icon: Activity, description: 'Normality & assumption tests' },
  ];

  // File upload handler
  const handleFileUpload = async (file) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch(`${API_URL}/api/upload`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Upload failed');
      }

      const data = await response.json();
      setUploadedData(data);
      setIsUploadModalOpen(false);
      toast.success(`Uploaded ${data.filename}: ${data.rows} rows, ${data.columns.length} columns`);
    } catch (error) {
      toast.error('Failed to upload file: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Run analysis
  const runAnalysis = async (analysisType, parameters) => {
    if (!uploadedData) {
      toast.error('Please upload data first');
      return;
    }

    setIsLoading(true);
    try {
      const response = await fetch(`${API_URL}/api/analyze/${analysisType}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data_id: uploadedData.id,
          analysis_type: analysisType,
          parameters
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.detail || 'Analysis failed');
      }

      const results = await response.json();
      setAnalysisResults(results);
      setGeneratedCode(results.code);
      setIsCodePanelOpen(true);
      toast.success('Analysis complete!');
    } catch (error) {
      toast.error('Analysis failed: ' + error.message);
    } finally {
      setIsLoading(false);
    }
  };

  // Download code
  const downloadCode = () => {
    if (!generatedCode) return;
    const blob = new Blob([generatedCode], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'analysis_code.py';
    a.click();
    URL.revokeObjectURL(url);
    toast.success('Code downloaded!');
  };

  // Download plot
  const downloadPlot = () => {
    if (!analysisResults?.plot_base64) return;
    const a = document.createElement('a');
    a.href = `data:image/png;base64,${analysisResults.plot_base64}`;
    a.download = 'analysis_plot.png';
    a.click();
    toast.success('Plot downloaded!');
  };

  return (
    <div className="h-screen bg-[#020408] flex flex-col overflow-hidden">
      {/* Header */}
      <header className="flex-shrink-0 h-14 border-b border-[#1E293B] bg-[#0B1121]/80 backdrop-blur-sm px-4 flex items-center justify-between z-50">
        <div className="flex items-center gap-4">
          <button
            data-testid="home-btn"
            onClick={() => navigate('/')}
            className="font-['Chivo'] font-bold text-lg hover:opacity-80 transition-opacity"
          >
            <span className="text-[#F8FAFC]">AG</span>
            <span className="text-[#00F0FF]">stat</span>
          </button>
          
          {/* Desktop Nav */}
          <nav className="hidden md:flex items-center gap-2">
            {uploadedData && (
              <div className="flex items-center gap-2 px-3 py-1 rounded bg-[#1E293B] text-sm">
                <FileSpreadsheet className="w-4 h-4 text-[#00F0FF]" />
                <span className="text-[#94A3B8]">{uploadedData.filename}</span>
                <span className="text-[#64748B]">({uploadedData.rows} rows)</span>
              </div>
            )}
          </nav>
        </div>

        <div className="flex items-center gap-2">
          <Button
            data-testid="upload-btn"
            onClick={() => setIsUploadModalOpen(true)}
            variant="outline"
            size="sm"
            className="bg-transparent border-[#1E293B] text-[#94A3B8] hover:border-[#00F0FF] hover:text-[#00F0FF]"
          >
            <Upload className="w-4 h-4 mr-2" />
            <span className="hidden sm:inline">Upload Data</span>
          </Button>
          
          {analysisResults && (
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button
                  data-testid="download-menu-btn"
                  variant="outline"
                  size="sm"
                  className="bg-transparent border-[#1E293B] text-[#94A3B8] hover:border-[#00F0FF] hover:text-[#00F0FF]"
                >
                  <Download className="w-4 h-4 mr-2" />
                  <span className="hidden sm:inline">Export</span>
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-[#0B1121] border-[#1E293B]">
                <DropdownMenuItem 
                  data-testid="download-code-btn"
                  onClick={downloadCode}
                  className="text-[#F8FAFC] hover:bg-[#1E293B] cursor-pointer"
                >
                  <Code className="w-4 h-4 mr-2" />
                  Download Code (.py)
                </DropdownMenuItem>
                {analysisResults.plot_base64 && (
                  <DropdownMenuItem 
                    data-testid="download-plot-btn"
                    onClick={downloadPlot}
                    className="text-[#F8FAFC] hover:bg-[#1E293B] cursor-pointer"
                  >
                    <BarChart3 className="w-4 h-4 mr-2" />
                    Download Plot (.png)
                  </DropdownMenuItem>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          )}
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Desktop Layout */}
        <div className="hidden md:flex flex-1">
          {/* Chat Panel - Left */}
          <div className="w-[400px] border-r border-[#1E293B] flex flex-col">
            <ChatPanel 
              sessionId={sessionId} 
              dataId={uploadedData?.id}
              onAnalysisRequest={(type, params) => runAnalysis(type, params)}
            />
          </div>

          {/* Main Panel - Center/Right */}
          <div className="flex-1 flex flex-col overflow-hidden">
            {/* Analysis Controls */}
            <div className="flex-shrink-0 p-4 border-b border-[#1E293B] bg-[#0B1121]/50">
              <div className="flex flex-wrap gap-2">
                {analysisTypes.map((analysis) => (
                  <AnalysisButton
                    key={analysis.id}
                    analysis={analysis}
                    uploadedData={uploadedData}
                    onRun={runAnalysis}
                    isLoading={isLoading}
                  />
                ))}
              </div>
            </div>

            {/* Results Area */}
            <div className="flex-1 overflow-auto p-4">
              {isLoading ? (
                <div className="flex items-center justify-center h-full">
                  <div className="text-center">
                    <Loader2 className="w-8 h-8 text-[#00F0FF] animate-spin mx-auto mb-4" />
                    <p className="text-[#94A3B8]">Running analysis...</p>
                  </div>
                </div>
              ) : analysisResults ? (
                <AnalysisPanel results={analysisResults} />
              ) : uploadedData ? (
                <DataPreview data={uploadedData} />
              ) : (
                <EmptyState onUpload={() => setIsUploadModalOpen(true)} />
              )}
            </div>

            {/* Code Panel - Bottom */}
            <AnimatePresence>
              {isCodePanelOpen && generatedCode && (
                <CodePanel 
                  code={generatedCode} 
                  onClose={() => setIsCodePanelOpen(false)}
                  onDownload={downloadCode}
                />
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Mobile Layout */}
        <div className="md:hidden flex-1 flex flex-col">
          <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1 flex flex-col">
            <TabsList className="flex-shrink-0 w-full justify-start bg-[#0B1121] border-b border-[#1E293B] rounded-none p-1">
              <TabsTrigger 
                value="chat" 
                className="data-[state=active]:bg-[#1E293B] data-[state=active]:text-[#00F0FF]"
              >
                Chat
              </TabsTrigger>
              <TabsTrigger 
                value="data" 
                className="data-[state=active]:bg-[#1E293B] data-[state=active]:text-[#00F0FF]"
              >
                Data
              </TabsTrigger>
              <TabsTrigger 
                value="results" 
                className="data-[state=active]:bg-[#1E293B] data-[state=active]:text-[#00F0FF]"
              >
                Results
              </TabsTrigger>
            </TabsList>

            <TabsContent value="chat" className="flex-1 m-0">
              <ChatPanel 
                sessionId={sessionId} 
                dataId={uploadedData?.id}
                onAnalysisRequest={(type, params) => runAnalysis(type, params)}
              />
            </TabsContent>

            <TabsContent value="data" className="flex-1 m-0 p-4 overflow-auto">
              {uploadedData ? (
                <DataPreview data={uploadedData} />
              ) : (
                <EmptyState onUpload={() => setIsUploadModalOpen(true)} />
              )}
            </TabsContent>

            <TabsContent value="results" className="flex-1 m-0 overflow-auto">
              {/* Mobile Analysis Controls */}
              <div className="p-4 border-b border-[#1E293B]">
                <div className="flex flex-wrap gap-2">
                  {analysisTypes.slice(0, 4).map((analysis) => (
                    <AnalysisButton
                      key={analysis.id}
                      analysis={analysis}
                      uploadedData={uploadedData}
                      onRun={runAnalysis}
                      isLoading={isLoading}
                      compact
                    />
                  ))}
                </div>
              </div>
              
              <div className="p-4">
                {isLoading ? (
                  <div className="flex items-center justify-center py-20">
                    <Loader2 className="w-8 h-8 text-[#00F0FF] animate-spin" />
                  </div>
                ) : analysisResults ? (
                  <AnalysisPanel results={analysisResults} />
                ) : (
                  <p className="text-center text-[#64748B] py-10">
                    Run an analysis to see results
                  </p>
                )}
              </div>
            </TabsContent>
          </Tabs>
        </div>
      </div>

      {/* File Upload Modal */}
      <FileUploadModal 
        isOpen={isUploadModalOpen}
        onClose={() => setIsUploadModalOpen(false)}
        onUpload={handleFileUpload}
        isLoading={isLoading}
      />
    </div>
  );
}

// Analysis Button Component
function AnalysisButton({ analysis, uploadedData, onRun, isLoading, compact }) {
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  const [params, setParams] = useState({});

  const handleRun = () => {
    if (!uploadedData) {
      toast.error('Please upload data first');
      return;
    }
    
    // Set default parameters based on analysis type
    let defaultParams = { ...params };
    
    if (analysis.id === 'anova') {
      if (!defaultParams.dependent_var && uploadedData.column_info) {
        const numericCols = uploadedData.column_info.filter(c => c.type === 'numeric');
        const categoricalCols = uploadedData.column_info.filter(c => c.type === 'categorical');
        if (numericCols.length > 0) defaultParams.dependent_var = numericCols[0].name;
        if (categoricalCols.length > 0) defaultParams.independent_var = categoricalCols[0].name;
      }
    } else if (analysis.id === 'pca' || analysis.id === 'clustering') {
      if (!defaultParams.numeric_cols && uploadedData.column_info) {
        defaultParams.numeric_cols = uploadedData.column_info
          .filter(c => c.type === 'numeric')
          .map(c => c.name);
      }
      if (analysis.id === 'clustering' && !defaultParams.n_clusters) {
        defaultParams.n_clusters = 3;
      }
    } else if (analysis.id === 'descriptive' || analysis.id === 'correlation') {
      if (!defaultParams.columns && uploadedData.column_info) {
        defaultParams.columns = uploadedData.column_info
          .filter(c => c.type === 'numeric')
          .map(c => c.name);
      }
    } else if (analysis.id === 'normality') {
      if (!defaultParams.column && uploadedData.column_info) {
        const numericCols = uploadedData.column_info.filter(c => c.type === 'numeric');
        if (numericCols.length > 0) defaultParams.column = numericCols[0].name;
      }
    }
    
    onRun(analysis.id, defaultParams);
  };

  return (
    <Button
      data-testid={`analysis-btn-${analysis.id}`}
      onClick={handleRun}
      disabled={isLoading || !uploadedData}
      variant="outline"
      size={compact ? "sm" : "default"}
      className={`
        bg-[#1E293B]/50 border-[#1E293B] text-[#94A3B8] 
        hover:border-[#00F0FF] hover:text-[#00F0FF] hover:bg-[#00F0FF]/10
        disabled:opacity-50 disabled:cursor-not-allowed
        ${compact ? 'text-xs px-2 py-1' : ''}
      `}
    >
      <analysis.icon className={`${compact ? 'w-3 h-3 mr-1' : 'w-4 h-4 mr-2'}`} />
      {analysis.name}
    </Button>
  );
}

// Empty State Component
function EmptyState({ onUpload }) {
  return (
    <div className="flex flex-col items-center justify-center h-full text-center p-8">
      <div className="w-20 h-20 rounded-full bg-[#1E293B] flex items-center justify-center mb-6">
        <FileSpreadsheet className="w-10 h-10 text-[#00F0FF]" />
      </div>
      <h3 className="font-['Chivo'] font-bold text-xl text-[#F8FAFC] mb-2">
        No Data Loaded
      </h3>
      <p className="text-[#94A3B8] mb-6 max-w-md">
        Upload a CSV or Excel file to start your statistical analysis
      </p>
      <Button
        data-testid="empty-state-upload-btn"
        onClick={onUpload}
        className="bg-[#00F0FF] text-black hover:bg-[#00CCFF] font-bold rounded-sm"
      >
        <Upload className="w-4 h-4 mr-2" />
        Upload Data
      </Button>
    </div>
  );
}
