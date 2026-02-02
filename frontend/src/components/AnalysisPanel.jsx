import React from 'react';
import { motion } from 'framer-motion';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Badge } from '@/components/ui/badge';
import { CheckCircle, XCircle, AlertTriangle } from 'lucide-react';

export default function AnalysisPanel({ results }) {
  if (!results) return null;

  const { analysis_type, results: data, plot_base64 } = results;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="space-y-6"
    >
      {/* Analysis Type Header */}
      <div className="flex items-center gap-3">
        <h2 className="font-['Chivo'] font-bold text-xl text-[#F8FAFC] uppercase tracking-wider">
          {analysis_type} Results
        </h2>
        <Badge className="bg-[#10B981]/20 text-[#10B981] border-[#10B981]/30">
          Complete
        </Badge>
      </div>

      {/* Plot */}
      {plot_base64 && (
        <div data-testid="analysis-plot" className="border border-[#1E293B] rounded-sm overflow-hidden bg-[#020408]">
          <img 
            src={`data:image/png;base64,${plot_base64}`}
            alt="Analysis Plot"
            className="w-full h-auto"
          />
        </div>
      )}

      {/* Results based on analysis type */}
      {analysis_type === 'anova' && <AnovaResults data={data} />}
      {analysis_type === 'pca' && <PCAResults data={data} />}
      {analysis_type === 'clustering' && <ClusteringResults data={data} />}
      {analysis_type === 'descriptive' && <DescriptiveResults data={data} />}
      {analysis_type === 'correlation' && <CorrelationResults data={data} />}
      {analysis_type === 'normality' && <NormalityResults data={data} />}
    </motion.div>
  );
}

function AnovaResults({ data }) {
  const { anova_table, assumptions, tukey_hsd, interpretation } = data;

  return (
    <div className="space-y-6">
      {/* Interpretation */}
      {interpretation && (
        <div className="p-4 bg-[#0B1121] border border-[#1E293B] rounded-sm">
          <p className="text-[#94A3B8]">{interpretation}</p>
        </div>
      )}

      {/* Assumption Checks */}
      {assumptions && (
        <div className="space-y-3">
          <h3 className="font-['Chivo'] font-bold text-sm text-[#00F0FF] uppercase tracking-wider">
            Assumption Checks
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <AssumptionCard 
              title="Normality (Shapiro-Wilk)"
              passed={assumptions.shapiro_wilk?.normal}
              statistic={assumptions.shapiro_wilk?.statistic}
              pValue={assumptions.shapiro_wilk?.p_value}
            />
            <AssumptionCard 
              title="Homogeneity (Levene)"
              passed={assumptions.levene?.homogeneous}
              statistic={assumptions.levene?.statistic}
              pValue={assumptions.levene?.p_value}
            />
          </div>
        </div>
      )}

      {/* ANOVA Table */}
      {anova_table && (
        <div className="space-y-3">
          <h3 className="font-['Chivo'] font-bold text-sm text-[#00F0FF] uppercase tracking-wider">
            ANOVA Table
          </h3>
          <div className="border border-[#1E293B] rounded-sm overflow-hidden">
            <table className="w-full">
              <thead>
                <tr className="bg-[#1E293B]">
                  <th className="px-4 py-2 text-left text-[#00F0FF] text-sm font-semibold">Source</th>
                  <th className="px-4 py-2 text-right text-[#00F0FF] text-sm font-semibold">Sum Sq</th>
                  <th className="px-4 py-2 text-right text-[#00F0FF] text-sm font-semibold">df</th>
                  <th className="px-4 py-2 text-right text-[#00F0FF] text-sm font-semibold">F</th>
                  <th className="px-4 py-2 text-right text-[#00F0FF] text-sm font-semibold">p-value</th>
                </tr>
              </thead>
              <tbody>
                {anova_table.map((row, index) => (
                  <tr key={index} className="border-t border-[#1E293B] hover:bg-[#1E293B]/30">
                    <td className="px-4 py-2 text-[#F8FAFC] text-sm">{row.source || row.index}</td>
                    <td className="px-4 py-2 text-right text-[#94A3B8] font-mono text-sm">
                      {row.sum_sq?.toFixed(3) || '—'}
                    </td>
                    <td className="px-4 py-2 text-right text-[#94A3B8] font-mono text-sm">
                      {row.df?.toFixed(0) || '—'}
                    </td>
                    <td className="px-4 py-2 text-right text-[#94A3B8] font-mono text-sm">
                      {row.F?.toFixed(3) || '—'}
                    </td>
                    <td className={`px-4 py-2 text-right font-mono text-sm ${
                      (row.p_value !== null && row.p_value < 0.05) ? 'text-[#00F0FF] font-semibold' : 'text-[#94A3B8]'
                    }`}>
                      {row.p_value !== null ? row.p_value.toExponential(3) : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Tukey HSD */}
      {tukey_hsd && (
        <div className="space-y-3">
          <h3 className="font-['Chivo'] font-bold text-sm text-[#00F0FF] uppercase tracking-wider">
            Tukey HSD Post-hoc
          </h3>
          <pre className="p-4 bg-[#0B1121] border border-[#1E293B] rounded-sm text-[#94A3B8] text-xs font-mono overflow-x-auto">
            {tukey_hsd}
          </pre>
        </div>
      )}
    </div>
  );
}

function PCAResults({ data }) {
  const { explained_variance_ratio, cumulative_variance, loadings, n_components } = data;

  return (
    <div className="space-y-6">
      {/* Variance Explained */}
      <div className="space-y-3">
        <h3 className="font-['Chivo'] font-bold text-sm text-[#00F0FF] uppercase tracking-wider">
          Variance Explained
        </h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {explained_variance_ratio?.map((ratio, index) => (
            <div key={index} className="p-4 bg-[#0B1121] border border-[#1E293B] rounded-sm text-center">
              <div className="text-2xl font-['Chivo'] font-bold text-[#00F0FF]">
                {(ratio * 100).toFixed(1)}%
              </div>
              <div className="text-xs text-[#64748B] uppercase">PC{index + 1}</div>
            </div>
          ))}
          <div className="p-4 bg-[#10B981]/10 border border-[#10B981]/30 rounded-sm text-center">
            <div className="text-2xl font-['Chivo'] font-bold text-[#10B981]">
              {(cumulative_variance * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-[#64748B] uppercase">Cumulative</div>
          </div>
        </div>
      </div>

      {/* Loadings */}
      {loadings && (
        <div className="space-y-3">
          <h3 className="font-['Chivo'] font-bold text-sm text-[#00F0FF] uppercase tracking-wider">
            Component Loadings
          </h3>
          <div className="border border-[#1E293B] rounded-sm overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="bg-[#1E293B]">
                  <th className="px-4 py-2 text-left text-[#00F0FF] text-sm font-semibold">Variable</th>
                  {Object.keys(loadings).map(pc => (
                    <th key={pc} className="px-4 py-2 text-right text-[#00F0FF] text-sm font-semibold">{pc}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.keys(loadings.PC1 || {}).map((variable, index) => (
                  <tr key={index} className="border-t border-[#1E293B] hover:bg-[#1E293B]/30">
                    <td className="px-4 py-2 text-[#F8FAFC] text-sm">{variable}</td>
                    {Object.keys(loadings).map(pc => (
                      <td key={pc} className={`px-4 py-2 text-right font-mono text-sm ${
                        Math.abs(loadings[pc][variable]) > 0.5 ? 'text-[#00F0FF] font-semibold' : 'text-[#94A3B8]'
                      }`}>
                        {loadings[pc][variable]?.toFixed(3)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

function ClusteringResults({ data }) {
  const { method, n_clusters, silhouette_score, cluster_sizes } = data;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
        <div className="p-4 bg-[#0B1121] border border-[#1E293B] rounded-sm text-center">
          <div className="text-xl font-['Chivo'] font-bold text-[#F8FAFC] capitalize">{method}</div>
          <div className="text-xs text-[#64748B] uppercase">Method</div>
        </div>
        <div className="p-4 bg-[#0B1121] border border-[#1E293B] rounded-sm text-center">
          <div className="text-xl font-['Chivo'] font-bold text-[#00F0FF]">{n_clusters}</div>
          <div className="text-xs text-[#64748B] uppercase">Clusters</div>
        </div>
        {silhouette_score !== undefined && (
          <div className="p-4 bg-[#0B1121] border border-[#1E293B] rounded-sm text-center">
            <div className={`text-xl font-['Chivo'] font-bold ${
              silhouette_score > 0.5 ? 'text-[#10B981]' : silhouette_score > 0.25 ? 'text-[#F59E0B]' : 'text-[#EF4444]'
            }`}>
              {silhouette_score.toFixed(3)}
            </div>
            <div className="text-xs text-[#64748B] uppercase">Silhouette</div>
          </div>
        )}
      </div>

      {cluster_sizes && (
        <div className="space-y-3">
          <h3 className="font-['Chivo'] font-bold text-sm text-[#00F0FF] uppercase tracking-wider">
            Cluster Sizes
          </h3>
          <div className="flex flex-wrap gap-3">
            {Object.entries(cluster_sizes).map(([cluster, size]) => (
              <div key={cluster} className="px-4 py-2 bg-[#1E293B] rounded-sm">
                <span className="text-[#00F0FF] font-semibold">{cluster}:</span>
                <span className="text-[#F8FAFC] ml-2">{size} samples</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

function DescriptiveResults({ data }) {
  const { statistics, grouped_by } = data;
  
  return (
    <div className="space-y-3">
      <h3 className="font-['Chivo'] font-bold text-sm text-[#00F0FF] uppercase tracking-wider">
        Descriptive Statistics {grouped_by && `(by ${grouped_by})`}
      </h3>
      <pre className="p-4 bg-[#0B1121] border border-[#1E293B] rounded-sm text-[#94A3B8] text-xs font-mono overflow-x-auto">
        {JSON.stringify(statistics, null, 2)}
      </pre>
    </div>
  );
}

function CorrelationResults({ data }) {
  const { method, correlation_matrix } = data;
  
  return (
    <div className="space-y-3">
      <h3 className="font-['Chivo'] font-bold text-sm text-[#00F0FF] uppercase tracking-wider">
        {method?.charAt(0).toUpperCase() + method?.slice(1)} Correlation Matrix
      </h3>
      <div className="border border-[#1E293B] rounded-sm overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-[#1E293B]">
              <th className="px-4 py-2 text-left text-[#00F0FF] text-sm font-semibold">Variable</th>
              {correlation_matrix && Object.keys(correlation_matrix).map(col => (
                <th key={col} className="px-4 py-2 text-right text-[#00F0FF] text-sm font-semibold">{col}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {correlation_matrix && Object.entries(correlation_matrix).map(([rowVar, values]) => (
              <tr key={rowVar} className="border-t border-[#1E293B] hover:bg-[#1E293B]/30">
                <td className="px-4 py-2 text-[#F8FAFC] text-sm font-medium">{rowVar}</td>
                {Object.values(values).map((val, i) => (
                  <td key={i} className={`px-4 py-2 text-right font-mono text-sm ${
                    Math.abs(val) > 0.7 ? 'text-[#00F0FF] font-semibold' : 
                    Math.abs(val) > 0.4 ? 'text-[#F59E0B]' : 'text-[#94A3B8]'
                  }`}>
                    {val?.toFixed(3)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function NormalityResults({ data }) {
  const { tests } = data;

  return (
    <div className="space-y-3">
      <h3 className="font-['Chivo'] font-bold text-sm text-[#00F0FF] uppercase tracking-wider">
        Normality Test Results
      </h3>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {tests && Object.entries(tests).map(([group, result]) => (
          <AssumptionCard 
            key={group}
            title={`${group} (n=${result.n})`}
            passed={result.normal}
            statistic={result.shapiro_statistic}
            pValue={result.p_value}
          />
        ))}
      </div>
    </div>
  );
}

function AssumptionCard({ title, passed, statistic, pValue }) {
  return (
    <div className={`p-4 rounded-sm border ${
      passed ? 'bg-[#10B981]/10 border-[#10B981]/30' : 'bg-[#EF4444]/10 border-[#EF4444]/30'
    }`}>
      <div className="flex items-center justify-between mb-2">
        <span className="text-sm font-medium text-[#F8FAFC]">{title}</span>
        {passed ? (
          <CheckCircle className="w-5 h-5 text-[#10B981]" />
        ) : (
          <AlertTriangle className="w-5 h-5 text-[#EF4444]" />
        )}
      </div>
      <div className="flex gap-4 text-xs">
        <div>
          <span className="text-[#64748B]">Statistic: </span>
          <span className="text-[#94A3B8] font-mono">{statistic?.toFixed(4)}</span>
        </div>
        <div>
          <span className="text-[#64748B]">p-value: </span>
          <span className={`font-mono ${passed ? 'text-[#10B981]' : 'text-[#EF4444]'}`}>
            {pValue?.toFixed(4)}
          </span>
        </div>
      </div>
      <p className="text-xs text-[#64748B] mt-2">
        {passed ? '✓ Assumption met (p > 0.05)' : '⚠ Assumption violated (p < 0.05)'}
      </p>
    </div>
  );
}
