import React, { useState } from 'react';
import {
  BookOpen,
  ShieldCheck,
  Globe,
  Brain,
  Hash,
  Sparkles,
} from 'lucide-react';
import { ResearchResponse } from '../../types/api';
import { ReportView } from './ReportView';
import { ClaimsPanel } from './ClaimsPanel';
import { SourcesPanel } from './SourcesPanel';
import { PlanPanel } from './PlanPanel';

interface ResearchDossierProps {
  data: ResearchResponse;
}

export type DossierTab = 'report' | 'claims' | 'sources' | 'plan';

export const ResearchDossier: React.FC<ResearchDossierProps> = ({ data }) => {
  const [activeTab, setActiveTab] = useState<DossierTab>('report');
  const [highlightedSourceId, setHighlightedSourceId] = useState<string | null>(null);

  // Handler when clicking inline citation in report: switch to sources tab & highlight
  const handleSelectCitation = (sourceId: string) => {
    setHighlightedSourceId(sourceId);
    setActiveTab('sources');
  };

  const claimsCount = data.claims ? data.claims.length : 0;
  const sourcesCount = data.sources ? data.sources.length : 0;
  const planSubQCount = data.plan?.sub_questions ? data.plan.sub_questions.length : 0;

  return (
    <div className="rounded-2xl border border-slate-800 bg-slate-900/60 shadow-2xl overflow-hidden">
      {/* Session Metadata Banner */}
      <div className="bg-slate-850/80 border-b border-slate-800 px-5 py-4 sm:px-6 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400"></span>
            <span>Completed</span>
          </span>

          {data.session_id && (
            <span className="inline-flex items-center gap-1 text-[11px] font-mono text-slate-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
              <Hash size={11} className="text-slate-500" />
              <span className="truncate max-w-[160px] sm:max-w-none">{data.session_id}</span>
            </span>
          )}
        </div>

        <div className="flex items-center gap-4 text-xs text-slate-400">
          <span>
            <strong className="text-slate-200">{sourcesCount}</strong> Sources
          </span>
          <span>•</span>
          <span>
            <strong className="text-slate-200">{claimsCount}</strong> Verified Claims
          </span>
          {planSubQCount > 0 && (
            <>
              <span>•</span>
              <span>
                <strong className="text-slate-200">{planSubQCount}</strong> Sub-Questions
              </span>
            </>
          )}
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="border-b border-slate-800 bg-slate-900/40 px-4 sm:px-6 flex items-center gap-2 overflow-x-auto">
        <button
          type="button"
          onClick={() => setActiveTab('report')}
          className={`flex items-center gap-2 py-3.5 px-3 text-xs sm:text-sm font-medium border-b-2 whitespace-nowrap transition-colors ${
            activeTab === 'report'
              ? 'border-indigo-500 text-indigo-300 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <BookOpen size={15} />
          <span>Research Report</span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('claims')}
          className={`flex items-center gap-2 py-3.5 px-3 text-xs sm:text-sm font-medium border-b-2 whitespace-nowrap transition-colors ${
            activeTab === 'claims'
              ? 'border-indigo-500 text-indigo-300 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <ShieldCheck size={15} />
          <span>Verified Claims</span>
          <span className="ml-0.5 px-1.5 py-0.2 rounded-full text-[10px] font-mono bg-slate-800 text-slate-300">
            {claimsCount}
          </span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('sources')}
          className={`flex items-center gap-2 py-3.5 px-3 text-xs sm:text-sm font-medium border-b-2 whitespace-nowrap transition-colors ${
            activeTab === 'sources'
              ? 'border-indigo-500 text-indigo-300 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Globe size={15} />
          <span>Sources</span>
          <span className="ml-0.5 px-1.5 py-0.2 rounded-full text-[10px] font-mono bg-slate-800 text-slate-300">
            {sourcesCount}
          </span>
        </button>

        <button
          type="button"
          onClick={() => setActiveTab('plan')}
          className={`flex items-center gap-2 py-3.5 px-3 text-xs sm:text-sm font-medium border-b-2 whitespace-nowrap transition-colors ${
            activeTab === 'plan'
              ? 'border-indigo-500 text-indigo-300 font-semibold'
              : 'border-transparent text-slate-400 hover:text-slate-200'
          }`}
        >
          <Brain size={15} />
          <span>Research Plan</span>
          {planSubQCount > 0 && (
            <span className="ml-0.5 px-1.5 py-0.2 rounded-full text-[10px] font-mono bg-slate-800 text-slate-300">
              {planSubQCount}
            </span>
          )}
        </button>
      </div>

      {/* Tab Content Panel */}
      <div className="p-5 sm:p-8">
        {activeTab === 'report' && (
          data.report ? (
            <ReportView
              report={data.report}
              onSelectCitation={handleSelectCitation}
            />
          ) : (
            <div className="text-center py-12 text-slate-400 text-sm">
              <Sparkles className="mx-auto h-8 w-8 text-slate-500 mb-2" />
              <p>No synthesized report returned for this session.</p>
            </div>
          )
        )}

        {activeTab === 'claims' && (
          <ClaimsPanel claims={data.claims || []} />
        )}

        {activeTab === 'sources' && (
          <SourcesPanel
            sources={data.sources || []}
            documents={data.documents || []}
            reportSources={data.report?.sources || []}
            highlightedSourceId={highlightedSourceId}
          />
        )}

        {activeTab === 'plan' && (
          <PlanPanel plan={data.plan || null} />
        )}
      </div>
    </div>
  );
};
