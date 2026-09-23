import React from 'react';
import { BookOpen, ExternalLink, Bookmark } from 'lucide-react';
import { ResearchReport } from '../../types/api';
import { ReportSectionView } from './ReportSection';
import { CopyButton } from '../common/CopyButton';

interface ReportViewProps {
  report: ResearchReport;
  onSelectCitation: (sourceId: string) => void;
}

export const ReportView: React.FC<ReportViewProps> = ({
  report,
  onSelectCitation,
}) => {
  return (
    <div className="space-y-8">
      {/* Report Header */}
      <div className="flex flex-col sm:flex-row sm:items-start justify-between gap-4 pb-6 border-b border-slate-800">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-indigo-500/10 text-indigo-400 border border-indigo-500/20 text-xs font-mono mb-2">
            <BookOpen size={12} />
            <span>Synthesized Research Report</span>
          </div>
          <h2 className="text-xl sm:text-2xl font-bold text-slate-100 tracking-tight">
            {report.title}
          </h2>
        </div>

        <div className="shrink-0">
          <CopyButton report={report} />
        </div>
      </div>

      {/* Executive Summary */}
      {report.summary && (
        <div className="rounded-xl border border-indigo-500/20 bg-indigo-500/5 p-5 sm:p-6">
          <h3 className="text-xs uppercase font-mono tracking-wider font-semibold text-indigo-400 mb-2.5">
            Executive Summary
          </h3>
          <p className="text-sm sm:text-base text-slate-200 leading-relaxed font-normal">
            {report.summary}
          </p>
        </div>
      )}

      {/* Structured Sections */}
      {report.sections && report.sections.length > 0 && (
        <div className="space-y-8">
          {report.sections.map((section, idx) => (
            <ReportSectionView
              key={idx}
              section={section}
              onSelectCitation={onSelectCitation}
            />
          ))}
        </div>
      )}

      {/* References Index */}
      {report.sources && report.sources.length > 0 && (
        <div className="pt-6 border-t border-slate-800">
          <div className="flex items-center gap-2 mb-4 text-slate-400">
            <Bookmark size={15} className="text-indigo-400" />
            <h3 className="text-xs uppercase font-mono tracking-wider font-semibold text-slate-400">
              Report References ({report.sources.length})
            </h3>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            {report.sources.map((src) => (
              <div
                key={src.id}
                onClick={() => onSelectCitation(src.id)}
                className="group p-3 rounded-lg border border-slate-800 bg-slate-800/40 hover:bg-slate-800/80 hover:border-slate-700 cursor-pointer transition-all flex items-start gap-2.5"
              >
                <span className="font-mono text-xs font-semibold px-1.5 py-0.5 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/30 shrink-0">
                  {src.id}
                </span>
                <div className="flex-1 min-w-0">
                  <h4 className="text-xs font-medium text-slate-200 group-hover:text-indigo-300 truncate">
                    {src.title}
                  </h4>
                  <a
                    href={src.url}
                    target="_blank"
                    rel="noreferrer noopener"
                    onClick={(e) => e.stopPropagation()}
                    className="inline-flex items-center gap-1 text-[11px] text-slate-400 hover:text-indigo-400 transition-colors mt-0.5 truncate max-w-full"
                  >
                    <span className="truncate">{src.url}</span>
                    <ExternalLink size={10} className="shrink-0" />
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
