import React from 'react';
import { Brain, Search, HelpCircle } from 'lucide-react';
import { ResearchPlan } from '../../types/api';

interface PlanPanelProps {
  plan: ResearchPlan | null;
}

export const PlanPanel: React.FC<PlanPanelProps> = ({ plan }) => {
  if (!plan || !plan.sub_questions || plan.sub_questions.length === 0) {
    return (
      <div className="text-center py-12 border border-slate-800 rounded-xl bg-slate-900/40">
        <Brain className="mx-auto h-8 w-8 text-slate-500 mb-2" />
        <p className="text-sm text-slate-400">No structured research plan available for this session.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="pb-3 border-b border-slate-800">
        <h3 className="text-base font-semibold text-slate-100">
          Decomposed Research Plan ({plan.sub_questions.length} Sub-Questions)
        </h3>
        <p className="text-xs text-slate-400 mt-0.5">
          The planning engine broke down the original inquiry into orthogonal sub-questions with targeted web queries.
        </p>
      </div>

      {/* Original Question Card */}
      <div className="p-4 rounded-xl border border-indigo-500/20 bg-indigo-500/5">
        <span className="text-xs uppercase font-mono tracking-wider font-semibold text-indigo-400 block mb-1">
          Target Objective
        </span>
        <p className="text-sm font-medium text-slate-200">
          {plan.original_question}
        </p>
      </div>

      {/* Sub-Questions Breakdown */}
      <div className="space-y-3.5">
        {plan.sub_questions.map((subQ, idx) => (
          <div
            key={subQ.id || idx}
            className="rounded-xl border border-slate-800 bg-slate-850/60 p-4 sm:p-5"
          >
            <div className="flex items-center gap-2 mb-2">
              <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
                {subQ.id || `Q${idx + 1}`}
              </span>
              <h4 className="text-sm font-semibold text-slate-100">
                {subQ.question}
              </h4>
            </div>

            {/* Keyword Search Query */}
            <div className="mt-2.5 flex items-start gap-2 text-xs">
              <Search size={13} className="text-indigo-400 shrink-0 mt-0.5" />
              <div className="text-slate-300">
                <span className="text-slate-400 font-medium">Search Query: </span>
                <code className="px-1.5 py-0.5 rounded bg-slate-900 border border-slate-800 font-mono text-indigo-300">
                  {subQ.search_query}
                </code>
              </div>
            </div>

            {/* Reasoning / Rationale */}
            {subQ.reason && (
              <div className="mt-2 flex items-start gap-2 text-xs">
                <HelpCircle size={13} className="text-slate-500 shrink-0 mt-0.5" />
                <p className="text-slate-400">
                  <span className="font-medium text-slate-300">Rationale: </span>
                  {subQ.reason}
                </p>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};
