import React, { useEffect, useState } from 'react';
import {
  Brain,
  Globe,
  FileText,
  Database,
  ShieldCheck,
  PenTool,
  Clock,
  CheckCircle2,
  Loader2,
  XCircle,
  LucideIcon,
} from 'lucide-react';

interface ResearchProgressProps {
  startTime: number;
  onCancel?: () => void;
}

interface StageConfig {
  id: string;
  name: string;
  description: string;
  icon: LucideIcon;
  approxStartSec: number;
}

const STAGES: StageConfig[] = [
  {
    id: 'planning',
    name: 'Planning',
    description: 'Decomposing question into structured sub-questions & queries',
    icon: Brain,
    approxStartSec: 0,
  },
  {
    id: 'searching',
    name: 'Searching',
    description: 'Querying Tavily search API across multiple angles',
    icon: Globe,
    approxStartSec: 4,
  },
  {
    id: 'extracting',
    name: 'Extracting',
    description: 'Scraping source contents and stripping HTML boilerplate',
    icon: FileText,
    approxStartSec: 9,
  },
  {
    id: 'retrieving',
    name: 'Retrieving evidence',
    description: 'Chunking, embedding & vector similarity search via pgvector',
    icon: Database,
    approxStartSec: 15,
  },
  {
    id: 'verifying',
    name: 'Verifying claims',
    description: 'Extracting factual claims & independent claim verification',
    icon: ShieldCheck,
    approxStartSec: 22,
  },
  {
    id: 'writing',
    name: 'Writing report',
    description: 'Synthesizing comprehensive findings with grounded citations',
    icon: PenTool,
    approxStartSec: 30,
  },
];

export const ResearchProgress: React.FC<ResearchProgressProps> = ({
  startTime,
  onCancel,
}) => {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    const updateElapsed = () => {
      const now = Date.now();
      setElapsed(Math.floor((now - startTime) / 1000));
    };

    updateElapsed();
    const interval = setInterval(updateElapsed, 500);
    return () => clearInterval(interval);
  }, [startTime]);

  // Determine current active stage index based on elapsed seconds
  let currentStageIndex = 0;
  for (let i = STAGES.length - 1; i >= 0; i--) {
    if (elapsed >= STAGES[i].approxStartSec) {
      currentStageIndex = i;
      break;
    }
  }

  const formatTime = (seconds: number) => {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${s.toString().padStart(2, '0')}`;
  };

  return (
    <div className="rounded-2xl border border-slate-700/80 bg-slate-800/70 backdrop-blur-sm p-6 sm:p-8 shadow-xl">
      {/* Top Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 border-b border-slate-700/60">
        <div>
          <div className="flex items-center gap-2">
            <span className="relative flex h-2.5 w-2.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-indigo-400 opacity-75"></span>
              <span className="relative inline-flex rounded-full h-2.5 w-2.5 bg-indigo-500"></span>
            </span>
            <h3 className="font-semibold text-slate-100 text-base">
              Autonomous Research Pipeline Active
            </h3>
          </div>
          <p className="text-xs text-slate-400 mt-1">
            Running multi-step research workflow (estimated progress visualization)
          </p>
        </div>

        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-900/60 border border-slate-700 font-mono text-xs text-indigo-300">
            <Clock size={13} className="text-indigo-400" />
            <span>Elapsed: {formatTime(elapsed)}</span>
          </div>

          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium text-slate-400 hover:text-rose-400 hover:bg-rose-500/10 border border-transparent hover:border-rose-500/20 transition-all"
            >
              <XCircle size={13} />
              <span>Cancel</span>
            </button>
          )}
        </div>
      </div>

      {/* Progress Steps List */}
      <div className="mt-6 space-y-3.5">
        {STAGES.map((stage, idx) => {
          const isDone = idx < currentStageIndex;
          const isCurrent = idx === currentStageIndex;
          const Icon = stage.icon;

          return (
            <div
              key={stage.id}
              className={`flex items-start gap-4 p-3 rounded-xl transition-all ${
                isCurrent
                  ? 'bg-indigo-500/10 border border-indigo-500/30 shadow-sm'
                  : isDone
                  ? 'bg-slate-900/30 border border-transparent opacity-85'
                  : 'border border-transparent opacity-40'
              }`}
            >
              {/* Step indicator icon */}
              <div
                className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 transition-colors ${
                  isDone
                    ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                    : isCurrent
                    ? 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/40 ring-2 ring-indigo-500/20'
                    : 'bg-slate-800 text-slate-500 border border-slate-700'
                }`}
              >
                {isDone ? (
                  <CheckCircle2 size={16} className="text-emerald-400" />
                ) : isCurrent ? (
                  <Loader2 size={16} className="animate-spin text-indigo-400" />
                ) : (
                  <Icon size={15} />
                )}
              </div>

              {/* Step content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <span
                    className={`text-sm font-medium ${
                      isCurrent
                        ? 'text-indigo-200 font-semibold'
                        : isDone
                        ? 'text-slate-300'
                        : 'text-slate-500'
                    }`}
                  >
                    {stage.name}
                  </span>
                  {isCurrent && (
                    <span className="text-[10px] font-mono uppercase px-1.5 py-0.2 rounded bg-indigo-500/20 text-indigo-300">
                      Processing
                    </span>
                  )}
                </div>
                <p
                  className={`text-xs mt-0.5 truncate ${
                    isCurrent ? 'text-indigo-300/80' : 'text-slate-400'
                  }`}
                >
                  {stage.description}
                </p>
              </div>
            </div>
          );
        })}
      </div>

      {/* Explanatory footer note */}
      <div className="mt-6 pt-4 border-t border-slate-700/60 flex items-center justify-between text-[11px] text-slate-400">
        <span>Typical research synthesis takes 15–35 seconds.</span>
        <span className="font-mono text-slate-500">FastAPI Pipeline</span>
      </div>
    </div>
  );
};
