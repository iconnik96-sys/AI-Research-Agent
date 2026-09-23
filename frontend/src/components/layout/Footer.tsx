import React from 'react';

export const Footer: React.FC = () => {
  return (
    <footer className="mt-auto border-t border-slate-800/80 py-6 text-center text-xs text-slate-500">
      <div className="max-w-6xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-3">
        <p>
          AI Research Agent • Autonomous Web Search, Vector RAG & Claim Verification
        </p>
        <div className="flex items-center gap-4 text-slate-400 font-mono text-[11px]">
          <span>FastAPI</span>
          <span>•</span>
          <span>pgvector</span>
          <span>•</span>
          <span>Tavily</span>
          <span>•</span>
          <span>OpenAI</span>
        </div>
      </div>
    </footer>
  );
};
