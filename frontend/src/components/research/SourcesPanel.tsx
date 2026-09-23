import React, { useState, useEffect, useRef } from 'react';
import { ExternalLink, ChevronDown, ChevronUp, Globe, FileText } from 'lucide-react';
import { SourceItem, Document, SourceReference } from '../../types/api';

interface SourcesPanelProps {
  sources: SourceItem[];
  documents?: Document[];
  reportSources?: SourceReference[];
  highlightedSourceId?: string | null;
}

export const SourcesPanel: React.FC<SourcesPanelProps> = ({
  sources,
  documents = [],
  reportSources = [],
  highlightedSourceId = null,
}) => {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const cardRefs = useRef<Record<string, HTMLDivElement | null>>({});

  // Helper to extract clean domain from URL
  const getDomain = (urlStr: string) => {
    try {
      const parsed = new URL(urlStr);
      return parsed.hostname.replace(/^www\./, '');
    } catch {
      return urlStr;
    }
  };

  // Map report source IDs (S1, S2, etc.) to sources by URL
  const getSourceIdForUrl = (url: string, index: number): string => {
    const matched = reportSources.find(
      (rs) => rs.url.toLowerCase() === url.toLowerCase()
    );
    return matched ? matched.id : `S${index + 1}`;
  };

  // Scroll to and pulse highlighted source if selected
  useEffect(() => {
    if (highlightedSourceId && cardRefs.current[highlightedSourceId]) {
      const el = cardRefs.current[highlightedSourceId];
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }
  }, [highlightedSourceId]);

  if (!sources || sources.length === 0) {
    return (
      <div className="text-center py-12 border border-slate-800 rounded-xl bg-slate-900/40">
        <Globe className="mx-auto h-8 w-8 text-slate-500 mb-2" />
        <p className="text-sm text-slate-400">No web sources collected for this research question.</p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="pb-3 border-b border-slate-800">
        <h3 className="text-base font-semibold text-slate-100">
          Collected Web Sources ({sources.length})
        </h3>
        <p className="text-xs text-slate-400 mt-0.5">
          Sources discovered through multi-query web search, scraped and indexed into vector embeddings.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-3.5">
        {sources.map((source, index) => {
          const sourceId = getSourceIdForUrl(source.url, index);
          const isHighlighted = highlightedSourceId === sourceId;
          const isExpanded = expandedIndex === index;
          const domain = getDomain(source.url);

          // Find matching document for full text preview if requested
          const matchedDoc = documents.find((d) => d.url === source.url);

          return (
            <div
              key={index}
              ref={(el) => {
                cardRefs.current[sourceId] = el;
              }}
              className={`rounded-xl border transition-all p-4 sm:p-5 ${
                isHighlighted
                  ? 'border-indigo-500 bg-indigo-950/30 ring-2 ring-indigo-500/30 citation-highlight-target'
                  : 'border-slate-800 bg-slate-850/60 hover:border-slate-700'
              }`}
            >
              {/* Header row: Source ID, domain, relevance score */}
              <div className="flex items-center justify-between flex-wrap gap-2 mb-2">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-xs font-bold px-2 py-0.5 rounded bg-indigo-500/15 text-indigo-300 border border-indigo-500/30">
                    {sourceId}
                  </span>
                  <span className="text-xs font-medium text-slate-400 flex items-center gap-1">
                    <Globe size={12} className="text-slate-500" />
                    {domain}
                  </span>
                </div>

                <div className="flex items-center gap-3">
                  {source.score !== null && source.score !== undefined && (
                    <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700/80">
                      Score: {(source.score * 100).toFixed(0)}%
                    </span>
                  )}
                  <a
                    href={source.url}
                    target="_blank"
                    rel="noreferrer noopener"
                    className="inline-flex items-center gap-1 text-xs text-indigo-400 hover:text-indigo-300 transition-colors"
                  >
                    <span>Visit Source</span>
                    <ExternalLink size={12} />
                  </a>
                </div>
              </div>

              {/* Title */}
              <h4 className="text-sm sm:text-base font-semibold text-slate-100 mb-2 leading-snug">
                {source.title}
              </h4>

              {/* Snippet summary */}
              {source.content && (
                <p className="text-xs sm:text-sm text-slate-300/90 leading-relaxed font-normal">
                  {source.content}
                </p>
              )}

              {/* Expandable extracted document content */}
              {matchedDoc && matchedDoc.text && (
                <div className="mt-3 pt-3 border-t border-slate-800/80">
                  <button
                    type="button"
                    onClick={() => setExpandedIndex(isExpanded ? null : index)}
                    className="inline-flex items-center gap-1.5 text-xs text-slate-400 hover:text-slate-200 transition-colors"
                  >
                    <FileText size={12} />
                    <span>
                      {isExpanded ? 'Hide extracted text' : `View extracted document (${matchedDoc.char_count} chars)`}
                    </span>
                    {isExpanded ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
                  </button>

                  {isExpanded && (
                    <div className="mt-2.5 p-3 rounded-lg bg-slate-900/80 border border-slate-800 text-xs text-slate-300 font-mono max-h-60 overflow-y-auto whitespace-pre-wrap leading-relaxed">
                      {matchedDoc.text}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};
