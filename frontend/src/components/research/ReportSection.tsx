import React from 'react';
import { ResearchSection } from '../../types/api';
import { CitationBadge } from '../common/CitationBadge';

interface ReportSectionProps {
  section: ResearchSection;
  onSelectCitation?: (sourceId: string) => void;
}

export const ReportSectionView: React.FC<ReportSectionProps> = ({
  section,
  onSelectCitation,
}) => {
  // Parse inline citations like [S1], [S2] within the paragraph content
  const renderContentWithCitations = (text: string) => {
    // Split by citation pattern [S1], [S2], etc.
    const parts = text.split(/(\[S\d+\])/g);

    return parts.map((part, index) => {
      const match = part.match(/^\[(S\d+)\]$/);
      if (match) {
        const sourceId = match[1];
        return (
          <CitationBadge
            key={index}
            id={sourceId}
            onClick={onSelectCitation}
          />
        );
      }
      return <span key={index}>{part}</span>;
    });
  };

  // Split section content into paragraphs
  const paragraphs = section.content.split('\n\n').filter((p) => p.trim());

  return (
    <article className="border-b border-slate-800/80 pb-6 last:border-b-0 last:pb-0">
      <h3 className="text-lg font-semibold text-slate-100 tracking-tight mb-3">
        {section.heading}
      </h3>

      <div className="space-y-3.5 text-sm sm:text-base text-slate-300 leading-relaxed font-normal">
        {paragraphs.map((para, idx) => (
          <p key={idx}>{renderContentWithCitations(para)}</p>
        ))}
      </div>

      {/* Section footer citations if provided in metadata */}
      {section.citations && section.citations.length > 0 && (
        <div className="mt-4 pt-3 flex items-center flex-wrap gap-2 text-xs text-slate-400">
          <span className="font-medium text-slate-500">Cited in section:</span>
          {section.citations.map((citeId, idx) => (
            <CitationBadge
              key={idx}
              id={citeId}
              onClick={onSelectCitation}
            />
          ))}
        </div>
      )}
    </article>
  );
};
