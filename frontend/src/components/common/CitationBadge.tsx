import React from 'react';

interface CitationBadgeProps {
  id: string;
  onClick?: (id: string) => void;
  title?: string;
}

export const CitationBadge: React.FC<CitationBadgeProps> = ({ id, onClick, title }) => {
  // Normalize id: if id doesn't start with 'S', keep as is, or strip brackets
  const cleanId = id.replace(/[[\]]/g, '').trim();

  return (
    <button
      type="button"
      onClick={() => onClick && onClick(cleanId)}
      title={title || `View source [${cleanId}]`}
      className="inline-flex items-center justify-center px-1.5 py-0.5 mx-0.5 text-xs font-mono font-medium rounded bg-indigo-500/15 text-indigo-300 hover:bg-indigo-500/25 hover:text-indigo-200 border border-indigo-500/30 transition-colors focus:outline-none focus:ring-1 focus:ring-indigo-400"
    >
      [{cleanId}]
    </button>
  );
};
