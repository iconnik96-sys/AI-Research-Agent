import React, { useState } from 'react';
import { Copy, Check } from 'lucide-react';
import { ResearchReport } from '../../types/api';

interface CopyButtonProps {
  report: ResearchReport | null;
  className?: string;
}

export function formatReportToMarkdown(report: ResearchReport): string {
  const lines: string[] = [];

  lines.push(`# ${report.title}`);
  lines.push('');
  lines.push('## Executive Summary');
  lines.push('');
  lines.push(report.summary);
  lines.push('');

  if (report.sections && report.sections.length > 0) {
    for (const section of report.sections) {
      lines.push(`## ${section.heading}`);
      lines.push('');
      lines.push(section.content);
      lines.push('');
      if (section.citations && section.citations.length > 0) {
        lines.push(`*Citations: ${section.citations.map((c) => `[${c}]`).join(', ')}*`);
        lines.push('');
      }
    }
  }

  if (report.sources && report.sources.length > 0) {
    lines.push('## References');
    lines.push('');
    for (const src of report.sources) {
      lines.push(`- **[${src.id}]** [${src.title}](${src.url})`);
    }
    lines.push('');
  }

  return lines.join('\n');
}

export const CopyButton: React.FC<CopyButtonProps> = ({ report, className = '' }) => {
  const [copied, setCopied] = useState(false);

  if (!report) return null;

  const handleCopy = async () => {
    try {
      const markdown = formatReportToMarkdown(report);
      await navigator.clipboard.writeText(markdown);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy markdown to clipboard', err);
    }
  };

  return (
    <button
      type="button"
      onClick={handleCopy}
      className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium border border-slate-700 bg-slate-800/80 hover:bg-slate-700/80 text-slate-200 transition-colors focus:outline-none focus:ring-2 focus:ring-indigo-500 ${className}`}
      title="Copy research report as formatted Markdown"
    >
      {copied ? (
        <>
          <Check size={13} className="text-emerald-400" />
          <span className="text-emerald-400 font-semibold">Copied Markdown</span>
        </>
      ) : (
        <>
          <Copy size={13} className="text-slate-400" />
          <span>Copy Markdown</span>
        </>
      )}
    </button>
  );
};
