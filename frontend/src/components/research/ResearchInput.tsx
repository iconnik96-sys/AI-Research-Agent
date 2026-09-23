import React, { useState, useRef, useEffect } from 'react';
import { Search, CornerDownLeft, Sparkles } from 'lucide-react';

interface ResearchInputProps {
  onSubmit: (question: string) => void;
  isLoading: boolean;
  initialQuestion?: string;
}

const EXAMPLE_QUESTIONS = [
  'What are the latest breakthroughs in nuclear fusion energy in 2024–2025?',
  'How do modern solid-state batteries compare to traditional lithium-ion for EVs?',
  'What is the clinical evidence for low-dose aspirin in primary cardiovascular prevention?',
  'What architectural advancements allow DeepSeek-R1 to achieve competitive reasoning efficiency?',
];

export const ResearchInput: React.FC<ResearchInputProps> = ({
  onSubmit,
  isLoading,
  initialQuestion = '',
}) => {
  const [question, setQuestion] = useState(initialQuestion);
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (initialQuestion) {
      setQuestion(initialQuestion);
    }
  }, [initialQuestion]);

  // Auto-resize textarea height
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = 'auto';
      textarea.style.height = `${Math.min(textarea.scrollHeight, 180)}px`;
    }
  }, [question]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    const trimmed = question.trim();
    if (!trimmed) {
      setError('Please enter a research question.');
      textareaRef.current?.focus();
      return;
    }
    setError(null);
    onSubmit(trimmed);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleSelectExample = (example: string) => {
    setQuestion(example);
    setError(null);
    textareaRef.current?.focus();
  };

  return (
    <div className="w-full">
      <form onSubmit={handleSubmit} className="relative">
        <div
          className={`relative rounded-2xl border transition-all duration-200 bg-slate-800/80 shadow-xl ${
            error
              ? 'border-rose-500/50 shadow-rose-500/5 ring-1 ring-rose-500/20'
              : 'border-slate-700/80 hover:border-slate-600 focus-within:border-indigo-500/80 focus-within:ring-2 focus-within:ring-indigo-500/20'
          }`}
        >
          <div className="p-4 sm:p-5 flex items-start gap-3">
            <Search className="w-5 h-5 text-indigo-400 mt-1 shrink-0" />
            <div className="flex-1">
              <textarea
                ref={textareaRef}
                value={question}
                onChange={(e) => {
                  setQuestion(e.target.value);
                  if (error) setError(null);
                }}
                onKeyDown={handleKeyDown}
                disabled={isLoading}
                placeholder="Ask any complex research question..."
                rows={2}
                className="w-full bg-transparent text-slate-100 placeholder-slate-400 text-sm sm:text-base focus:outline-none resize-none leading-relaxed disabled:opacity-50"
              />
            </div>
          </div>

          {/* Action bar inside input box */}
          <div className="px-4 pb-3 sm:px-5 sm:pb-4 flex items-center justify-between border-t border-slate-700/40 pt-3">
            <span className="text-[11px] text-slate-400 hidden sm:inline-flex items-center gap-1 font-mono">
              Press <kbd className="px-1.5 py-0.5 rounded bg-slate-700 text-slate-300 text-[10px]">Ctrl</kbd> + <kbd className="px-1.5 py-0.5 rounded bg-slate-700 text-slate-300 text-[10px]">Enter</kbd> to submit
            </span>
            <div className="ml-auto flex items-center gap-2">
              <button
                type="submit"
                disabled={isLoading || !question.trim()}
                className="inline-flex items-center gap-2 px-4 py-2 rounded-xl text-xs sm:text-sm font-medium bg-indigo-600 hover:bg-indigo-500 active:bg-indigo-700 text-white shadow-md shadow-indigo-600/30 transition-all disabled:opacity-40 disabled:cursor-not-allowed focus:outline-none focus:ring-2 focus:ring-indigo-400"
              >
                <span>{isLoading ? 'Researching...' : 'Start Research'}</span>
                {!isLoading && <CornerDownLeft size={14} className="opacity-70" />}
              </button>
            </div>
          </div>
        </div>

        {error && (
          <p className="mt-2 text-xs text-rose-400 pl-1">{error}</p>
        )}
      </form>

      {/* Example Prompt Pills */}
      {!isLoading && (
        <div className="mt-4">
          <div className="flex items-center gap-1.5 text-xs text-slate-400 font-medium mb-2 pl-1">
            <Sparkles size={13} className="text-indigo-400" />
            <span>Example investigations</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {EXAMPLE_QUESTIONS.map((ex, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => handleSelectExample(ex)}
                className="text-left text-xs px-3 py-1.5 rounded-xl border border-slate-800 bg-slate-800/40 hover:bg-slate-800 hover:border-slate-700 text-slate-300 hover:text-white transition-all text-ellipsis"
              >
                {ex}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};
