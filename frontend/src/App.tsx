import React, { useState, useRef } from 'react';
import { Header } from './components/layout/Header';
import { Footer } from './components/layout/Footer';
import { ResearchInput } from './components/research/ResearchInput';
import { ResearchProgress } from './components/research/ResearchProgress';
import { ResearchDossier } from './components/research/ResearchDossier';
import { ErrorAlert } from './components/common/ErrorAlert';
import { submitResearch, ResearchApiError } from './services/api';
import { ApiError, ResearchResponse } from './types/api';

export const App: React.FC = () => {
  const [currentQuestion, setCurrentQuestion] = useState<string>('');
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [startTime, setStartTime] = useState<number>(0);
  const [result, setResult] = useState<ResearchResponse | null>(null);
  const [error, setError] = useState<ApiError | null>(null);

  const abortControllerRef = useRef<AbortController | null>(null);

  const handleStartResearch = async (question: string) => {
    // Cancel any ongoing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    const controller = new AbortController();
    abortControllerRef.current = controller;

    setCurrentQuestion(question);
    setIsLoading(true);
    setStartTime(Date.now());
    setError(null);
    setResult(null);

    try {
      const response = await submitResearch(question, controller.signal);
      setResult(response);
    } catch (err: unknown) {
      if (err instanceof ResearchApiError) {
        // If aborted, don't show an error banner unless user-initiated
        if (err.message.includes('cancelled')) {
          return;
        }
        setError({
          message: err.message,
          status: err.status,
          requestId: err.requestId,
          retryAfter: err.retryAfter,
        });
      } else {
        setError({
          message: 'An unexpected application error occurred while processing research.',
        });
      }
    } finally {
      setIsLoading(false);
      abortControllerRef.current = null;
    }
  };

  const handleCancel = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    setIsLoading(false);
  };

  const handleRetry = () => {
    if (currentQuestion) {
      handleStartResearch(currentQuestion);
    }
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-900 text-slate-100 selection:bg-indigo-500/30 selection:text-indigo-200">
      <Header />

      <main className="flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 py-8 sm:py-12 flex flex-col gap-8">
        {/* Landing Hero (shown in initial state, or minimized when result is present) */}
        {!result && !isLoading && (
          <div className="text-center max-w-2xl mx-auto space-y-3 pt-4 sm:pt-8">
            <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-white leading-tight">
              Autonomous AI Research Agent
            </h1>
            <p className="text-sm sm:text-base text-slate-400 leading-relaxed font-normal">
              Formulates multi-angle search plans, gathers live web evidence, indexes vector embeddings, independently verifies factual claims, and synthesizes grounded research reports with exact citations.
            </p>
          </div>
        )}

        {/* Input Bar */}
        <section aria-label="Research Question Input">
          <ResearchInput
            onSubmit={handleStartResearch}
            isLoading={isLoading}
            initialQuestion={currentQuestion}
          />
        </section>

        {/* Progress Tracker (while request is processing) */}
        {isLoading && (
          <section aria-label="Research Progress" className="animate-fadeIn">
            <ResearchProgress
              startTime={startTime}
              onCancel={handleCancel}
            />
          </section>
        )}

        {/* Error Alert with Retry & Request ID */}
        {error && !isLoading && (
          <section aria-label="Error Alert">
            <ErrorAlert error={error} onRetry={handleRetry} />
          </section>
        )}

        {/* Completed Research Dossier */}
        {result && !isLoading && (
          <section aria-label="Research Results" className="animate-fadeIn">
            <ResearchDossier data={result} />
          </section>
        )}
      </main>

      <Footer />
    </div>
  );
};

export default App;
