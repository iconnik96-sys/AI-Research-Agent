import React from 'react';
import { AlertCircle, RotateCcw } from 'lucide-react';
import { ApiError } from '../../types/api';

interface ErrorAlertProps {
  error: ApiError;
  onRetry?: () => void;
}

export const ErrorAlert: React.FC<ErrorAlertProps> = ({ error, onRetry }) => {
  return (
    <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-5 text-rose-200">
      <div className="flex items-start gap-3">
        <AlertCircle className="w-5 h-5 text-rose-400 shrink-0 mt-0.5" />
        <div className="flex-1 space-y-2">
          <div className="flex items-center justify-between flex-wrap gap-2">
            <h3 className="font-semibold text-rose-100 text-sm tracking-wide">
              {error.status === 429
                ? 'Rate Limit Reached'
                : error.status === 0
                ? 'Backend Connection Failed'
                : 'Research Request Error'}
            </h3>
            {error.status ? (
              <span className="text-xs font-mono px-2 py-0.5 rounded bg-rose-500/20 text-rose-300">
                HTTP {error.status}
              </span>
            ) : null}
          </div>

          <p className="text-sm text-rose-200/90 leading-relaxed">{error.message}</p>

          {error.requestId && (
            <div className="pt-1 text-xs text-rose-300/70 font-mono">
              Request ID:{' '}
              <span className="select-all text-rose-300 font-semibold">{error.requestId}</span>
            </div>
          )}

          {onRetry && (
            <div className="pt-2">
              <button
                type="button"
                onClick={onRetry}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium bg-rose-500/20 hover:bg-rose-500/30 text-rose-100 border border-rose-500/30 transition-colors focus:outline-none focus:ring-2 focus:ring-rose-400"
              >
                <RotateCcw size={13} />
                <span>Retry Request</span>
              </button>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
