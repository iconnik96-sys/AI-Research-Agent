import React, { useState } from 'react';
import { ShieldCheck, Filter } from 'lucide-react';
import { VerifiedClaim } from '../../types/api';
import { StatusBadge } from '../common/StatusBadge';

interface ClaimsPanelProps {
  claims: VerifiedClaim[];
}

type ClaimFilter = 'all' | 'supported' | 'flagged';

export const ClaimsPanel: React.FC<ClaimsPanelProps> = ({ claims }) => {
  const [filter, setFilter] = useState<ClaimFilter>('all');

  if (!claims || claims.length === 0) {
    return (
      <div className="text-center py-12 border border-slate-800 rounded-xl bg-slate-900/40">
        <ShieldCheck className="mx-auto h-8 w-8 text-slate-500 mb-2" />
        <p className="text-sm text-slate-400">No claim verification records generated for this session.</p>
      </div>
    );
  }

  const supportedCount = claims.filter((c) => c.status === 'SUPPORTED').length;
  const flaggedCount = claims.filter(
    (c) =>
      c.status === 'PARTIALLY_SUPPORTED' ||
      c.status === 'UNSUPPORTED' ||
      c.status === 'INSUFFICIENT_EVIDENCE'
  ).length;

  const filteredClaims = claims.filter((c) => {
    if (filter === 'supported') return c.status === 'SUPPORTED';
    if (filter === 'flagged') {
      return (
        c.status === 'PARTIALLY_SUPPORTED' ||
        c.status === 'UNSUPPORTED' ||
        c.status === 'INSUFFICIENT_EVIDENCE'
      );
    }
    return true;
  });

  return (
    <div className="space-y-6">
      {/* Top Controls & Statistics */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <h3 className="text-base font-semibold text-slate-100">
            Fact-Checking & Claim Verification
          </h3>
          <p className="text-xs text-slate-400 mt-0.5">
            Every candidate factual claim is independently verified against retrieved evidence chunks.
          </p>
        </div>

        {/* Filter Pills */}
        <div className="flex items-center gap-1.5 p-1 rounded-xl bg-slate-900 border border-slate-800 self-start sm:self-auto">
          <div className="flex items-center gap-1 px-2 text-xs text-slate-500 font-medium">
            <Filter size={11} />
            <span>Filter:</span>
          </div>

          <button
            type="button"
            onClick={() => setFilter('all')}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
              filter === 'all'
                ? 'bg-indigo-600 text-white'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            All ({claims.length})
          </button>

          <button
            type="button"
            onClick={() => setFilter('supported')}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
              filter === 'supported'
                ? 'bg-emerald-600 text-white'
                : 'text-slate-400 hover:text-emerald-400'
            }`}
          >
            Supported ({supportedCount})
          </button>

          <button
            type="button"
            onClick={() => setFilter('flagged')}
            className={`px-2.5 py-1 rounded-lg text-xs font-medium transition-colors ${
              filter === 'flagged'
                ? 'bg-amber-600 text-white'
                : 'text-slate-400 hover:text-amber-400'
            }`}
          >
            Flagged ({flaggedCount})
          </button>
        </div>
      </div>

      {/* Claims List */}
      <div className="space-y-4">
        {filteredClaims.length === 0 ? (
          <div className="text-center py-8 text-sm text-slate-400">
            No claims matching the selected filter.
          </div>
        ) : (
          filteredClaims.map((claim) => (
            <div
              key={claim.id}
              className="rounded-xl border border-slate-800 bg-slate-850/60 p-4 sm:p-5 hover:border-slate-700 transition-colors"
            >
              {/* Header row: Claim ID + Status badge */}
              <div className="flex items-start justify-between gap-3 mb-2.5">
                <span className="font-mono text-xs font-semibold px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700 shrink-0">
                  {claim.id}
                </span>
                <StatusBadge status={claim.status} />
              </div>

              {/* Claim Statement */}
              <h4 className="text-sm sm:text-base font-medium text-slate-100 leading-snug mb-3">
                &ldquo;{claim.claim}&rdquo;
              </h4>

              {/* Verifier Reasoning */}
              {claim.reason && (
                <div className="rounded-lg bg-slate-900/60 border border-slate-800/80 p-3 text-xs text-slate-300 leading-relaxed mb-3">
                  <span className="font-semibold text-slate-400 block mb-1">
                    Verification Justification:
                  </span>
                  {claim.reason}
                </div>
              )}

              {/* Evidence IDs */}
              <div className="flex flex-wrap items-center gap-x-4 gap-y-2 text-xs text-slate-400 pt-2 border-t border-slate-800/60 font-mono">
                {claim.supporting_evidence_ids && claim.supporting_evidence_ids.length > 0 && (
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-400">Directly Supported By:</span>
                    <div className="flex items-center gap-1">
                      {claim.supporting_evidence_ids.map((eid) => (
                        <span
                          key={eid}
                          className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 font-semibold"
                        >
                          {eid}
                        </span>
                      ))}
                    </div>
                  </div>
                )}

                {claim.evidence_ids && claim.evidence_ids.length > 0 && (
                  <div className="flex items-center gap-1.5">
                    <span className="text-slate-400">Total Considered:</span>
                    <div className="flex items-center gap-1">
                      {claim.evidence_ids.map((eid) => (
                        <span
                          key={eid}
                          className="px-1.5 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700"
                        >
                          {eid}
                        </span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))
        )}
      </div>
    </div>
  );
};
