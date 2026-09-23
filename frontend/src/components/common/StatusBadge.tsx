import React from 'react';
import { CheckCircle2, AlertTriangle, XCircle, HelpCircle } from 'lucide-react';
import { ClaimVerificationStatus } from '../../types/api';

interface StatusBadgeProps {
  status: ClaimVerificationStatus;
  size?: 'sm' | 'md';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'md' }) => {
  const isSm = size === 'sm';
  const iconSize = isSm ? 12 : 14;

  switch (status) {
    case 'SUPPORTED':
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 ${
            isSm ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs tracking-wide'
          }`}
        >
          <CheckCircle2 size={iconSize} className="text-emerald-400 shrink-0" />
          <span>SUPPORTED</span>
        </span>
      );

    case 'PARTIALLY_SUPPORTED':
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-amber-500/10 text-amber-400 border border-amber-500/20 ${
            isSm ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs tracking-wide'
          }`}
        >
          <AlertTriangle size={iconSize} className="text-amber-400 shrink-0" />
          <span>PARTIALLY SUPPORTED</span>
        </span>
      );

    case 'UNSUPPORTED':
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-rose-500/10 text-rose-400 border border-rose-500/20 ${
            isSm ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs tracking-wide'
          }`}
        >
          <XCircle size={iconSize} className="text-rose-400 shrink-0" />
          <span>UNSUPPORTED</span>
        </span>
      );

    case 'INSUFFICIENT_EVIDENCE':
    default:
      return (
        <span
          className={`inline-flex items-center gap-1.5 font-medium rounded-full bg-slate-500/10 text-slate-400 border border-slate-500/20 ${
            isSm ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs tracking-wide'
          }`}
        >
          <HelpCircle size={iconSize} className="text-slate-400 shrink-0" />
          <span>INSUFFICIENT EVIDENCE</span>
        </span>
      );
  }
};
