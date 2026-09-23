import React, { useEffect, useState } from 'react';
import { Sparkles, Activity, ShieldCheck, AlertCircle } from 'lucide-react';
import { checkHealth, checkReadiness } from '../../services/api';

export const Header: React.FC = () => {
  const [healthStatus, setHealthStatus] = useState<'checking' | 'ready' | 'degraded' | 'offline'>('checking');
  const [dbStatus, setDbStatus] = useState<string>('');

  useEffect(() => {
    let isMounted = true;

    async function probeSystem() {
      try {
        await checkHealth();
        try {
          const ready = await checkReadiness();
          if (isMounted) {
            setHealthStatus('ready');
            setDbStatus(ready.database || 'connected');
          }
        } catch {
          if (isMounted) {
            setHealthStatus('degraded');
            setDbStatus('db disconnected');
          }
        }
      } catch {
        if (isMounted) {
          setHealthStatus('offline');
          setDbStatus('unreachable');
        }
      }
    }

    probeSystem();

    // Modest non-aggressive 60-second polling interval
    const interval = setInterval(probeSystem, 60000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-md sticky top-0 z-30">
      <div className="max-w-6xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-violet-600 flex items-center justify-center shadow-lg shadow-indigo-500/20 ring-1 ring-white/20">
            <Sparkles className="w-5 h-5 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-bold text-base tracking-tight text-white">
                AI Research Agent
              </span>
              <span className="text-[10px] uppercase font-mono px-1.5 py-0.5 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">
                v1.0
              </span>
            </div>
            <p className="text-xs text-slate-400 hidden sm:block">
              Grounded synthesis with factual claim verification
            </p>
          </div>
        </div>

        {/* System Health Probe Badge */}
        <div className="flex items-center gap-3">
          <div
            className="flex items-center gap-2 px-2.5 py-1 rounded-full text-xs font-medium border bg-slate-800/60 transition-colors"
            title={
              healthStatus === 'ready'
                ? `FastAPI + PostgreSQL (${dbStatus}) active`
                : healthStatus === 'degraded'
                ? 'Backend online, database pending'
                : healthStatus === 'offline'
                ? 'Backend not responding at configured host'
                : 'Verifying system connectivity...'
            }
          >
            {healthStatus === 'ready' && (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                </span>
                <span className="text-emerald-400 font-mono text-[11px]">System Ready</span>
              </>
            )}

            {healthStatus === 'degraded' && (
              <>
                <Activity size={12} className="text-amber-400 animate-pulse" />
                <span className="text-amber-400 font-mono text-[11px]">Degraded</span>
              </>
            )}

            {healthStatus === 'offline' && (
              <>
                <AlertCircle size={12} className="text-rose-400" />
                <span className="text-rose-400 font-mono text-[11px]">Backend Offline</span>
              </>
            )}

            {healthStatus === 'checking' && (
              <>
                <ShieldCheck size={12} className="text-slate-400 animate-pulse" />
                <span className="text-slate-400 font-mono text-[11px]">Probing...</span>
              </>
            )}
          </div>
        </div>
      </div>
    </header>
  );
};
