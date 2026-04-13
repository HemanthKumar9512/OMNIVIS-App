/**
 * OMNIVIS — Alert Banner Component
 * Displays anomaly alerts with severity levels and history log.
 */
import React from 'react'
import { useOmnivisStore } from '../store/omnivis.store'

const SEVERITY_CONFIG = {
  green: { bg: 'alert-green', icon: '✓', label: 'NORMAL' },
  yellow: { bg: 'alert-yellow', icon: '⚠', label: 'SUSPICIOUS' },
  red: { bg: 'alert-red', icon: '🚨', label: 'ANOMALY' },
}

export const AlertBanner: React.FC = () => {
  const { alertLevel, anomalies, clearAnomalies } = useOmnivisStore()
  const config = SEVERITY_CONFIG[alertLevel]

  return (
    <div className="flex flex-col gap-2">
      {/* Current Status */}
      <div className={`alert-banner ${config.bg}`}>
        <span className="text-lg">{config.icon}</span>
        <div className="flex-1">
          <div className="text-xs font-bold uppercase tracking-wider">{config.label}</div>
          <div className="text-[10px] opacity-70">
            {alertLevel === 'green' ? 'All systems normal' :
             alertLevel === 'yellow' ? 'Potential anomaly detected' :
             'Confirmed anomaly — immediate attention required'}
          </div>
        </div>
        <div className={`status-dot ${
          alertLevel === 'green' ? 'status-green' :
          alertLevel === 'yellow' ? 'status-yellow' :
          'status-red'
        }`} />
      </div>

      {/* Alert History */}
      {anomalies.length > 0 && (
        <div className="glass p-3">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-medium text-white/50 uppercase tracking-wider">Alert History</span>
            <button
              onClick={clearAnomalies}
              className="text-[10px] text-white/30 hover:text-white/60 transition-colors"
            >
              Clear
            </button>
          </div>
          <div className="max-h-[120px] overflow-y-auto space-y-1">
            {anomalies.slice(0, 20).map((alert) => (
              <div
                key={alert.id}
                className={`flex items-start gap-2 px-2 py-1.5 rounded-lg text-[10px] animate-slide-in
                  ${alert.severity === 'red' ? 'bg-red-500/5' :
                    alert.severity === 'yellow' ? 'bg-amber-500/5' :
                    'bg-emerald-500/5'}`}
              >
                <span className={`mt-0.5 w-1.5 h-1.5 rounded-full flex-shrink-0
                  ${alert.severity === 'red' ? 'bg-red-400' :
                    alert.severity === 'yellow' ? 'bg-amber-400' :
                    'bg-emerald-400'}`}
                />
                <div className="flex-1 min-w-0">
                  <div className="font-medium text-white/60 truncate">{alert.description}</div>
                  <div className="text-white/30 mt-0.5">
                    Score: {(alert.score * 100).toFixed(1)}% • {new Date(alert.timestamp * 1000).toLocaleTimeString()}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
