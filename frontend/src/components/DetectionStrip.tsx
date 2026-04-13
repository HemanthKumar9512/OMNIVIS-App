/**
 * OMNIVIS — Detection Strip Component
 * Bottom bar showing all current detections as chips with class + confidence + track ID.
 */
import React from 'react'
import { useOmnivisStore } from '../store/omnivis.store'

const CLASS_COLORS: Record<string, string> = {
  person: 'from-blue-500/20 to-blue-600/10 border-blue-500/30 text-blue-300',
  car: 'from-emerald-500/20 to-emerald-600/10 border-emerald-500/30 text-emerald-300',
  truck: 'from-amber-500/20 to-amber-600/10 border-amber-500/30 text-amber-300',
  dog: 'from-pink-500/20 to-pink-600/10 border-pink-500/30 text-pink-300',
  cat: 'from-purple-500/20 to-purple-600/10 border-purple-500/30 text-purple-300',
  bicycle: 'from-cyan-500/20 to-cyan-600/10 border-cyan-500/30 text-cyan-300',
  bus: 'from-orange-500/20 to-orange-600/10 border-orange-500/30 text-orange-300',
  motorcycle: 'from-red-500/20 to-red-600/10 border-red-500/30 text-red-300',
  chair: 'from-teal-500/20 to-teal-600/10 border-teal-500/30 text-teal-300',
  default: 'from-omni-500/20 to-omni-600/10 border-omni-500/30 text-omni-300',
}

function getChipColor(className: string): string {
  return CLASS_COLORS[className.toLowerCase()] || CLASS_COLORS.default
}

export const DetectionStrip: React.FC = () => {
  const { detections, actions } = useOmnivisStore()

  return (
    <div className="glass-sidebar border-t border-white/5 px-4 py-2">
      <div className="flex items-center gap-3">
        {/* Detection count */}
        <div className="flex items-center gap-2 flex-shrink-0">
          <div className="w-6 h-6 rounded-lg bg-omni-500/20 flex items-center justify-center">
            <span className="text-xs font-bold text-omni-300">{detections.length}</span>
          </div>
          <span className="text-[10px] text-white/30 uppercase tracking-wider">Detections</span>
        </div>

        <div className="h-4 w-px bg-white/10" />

        {/* Detection chips */}
        <div className="flex-1 overflow-x-auto flex items-center gap-1.5 min-w-0 scrollbar-hide">
          {detections.length === 0 ? (
            <span className="text-xs text-white/20 italic">No detections</span>
          ) : (
            detections.slice(0, 30).map((det, i) => (
              <div
                key={`${det.class_name}-${i}`}
                className={`detection-chip bg-gradient-to-r border flex-shrink-0 ${getChipColor(det.class_name)}`}
              >
                <span className="font-medium text-[11px]">{det.class_name}</span>
                <span className="text-[10px] opacity-60">{(det.confidence * 100).toFixed(0)}%</span>
                {det.track_id !== undefined && (
                  <span className="text-[9px] px-1 py-0 rounded bg-white/10 opacity-50">#{det.track_id}</span>
                )}
              </div>
            ))
          )}
        </div>

        {/* Action predictions */}
        {actions.length > 0 && (
          <>
            <div className="h-4 w-px bg-white/10" />
            <div className="flex items-center gap-1.5 flex-shrink-0">
              <span className="text-[10px] text-white/30">Actions:</span>
              {actions.slice(0, 2).map((a, i) => (
                <span key={i} className="detection-chip bg-gradient-to-r from-violet-500/20 to-violet-600/10 border-violet-500/30 text-violet-300 flex-shrink-0">
                  <span className="text-[11px] font-medium">{a.action}</span>
                  <span className="text-[10px] opacity-60">{(a.confidence * 100).toFixed(0)}%</span>
                </span>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
