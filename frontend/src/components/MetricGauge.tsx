/**
 * OMNIVIS — Metric Gauge Component
 * Circular gauge charts for GPU utilization, memory, and inference latency.
 */
import React from 'react'
import { useOmnivisStore } from '../store/omnivis.store'

interface GaugeProps {
  value: number
  max: number
  label: string
  unit: string
  color: string
  size?: number
}

export const MetricGauge: React.FC<GaugeProps> = ({
  value, max, label, unit, color, size = 80,
}) => {
  const percentage = Math.min(100, Math.max(0, (value / max) * 100))
  const radius = (size - 12) / 2
  const circumference = 2 * Math.PI * radius
  const strokeDashoffset = circumference * (1 - percentage / 100)

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.05)"
          strokeWidth={4}
        />
        {/* Progress arc */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth={4}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className="transition-all duration-500 ease-out"
          style={{
            filter: `drop-shadow(0 0 6px ${color}40)`,
          }}
        />
        {/* Center text */}
        <text
          x={size / 2}
          y={size / 2}
          textAnchor="middle"
          dominantBaseline="central"
          fill="white"
          fontSize="14"
          fontWeight="700"
          fontFamily="JetBrains Mono"
          className="transform rotate-90"
          style={{ transformOrigin: `${size / 2}px ${size / 2}px` }}
        >
          {value.toFixed(0)}
        </text>
      </svg>
      <div className="text-center">
        <div className="text-[9px] text-white/40 uppercase tracking-widest">{label}</div>
        <div className="text-[10px] font-mono text-white/25">{unit}</div>
      </div>
    </div>
  )
}

export const MetricGaugeRow: React.FC = () => {
  const { currentFps, currentInferenceMs, detections, tracks, metricsHistory } = useOmnivisStore()
  
  const latestMetrics = metricsHistory.length > 0 
    ? metricsHistory[metricsHistory.length - 1] 
    : null
  
  const gpuUtil = latestMetrics?.gpu_util ?? 0
  const gpuMem = latestMetrics?.gpu_memory ?? 0
  const cpuUtil = latestMetrics?.cpu_util ?? 0
  const latency = currentInferenceMs > 0 ? currentInferenceMs : (latestMetrics?.inference_ms ?? 0)

  return (
    <div className="glass-card p-3">
      <div className="text-[11px] font-medium text-white/50 uppercase tracking-wider mb-3">
        System Resources
      </div>
      <div className="flex items-center justify-around">
        <MetricGauge value={gpuUtil} max={100} label="GPU" unit="%" color="#6366f1" />
        <MetricGauge value={gpuMem} max={8192} label="VRAM" unit="MB" color="#10b981" />
        <MetricGauge value={cpuUtil} max={100} label="CPU" unit="%" color="#f59e0b" />
        <MetricGauge value={latency} max={200} label="Latency" unit="ms" color="#ec4899" />
      </div>
    </div>
  )
}
