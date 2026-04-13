/**
 * OMNIVIS — Sidebar Component
 * Left sidebar: Module toggles, model selectors, threshold sliders.
 * Right sidebar: Metric charts, detection list, alert log.
 */
import React, { useState } from 'react'
import { useOmnivisStore } from '../store/omnivis.store'
import { MetricCharts } from './MetricChart'
import { MetricGaugeRow } from './MetricGauge'
import { AlertBanner } from './AlertBanner'
import { SceneGraph3D } from './SceneGraph3D'
import { PointCloud3D } from './PointCloud3D'

const MODULE_INFO: Record<string, { icon: string; label: string; desc: string }> = {
  detection: { icon: '🎯', label: 'Detection', desc: 'YOLOv8 object detection' },
  segmentation: { icon: '🧩', label: 'Segmentation', desc: 'DeepLabV3+ / Mask R-CNN' },
  face: { icon: '👤', label: 'Face Analysis', desc: 'Detection + recognition + attributes' },
  optical_flow: { icon: '🌊', label: 'Optical Flow', desc: 'RAFT / Farneback motion' },
  depth: { icon: '🏔️', label: 'Depth', desc: 'MiDaS monocular depth' },
  reconstruction: { icon: '🧊', label: '3D Recon', desc: 'Structure from Motion' },
  tracking: { icon: '📍', label: 'Tracking', desc: 'ByteTrack multi-object' },
  scene_graph: { icon: '🕸️', label: 'Scene Graph', desc: 'Object relationships' },
  trajectory: { icon: '📐', label: 'Trajectory', desc: 'Path prediction' },
  anomaly: { icon: '🚨', label: 'Anomaly', desc: 'SVM + Autoencoder ensemble' },
  gait: { icon: '🚶', label: 'Gait', desc: 'MediaPipe pose analysis' },
  action: { icon: '🎬', label: 'Action', desc: 'Video action recognition' },
}

const MODEL_OPTIONS = [
  { value: 'yolov8n', label: 'YOLOv8n (Fast)' },
  { value: 'yolov8s', label: 'YOLOv8s (Small)' },
  { value: 'yolov8m', label: 'YOLOv8m (Medium)' },
  { value: 'yolov8l', label: 'YOLOv8l (Large)' },
  { value: 'yolov8x', label: 'YOLOv8x (Extreme)' },
]

export const LeftSidebar: React.FC = () => {
  const {
    modules, toggleModule,
    confidenceThreshold, setConfidenceThreshold,
    nmsThreshold, setNmsThreshold,
    selectedModel, setSelectedModel,
    inputSource, setInputSource, inputUrl, setInputUrl,
  } = useOmnivisStore()

  const [collapsed, setCollapsed] = useState<Record<string, boolean>>({})

  return (
    <div className="glass-sidebar w-[260px] flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/5">
        <h2 className="text-xs font-bold text-white/60 uppercase tracking-widest">Controls</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* Input Source */}
        <div className="glass p-3 space-y-2">
          <span className="text-[11px] font-semibold text-white/50 uppercase tracking-wider">📡 Input Source</span>
          <select
            value={inputSource}
            onChange={e => setInputSource(e.target.value as any)}
            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white/80 focus:outline-none focus:border-omni-500/50"
          >
            <option value="webcam">Webcam</option>
            <option value="rtsp">RTSP Stream</option>
            <option value="youtube">YouTube URL</option>
            <option value="file">File Upload</option>
          </select>
          {(inputSource === 'rtsp' || inputSource === 'youtube') && (
            <input
              type="text"
              value={inputUrl}
              onChange={e => setInputUrl(e.target.value)}
              placeholder={inputSource === 'rtsp' ? 'rtsp://...' : 'https://youtube.com/...'}
              className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white/80 placeholder-white/20 focus:outline-none focus:border-omni-500/50"
            />
          )}
        </div>

        {/* Model Selector */}
        <div className="glass p-3 space-y-2">
          <span className="text-[11px] font-semibold text-white/50 uppercase tracking-wider">🧠 Detection Model</span>
          <select
            value={selectedModel}
            onChange={e => setSelectedModel(e.target.value)}
            className="w-full bg-white/5 border border-white/10 rounded-lg px-3 py-1.5 text-xs text-white/80 focus:outline-none focus:border-omni-500/50"
          >
            {MODEL_OPTIONS.map(opt => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
        </div>

        {/* Thresholds */}
        <div className="glass p-3 space-y-3">
          <span className="text-[11px] font-semibold text-white/50 uppercase tracking-wider">⚙️ Thresholds</span>

          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] text-white/40">Confidence</span>
              <span className="text-[10px] font-mono text-omni-300">{confidenceThreshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.95"
              step="0.05"
              value={confidenceThreshold}
              onChange={e => setConfidenceThreshold(parseFloat(e.target.value))}
              className="omni-slider"
            />
          </div>

          <div>
            <div className="flex items-center justify-between mb-1">
              <span className="text-[10px] text-white/40">NMS IoU</span>
              <span className="text-[10px] font-mono text-omni-300">{nmsThreshold.toFixed(2)}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="0.9"
              step="0.05"
              value={nmsThreshold}
              onChange={e => setNmsThreshold(parseFloat(e.target.value))}
              className="omni-slider"
            />
          </div>
        </div>

        {/* Module Toggles */}
        <div className="space-y-1">
          <span className="text-[11px] font-semibold text-white/50 uppercase tracking-wider px-1">🔌 Modules</span>
          {Object.entries(MODULE_INFO).map(([key, info]) => {
            const mod = modules[key]
            if (!mod) return null
            return (
              <button
                key={key}
                onClick={() => toggleModule(key)}
                className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg transition-all duration-200 text-left
                  ${mod.enabled
                    ? 'bg-omni-500/10 border border-omni-500/20 text-white/80'
                    : 'bg-white/[0.02] border border-transparent text-white/30 hover:bg-white/5'
                  }`}
              >
                <span className="text-base">{info.icon}</span>
                <div className="flex-1 min-w-0">
                  <div className="text-[11px] font-medium">{info.label}</div>
                  <div className="text-[9px] opacity-50 truncate">{info.desc}</div>
                </div>
                <div className={`toggle-switch ${mod.enabled ? 'active' : ''}`} />
              </button>
            )
          })}
        </div>
      </div>
    </div>
  )
}

export const RightSidebar: React.FC = () => {
  return (
    <div className="glass-sidebar w-[300px] flex flex-col h-full overflow-hidden border-l border-white/5">
      {/* Header */}
      <div className="px-4 py-3 border-b border-white/5">
        <h2 className="text-xs font-bold text-white/60 uppercase tracking-widest">Analytics</h2>
      </div>

      <div className="flex-1 overflow-y-auto p-3 space-y-4">
        {/* System Gauges */}
        <MetricGaugeRow />

        {/* Alert Banner */}
        <AlertBanner />

        {/* Metric Charts */}
        <MetricCharts />

        {/* Scene Graph */}
        <SceneGraph3D />

        {/* Point Cloud */}
        <PointCloud3D />
      </div>
    </div>
  )
}
