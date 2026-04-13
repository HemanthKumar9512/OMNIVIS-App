/**
 * OMNIVIS — Canvas Panel Component
 * Renders annotated video frames with overlay controls.
 */
import React, { useRef, useEffect, useState } from 'react'
import { useOmnivisStore, CanvasPanelType } from '../store/omnivis.store'

const PANEL_LABELS: Record<CanvasPanelType, string> = {
  detection: '🎯 Object Detection',
  segmentation: '🧩 Segmentation',
  flow: '🌊 Optical Flow',
  depth: '🏔️ Depth Estimation',
  tracking: '📍 Multi-Object Tracking',
  face: '👤 Face Analysis',
  gait: '🚶 Gait Analysis',
  '3d': '🧊 3D Reconstruction',
  scene_graph: '🕸️ Scene Graph',
}

const PANEL_TYPES: CanvasPanelType[] = ['detection', 'segmentation', 'flow', 'depth', 'tracking', 'face', 'gait', '3d', 'scene_graph']

interface CanvasPanelProps {
  panelIndex: number
}

export const CanvasPanel: React.FC<CanvasPanelProps> = ({ panelIndex }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [fps, setFps] = useState(0)
  const [showTypeMenu, setShowTypeMenu] = useState(false)
  const frameCountRef = useRef(0)
  const lastTimeRef = useRef(Date.now())

  const { panels, setPanelType, togglePanelFullscreen, currentFrame, currentFps } = useOmnivisStore()
  const panel = panels[panelIndex]

  // Draw frame on canvas
  useEffect(() => {
    if (!currentFrame || !canvasRef.current) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const img = new Image()
    img.onload = () => {
      canvas.width = img.width
      canvas.height = img.height
      ctx.drawImage(img, 0, 0)

      // FPS counter
      frameCountRef.current++
      const now = Date.now()
      if (now - lastTimeRef.current > 1000) {
        setFps(frameCountRef.current)
        frameCountRef.current = 0
        lastTimeRef.current = now
      }
    }
    img.src = `data:image/jpeg;base64,${currentFrame}`
  }, [currentFrame])

  const handleScreenshot = () => {
    if (!canvasRef.current) return
    const link = document.createElement('a')
    link.download = `omnivis_${panel.type}_${Date.now()}.png`
    link.href = canvasRef.current.toDataURL('image/png')
    link.click()
  }

  return (
    <div className={`canvas-panel ${panel.isFullscreen ? 'fullscreen' : ''} flex flex-col`}>
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-white/5 bg-black/20">
        <div className="flex items-center gap-2">
          <div className="relative">
            <button
              onClick={() => setShowTypeMenu(!showTypeMenu)}
              className="text-xs font-semibold text-white/70 hover:text-white transition-colors flex items-center gap-1"
            >
              {PANEL_LABELS[panel.type]}
              <svg className="w-3 h-3 opacity-50" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
              </svg>
            </button>
            {showTypeMenu && (
              <div className="absolute top-full left-0 mt-1 z-50 glass-strong p-1 min-w-[180px] animate-fade-in">
                {PANEL_TYPES.map(type => (
                  <button
                    key={type}
                    onClick={() => { setPanelType(panelIndex, type); setShowTypeMenu(false) }}
                    className={`block w-full text-left px-3 py-1.5 text-xs rounded-lg transition-colors
                      ${panel.type === type ? 'bg-omni-500/20 text-omni-300' : 'hover:bg-white/5 text-white/60 hover:text-white'}`}
                  >
                    {PANEL_LABELS[type]}
                  </button>
                ))}
              </div>
            )}
          </div>
        </div>

        <div className="flex items-center gap-2">
          {/* FPS Badge */}
          <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-omni-500/20 text-omni-300">
            {currentFps.toFixed(1)} FPS
          </span>

          {/* Screenshot */}
          <button onClick={handleScreenshot} className="p-1 hover:bg-white/10 rounded transition-colors" title="Screenshot">
            <svg className="w-3.5 h-3.5 text-white/50 hover:text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>

          {/* Fullscreen */}
          <button onClick={() => togglePanelFullscreen(panelIndex)} className="p-1 hover:bg-white/10 rounded transition-colors" title="Fullscreen">
            <svg className="w-3.5 h-3.5 text-white/50 hover:text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 8V4m0 0h4M4 4l5 5m11-1V4m0 0h-4m4 0l-5 5M4 16v4m0 0h4m-4 0l5-5m11 5l-5-5m5 5v-4m0 4h-4" />
            </svg>
          </button>
        </div>
      </div>

      {/* Canvas Area */}
      <div className="flex-1 relative flex items-center justify-center bg-black/60 min-h-0">
        {currentFrame ? (
          <canvas
            ref={canvasRef}
            className="max-w-full max-h-full object-contain"
          />
        ) : (
          <div className="flex flex-col items-center gap-3 text-white/30">
            <div className="w-16 h-16 rounded-2xl bg-white/5 flex items-center justify-center">
              <svg className="w-8 h-8" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
            </div>
            <span className="text-xs">Connect a source to begin</span>
          </div>
        )}

        {/* Timestamp overlay */}
        {currentFrame && (
          <div className="absolute bottom-2 left-2 text-[10px] font-mono text-white/40 bg-black/40 px-1.5 py-0.5 rounded">
            {new Date().toLocaleTimeString()}
          </div>
        )}
      </div>
    </div>
  )
}
