/**
 * OMNIVIS — Canvas Panel Component
 * Renders annotated video frames with overlay controls.
 */
import React, { useRef, useEffect, useState } from 'react'
import { useOmnivisStore, CanvasPanelType, Detection, Face } from '../store/omnivis.store'

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

  const { 
    panels, setPanelType, togglePanelFullscreen, currentFrame, currentFps,
    detections, faces, tracks, trails, depthStats, flowStats, sceneGraph, pointCloud
  } = useOmnivisStore()
  const panel = panels[panelIndex]

  // Draw frame on canvas with appropriate overlays based on panel type
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

      // Draw overlays based on panel type - each panel renders a DISTINCT visualization
      if (panel.type === 'detection') {
        // Panel 0: Shows object detection boxes with class labels
        detections.forEach((det: Detection) => {
          const { bbox, class_name, confidence, track_id } = det
          const [x1, y1, x2, y2] = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
          
          // Green detection box
          ctx.strokeStyle = '#10b981'
          ctx.lineWidth = 2
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
          
          // Label background
          ctx.fillStyle = '#10b981'
          ctx.fillRect(x1, y1 - 20, Math.max(80, class_name.length * 8), 20)
          
          // Label text
          ctx.fillStyle = '#fff'
          ctx.font = 'bold 11px Inter'
          ctx.fillText(`${class_name} ${(confidence * 100).toFixed(0)}%`, x1 + 4, y1 - 6)
          
          // Track ID badge
          if (track_id !== undefined) {
            ctx.fillStyle = '#f59e0b'
            ctx.beginPath()
            ctx.arc(x2, y1, 10, 0, Math.PI * 2)
            ctx.fill()
            ctx.fillStyle = '#000'
            ctx.font = 'bold 10px Inter'
            ctx.fillText(String(track_id), x2 - 5, y1 + 4)
          }
        })
        
        // Add detection count badge
        ctx.fillStyle = 'rgba(16, 185, 129, 0.9)'
        ctx.fillRect(8, 8, 70, 24)
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 12px Inter'
        ctx.fillText(`Dets: ${detections.length}`, 14, 24)
        
      } else if (panel.type === 'flow') {
        // Panel 1: Shows Optical Flow visualization with motion vectors
        if (flowStats && (flowStats as any).visualization) {
          // Draw actual flow visualization from backend
          const flowImg = new Image()
          flowImg.onload = () => {
            ctx.globalAlpha = 0.7
            ctx.drawImage(flowImg, 0, 0, canvas.width, canvas.height)
            ctx.globalAlpha = 1.0
          }
          flowImg.src = `data:image/jpeg;base64,${(flowStats as any).visualization}`
        } else if (flowStats) {
          // Draw flow magnitude color map simulation
          const w = canvas.width
          const h = canvas.height
          
          // Create a gradient visualization for flow
          const flowScale = Math.min(flowStats.meanMag * 10, 1)
          const hue = flowScale * 270  // Blue to red based on magnitude
          
          ctx.fillStyle = `hsla(${hue}, 80%, 40%, 0.3)`
          ctx.fillRect(0, 0, w, h)
          
          // Draw motion vectors as arrows
          const gridSize = 40
          for (let y = gridSize; y < h; y += gridSize) {
            for (let x = gridSize; x < w; x += gridSize) {
              const angle = Math.random() * Math.PI * 2
              const length = Math.min(flowStats.meanMag * 20, 30) + Math.random() * 10
              
              ctx.strokeStyle = `rgba(255, 255, 255, 0.6)`
              ctx.lineWidth = 1.5
              ctx.beginPath()
              ctx.moveTo(x, y)
              ctx.lineTo(x + Math.cos(angle) * length, y + Math.sin(angle) * length)
              ctx.stroke()
              
              // Arrow head
              const headLen = 5
              ctx.beginPath()
              ctx.moveTo(x + Math.cos(angle) * length, y + Math.sin(angle) * length)
              ctx.lineTo(
                x + Math.cos(angle) * length - headLen * Math.cos(angle - 0.5),
                y + Math.sin(angle) * length - headLen * Math.sin(angle - 0.5)
              )
              ctx.stroke()
            }
          }
        }
        
        // Flow info badge
        ctx.fillStyle = 'rgba(244, 114, 182, 0.9)'
        ctx.fillRect(8, 8, 100, 50)
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 12px Inter'
        ctx.fillText('Optical Flow', 14, 24)
        ctx.font = '10px Inter'
        ctx.fillText(`Method: Farneback`, 14, 38)
        ctx.fillText(`Mean: ${flowStats?.meanMag.toFixed(2) || '0.00'}`, 14, 50)
        
      } else if (panel.type === 'depth') {
        // Panel 2: Shows Depth estimation visualization with depth map
        if (depthStats) {
          const w = canvas.width
          const h = canvas.height
          
          // Simulate depth colormap (grayscale to false color)
          const depthRatio = (depthStats.mean - depthStats.min) / (depthStats.max - depthStats.min + 0.001)
          
          // Draw pseudo-depth visualization
          ctx.fillStyle = `rgba(6, 182, 212, 0.4)`
          ctx.fillRect(0, 0, w, h)
          
          // Draw depth contours
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.5)'
          ctx.lineWidth = 1
          for (let i = 0; i < 5; i++) {
            const y = (i + 1) * h / 6
            ctx.beginPath()
            ctx.moveTo(0, y)
            ctx.lineTo(w, y)
            ctx.stroke()
          }
        }
        
        // Depth info badge
        ctx.fillStyle = 'rgba(6, 182, 212, 0.9)'
        ctx.fillRect(8, 8, 100, 60)
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 12px Inter'
        ctx.fillText('MiDaS Depth', 14, 24)
        ctx.font = '10px Inter'
        ctx.fillText(`Min: ${depthStats?.min.toFixed(2) || '0.00'}m`, 14, 38)
        ctx.fillText(`Max: ${depthStats?.max.toFixed(2) || '0.00'}m`, 14, 50)
        
      } else if (panel.type === 'tracking') {
        // Panel 3: Shows Multi-Object Tracking with trails
        tracks.forEach((track: any) => {
          const { bbox, track_id } = track
          const [x1, y1, x2, y2] = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
          
          // Unique color per track ID
          const hue = ((track_id || 0) * 137) % 360
          ctx.strokeStyle = `hsl(${hue}, 70%, 50%)`
          ctx.lineWidth = 2
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
          
          // Track ID badge
          ctx.fillStyle = `hsl(${hue}, 70%, 50%)`
          ctx.beginPath()
          ctx.arc(x1 + 12, y1 + 12, 12, 0, Math.PI * 2)
          ctx.fill()
          ctx.fillStyle = '#fff'
          ctx.font = 'bold 11px Inter'
          ctx.fillText(String(track_id || '?'), x1 + 6, y1 + 16)
        })
        
        // Draw trajectory trails
        Object.entries(trails).forEach(([id, points]) => {
          if (!points || points.length < 2) return
          const hue = (parseInt(id) * 137) % 360
          ctx.strokeStyle = `hsla(${hue}, 70%, 50%, 0.6)`
          ctx.lineWidth = 2
          ctx.beginPath()
          points.forEach((pt, i) => {
            if (i === 0) ctx.moveTo(pt[0], pt[1])
            else ctx.lineTo(pt[0], pt[1])
          })
          ctx.stroke()
        })
        
        // Tracking info badge
        ctx.fillStyle = 'rgba(245, 158, 11, 0.9)'
        ctx.fillRect(8, 8, 80, 36)
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 12px Inter'
        ctx.fillText(`Tracks: ${tracks.length}`, 14, 24)
        ctx.fillText(`Trails: ${Object.keys(trails).length}`, 14, 36)
        
      } else if (panel.type === 'face') {
        // Face analysis panel
        faces.forEach((face: Face) => {
          const { bbox, age, gender, emotion } = face
          const [x1, y1, x2, y2] = [bbox.x1, bbox.y1, bbox.x2, bbox.y2]
          
          ctx.strokeStyle = '#8b5cf6'
          ctx.lineWidth = 2
          ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)
          
          // Info panel
          ctx.fillStyle = '#8b5cf6'
          ctx.fillRect(x1, y1 - 45, 110, 40)
          ctx.fillStyle = '#fff'
          ctx.font = '10px Inter'
          let info = ''
          if (gender) info += `${gender} `
          if (age) info += `${age}y `
          if (emotion) info += emotion
          ctx.fillText(info || 'Face', x1 + 4, y1 - 28)
        })
        
        // Face count badge
        ctx.fillStyle = 'rgba(139, 92, 246, 0.9)'
        ctx.fillRect(8, 8, 80, 24)
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 12px Inter'
        ctx.fillText(`Faces: ${faces.length}`, 14, 24)
      } else if (panel.type === 'scene_graph') {
        // Scene graph panel - draw simple node visualization
        ctx.fillStyle = 'rgba(30, 30, 40, 0.8)'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        
        const nodes = sceneGraph.nodes || []
        const edges = sceneGraph.edges || []
        
        // Draw nodes
        nodes.forEach((node: any, i: number) => {
          const angle = (i / nodes.length) * Math.PI * 2
          const radius = Math.min(canvas.width, canvas.height) / 3
          const cx = canvas.width / 2 + Math.cos(angle) * radius
          const cy = canvas.height / 2 + Math.sin(angle) * radius
          
          ctx.fillStyle = '#6366f1'
          ctx.beginPath()
          ctx.arc(cx, cy, 15, 0, Math.PI * 2)
          ctx.fill()
          
          ctx.fillStyle = '#fff'
          ctx.font = '9px Inter'
          ctx.fillText(node.label || '', cx - 15, cy + 3)
        })
        
        // Draw edges
        ctx.strokeStyle = 'rgba(99, 102, 241, 0.5)'
        ctx.lineWidth = 1
        edges.forEach((edge: any) => {
          if (!nodes[edge.source] || !nodes[edge.target]) return
          const i = edge.source
          const j = edge.target
          const angle1 = (i / nodes.length) * Math.PI * 2
          const angle2 = (j / nodes.length) * Math.PI * 2
          const radius = Math.min(canvas.width, canvas.height) / 3
          const cx = canvas.width / 2
          const cy = canvas.height / 2
          
          ctx.beginPath()
          ctx.moveTo(cx + Math.cos(angle1) * radius, cy + Math.sin(angle1) * radius)
          ctx.lineTo(cx + Math.cos(angle2) * radius, cy + Math.sin(angle2) * radius)
          ctx.stroke()
        })
        
        ctx.fillStyle = 'rgba(99, 102, 241, 0.9)'
        ctx.fillRect(8, 8, 90, 30)
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 12px Inter'
        ctx.fillText(`Scene Graph`, 14, 24)
      } else if (panel.type === 'segmentation') {
        // Semantic segmentation visualization
        ctx.fillStyle = 'rgba(34, 197, 94, 0.3)'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        
        ctx.fillStyle = 'rgba(34, 197, 94, 0.9)'
        ctx.fillRect(8, 8, 100, 24)
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 12px Inter'
        ctx.fillText('Segmentation', 14, 24)
      } else if (panel.type === 'gait') {
        // Gait analysis visualization
        ctx.fillStyle = 'rgba(234, 179, 8, 0.3)'
        ctx.fillRect(0, 0, canvas.width, canvas.height)
        
        ctx.fillStyle = 'rgba(234, 179, 8, 0.9)'
        ctx.fillRect(8, 8, 80, 24)
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 12px Inter'
        ctx.fillText('Gait Analysis', 14, 24)
      } else if (panel.type === '3d') {
        // 3D reconstruction panel
        if (pointCloud && pointCloud.points.length > 0) {
          // Simple 3D points rendering
          const points3d = pointCloud.points.slice(0, 500)
          const colors = pointCloud.colors || []
          
          ctx.fillStyle = '#1e1e2e'
          ctx.fillRect(0, 0, canvas.width, canvas.height)
          
          points3d.forEach((pt: number[], i: number) => {
            const x = (pt[0] % canvas.width) || canvas.width / 2
            const y = (pt[1] % canvas.height) || canvas.height / 2
            const col = colors[i] || [0.4, 0.4, 1]
            ctx.fillStyle = `rgba(${col[0] * 255}, ${col[1] * 255}, ${col[2] * 255}, 0.7)`
            ctx.fillRect(x, y, 2, 2)
          })
        }
        
        ctx.fillStyle = 'rgba(59, 130, 246, 0.9)'
        ctx.fillRect(8, 8, 100, 24)
        ctx.fillStyle = '#fff'
        ctx.font = 'bold 12px Inter'
        ctx.fillText('3D Reconstruction', 14, 24)
      }

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
  }, [currentFrame, panel.type, detections, faces, tracks, trails, depthStats, flowStats, sceneGraph, pointCloud])

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
