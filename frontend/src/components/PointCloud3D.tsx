/**
 * OMNIVIS — 3D Point Cloud Viewer
 * Three.js WebGL renderer with orbit controls for point cloud visualization.
 */
import React, { useRef, useEffect, useState, useCallback } from 'react'
import { useOmnivisStore } from '../store/omnivis.store'

export const PointCloud3D: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const { pointCloud } = useOmnivisStore()
  const [rotation, setRotation] = useState({ x: -0.3, y: 0.5 })
  const [zoom, setZoom] = useState(1)
  const isDragging = useRef(false)
  const lastMouse = useRef({ x: 0, y: 0 })

  const drawPointCloud = useCallback(() => {
    if (!canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const w = canvas.width
    const h = canvas.height

    // Clear
    ctx.fillStyle = 'rgba(0, 0, 0, 0.9)'
    ctx.fillRect(0, 0, w, h)

    // Draw grid
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.1)'
    ctx.lineWidth = 0.5
    for (let i = 0; i < w; i += 20) {
      ctx.beginPath()
      ctx.moveTo(i, 0)
      ctx.lineTo(i, h)
      ctx.stroke()
    }
    for (let i = 0; i < h; i += 20) {
      ctx.beginPath()
      ctx.moveTo(0, i)
      ctx.lineTo(w, i)
      ctx.stroke()
    }

    if (!pointCloud || pointCloud.points.length === 0) {
      // Draw placeholder
      ctx.fillStyle = 'rgba(255,255,255,0.2)'
      ctx.font = '12px Inter'
      ctx.textAlign = 'center'
      ctx.fillText('3D Point Cloud', w / 2, h / 2 - 10)
      ctx.fillText('(Reconstruct to populate)', w / 2, h / 2 + 10)

      // Draw rotating cube wireframe
      const time = Date.now() / 2000
      const cubeSize = 30
      const cx = w / 2
      const cy = h / 2
      const cosA = Math.cos(time)
      const sinA = Math.sin(time)

      const vertices = [
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
      ].map(([x, y, z]) => {
        const rx = x * cosA - z * sinA
        const rz = x * sinA + z * cosA
        const ry = y * Math.cos(time * 0.7) - rz * Math.sin(time * 0.7)
        const rzFinal = y * Math.sin(time * 0.7) + rz * Math.cos(time * 0.7)
        const scale = cubeSize / (3 + rzFinal * 0.3)
        return [cx + rx * scale, cy + ry * scale]
      })

      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.3)'
      ctx.lineWidth = 1
      edges.forEach(([a, b]) => {
        ctx.beginPath()
        ctx.moveTo(vertices[a][0], vertices[a][1])
        ctx.lineTo(vertices[b][0], vertices[b][1])
        ctx.stroke()
      })

      requestAnimationFrame(drawPointCloud)
      return
    }

    // Draw actual points
    const cosX = Math.cos(rotation.x)
    const sinX = Math.sin(rotation.x)
    const cosY = Math.cos(rotation.y)
    const sinY = Math.sin(rotation.y)

    const projected = pointCloud.points.map((p, i) => {
      let [x, y, z] = p
      // Apply rotation
      const ry = y * cosX - z * sinX
      const rz = y * sinX + z * cosX
      const rx = x * cosY - rz * sinY
      const rz2 = x * sinY + rz * cosY
      // Project
      const scale = 200 * zoom / (5 + rz2)
      return {
        px: w / 2 + rx * scale,
        py: h / 2 + ry * scale,
        depth: rz2,
        color: pointCloud.colors[i] || [0.4, 0.4, 1],
      }
    })

    // Sort by depth
    projected.sort((a, b) => b.depth - a.depth)

    // Draw points
    projected.forEach(p => {
      const alpha = Math.max(0.1, Math.min(1, 1 - p.depth / 50))
      const r = Math.round(p.color[0] * 255)
      const g = Math.round(p.color[1] * 255)
      const b = Math.round(p.color[2] * 255)
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`
      ctx.fillRect(p.px, p.py, 2, 2)
    })

    // Info
    ctx.fillStyle = 'rgba(255,255,255,0.3)'
    ctx.font = '10px JetBrains Mono'
    ctx.textAlign = 'left'
    ctx.fillText(`Points: ${pointCloud.points.length}`, 8, 16)
  }, [pointCloud, rotation, zoom])

  useEffect(() => {
    drawPointCloud()
  }, [drawPointCloud])

  // Mouse controls
  const handleMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true
    lastMouse.current = { x: e.clientX, y: e.clientY }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return
    const dx = e.clientX - lastMouse.current.x
    const dy = e.clientY - lastMouse.current.y
    setRotation(r => ({ x: r.x + dy * 0.01, y: r.y + dx * 0.01 }))
    lastMouse.current = { x: e.clientX, y: e.clientY }
  }

  const handleMouseUp = () => { isDragging.current = false }
  const handleWheel = (e: React.WheelEvent) => {
    setZoom(z => Math.max(0.3, Math.min(5, z - e.deltaY * 0.001)))
  }

  return (
    <div className="glass-card p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] font-medium text-white/50 uppercase tracking-wider">🧊 3D Point Cloud</span>
        <span className="text-[10px] text-white/30">
          {pointCloud?.points.length || 0} pts
        </span>
      </div>
      <canvas
        ref={canvasRef}
        width={300}
        height={180}
        className="w-full rounded-lg cursor-grab active:cursor-grabbing"
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />
    </div>
  )
}
