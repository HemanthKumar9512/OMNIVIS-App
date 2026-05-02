/**
 * OMNIVIS — 3D Point Cloud Viewer
 * Canvas 2D renderer with auto-rotation and orbit controls.
 */
import React, { useRef, useEffect, useState, useCallback } from 'react'
import { useOmnivisStore } from '../store/omnivis.store'

export const PointCloud3D: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const animFrameRef = useRef<number>(0)
  const { pointCloud } = useOmnivisStore()
  const [rotation, setRotation] = useState({ x: -0.4, y: 0 })
  const [zoom, setZoom] = useState(1.0)
  const isDragging = useRef(false)
  const lastMouse = useRef({ x: 0, y: 0 })
  const autoRotate = useRef(true)

  const drawPointCloud = useCallback(() => {
    if (!canvasRef.current) return
    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    if (canvas.width !== rect.width * dpr || canvas.height !== rect.height * dpr) {
      canvas.width = rect.width * dpr
      canvas.height = rect.height * dpr
      ctx.scale(dpr, dpr)
    }

    const w = rect.width
    const h = rect.height

    ctx.fillStyle = 'rgba(5, 5, 20, 0.95)'
    ctx.fillRect(0, 0, w, h)

    // Grid
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.08)'
    ctx.lineWidth = 0.5
    for (let i = 0; i < w; i += 30) {
      ctx.beginPath(); ctx.moveTo(i, 0); ctx.lineTo(i, h); ctx.stroke()
    }
    for (let i = 0; i < h; i += 30) {
      ctx.beginPath(); ctx.moveTo(0, i); ctx.lineTo(w, i); ctx.stroke()
    }

    const hasPoints = pointCloud && pointCloud.points.length > 0

    if (!hasPoints) {
      ctx.fillStyle = 'rgba(255,255,255,0.15)'
      ctx.font = '11px Inter, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText('3D Point Cloud', w / 2, h / 2 - 8)
      ctx.fillText('Enable reconstruction to populate', w / 2, h / 2 + 8)

      // Rotating cube
      const time = Date.now() / 2000
      const cubeSize = 25
      const cx = w / 2, cy = h / 2
      const cosA = Math.cos(time), sinA = Math.sin(time)
      const cosB = Math.cos(time * 0.7), sinB = Math.sin(time * 0.7)

      const vertices = [
        [-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
        [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1],
      ].map(([x, y, z]) => {
        const rx = x * cosA - z * sinA
        const rz = x * sinA + z * cosA
        const ry = y * cosB - rz * sinB
        const rz2 = y * sinB + rz * cosB
        const scale = cubeSize / (2.5 + rz2 * 0.3)
        return [cx + rx * scale, cy + ry * scale]
      })

      const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]]
      ctx.strokeStyle = 'rgba(99, 102, 241, 0.25)'
      ctx.lineWidth = 1
      edges.forEach(([a, b]) => {
        ctx.beginPath()
        ctx.moveTo(vertices[a][0], vertices[a][1])
        ctx.lineTo(vertices[b][0], vertices[b][1])
        ctx.stroke()
      })

      animFrameRef.current = requestAnimationFrame(drawPointCloud)
      return
    }

    // Auto-rotate
    if (autoRotate.current && !isDragging.current) {
      setRotation(r => ({ x: r.x, y: r.y + 0.008 }))
    }

    const cosX = Math.cos(rotation.x), sinX = Math.sin(rotation.x)
    const cosY = Math.cos(rotation.y), sinY = Math.sin(rotation.y)

    // Compute bounding box for auto-scaling
    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity, minZ = Infinity, maxZ = -Infinity
    for (const p of pointCloud.points) {
      if (p[0] < minX) minX = p[0]; if (p[0] > maxX) maxX = p[0]
      if (p[1] < minY) minY = p[1]; if (p[1] > maxY) maxY = p[1]
      if (p[2] < minZ) minZ = p[2]; if (p[2] > maxZ) maxZ = p[2]
    }
    const rangeX = maxX - minX || 1, rangeY = maxY - minY || 1, rangeZ = maxZ - minZ || 1
    const maxRange = Math.max(rangeX, rangeY, rangeZ)
    const autoScale = Math.min(w, h) / (maxRange * 1.5) * zoom

    const centerX = (minX + maxX) / 2
    const centerY = (minY + maxY) / 2
    const centerZ = (minZ + maxZ) / 2

    const projected = pointCloud.points.map((p, i) => {
      let x = p[0] - centerX
      let y = p[1] - centerY
      let z = p[2] - centerZ

      // Rotate Y
      const rx1 = x * cosY - z * sinY
      const rz1 = x * sinY + z * cosY
      // Rotate X
      const ry = y * cosX - rz1 * sinX
      const rz2 = y * sinX + rz1 * cosX

      const scale = autoScale / (1 + rz2 * 0.002)
      return {
        px: w / 2 + rx1 * scale,
        py: h / 2 + ry * scale,
        depth: rz2,
        color: pointCloud.colors[i] || [0.4, 0.5, 1.0],
      }
    })

    projected.sort((a, b) => b.depth - a.depth)

    // Draw points
    const maxDepth = Math.max(...projected.map(p => Math.abs(p.depth))) || 1
    projected.forEach(p => {
      const alpha = Math.max(0.2, Math.min(1, 1 - Math.abs(p.depth) / (maxDepth * 1.5)))
      const size = Math.max(1.5, 3 * alpha)
      const r = Math.round(Math.min(255, p.color[0] * 255))
      const g = Math.round(Math.min(255, p.color[1] * 255))
      const b = Math.round(Math.min(255, p.color[2] * 255))
      ctx.fillStyle = `rgba(${r},${g},${b},${alpha})`
      ctx.beginPath()
      ctx.arc(p.px, p.py, size, 0, Math.PI * 2)
      ctx.fill()
    })

    // Draw connecting lines for nearby points (sparse)
    ctx.strokeStyle = 'rgba(99, 102, 241, 0.06)'
    ctx.lineWidth = 0.5
    const step = Math.max(1, Math.floor(projected.length / 100))
    for (let i = 0; i < projected.length - step; i += step) {
      const a = projected[i], b = projected[i + step]
      const dist = Math.sqrt((a.px - b.px) ** 2 + (a.py - b.py) ** 2)
      if (dist < 40) {
        ctx.beginPath()
        ctx.moveTo(a.px, a.py)
        ctx.lineTo(b.px, b.py)
        ctx.stroke()
      }
    }

    // Info overlay
    ctx.fillStyle = 'rgba(255,255,255,0.3)'
    ctx.font = '10px JetBrains Mono, monospace'
    ctx.textAlign = 'left'
    ctx.fillText(`Points: ${pointCloud.points.length}`, 8, 14)
    ctx.fillText(`Zoom: ${zoom.toFixed(1)}x`, 8, 26)

    animFrameRef.current = requestAnimationFrame(drawPointCloud)
  }, [pointCloud, rotation, zoom])

  useEffect(() => {
    animFrameRef.current = requestAnimationFrame(drawPointCloud)
    return () => cancelAnimationFrame(animFrameRef.current)
  }, [drawPointCloud])

  const handleMouseDown = (e: React.MouseEvent) => {
    isDragging.current = true
    autoRotate.current = false
    lastMouse.current = { x: e.clientX, y: e.clientY }
  }

  const handleMouseMove = (e: React.MouseEvent) => {
    if (!isDragging.current) return
    const dx = e.clientX - lastMouse.current.x
    const dy = e.clientY - lastMouse.current.y
    setRotation(r => ({ x: r.x + dy * 0.008, y: r.y + dx * 0.008 }))
    lastMouse.current = { x: e.clientX, y: e.clientY }
  }

  const handleMouseUp = () => {
    isDragging.current = false
    setTimeout(() => { autoRotate.current = true }, 3000)
  }

  const handleWheel = (e: React.WheelEvent) => {
    setZoom(z => Math.max(0.2, Math.min(8, z - e.deltaY * 0.002)))
  }

  return (
    <div className="glass-card p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] font-medium text-white/50 uppercase tracking-wider">3D Point Cloud</span>
        <span className="text-[10px] text-white/30">
          {pointCloud?.points.length || 0} pts
        </span>
      </div>
      <canvas
        ref={canvasRef}
        className="w-full rounded-lg cursor-grab active:cursor-grabbing"
        style={{ height: '180px' }}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        onMouseLeave={handleMouseUp}
        onWheel={handleWheel}
      />
    </div>
  )
}
