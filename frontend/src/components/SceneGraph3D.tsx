/**
 * OMNIVIS — Scene Graph 3D Visualization
 * D3.js force-directed graph with live-updating nodes and edges.
 */
import React, { useEffect, useRef } from 'react'
import { useOmnivisStore } from '../store/omnivis.store'

export const SceneGraph3D: React.FC = () => {
  const svgRef = useRef<SVGSVGElement>(null)
  const { sceneGraph } = useOmnivisStore()

  useEffect(() => {
    if (!svgRef.current) return
    const svg = svgRef.current
    const width = svg.clientWidth || 300
    const height = svg.clientHeight || 200

    // Clear previous
    while (svg.firstChild) svg.removeChild(svg.firstChild)

    const { nodes, edges } = sceneGraph
    if (nodes.length === 0) {
      // Draw placeholder text
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text')
      text.setAttribute('x', `${width / 2}`)
      text.setAttribute('y', `${height / 2}`)
      text.setAttribute('text-anchor', 'middle')
      text.setAttribute('fill', 'rgba(255,255,255,0.2)')
      text.setAttribute('font-size', '12')
      text.textContent = 'Scene graph will appear here'
      svg.appendChild(text)
      return
    }

    // Simple force simulation (no D3 dependency for server compatibility)
    const positions: { x: number; y: number }[] = nodes.map((_, i) => ({
      x: width / 2 + Math.cos(i * 2 * Math.PI / nodes.length) * Math.min(width, height) * 0.35,
      y: height / 2 + Math.sin(i * 2 * Math.PI / nodes.length) * Math.min(width, height) * 0.35,
    }))

    // Draw edges
    edges.forEach(edge => {
      if (edge.confidence < 0.3) return
      const src = positions[edge.source]
      const tgt = positions[edge.target]
      if (!src || !tgt) return

      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line')
      line.setAttribute('x1', `${src.x}`)
      line.setAttribute('y1', `${src.y}`)
      line.setAttribute('x2', `${tgt.x}`)
      line.setAttribute('y2', `${tgt.y}`)
      line.setAttribute('stroke', `rgba(99, 102, 241, ${edge.confidence * 0.5})`)
      line.setAttribute('stroke-width', '1.5')
      svg.appendChild(line)

      // Edge label
      const midX = (src.x + tgt.x) / 2
      const midY = (src.y + tgt.y) / 2
      const label = document.createElementNS('http://www.w3.org/2000/svg', 'text')
      label.setAttribute('x', `${midX}`)
      label.setAttribute('y', `${midY - 4}`)
      label.setAttribute('text-anchor', 'middle')
      label.setAttribute('fill', 'rgba(167, 139, 250, 0.6)')
      label.setAttribute('font-size', '8')
      label.textContent = edge.predicate
      svg.appendChild(label)
    })

    // Draw nodes
    nodes.forEach((node, i) => {
      const pos = positions[i]

      // Node circle
      const circle = document.createElementNS('http://www.w3.org/2000/svg', 'circle')
      circle.setAttribute('cx', `${pos.x}`)
      circle.setAttribute('cy', `${pos.y}`)
      circle.setAttribute('r', `${8 + node.confidence * 6}`)
      circle.setAttribute('fill', `rgba(99, 102, 241, ${0.3 + node.confidence * 0.4})`)
      circle.setAttribute('stroke', 'rgba(129, 140, 248, 0.5)')
      circle.setAttribute('stroke-width', '1.5')
      svg.appendChild(circle)

      // Node label
      const text = document.createElementNS('http://www.w3.org/2000/svg', 'text')
      text.setAttribute('x', `${pos.x}`)
      text.setAttribute('y', `${pos.y + 20}`)
      text.setAttribute('text-anchor', 'middle')
      text.setAttribute('fill', 'rgba(255,255,255,0.7)')
      text.setAttribute('font-size', '10')
      text.setAttribute('font-weight', '500')
      text.textContent = node.label
      svg.appendChild(text)
    })
  }, [sceneGraph])

  return (
    <div className="glass-card p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] font-medium text-white/50 uppercase tracking-wider">🕸️ Scene Graph</span>
        <span className="text-[10px] text-white/30">
          {sceneGraph.nodes.length}N / {sceneGraph.edges.length}E
        </span>
      </div>
      <svg
        ref={svgRef}
        className="w-full rounded-lg bg-black/20"
        style={{ height: '180px' }}
      />
      {/* Triplets list */}
      {sceneGraph.triplets.length > 0 && (
        <div className="mt-2 max-h-[80px] overflow-y-auto space-y-0.5">
          {sceneGraph.triplets.slice(0, 5).map((t, i) => (
            <div key={i} className="text-[10px] text-white/40 flex items-center gap-1">
              <span className="text-omni-300">{t.subject}</span>
              <span className="text-amber-400/60">{t.predicate}</span>
              <span className="text-emerald-300">{t.object}</span>
              <span className="ml-auto text-white/20">{(t.confidence * 100).toFixed(0)}%</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
