/**
 * OMNIVIS — WebSocket Connection Manager Hook
 * Handles connection, reconnection, message parsing, and state updates.
 */
import { useEffect, useRef, useCallback } from 'react'
import { useOmnivisStore } from '../store/omnivis.store'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname}:8000`
const WS_URL = `${BACKEND_URL}/ws/stream`
const RECONNECT_DELAY = 3000
const MAX_RECONNECT_ATTEMPTS = 10

export function useWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectCount = useRef(0)
  const reconnectTimer = useRef<ReturnType<typeof setTimeout>>()

  const {
    setConnected, setCurrentFrame, setDetections, setFaces,
    setTracks, setTrails, setSceneGraph, addAnomaly,
    setAlertLevel, setActions, addMetrics, setCurrentPerf,
    setDepthStats, setFlowStats, setPointCloud,
  } = useOmnivisStore()

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    try {
      const ws = new WebSocket(WS_URL)
      wsRef.current = ws

      ws.onopen = () => {
        console.log('[OMNIVIS] WebSocket connected')
        setConnected(true)
        reconnectCount.current = 0
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          handleMessage(message)
        } catch (err) {
          console.error('[OMNIVIS] Message parse error:', err)
        }
      }

      ws.onclose = () => {
        console.log('[OMNIVIS] WebSocket disconnected')
        setConnected(false)
        scheduleReconnect()
      }

      ws.onerror = (err) => {
        console.error('[OMNIVIS] WebSocket error:', err)
        ws.close()
      }
    } catch (err) {
      console.error('[OMNIVIS] Connection failed:', err)
      scheduleReconnect()
    }
  }, [])

  const scheduleReconnect = useCallback(() => {
    if (reconnectCount.current >= MAX_RECONNECT_ATTEMPTS) {
      console.log('[OMNIVIS] Max reconnection attempts reached')
      return
    }
    reconnectCount.current++
    const delay = RECONNECT_DELAY * Math.min(reconnectCount.current, 5)
    console.log(`[OMNIVIS] Reconnecting in ${delay}ms (attempt ${reconnectCount.current})`)
    reconnectTimer.current = setTimeout(connect, delay)
  }, [connect])

  const handleMessage = useCallback((message: any) => {
    const { type, data } = message
    console.log('[WS] Received message type:', type, 'keys:', Object.keys(data || {}))
    
    if (type !== 'result') {
      console.log('[WS] Unknown message type:', type)
      return
    }
    if (!data) {
      console.log('[WS] No data in message')
      return
    }

    // Annotated frame
    if (data.annotated_frame) {
      setCurrentFrame(data.annotated_frame)
      console.log('[WS] Frame updated, length:', data.annotated_frame.length)
    }

    // Detection results
    if (data.detection) {
      setDetections(data.detection.detections || [])
      console.log('[WS] Detections:', data.detection.count || 0)
    }

    // Face results
    if (data.face) {
      setFaces(data.face.faces || [])
      console.log('[WS] Faces:', data.face.face_count || 0)
    }

    // Tracking results
    if (data.tracking) {
      setTracks(data.tracking.tracks || [])
      setTrails(data.tracking.trails || {})
      console.log('[WS] Tracks:', data.tracking.track_count || 0)
    }

    // Scene graph
    if (data.scene_graph) {
      setSceneGraph({
        nodes: data.scene_graph.nodes || [],
        edges: data.scene_graph.edges || [],
        triplets: data.scene_graph.triplets || [],
      })
      console.log('[WS] Scene graph nodes:', data.scene_graph.nodes?.length || 0)
    }

    // Anomaly detection
    if (data.anomaly) {
      setAlertLevel(data.anomaly.alert_level || 'green')
      if (data.anomaly.anomalies?.length > 0) {
        for (const a of data.anomaly.anomalies) {
          addAnomaly({
            id: `${Date.now()}-${Math.random().toString(36).slice(2)}`,
            timestamp: data.timestamp || Date.now() / 1000,
            type: a.type,
            severity: a.severity,
            score: a.score,
            description: a.description,
          })
        }
      }
      console.log('[WS] Anomaly level:', data.anomaly.alert_level)
    }

    // Action recognition
    if (data.action) {
      setActions(data.action.actions || [])
    }

    // Depth stats
    if (data.depth) {
      setDepthStats({
        min: data.depth.min_depth,
        max: data.depth.max_depth,
        mean: data.depth.mean_depth,
      })
      console.log('[WS] Depth:', data.depth.mean_depth?.toFixed(2))
    }

    // Flow stats
    if (data.optical_flow) {
      setFlowStats({
        meanMag: data.optical_flow.mean_magnitude,
        maxMag: data.optical_flow.max_magnitude,
        method: data.optical_flow.method,
        visualization: data.optical_flow.visualization,
      })
      console.log('[WS] Flow:', data.optical_flow.mean_magnitude?.toFixed(2))
    }

    // Reconstruction / point cloud
    if (data.reconstruction) {
      if (data.reconstruction.total_points > 0 && data.reconstruction.points) {
        setPointCloud({
          points: data.reconstruction.points,
          colors: data.reconstruction.colors || [],
        })
      }
      console.log('[WS] 3D Points:', data.reconstruction.total_points || 0)
    }

    // Performance metrics
    const fps = data.fps || 0
    const inferenceMs = data.total_inference_ms || 0
    setCurrentPerf(fps, inferenceMs)

    const sysMetrics = data.system_metrics || {}
    addMetrics({
      timestamp: data.timestamp || Date.now() / 1000,
      fps,
      inference_ms: inferenceMs,
      detection_count: sysMetrics.detection_count || data.detection?.count || 0,
      track_count: sysMetrics.track_count || data.tracking?.track_count || 0,
      gpu_util: sysMetrics.gpu_util ?? data.gpu_util ?? 0,
      gpu_memory: sysMetrics.gpu_memory ?? data.gpu_memory ?? 0,
      cpu_util: sysMetrics.cpu_percent ?? data.cpu_util ?? 0,
      map50: data.detection?.map50,
      iou: data.detection?.iou,
      epe: data.optical_flow?.epe,
    })
  }, [])

  const sendFrame = useCallback((frameBase64: string) => {
    console.log('[OMNIVIS] Sending frame to backend, WS state:', wsRef.current?.readyState)
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        data: { frame: frameBase64 },
      }))
    } else {
      console.warn('[OMNIVIS] WebSocket not open, cannot send frame')
    }
  }, [])

  const sendConfig = useCallback((module: string, config: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'config',
        data: { module, ...config },
      }))
    }
  }, [])

  const disconnect = useCallback(() => {
    if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
    wsRef.current?.close()
    setConnected(false)
  }, [])

  useEffect(() => {
    connect()
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  // Expose WebSocket ref for external access
  const getWsState = useCallback(() => wsRef.current?.readyState, [])

  return { sendFrame, sendConfig, connect, disconnect, wsRef: wsRef as React.MutableRefObject<WebSocket | null>, getWsState }
}
