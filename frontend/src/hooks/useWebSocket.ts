/**
 * OMNIVIS — WebSocket Connection Manager Hook
 * Handles connection, reconnection, message parsing, and state updates.
 */
import { useEffect, useRef, useCallback } from 'react'
import { useOmnivisStore } from '../store/omnivis.store'

const WS_URL = `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.hostname}:8000/ws/stream`
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
    if (type !== 'result') return
    if (!data) return

    // Annotated frame
    if (data.annotated_frame) {
      setCurrentFrame(data.annotated_frame)
    }

    // Detection results
    if (data.detection) {
      setDetections(data.detection.detections || [])
    }

    // Face results
    if (data.face) {
      setFaces(data.face.faces || [])
    }

    // Tracking results
    if (data.tracking) {
      setTracks(data.tracking.tracks || [])
      setTrails(data.tracking.trails || {})
    }

    // Scene graph
    if (data.scene_graph) {
      setSceneGraph({
        nodes: data.scene_graph.nodes || [],
        edges: data.scene_graph.edges || [],
        triplets: data.scene_graph.triplets || [],
      })
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
    }

    // Flow stats
    if (data.optical_flow) {
      setFlowStats({
        meanMag: data.optical_flow.mean_magnitude,
        maxMag: data.optical_flow.max_magnitude,
        method: data.optical_flow.method,
      })
    }

    // Reconstruction / point cloud
    if (data.reconstruction) {
      if (data.reconstruction.total_points > 0) {
        // Point cloud data will be fetched separately
      }
    }

    // Performance metrics
    const fps = data.fps || 0
    const inferenceMs = data.total_inference_ms || 0
    setCurrentPerf(fps, inferenceMs)

    addMetrics({
      timestamp: data.timestamp || Date.now() / 1000,
      fps,
      inference_ms: inferenceMs,
      detection_count: data.detection?.count || 0,
      track_count: data.tracking?.track_count || 0,
      gpu_util: 0,
      gpu_memory: 0,
      cpu_util: 0,
      map50: data.detection?.map50,
      iou: data.detection?.iou,
      epe: data.optical_flow?.epe,
    })
  }, [])

  const sendFrame = useCallback((frameBase64: string) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({
        type: 'frame',
        data: { frame: frameBase64 },
      }))
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

  return { sendFrame, sendConfig, connect, disconnect }
}
