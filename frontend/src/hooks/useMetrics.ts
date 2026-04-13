/**
 * OMNIVIS — Metrics Hook
 * Manages performance metrics history and computed statistics.
 */
import { useMemo } from 'react'
import { useOmnivisStore } from '../store/omnivis.store'

export function useMetrics() {
  const { metricsHistory, currentFps, currentInferenceMs } = useOmnivisStore()

  const stats = useMemo(() => {
    if (metricsHistory.length === 0) {
      return {
        avgFps: 0,
        maxFps: 0,
        minFps: 0,
        avgInferenceMs: 0,
        avgDetections: 0,
        avgTracks: 0,
        totalDetections: 0,
      }
    }

    const recent = metricsHistory.slice(-60) // Last 60 samples
    const fpsValues = recent.map(m => m.fps)
    const infValues = recent.map(m => m.inference_ms)
    const detValues = recent.map(m => m.detection_count)
    const trkValues = recent.map(m => m.track_count)

    return {
      avgFps: fpsValues.reduce((a, b) => a + b, 0) / fpsValues.length,
      maxFps: Math.max(...fpsValues),
      minFps: Math.min(...fpsValues),
      avgInferenceMs: infValues.reduce((a, b) => a + b, 0) / infValues.length,
      avgDetections: detValues.reduce((a, b) => a + b, 0) / detValues.length,
      avgTracks: trkValues.reduce((a, b) => a + b, 0) / trkValues.length,
      totalDetections: detValues.reduce((a, b) => a + b, 0),
    }
  }, [metricsHistory])

  // Chart data for Recharts (last 60 seconds)
  const chartData = useMemo(() => {
    return metricsHistory.slice(-60).map((m, i) => ({
      time: i,
      fps: Math.round(m.fps * 10) / 10,
      latency: Math.round(m.inference_ms * 10) / 10,
      detections: m.detection_count,
      tracks: m.track_count,
      gpu: m.gpu_util,
    }))
  }, [metricsHistory])

  return {
    currentFps,
    currentInferenceMs,
    stats,
    chartData,
    metricsHistory,
  }
}
