/**
 * OMNIVIS — Video Processor Hook
 * Processes video files frame-by-frame for inference.
 */
import { useEffect, useRef, useCallback, useState } from 'react'

interface UseVideoProcessorOptions {
  videoUrl: string | null
  enabled?: boolean
  fps?: number
  onFrame?: (frameBase64: string) => void
}

interface VideoProcessorResult {
  videoRef: React.RefObject<HTMLVideoElement>
  isPlaying: boolean
  error: string | null
  duration: number
  currentTime: number
  startPlayback: () => Promise<void>
  stopPlayback: () => void
  captureFrame: () => void
}

export function useVideoProcessor({ videoUrl, enabled = false, fps = 8, onFrame }: UseVideoProcessorOptions): VideoProcessorResult {
  const internalVideoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const onFrameRef = useRef(onFrame)
  const [isPlaying, setIsPlaying] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [duration, setDuration] = useState(0)
  const [currentTime, setCurrentTime] = useState(0)

  // Keep onFrame ref updated
  useEffect(() => {
    onFrameRef.current = onFrame
  }, [onFrame])

  const captureFrame = useCallback(() => {
    if (!internalVideoRef.current || !canvasRef.current || !onFrameRef.current) return
    if (internalVideoRef.current.readyState < 2) return

    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) return

    ctx.drawImage(internalVideoRef.current, 0, 0)
    const dataUrl = canvasRef.current.toDataURL('image/jpeg', 0.75)  // Lower quality for faster transfer
    const base64 = dataUrl.split(',')[1]
    onFrameRef.current(base64)
  }, [])

  const startPlayback = useCallback(async () => {
    if (!videoUrl || !internalVideoRef.current) return

    try {
      setError(null)
      internalVideoRef.current.src = videoUrl
      await internalVideoRef.current.play()
      setIsPlaying(true)

      const intervalMs = 1000 / fps
      intervalRef.current = setInterval(() => {
        captureFrame()
        setCurrentTime(internalVideoRef.current?.currentTime || 0)
      }, intervalMs)
    } catch (err: unknown) {
      const errMsg = err instanceof Error ? err.message : 'Video playback failed'
      setError(errMsg)
    }
  }, [videoUrl, fps, captureFrame])

  const stopPlayback = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    if (internalVideoRef.current) {
      internalVideoRef.current.pause()
      internalVideoRef.current.currentTime = 0
    }
    setIsPlaying(false)
  }, [])

  useEffect(() => {
    if (!canvasRef.current) {
      const canvas = document.createElement('canvas')
      canvas.width = 640
      canvas.height = 480
      canvasRef.current = canvas
    }
  }, [])

  useEffect(() => {
    if (enabled && videoUrl) {
      startPlayback()
    } else {
      stopPlayback()
    }
    return () => stopPlayback()
  }, [enabled, videoUrl, startPlayback, stopPlayback])

  useEffect(() => {
    const video = internalVideoRef.current
    if (!video) return

    const handleLoadedMetadata = () => {
      setDuration(video.duration)
    }
    const handleTimeUpdate = () => {
      setCurrentTime(video.currentTime)
    }
    const handleEnded = () => {
      stopPlayback()
    }

    video.addEventListener('loadedmetadata', handleLoadedMetadata)
    video.addEventListener('timeupdate', handleTimeUpdate)
    video.addEventListener('ended', handleEnded)

    return () => {
      video.removeEventListener('loadedmetadata', handleLoadedMetadata)
      video.removeEventListener('timeupdate', handleTimeUpdate)
      video.removeEventListener('ended', handleEnded)
    }
  }, [stopPlayback])

  return {
    videoRef: internalVideoRef,
    isPlaying,
    error,
    duration,
    currentTime,
    startPlayback,
    stopPlayback,
    captureFrame,
  }
}