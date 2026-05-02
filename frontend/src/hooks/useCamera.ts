/**
 * OMNIVIS — Camera Stream Hook
 * Manages webcam access, frame capture, and sending to WebSocket.
 */
import { useEffect, useRef, useCallback, useState } from 'react'

interface UseCameraOptions {
  enabled: boolean
  width?: number
  height?: number
  fps?: number
  onFrame?: (frameBase64: string) => void
}

export function useCamera({ enabled, width = 640, height = 480, fps = 8, onFrame }: UseCameraOptions) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const onFrameRef = useRef(onFrame)  // Store latest callback
  const [isActive, setIsActive] = useState(false)
  const [error, setError] = useState<string | null>(null)

  // Keep onFrame ref updated
  useEffect(() => {
    onFrameRef.current = onFrame
  }, [onFrame])

  const startCamera = useCallback(async () => {
    try {
      setError(null)
      console.log('[Camera] Requesting camera access...')
      
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: width },
          height: { ideal: height },
          facingMode: 'user',
        },
        audio: false,
      })

      console.log('[Camera] Camera access granted')
      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
        console.log('[Camera] Video playing')
      }

      // Create hidden canvas for frame capture
      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas')
      }
      canvasRef.current.width = width
      canvasRef.current.height = height

      // Start frame capture loop with error handling
      const captureInterval = 1000 / fps
      intervalRef.current = setInterval(() => {
        try {
          captureFrame()
        } catch (err) {
          console.error('[Camera] Capture error:', err)
        }
      }, captureInterval)

      setIsActive(true)
      console.log('[Camera] Started capturing at', fps, 'fps')
    } catch (err: any) {
      console.error('[Camera] Error:', err)
      setError(err.message || 'Camera access denied')
      setIsActive(false)
    }
  }, [width, height, fps])

  const stopCamera = useCallback(() => {
    console.log('[Camera] Stopping...')
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
      intervalRef.current = null
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsActive(false)
    console.log('[Camera] Stopped')
  }, [])

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !onFrameRef.current) return
    if (!videoRef.current.videoWidth || !videoRef.current.videoHeight) return

    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) return

    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height)
    const dataUrl = canvasRef.current.toDataURL('image/jpeg', 0.75)  // Lower quality for faster transfer
    const base64 = dataUrl.split(',')[1]
    onFrameRef.current(base64)
  }, [])

  useEffect(() => {
    console.log('[Camera] enabled changed:', enabled)
    if (enabled) {
      startCamera()
    } else {
      stopCamera()
    }
    return () => stopCamera()
  }, [enabled, startCamera, stopCamera])

  return {
    videoRef,
    isActive,
    error,
    startCamera,
    stopCamera,
  }
}
