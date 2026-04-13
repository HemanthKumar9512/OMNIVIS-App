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

export function useCamera({ enabled, width = 640, height = 480, fps = 15, onFrame }: UseCameraOptions) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const intervalRef = useRef<ReturnType<typeof setInterval>>()
  const [isActive, setIsActive] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const startCamera = useCallback(async () => {
    try {
      setError(null)
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: width },
          height: { ideal: height },
          facingMode: 'user',
        },
        audio: false,
      })

      streamRef.current = stream

      if (videoRef.current) {
        videoRef.current.srcObject = stream
        await videoRef.current.play()
      }

      // Create hidden canvas for frame capture
      if (!canvasRef.current) {
        canvasRef.current = document.createElement('canvas')
      }
      canvasRef.current.width = width
      canvasRef.current.height = height

      // Start frame capture loop
      const captureInterval = 1000 / fps
      intervalRef.current = setInterval(() => {
        captureFrame()
      }, captureInterval)

      setIsActive(true)
    } catch (err: any) {
      console.error('[Camera] Error:', err)
      setError(err.message || 'Camera access denied')
      setIsActive(false)
    }
  }, [width, height, fps])

  const stopCamera = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach(track => track.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setIsActive(false)
  }, [])

  const captureFrame = useCallback(() => {
    if (!videoRef.current || !canvasRef.current || !onFrame) return
    if (videoRef.current.readyState < 2) return

    const ctx = canvasRef.current.getContext('2d')
    if (!ctx) return

    ctx.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height)
    const dataUrl = canvasRef.current.toDataURL('image/jpeg', 0.8)
    const base64 = dataUrl.split(',')[1]
    onFrame(base64)
  }, [onFrame])

  useEffect(() => {
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
