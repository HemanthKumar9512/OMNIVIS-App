/**
 * OMNIVIS — Main Application Component
 * Full dashboard layout: top bar, sidebars, 2x2 canvas grid, bottom detection strip.
 */
import React, { useEffect, useState } from 'react'
import { useOmnivisStore } from './store/omnivis.store'
import { useWebSocket } from './hooks/useWebSocket'
import { useCamera } from './hooks/useCamera'
import { CanvasPanel } from './components/CanvasPanel'
import { LeftSidebar, RightSidebar } from './components/Sidebar'
import { DetectionStrip } from './components/DetectionStrip'

const App: React.FC = () => {
  const {
    isConnected, theme, toggleTheme, currentFps, currentInferenceMs,
    isRecording, toggleRecording, detections, alertLevel,
    language, setLanguage, inputSource,
  } = useOmnivisStore()

  const { sendFrame, sendConfig } = useWebSocket()
  const [showSettings, setShowSettings] = useState(false)

  // Camera hook — sends frames to WebSocket
  const { videoRef, isActive: cameraActive, error: cameraError } = useCamera({
    enabled: inputSource === 'webcam' && isConnected,
    width: 640,
    height: 480,
    fps: 12,
    onFrame: sendFrame,
  })

  // Apply theme
  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark')
    document.documentElement.classList.toggle('light', theme === 'light')
  }, [theme])

  return (
    <div className="h-screen w-screen flex flex-col bg-surface-950 bg-grid overflow-hidden">
      {/* ═══ Top Bar ══════════════════════════════════════════════════ */}
      <header className="h-12 flex items-center justify-between px-4 border-b border-white/5 bg-surface-900/60 backdrop-blur-xl z-30 flex-shrink-0">
        {/* Logo + Status */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-omni-500 to-omni-700 flex items-center justify-center shadow-lg shadow-omni-500/30">
              <svg className="w-5 h-5 text-white" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
                <circle cx="12" cy="12" r="3" />
                <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
              </svg>
            </div>
            <div>
              <h1 className="text-sm font-black tracking-tight gradient-text">OMNIVIS</h1>
              <span className="text-[9px] text-white/30 font-medium tracking-widest uppercase">Vision Intelligence</span>
            </div>
          </div>

          {/* Connection Status */}
          <div className="flex items-center gap-2 px-3 py-1 rounded-full bg-white/5 border border-white/5">
            <div className={`status-dot ${isConnected ? 'status-green' : 'status-red'}`} />
            <span className="text-[10px] font-medium text-white/50">
              {isConnected ? 'Connected' : 'Disconnected'}
            </span>
          </div>

          {/* Alert Level */}
          <div className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full border text-[10px] font-bold uppercase tracking-wider
            ${alertLevel === 'green' ? 'bg-emerald-500/10 border-emerald-500/20 text-emerald-400' :
              alertLevel === 'yellow' ? 'bg-amber-500/10 border-amber-500/20 text-amber-400' :
              'bg-red-500/10 border-red-500/20 text-red-400 animate-pulse'}`}>
            <span>{alertLevel === 'green' ? '✓' : alertLevel === 'yellow' ? '⚠' : '🚨'}</span>
            {alertLevel.toUpperCase()}
          </div>
        </div>

        {/* Center — Performance */}
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-4 px-4 py-1 rounded-full bg-white/[0.03] border border-white/5">
            <div className="text-center">
              <div className="text-sm font-bold font-mono text-emerald-400">{currentFps.toFixed(1)}</div>
              <div className="text-[8px] text-white/30 uppercase">FPS</div>
            </div>
            <div className="w-px h-6 bg-white/10" />
            <div className="text-center">
              <div className="text-sm font-bold font-mono text-amber-400">{currentInferenceMs.toFixed(0)}</div>
              <div className="text-[8px] text-white/30 uppercase">MS</div>
            </div>
            <div className="w-px h-6 bg-white/10" />
            <div className="text-center">
              <div className="text-sm font-bold font-mono text-omni-400">{detections.length}</div>
              <div className="text-[8px] text-white/30 uppercase">DETS</div>
            </div>
          </div>
        </div>

        {/* Right — Controls */}
        <div className="flex items-center gap-2">
          {/* Camera indicator */}
          {cameraActive && (
            <div className="flex items-center gap-1.5 px-2 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20">
              <div className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
              <span className="text-[10px] text-emerald-400 font-medium">LIVE</span>
            </div>
          )}

          {/* Record */}
          <button
            onClick={toggleRecording}
            className={`p-2 rounded-lg transition-all ${isRecording ? 'bg-red-500/20 text-red-400' : 'hover:bg-white/5 text-white/40'}`}
            title={isRecording ? 'Stop Recording' : 'Start Recording'}
          >
            <div className={`w-3 h-3 rounded-full ${isRecording ? 'bg-red-500 animate-pulse' : 'border-2 border-current'}`} />
          </button>

          {/* Language */}
          <select
            value={language}
            onChange={e => setLanguage(e.target.value as any)}
            className="bg-white/5 border border-white/10 rounded-lg px-2 py-1 text-[10px] text-white/60 focus:outline-none"
          >
            <option value="en">EN</option>
            <option value="ta">தமிழ்</option>
            <option value="hi">हिंदी</option>
          </select>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-lg hover:bg-white/5 transition-colors text-white/40 hover:text-white/80"
            title="Toggle Theme"
          >
            {theme === 'dark' ? (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
              </svg>
            ) : (
              <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
              </svg>
            )}
          </button>

          {/* Settings */}
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="p-2 rounded-lg hover:bg-white/5 transition-colors text-white/40 hover:text-white/80"
            title="Settings"
          >
            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.066 2.573c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.573 1.066c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.066-2.573c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </button>
        </div>
      </header>

      {/* ═══ Main Content ════════════════════════════════════════════ */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Sidebar */}
        <LeftSidebar />

        {/* Center — 2×2 Canvas Grid */}
        <main className="flex-1 p-3 flex flex-col gap-3 min-w-0">
          <div className="flex-1 grid grid-cols-2 grid-rows-2 gap-3 min-h-0">
            <CanvasPanel panelIndex={0} />
            <CanvasPanel panelIndex={1} />
            <CanvasPanel panelIndex={2} />
            <CanvasPanel panelIndex={3} />
          </div>
        </main>

        {/* Right Sidebar */}
        <RightSidebar />
      </div>

      {/* ═══ Bottom Bar — Detection Strip ════════════════════════════ */}
      <DetectionStrip />

      {/* Hidden video element for camera */}
      <video ref={videoRef} className="hidden" muted playsInline />

      {/* Camera error toast */}
      {cameraError && (
        <div className="fixed bottom-16 left-1/2 -translate-x-1/2 z-50 animate-slide-in">
          <div className="alert-banner alert-yellow px-4 py-2 rounded-xl shadow-2xl">
            <span className="text-sm">📷 {cameraError}</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
