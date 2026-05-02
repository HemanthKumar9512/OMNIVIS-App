/**
 * OMNIVIS — Medical Scan Analysis Panel
 * Upload and analyze MRI, CT, X-ray, and Ultrasound scans for risk detection.
 */
import React, { useRef, useState, useCallback } from 'react'
import { useOmnivisStore } from '../store/omnivis.store'

export const MedicalScanPanel: React.FC = () => {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [dragOver, setDragOver] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const [scanType, setScanType] = useState('auto')
  const { medicalResult, isAnalyzingMedical, setMedicalResult, setIsAnalyzingMedical } = useOmnivisStore()

  const analyzeFile = useCallback(async (file: File) => {
    if (!file.type.startsWith('image/')) return

    const reader = new FileReader()
    reader.onload = async (e) => {
      const result = e.target?.result as string
      setPreview(result)
      setIsAnalyzingMedical(true)
      setMedicalResult(null)

      try {
        const formData = new FormData()
        formData.append('file', file)
        formData.append('scan_type', scanType)

        const port = window.location.port === '3000' ? '8000' : window.location.port
        const protocol = window.location.protocol === 'https:' ? 'https' : 'http'
        const response = await fetch(`${protocol}://${window.location.hostname}:${port}/api/medical/analyze`, {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.detail || 'Analysis failed')
        }

        const data = await response.json()
        setMedicalResult({
          scanType: data.scan_type,
          overallRisk: data.overall_risk,
          riskScore: data.risk_score,
          findings: data.findings,
          summary: data.summary,
          annotatedImage: data.annotated_image,
          inferenceMs: data.inference_ms,
        })
      } catch (err: any) {
        setMedicalResult({
          scanType: 'error',
          overallRisk: 'error',
          riskScore: 0,
          findings: [{
            type: 'analysis_error',
            severity: 'high',
            riskLevel: 'high',
            score: 0,
            description: err.message || 'Failed to analyze scan',
            recommendation: 'Check backend server and try again',
            confidence: 0,
          }],
          summary: 'Analysis failed. Please ensure the backend server is running.',
          annotatedImage: null,
          inferenceMs: 0,
        })
      } finally {
        setIsAnalyzingMedical(false)
      }
    }
    reader.readAsDataURL(file)
  }, [scanType, setMedicalResult, setIsAnalyzingMedical])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file) analyzeFile(file)
  }, [analyzeFile])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) analyzeFile(file)
  }, [analyzeFile])

  const riskColors: Record<string, string> = {
    low: 'text-emerald-400',
    medium: 'text-amber-400',
    high: 'text-orange-400',
    critical: 'text-red-400',
  }

  const riskBgColors: Record<string, string> = {
    low: 'bg-emerald-500/10 border-emerald-500/30',
    medium: 'bg-amber-500/10 border-amber-500/30',
    high: 'bg-orange-500/10 border-orange-500/30',
    critical: 'bg-red-500/10 border-red-500/30 animate-pulse',
  }

  const severityIcons: Record<string, string> = {
    low: '\u2705',
    medium: '\u26A0\uFE0F',
    high: '\uD83D\uDD36',
    critical: '\uD83D\uDEA8',
  }

  return (
    <div className="space-y-3">
      <div className="glass-card p-3">
        <div className="text-[11px] font-medium text-white/50 uppercase tracking-wider mb-3">
          Medical Scan Analysis
        </div>

        {/* Scan type selector */}
        <div className="mb-3">
          <label className="text-[10px] text-white/40 mb-1 block">Scan Type</label>
          <select
            value={scanType}
            onChange={e => setScanType(e.target.value)}
            className="w-full bg-white/5 border border-white/10 rounded-lg px-2 py-1.5 text-xs text-white/80 focus:outline-none focus:border-omni-500/50"
          >
            <option value="auto">Auto-detect</option>
            <option value="x-ray">X-Ray</option>
            <option value="mri">MRI</option>
            <option value="ct_scan">CT Scan</option>
            <option value="ultrasound">Ultrasound</option>
          </select>
        </div>

        {/* Upload area */}
        <div
          className={`border-2 border-dashed rounded-lg p-4 text-center cursor-pointer transition-all ${
            dragOver ? 'border-omni-500 bg-omni-500/10' : 'border-white/20 hover:border-omni-500/50'
          }`}
          onDrop={handleDrop}
          onDragOver={e => { e.preventDefault(); setDragOver(true) }}
          onDragLeave={() => setDragOver(false)}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            className="hidden"
            onChange={handleFileSelect}
          />
          {isAnalyzingMedical ? (
            <div className="text-omni-300">
              <div className="text-2xl mb-2 animate-pulse">Analyzing...</div>
              <div className="text-[10px]">Processing scan with AI analysis</div>
            </div>
          ) : preview ? (
            <div>
              <img src={preview} alt="Scan preview" className="max-h-32 mx-auto rounded-lg object-contain mb-2" />
              <div className="text-[10px] text-white/50">Click to upload new scan</div>
            </div>
          ) : (
            <div className="text-white/40">
              <div className="text-2xl mb-1">Scan</div>
              <div className="text-[10px]">Drop MRI/CT/X-ray or click to upload</div>
              <div className="text-[9px] text-white/20 mt-1">JPEG, PNG, WebP, BMP, TIFF</div>
            </div>
          )}
        </div>
      </div>

      {/* Results */}
      {medicalResult && (
        <>
          {/* Risk Banner */}
          <div className={`glass-card p-3 border ${riskBgColors[medicalResult.overallRisk] || 'bg-white/5'}`}>
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-bold uppercase tracking-wider">
                Risk Level
              </span>
              <span className={`text-sm font-bold ${riskColors[medicalResult.overallRisk] || 'text-white'}`}>
                {medicalResult.overallRisk.toUpperCase()}
              </span>
            </div>
            <div className="flex items-center gap-4 text-[10px] text-white/60">
              <span>Score: {(medicalResult.riskScore * 100).toFixed(0)}%</span>
              <span>Findings: {medicalResult.findings.length}</span>
              <span>Scan: {medicalResult.scanType}</span>
              <span>Time: {medicalResult.inferenceMs.toFixed(0)}ms</span>
            </div>
          </div>

          {/* Findings List */}
          <div className="glass-card p-3">
            <div className="text-[11px] font-medium text-white/50 uppercase tracking-wider mb-2">
              Findings ({medicalResult.findings.length})
            </div>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {medicalResult.findings.map((f, i) => (
                <div key={i} className="bg-white/5 rounded-lg p-2 border border-white/10">
                  <div className="flex items-center gap-2 mb-1">
                    <span className="text-sm">{severityIcons[f.riskLevel] || '\u2022'}</span>
                    <span className={`text-[10px] font-bold uppercase ${riskColors[f.riskLevel] || 'text-white'}`}>
                      {f.riskLevel}
                    </span>
                    <span className="text-[10px] text-white/30">({(f.confidence * 100).toFixed(0)}% conf)</span>
                  </div>
                  <div className="text-[10px] text-white/70 mb-1">{f.description}</div>
                  <div className="text-[10px] text-omni-300/70">
                    {f.recommendation}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Summary */}
          <div className="glass-card p-3">
            <div className="text-[11px] font-medium text-white/50 uppercase tracking-wider mb-2">
              Analysis Summary
            </div>
            <pre className="text-[10px] text-white/70 whitespace-pre-wrap font-sans leading-relaxed">
              {medicalResult.summary}
            </pre>
          </div>

          {/* Annotated Image */}
          {medicalResult.annotatedImage && (
            <div className="glass-card p-3">
              <div className="text-[11px] font-medium text-white/50 uppercase tracking-wider mb-2">
                Annotated Scan
              </div>
              <img
                src={`data:image/jpeg;base64,${medicalResult.annotatedImage}`}
                alt="Annotated scan"
                className="w-full rounded-lg"
              />
            </div>
          )}

          {/* Disclaimer */}
          <div className="bg-amber-500/10 border border-amber-500/20 rounded-lg p-2">
            <div className="text-[9px] text-amber-300/80">
              DISCLAIMER: This is an AI-assisted screening analysis only. All findings require clinical correlation by a qualified medical professional and should NOT replace professional medical diagnosis.
            </div>
          </div>
        </>
      )}
    </div>
  )
}
