/**
 * OMNIVIS — File Uploader Component
 * Handles photo/video file uploads with drag-and-drop and preview.
 */
import React, { useRef, useState, useCallback } from 'react'
import { useOmnivisStore } from '../store/omnivis.store'

interface FileUploaderProps {
  onFileProcessed?: (frames: string[]) => void
}

export const FileUploader: React.FC<FileUploaderProps> = ({ onFileProcessed }) => {
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [preview, setPreview] = useState<string | null>(null)
  const [fileName, setFileName] = useState<string>('')
  const [fileType, setFileType] = useState<string>('')
  
  const { setUploadedFile, uploadedFile, setInputApproved } = useOmnivisStore()

  const processFile = useCallback(async (file: File) => {
    setIsProcessing(true)
    setFileName(file.name)
    setFileType(file.type)

    if (file.type.startsWith('image/')) {
      const reader = new FileReader()
      reader.onload = (e) => {
        const result = e.target?.result as string
        setPreview(result)
        const base64 = result.split(',')[1]
        setUploadedFile({
          type: 'image',
          name: file.name,
          data: base64,
          contentType: file.type,
        })
        setIsProcessing(false)
        setInputApproved(true)
      }
      reader.readAsDataURL(file)
    } else if (file.type.startsWith('video/')) {
      const objectUrl = URL.createObjectURL(file)
      setPreview(objectUrl)
      setUploadedFile({
        type: 'video',
        name: file.name,
        data: objectUrl,
        contentType: file.type,
      })
      setInputApproved(true)
      setIsProcessing(false)
    }
  }, [setUploadedFile, setInputApproved])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    
    const file = e.dataTransfer.files[0]
    if (file) {
      processFile(file)
    }
  }, [processFile])

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
  }, [])

  const handleFileSelect = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (file) {
      processFile(file)
    }
  }, [processFile])

  const clearFile = useCallback(() => {
    setPreview(null)
    setFileName('')
    setFileType('')
    setUploadedFile(null)
    setInputApproved(false)
  }, [setUploadedFile, setInputApproved])

  return (
    <div className="space-y-2">
      {!uploadedFile ? (
        <div
          className="border border-dashed border-white/20 rounded-lg p-4 text-center cursor-pointer hover:border-omni-500/50 transition-colors"
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onClick={() => fileInputRef.current?.click()}
        >
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,video/*"
            className="hidden"
            onChange={handleFileSelect}
          />
          <div className="text-white/40">
            <div className="text-lg mb-1">📁</div>
            <div className="text-[10px]">Drop file or click to upload</div>
            <div className="text-[9px] text-white/20 mt-1">Photos: JPG, PNG, WebP • Videos: MP4, WebM</div>
          </div>
        </div>
      ) : (
        <div className="space-y-2">
          {preview && (
            <div className="relative rounded-lg overflow-hidden bg-black/40">
              {fileType.startsWith('image/') ? (
                <img src={preview} alt="Preview" className="w-full h-24 object-contain" />
              ) : (
                <video src={preview} className="w-full h-24 object-contain" muted />
              )}
              {isProcessing && (
                <div className="absolute inset-0 bg-black/60 flex items-center justify-center">
                  <div className="text-omni-300 text-xs animate-pulse">Processing...</div>
                </div>
              )}
            </div>
          )}
          
          <div className="flex items-center justify-between text-[10px] text-white/50">
            <div className="truncate flex-1">
              {fileType.startsWith('image/') ? '🖼️' : '🎬'} {fileName}
            </div>
            <button
              onClick={clearFile}
              className="text-red-400 hover:text-red-300 ml-2"
            >
              ✕
            </button>
          </div>
        </div>
      )}
    </div>
  )
}