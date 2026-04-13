/**
 * OMNIVIS — Global State Store (Zustand)
 * Centralized state management for all modules, settings, and real-time data.
 */
import { create } from 'zustand'

// ── Types ──────────────────────────────────────────────────
export interface BBox {
  x1: number; y1: number; x2: number; y2: number;
}

export interface Detection {
  class_name: string;
  class_id: number;
  confidence: number;
  bbox: BBox;
  track_id?: number;
}

export interface Face {
  bbox: BBox;
  confidence: number;
  landmarks: number[][];
  age?: number;
  gender?: string;
  emotion?: string;
}

export interface AnomalyAlert {
  id: string;
  timestamp: number;
  type: string;
  severity: 'green' | 'yellow' | 'red';
  score: number;
  description: string;
}

export interface ActionPrediction {
  action: string;
  confidence: number;
}

export interface SceneNode {
  id: number;
  label: string;
  confidence: number;
  center: number[];
}

export interface SceneEdge {
  source: number;
  target: number;
  predicate: string;
  confidence: number;
}

export interface Triplet {
  subject: string;
  predicate: string;
  object: string;
  confidence: number;
}

export interface MetricSample {
  timestamp: number;
  fps: number;
  inference_ms: number;
  detection_count: number;
  track_count: number;
  gpu_util: number;
  gpu_memory: number;
  cpu_util: number;
  map50?: number;
  iou?: number;
  epe?: number;
}

export interface ModuleState {
  enabled: boolean;
  loaded: boolean;
}

export type CanvasPanelType = 'detection' | 'segmentation' | 'flow' | 'depth' | 'tracking' | 'face' | 'gait' | '3d' | 'scene_graph';

export interface PanelConfig {
  type: CanvasPanelType;
  showBoxes: boolean;
  showMasks: boolean;
  showFlow: boolean;
  showDepth: boolean;
  showSkeleton: boolean;
  showTrackTrails: boolean;
  showSceneGraph: boolean;
  showLabels: boolean;
  showConfidence: boolean;
  isFullscreen: boolean;
}

export type ThemeMode = 'dark' | 'light';
export type Language = 'en' | 'ta' | 'hi';

// ── Store ──────────────────────────────────────────────────
interface OmnivisStore {
  // Connection
  isConnected: boolean;
  setConnected: (v: boolean) => void;

  // Theme
  theme: ThemeMode;
  toggleTheme: () => void;

  // Language
  language: Language;
  setLanguage: (l: Language) => void;

  // Input source
  inputSource: 'webcam' | 'rtsp' | 'youtube' | 'file';
  inputUrl: string;
  setInputSource: (s: 'webcam' | 'rtsp' | 'youtube' | 'file') => void;
  setInputUrl: (u: string) => void;

  // Modules
  modules: Record<string, ModuleState>;
  toggleModule: (name: string) => void;
  setModuleLoaded: (name: string, loaded: boolean) => void;

  // Detection config
  confidenceThreshold: number;
  nmsThreshold: number;
  setConfidenceThreshold: (v: number) => void;
  setNmsThreshold: (v: number) => void;
  selectedModel: string;
  setSelectedModel: (m: string) => void;

  // Panel configs
  panels: PanelConfig[];
  setPanelType: (idx: number, type: CanvasPanelType) => void;
  togglePanelOverlay: (idx: number, overlay: keyof PanelConfig) => void;
  togglePanelFullscreen: (idx: number) => void;

  // Real-time data
  currentFrame: string | null;
  setCurrentFrame: (f: string | null) => void;

  detections: Detection[];
  setDetections: (d: Detection[]) => void;

  faces: Face[];
  setFaces: (f: Face[]) => void;

  tracks: any[];
  setTracks: (t: any[]) => void;

  trails: Record<string, number[][]>;
  setTrails: (t: Record<string, number[][]>) => void;

  sceneGraph: { nodes: SceneNode[]; edges: SceneEdge[]; triplets: Triplet[] };
  setSceneGraph: (g: { nodes: SceneNode[]; edges: SceneEdge[]; triplets: Triplet[] }) => void;

  anomalies: AnomalyAlert[];
  addAnomaly: (a: AnomalyAlert) => void;
  clearAnomalies: () => void;
  alertLevel: 'green' | 'yellow' | 'red';
  setAlertLevel: (l: 'green' | 'yellow' | 'red') => void;

  actions: ActionPrediction[];
  setActions: (a: ActionPrediction[]) => void;

  // Metrics
  metricsHistory: MetricSample[];
  addMetrics: (m: MetricSample) => void;
  currentFps: number;
  currentInferenceMs: number;
  setCurrentPerf: (fps: number, ms: number) => void;

  // Recording
  isRecording: boolean;
  toggleRecording: () => void;

  // Point cloud data
  pointCloud: { points: number[][]; colors: number[][] } | null;
  setPointCloud: (pc: { points: number[][]; colors: number[][] } | null) => void;

  // Depth data
  depthStats: { min: number; max: number; mean: number } | null;
  setDepthStats: (s: { min: number; max: number; mean: number } | null) => void;

  // Flow data
  flowStats: { meanMag: number; maxMag: number; method: string } | null;
  setFlowStats: (s: { meanMag: number; maxMag: number; method: string } | null) => void;
}

export const useOmnivisStore = create<OmnivisStore>((set, get) => ({
  // Connection
  isConnected: false,
  setConnected: (v) => set({ isConnected: v }),

  // Theme
  theme: 'dark',
  toggleTheme: () => set((s) => ({ theme: s.theme === 'dark' ? 'light' : 'dark' })),

  // Language
  language: 'en',
  setLanguage: (l) => set({ language: l }),

  // Input source
  inputSource: 'webcam',
  inputUrl: '',
  setInputSource: (s) => set({ inputSource: s }),
  setInputUrl: (u) => set({ inputUrl: u }),

  // Modules
  modules: {
    detection: { enabled: true, loaded: false },
    segmentation: { enabled: true, loaded: false },
    face: { enabled: true, loaded: false },
    optical_flow: { enabled: true, loaded: false },
    depth: { enabled: true, loaded: false },
    reconstruction: { enabled: false, loaded: false },
    tracking: { enabled: true, loaded: false },
    scene_graph: { enabled: true, loaded: false },
    trajectory: { enabled: true, loaded: false },
    anomaly: { enabled: true, loaded: false },
    gait: { enabled: false, loaded: false },
    action: { enabled: false, loaded: false },
    gan: { enabled: false, loaded: false },
  },
  toggleModule: (name) => set((s) => ({
    modules: {
      ...s.modules,
      [name]: { ...s.modules[name], enabled: !s.modules[name]?.enabled },
    },
  })),
  setModuleLoaded: (name, loaded) => set((s) => ({
    modules: {
      ...s.modules,
      [name]: { ...s.modules[name], loaded },
    },
  })),

  // Detection config
  confidenceThreshold: 0.5,
  nmsThreshold: 0.45,
  setConfidenceThreshold: (v) => set({ confidenceThreshold: v }),
  setNmsThreshold: (v) => set({ nmsThreshold: v }),
  selectedModel: 'yolov8x',
  setSelectedModel: (m) => set({ selectedModel: m }),

  // Panel configs
  panels: [
    { type: 'detection', showBoxes: true, showMasks: false, showFlow: false, showDepth: false, showSkeleton: false, showTrackTrails: true, showSceneGraph: false, showLabels: true, showConfidence: true, isFullscreen: false },
    { type: 'flow', showBoxes: false, showMasks: false, showFlow: true, showDepth: false, showSkeleton: false, showTrackTrails: false, showSceneGraph: false, showLabels: false, showConfidence: false, isFullscreen: false },
    { type: 'depth', showBoxes: false, showMasks: false, showFlow: false, showDepth: true, showSkeleton: false, showTrackTrails: false, showSceneGraph: false, showLabels: false, showConfidence: false, isFullscreen: false },
    { type: 'tracking', showBoxes: true, showMasks: false, showFlow: false, showDepth: false, showSkeleton: false, showTrackTrails: true, showSceneGraph: false, showLabels: true, showConfidence: false, isFullscreen: false },
  ],
  setPanelType: (idx, type) => set((s) => {
    const panels = [...s.panels];
    panels[idx] = { ...panels[idx], type };
    return { panels };
  }),
  togglePanelOverlay: (idx, overlay) => set((s) => {
    const panels = [...s.panels];
    panels[idx] = { ...panels[idx], [overlay]: !panels[idx][overlay] };
    return { panels };
  }),
  togglePanelFullscreen: (idx) => set((s) => {
    const panels = [...s.panels];
    panels[idx] = { ...panels[idx], isFullscreen: !panels[idx].isFullscreen };
    return { panels };
  }),

  // Real-time data
  currentFrame: null,
  setCurrentFrame: (f) => set({ currentFrame: f }),

  detections: [],
  setDetections: (d) => set({ detections: d }),

  faces: [],
  setFaces: (f) => set({ faces: f }),

  tracks: [],
  setTracks: (t) => set({ tracks: t }),

  trails: {},
  setTrails: (t) => set({ trails: t }),

  sceneGraph: { nodes: [], edges: [], triplets: [] },
  setSceneGraph: (g) => set({ sceneGraph: g }),

  anomalies: [],
  addAnomaly: (a) => set((s) => ({
    anomalies: [a, ...s.anomalies].slice(0, 100),
  })),
  clearAnomalies: () => set({ anomalies: [] }),
  alertLevel: 'green',
  setAlertLevel: (l) => set({ alertLevel: l }),

  actions: [],
  setActions: (a) => set({ actions: a }),

  // Metrics
  metricsHistory: [],
  addMetrics: (m) => set((s) => ({
    metricsHistory: [...s.metricsHistory, m].slice(-120), // 2 minutes of data
  })),
  currentFps: 0,
  currentInferenceMs: 0,
  setCurrentPerf: (fps, ms) => set({ currentFps: fps, currentInferenceMs: ms }),

  // Recording
  isRecording: false,
  toggleRecording: () => set((s) => ({ isRecording: !s.isRecording })),

  // Point cloud
  pointCloud: null,
  setPointCloud: (pc) => set({ pointCloud: pc }),

  // Depth
  depthStats: null,
  setDepthStats: (s) => set({ depthStats: s }),

  // Flow
  flowStats: null,
  setFlowStats: (s) => set({ flowStats: s }),
}))
