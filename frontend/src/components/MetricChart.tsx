/**
 * OMNIVIS — Metric Chart Component
 * Rolling line charts for real-time performance metrics using Recharts.
 */
import React from 'react'
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid,
} from 'recharts'
import { useMetrics } from '../hooks/useMetrics'

interface MetricChartProps {
  dataKey: string
  label: string
  color: string
  gradientId: string
  unit?: string
}

const MetricChartSingle: React.FC<MetricChartProps> = ({ dataKey, label, color, gradientId, unit = '' }) => {
  const { chartData } = useMetrics()

  return (
    <div className="glass-card p-3">
      <div className="flex items-center justify-between mb-2">
        <span className="text-[11px] font-medium text-white/50 uppercase tracking-wider">{label}</span>
        {chartData.length > 0 && (
          <span className="text-sm font-bold" style={{ color }}>
            {(chartData[chartData.length - 1] as any)?.[dataKey]?.toFixed(1) || '0'}{unit}
          </span>
        )}
      </div>
      <div className="h-[80px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 0, right: 0, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={color} stopOpacity={0.3} />
                <stop offset="100%" stopColor={color} stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
            <XAxis dataKey="time" hide />
            <YAxis hide domain={['auto', 'auto']} />
            <Tooltip
              contentStyle={{
                background: 'rgba(15, 23, 42, 0.95)',
                border: '1px solid rgba(255,255,255,0.1)',
                borderRadius: '8px',
                fontSize: '11px',
                color: '#fff',
              }}
              labelStyle={{ display: 'none' }}
              formatter={(value: number) => [`${value.toFixed(1)}${unit}`, label]}
            />
            <Area
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              strokeWidth={1.5}
              fill={`url(#${gradientId})`}
              dot={false}
              isAnimationActive={false}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  )
}

export const MetricCharts: React.FC = () => {
  return (
    <div className="flex flex-col gap-3">
      <MetricChartSingle
        dataKey="fps"
        label="FPS"
        color="#10b981"
        gradientId="grad-fps"
        unit=""
      />
      <MetricChartSingle
        dataKey="latency"
        label="Latency"
        color="#f59e0b"
        gradientId="grad-latency"
        unit="ms"
      />
      <MetricChartSingle
        dataKey="detections"
        label="Detections"
        color="#6366f1"
        gradientId="grad-det"
      />
      <MetricChartSingle
        dataKey="tracks"
        label="Active Tracks"
        color="#ec4899"
        gradientId="grad-trk"
      />
    </div>
  )
}
