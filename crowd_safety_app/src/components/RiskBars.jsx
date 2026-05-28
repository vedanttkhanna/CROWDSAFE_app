import React from 'react'

const SEGMENTS = 20

function SegmentBar({ value = 0, color }) {
  const filled = Math.round((value / 100) * SEGMENTS)
  return (
    <div style={styles.barRow}>
      {Array.from({ length: SEGMENTS }).map((_, i) => {
        const active = i < filled
        const pct    = (i / SEGMENTS) * 100
        const segColor = pct < 40 ? '#22c55e'
                       : pct < 70 ? '#eab308' : '#ef4444'
        return (
          <div key={i} style={{
            ...styles.seg,
            background: active ? segColor : '#1c1c1f',
            opacity: active ? 1 : 0.3,
          }} />
        )
      })}
    </div>
  )
}

export default function RiskBars({ risk }) {
  const bars = [
    { label: 'DENSITY RISK',  value: risk?.density  ?? 0 },
    { label: 'VELOCITY RISK', value: risk?.velocity ?? 0 },
    { label: 'ANOMALY RISK',  value: risk?.anomaly  ?? 0 },
  ]

  return (
    <div style={styles.wrap}>
      {bars.map(({ label, value }) => (
        <div key={label} style={styles.row}>
          <div style={styles.header}>
            <span style={styles.label}>{label}</span>
            <span style={{
              ...styles.value,
              color: value > 70 ? '#ef4444'
                   : value > 40 ? '#eab308' : '#22c55e'
            }}>
              {Math.round(value)}
            </span>
          </div>
          <SegmentBar value={value} />
        </div>
      ))}
    </div>
  )
}

const styles = {
  wrap: {
    padding: '10px 16px',
    borderBottom: '1px solid #18181b',
    display: 'flex', flexDirection: 'column', gap: 10,
  },
  row:    { display: 'flex', flexDirection: 'column', gap: 4 },
  header: { display: 'flex', justifyContent: 'space-between', alignItems: 'center' },
  label:  {
    fontSize: 9, letterSpacing: '0.1em', color: '#71717a',
    fontFamily: "'JetBrains Mono', monospace",
  },
  value:  {
    fontSize: 11, fontWeight: 600,
    fontFamily: "'JetBrains Mono', monospace",
  },
  barRow: { display: 'flex', gap: 2 },
  seg:    {
    flex: 1, height: 6, borderRadius: 1,
    transition: 'background 0.3s ease',
  },
}