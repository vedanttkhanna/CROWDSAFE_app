import React from 'react'
import { LineChart, Line, ResponsiveContainer } from 'recharts'

export default function RiskGauge({ risk, history }) {
  const score = risk?.overall ?? 0
  const label = risk?.label  ?? 'Normal'

  const color = score < 30 ? '#22c55e'
              : score < 60 ? '#eab308'
              : score < 80 ? '#f97316' : '#ef4444'

  // SVG arc math
  const r = 70, cx = 100, cy = 95
  const startAngle = -210, endAngle = 30
  const totalAngle = endAngle - startAngle
  const filled     = startAngle + (score / 100) * totalAngle

  function polarToXY(angle, radius) {
    const rad = (angle * Math.PI) / 180
    return { x: cx + radius * Math.cos(rad), y: cy + radius * Math.sin(rad) }
  }

  function arcPath(start, end, r) {
    const s   = polarToXY(start, r)
    const e   = polarToXY(end,   r)
    const large = end - start > 180 ? 1 : 0
    return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`
  }

  const sparkData = history.map((v, i) => ({ i, v }))

  return (
    <div style={styles.wrap}>
      <div style={styles.label}>CROWD RISK</div>
      <div style={styles.gaugeWrap}>
        <svg width="200" height="120" style={{ overflow: 'visible' }}>
          {/* Track */}
          <path d={arcPath(-210, 30, r)} fill="none"
                stroke="#27272a" strokeWidth="6" strokeLinecap="round" />
          {/* Fill */}
          <path d={arcPath(-210, filled, r)} fill="none"
                stroke={color} strokeWidth="6" strokeLinecap="round"
                style={{ transition: 'all 0.5s ease' }} />
          {/* Score */}
          <text x={cx} y={cy - 8} textAnchor="middle"
                fill={color} fontSize="32" fontWeight="700"
                fontFamily="'JetBrains Mono', monospace">
            {Math.round(score)}
          </text>
          <text x={cx} y={cy + 12} textAnchor="middle"
                fill="#71717a" fontSize="9" letterSpacing="2"
                fontFamily="'JetBrains Mono', monospace">
            / 100
          </text>
        </svg>

        <div style={{ ...styles.riskLabel, color }}>
          {label.toUpperCase()}
        </div>
      </div>

      {/* Sparkline */}
      {sparkData.length > 1 && (
        <div style={styles.sparkWrap}>
          <span style={styles.sparkLabel}>30s history</span>
          <ResponsiveContainer width="100%" height={28}>
            <LineChart data={sparkData}>
              <Line type="monotone" dataKey="v" stroke={color}
                    strokeWidth={1.5} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  )
}

const styles = {
  wrap: {
    padding: '12px 16px 8px',
    borderBottom: '1px solid #18181b',
  },
  label: {
    fontSize: 9, letterSpacing: '0.12em',
    color: '#52525b', marginBottom: 4,
    fontFamily: "'JetBrains Mono', monospace",
  },
  gaugeWrap: {
    display: 'flex', alignItems: 'center',
    gap: 12,
  },
  riskLabel: {
    fontSize: 13, fontWeight: 600,
    letterSpacing: '0.08em',
    fontFamily: "'JetBrains Mono', monospace",
  },
  sparkWrap: { marginTop: 4 },
  sparkLabel: {
    fontSize: 9, color: '#3f3f46',
    fontFamily: "'JetBrains Mono', monospace",
  },
}