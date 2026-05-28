import React from 'react'

export default function FlaggedList({ items = [] }) {
  return (
    <div style={styles.wrap}>
      <div style={styles.header}>
        <span style={styles.title}>FLAGGED INDIVIDUALS</span>
        <span style={styles.count}>{items.length} active</span>
      </div>

      <div style={styles.list}>
        {items.length === 0 ? (
          <div style={styles.empty}>No anomalies detected</div>
        ) : (
          items.slice(0, 4).map((item) => (
            <div key={item.ped_id} style={styles.item}>
              <div style={styles.redBar} />
              <div style={styles.blink} />
              <div style={styles.info}>
                <span style={styles.id}>#{item.ped_id}</span>
                <span style={styles.reason}>
                  {item.anomaly_score > 0.5
                    ? 'Erratic movement'
                    : item.anomaly_score > 0.35
                    ? 'Counter-flow motion'
                    : 'Unusual trajectory'}
                </span>
              </div>
              <span style={styles.score}>
                {item.anomaly_score.toFixed(2)}
              </span>
            </div>
          ))
        )}
      </div>
    </div>
  )
}

const styles = {
  wrap: {
    padding: '10px 16px',
    borderBottom: '1px solid #18181b',
    flex: 1, overflow: 'hidden',
  },
  header: {
    display: 'flex', justifyContent: 'space-between',
    alignItems: 'center', marginBottom: 8,
  },
  title: {
    fontSize: 9, letterSpacing: '0.12em', color: '#52525b',
    fontFamily: "'JetBrains Mono', monospace",
  },
  count: {
    fontSize: 9, color: '#ef4444',
    fontFamily: "'JetBrains Mono', monospace",
  },
  list:  { display: 'flex', flexDirection: 'column', gap: 4 },
  empty: { fontSize: 11, color: '#3f3f46', padding: '4px 0' },
  item:  {
    display: 'flex', alignItems: 'center', gap: 8,
    padding: '6px 8px', background: '#0d0d12',
    border: '1px solid #1c1c1f', position: 'relative',
    overflow: 'hidden',
  },
  redBar: {
    position: 'absolute', left: 0, top: 0, bottom: 0,
    width: 2, background: '#ef4444',
  },
  blink: {
    width: 5, height: 5, borderRadius: '50%',
    background: '#ef4444', flexShrink: 0,
    animation: 'pulse 1.5s infinite',
  },
  info:   { flex: 1, display: 'flex', gap: 8, alignItems: 'center' },
  id:     {
    fontSize: 11, fontWeight: 600, color: '#e4e4e7',
    fontFamily: "'JetBrains Mono', monospace", minWidth: 32,
  },
  reason: { fontSize: 10, color: '#71717a' },
  score:  {
    fontSize: 11, color: '#ef4444', fontWeight: 600,
    fontFamily: "'JetBrains Mono', monospace",
  },
}