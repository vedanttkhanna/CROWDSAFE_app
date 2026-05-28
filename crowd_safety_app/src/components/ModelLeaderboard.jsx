import React from 'react'

export default function ModelLeaderboard({ models = [] }) {
  return (
    <div style={styles.wrap}>
      <div style={styles.title}>MODEL LEADERBOARD</div>
      <table style={styles.table}>
        <thead>
          <tr>
            {['MODEL', 'TYPE', 'ADE', 'FDE', 'STATUS'].map(h => (
              <th key={h} style={styles.th}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {models.map((m, i) => (
            <tr key={m.id} style={styles.tr}>
              <td style={styles.td}>
                <span style={styles.rank}>{i + 1}.</span>
                <span style={styles.modelName}>{m.name}</span>
              </td>
              <td style={{ ...styles.td, color: '#71717a' }}>
                {m.type}
              </td>
              <td style={styles.tdMono}>
                {m.test_ade != null ? m.test_ade.toFixed(2) : 'N/A'}
              </td>
              <td style={styles.tdMono}>
                {m.test_fde != null ? m.test_fde.toFixed(2) : 'N/A'}
              </td>
              <td style={styles.td}>
                <div style={styles.statusWrap}>
                  <div style={{
                    ...styles.dot,
                    background: m.trained ? '#22c55e' : '#3f3f46',
                  }} />
                  <span style={{
                    ...styles.statusText,
                    color: m.trained ? '#22c55e' : '#3f3f46',
                  }}>
                    {m.trained ? 'trained' : 'documented'}
                  </span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

const styles = {
  wrap:  { padding: '10px 16px', overflow: 'auto', flex: 1 },
  title: {
    fontSize: 9, letterSpacing: '0.12em', color: '#52525b',
    marginBottom: 8,
    fontFamily: "'JetBrains Mono', monospace",
  },
  table: { width: '100%', borderCollapse: 'collapse', fontSize: 11 },
  th: {
    textAlign: 'left', padding: '4px 6px',
    fontSize: 9, color: '#3f3f46', letterSpacing: '0.08em',
    borderBottom: '1px solid #18181b',
    fontFamily: "'JetBrains Mono', monospace",
  },
  tr: { borderBottom: '1px solid #18181b' },
  td: { padding: '5px 6px', color: '#a1a1aa', fontSize: 11 },
  tdMono: {
    padding: '5px 6px', color: '#e4e4e7',
    fontFamily: "'JetBrains Mono', monospace", fontSize: 11,
  },
  rank:      { color: '#3f3f46', marginRight: 4, fontSize: 10 },
  modelName: { color: '#e4e4e7', fontWeight: 500 },
  statusWrap: { display: 'flex', alignItems: 'center', gap: 5 },
  dot:        { width: 5, height: 5, borderRadius: '50%' },
  statusText: { fontSize: 9, fontFamily: "'JetBrains Mono', monospace" },
}