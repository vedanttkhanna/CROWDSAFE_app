import React, { useState, useEffect } from 'react'
import RiskGauge from './components/RiskGauge'
import RiskBars from './components/RiskBars'
import FlaggedList from './components/FlaggedList'
import ModelLeaderboard from './components/ModelLeaderboard'
import LiveFeed from './components/LiveFeed'

const API = 'http://localhost:8000'

export default function App() {
  const [riskData, setRiskData]       = useState(null)
  const [flagged, setFlagged]         = useState([])
  const [models, setModels]           = useState([])
  const [apiStatus, setApiStatus]     = useState('connecting')
  const [riskHistory, setRiskHistory] = useState([])
  const [isLive, setIsLive]           = useState(false)

  // Fetch models once on load
  useEffect(() => {
    async function fetchModels() {
      try {
        const res  = await fetch(`${API}/models`)
        const data = await res.json()
        setModels(data.models)
        setApiStatus('connected')
      } catch {
        setApiStatus('disconnected')
      }
    }
    fetchModels()
  }, [])

  // Just keep connection alive — no demo data
  useEffect(() => {
    if (isLive) return
    async function checkConnection() {
      try {
        await fetch(`${API}/`)
        setApiStatus('connected')
      } catch {
        setApiStatus('disconnected')
      }
    }
    checkConnection()
    const id = setInterval(checkConnection, 5000)
    return () => clearInterval(id)
  }, [isLive])

  function handleRiskUpdate(risk) {
    setIsLive(true)
    setRiskData(risk)
    setRiskHistory(prev => [...prev, risk.overall].slice(-30))
  }

  function handleFlaggedUpdate(tracks) {
    setFlagged(tracks.map(t => ({
      ped_id:        t.id,
      anomaly_score: t.score,
      flagged:       true,
    })))
  }

  return (
    <div style={styles.root}>
      <div style={styles.titlebar}>
        <div style={styles.titleLeft}>
          <div style={styles.dot} />
          <span style={styles.title}>CROWD SAFETY MONITOR</span>
          <span style={styles.version}>v1.0</span>
          {isLive && <span style={styles.liveBadge}>● LIVE ANALYSIS</span>}
        </div>
        <div style={styles.titleRight}>
          <div style={{
            ...styles.statusDot,
            background: apiStatus === 'connected'   ? '#22c55e'
                       : apiStatus === 'connecting' ? '#eab308' : '#ef4444'
          }} />
          <span style={styles.statusText}>
            {apiStatus === 'connected'   ? 'API CONNECTED'
           : apiStatus === 'connecting' ? 'CONNECTING...' : 'API OFFLINE'}
          </span>
          <span style={styles.apiUrl}>localhost:8000</span>
        </div>
      </div>

      <div style={styles.body}>
        <div style={styles.left}>
          <LiveFeed
            api={API}
            onRiskUpdate={handleRiskUpdate}
            onFlaggedUpdate={handleFlaggedUpdate}
          />
        </div>
        <div style={styles.right}>
          <div style={styles.topPanel}>
            {riskData ? (
              <>
                <RiskGauge risk={riskData} history={riskHistory} />
                <RiskBars  risk={riskData} />
                <FlaggedList items={flagged} />
              </>
            ) : (
              <div style={styles.idle}>
                <div style={styles.idleIcon}>⬡</div>
                <p style={styles.idleText}>AWAITING VIDEO INPUT</p>
                <p style={styles.idleSub}>Upload a crowd video to begin live analysis</p>
              </div>
            )}
          </div>
          <div style={styles.leaderboardPanel}>
            <ModelLeaderboard models={models} />
          </div>
        </div>
      </div>
    </div>
  )
}

const styles = {
  root: { display: 'flex', flexDirection: 'column', height: '100vh', background: '#0a0a0f' },
  titlebar: {
    display: 'flex', alignItems: 'center', justifyContent: 'space-between',
    padding: '0 16px', height: 40, background: '#0d0d12',
    borderBottom: '1px solid #18181b', flexShrink: 0,
  },
  titleLeft:  { display: 'flex', alignItems: 'center', gap: 10 },
  titleRight: { display: 'flex', alignItems: 'center', gap: 8 },
  dot:        { width: 8, height: 8, borderRadius: '50%', background: '#ef4444' },
  title:      { fontSize: 11, fontWeight: 600, letterSpacing: '0.12em', color: '#e4e4e7', fontFamily: 'monospace' },
  version:    { fontSize: 10, color: '#52525b' },
  liveBadge:  { fontSize: 9, color: '#ef4444', fontFamily: 'monospace', letterSpacing: '0.08em' },
  statusDot:  { width: 6, height: 6, borderRadius: '50%' },
  statusText: { fontSize: 10, color: '#71717a', fontFamily: 'monospace', letterSpacing: '0.08em' },
  apiUrl:     { fontSize: 10, color: '#3f3f46', fontFamily: 'monospace' },
  body:       { display: 'flex', flex: 1, overflow: 'hidden', gap: 1 },
  left:       {flex: '0 0 60%',borderRight: '1px solid #18181b',overflow: 'auto'},
  right:      { flex: 1, display: 'flex', flexDirection: 'column', gap: 1, overflow: 'hidden' },
  topPanel: {
    flex: '0 0 auto', display: 'flex',
    flexDirection: 'column', gap: 1,
    borderBottom: '1px solid #18181b',
  },
  leaderboardPanel: { flex: 1, overflow: 'auto', minHeight: 200 },
  idle: {
    display: 'flex', flexDirection: 'column',
    alignItems: 'center', justifyContent: 'center',
    gap: 12, padding: 40, minHeight: 200,
  },
  idleIcon: { fontSize: 48, color: '#27272a' },
  idleText: { fontSize: 11, letterSpacing: '0.12em', color: '#3f3f46', fontFamily: 'monospace' },
  idleSub:  { fontSize: 11, color: '#27272a', textAlign: 'center', lineHeight: 1.6 },
}