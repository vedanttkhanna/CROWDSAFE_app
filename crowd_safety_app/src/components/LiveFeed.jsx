import React, { useRef, useEffect, useState, useCallback } from 'react'

const ipcRenderer = window.require ? window.require('electron').ipcRenderer : null

export default function LiveFeed({ api, onRiskUpdate, onFlaggedUpdate }) {
  const canvasRef        = useRef(null)
  const videoRef         = useRef(null)
  const animRef          = useRef(null)
  const lastAnnotatedRef = useRef(null)
  const processingRef    = useRef(false)
  const [status, setStatus]         = useState('SIMULATED')
  const [frameCount, setFrameCount] = useState(0)
  const [videoSrc, setVideoSrc]     = useState(null)
  const [isLive, setIsLive]         = useState(false)
  const [fps, setFps]               = useState(0)
  const fpsRef = useRef({ count: 0, last: Date.now() })

  // ── Send frame to API ─────────────────────────────────────────────────────
  const processFrame = useCallback(async (W, H) => {
    if (processingRef.current) return
    processingRef.current = true

    try {
      const tempCanvas  = document.createElement('canvas')
      tempCanvas.width  = W
      tempCanvas.height = H
      const tempCtx     = tempCanvas.getContext('2d')
      const video       = videoRef.current
      if (video && video.readyState >= 2) {
        tempCtx.drawImage(video, 0, 0, W, H)
      }

      const b64 = tempCanvas.toDataURL('image/jpeg', 0.7).split(',')[1]

      const res = await fetch(`${api}/video/frame`, {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ frame_b64: b64 }),
      })

      if (!res.ok) throw new Error(`API error ${res.status}`)
      const data = await res.json()

      const img  = new Image()
      img.onload = () => { lastAnnotatedRef.current = img }
      img.src    = `data:image/jpeg;base64,${data.frame_b64}`

      if (onRiskUpdate)    onRiskUpdate(data.risk)
      if (onFlaggedUpdate) onFlaggedUpdate(data.tracks.filter(t => t.flagged))

      fpsRef.current.count++
      const now = Date.now()
      if (now - fpsRef.current.last > 1000) {
        setFps(fpsRef.current.count)
        fpsRef.current = { count: 0, last: now }
      }
    } catch (e) {
      console.error('Frame processing error:', e)
    } finally {
      processingRef.current = false
    }
  }, [api, onRiskUpdate, onFlaggedUpdate])

  // ── Canvas resize — respects video aspect ratio ───────────────────────────
  const resizeCanvas = useCallback(() => {
    const canvas = canvasRef.current
    const video  = videoRef.current
    if (!canvas) return

    const rect = canvas.parentElement.getBoundingClientRect()
    const availW = rect.width

    if (video && video.videoWidth > 0) {
      const ratio   = video.videoHeight / video.videoWidth
      canvas.width  = availW
      canvas.height = availW * ratio
    } else {
      canvas.width  = availW
      canvas.height = availW * (9 / 16)
    }
  }, [])

  useEffect(() => {
    resizeCanvas()
    window.addEventListener('resize', resizeCanvas)
    return () => window.removeEventListener('resize', resizeCanvas)
  }, [resizeCanvas])

  // ── Animation loop ────────────────────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')

    let lastApiCall    = 0
    const API_INTERVAL = 150

    function draw() {
      const W     = canvas.width
      const H     = canvas.height
      const video = videoRef.current
      const now   = Date.now()

      if (isLive && video && video.readyState >= 2) {
        ctx.drawImage(video, 0, 0, W, H)

        if (lastAnnotatedRef.current) {
          ctx.globalAlpha = 0.88
          ctx.drawImage(lastAnnotatedRef.current, 0, 0, W, H)
          ctx.globalAlpha = 1.0
        }

        if (now - lastApiCall > API_INTERVAL && !processingRef.current) {
          lastApiCall = now
          processFrame(W, H)
        }
      } else {
        ctx.fillStyle = '#0a0a0f'
        ctx.fillRect(0, 0, W, H)

        ctx.strokeStyle = '#0d1117'
        ctx.lineWidth   = 0.5
        for (let x = 0; x < W; x += 40) {
          ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke()
        }
        for (let y = 0; y < H; y += 40) {
          ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke()
        }

        const scanY = (now / 20) % H
        const grad  = ctx.createLinearGradient(0, scanY - 20, 0, scanY + 4)
        grad.addColorStop(0, 'rgba(34,197,94,0)')
        grad.addColorStop(1, 'rgba(34,197,94,0.04)')
        ctx.fillStyle = grad
        ctx.fillRect(0, scanY - 20, W, 24)

        ctx.fillStyle = '#3f3f46'
        ctx.font      = '13px monospace'
        ctx.textAlign = 'center'
        ctx.fillText('Upload a video to start live analysis', W / 2, H / 2)
        ctx.textAlign = 'left'
      }

      setFrameCount(f => f + 1)
      animRef.current = requestAnimationFrame(draw)
    }

    animRef.current = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(animRef.current)
  }, [isLive, processFrame])

  // ── File upload ───────────────────────────────────────────────────────────
  async function handleUpload() {
    if (!ipcRenderer) return
    const filePath = await ipcRenderer.invoke('open-file-dialog')
    if (!filePath) return

    const fileName = filePath.split('\\').pop()
    setStatus(`LIVE: ${fileName}`)
    setIsLive(true)
    lastAnnotatedRef.current = null

    const fileUrl = `file:///${filePath.replace(/\\/g, '/')}`
    setVideoSrc(fileUrl)

    setTimeout(() => {
      const video = videoRef.current
      if (video) video.play().catch(e => console.log('Play error:', e))
    }, 300)
  }

  return (
    <div style={styles.wrap}>
      <video
        ref={videoRef}
        src={videoSrc}
        style={{ display: 'none' }}
        loop muted crossOrigin="anonymous"
        onLoadedMetadata={resizeCanvas}
      />

      <div style={styles.header}>
        <div style={styles.headerLeft}>
          <span style={styles.label}>LIVE FEED</span>
          <span style={{
            ...styles.badge,
            color:      isLive ? '#22c55e' : '#71717a',
            border:     `1px solid ${isLive ? '#14532d' : '#27272a'}`,
            background: isLive ? '#052e16' : 'transparent',
          }}>
            {status}
          </span>
          {isLive && <span style={styles.fpsBadge}>{fps} fps</span>}
        </div>
        <div style={styles.headerRight}>
          <span style={styles.frames}>
            {frameCount.toString().padStart(6, '0')} frames
          </span>
          <button style={styles.btn} onClick={handleUpload}>
            ↑ Upload Video
          </button>
        </div>
      </div>

      <div style={styles.legend}>
        <span style={styles.legendItem}>
          <span style={{ ...styles.ldot, background: '#22c55e' }} /> Normal
        </span>
        <span style={styles.legendItem}>
          <span style={{ ...styles.ldot, background: '#ef4444' }} /> Flagged
        </span>
        {isLive && (
          <span style={styles.legendItem}>
            <span style={{ ...styles.ldot, background: '#3b82f6' }} /> YOLOv8 + LSTM
          </span>
        )}
      </div>

      {/* Scrollable canvas container */}
      <div style={styles.canvasWrap}>
        <canvas ref={canvasRef} style={styles.canvas} />
      </div>
    </div>
  )
}

const styles = {
  wrap: {
    display: 'flex', flexDirection: 'column',
    height: '100%', background: '#0a0a0f',
  },
  header: {
    display: 'flex', justifyContent: 'space-between',
    alignItems: 'center', padding: '10px 16px',
    borderBottom: '1px solid #18181b', flexShrink: 0,
  },
  headerLeft:  { display: 'flex', alignItems: 'center', gap: 10 },
  headerRight: { display: 'flex', alignItems: 'center', gap: 12 },
  label: {
    fontSize: 9, letterSpacing: '0.12em',
    color: '#52525b', fontFamily: 'monospace',
  },
  badge:  { fontSize: 9, fontFamily: 'monospace', padding: '2px 6px' },
  fpsBadge: {
    fontSize: 9, color: '#3b82f6', fontFamily: 'monospace',
    padding: '2px 6px', border: '1px solid #1d4ed8', background: '#0c1a3d',
  },
  frames: { fontSize: 9, color: '#3f3f46', fontFamily: 'monospace' },
  btn: {
    fontSize: 10, padding: '4px 10px', background: 'transparent',
    border: '1px solid #3f3f46', color: '#a1a1aa',
    cursor: 'pointer', fontFamily: 'monospace',
  },
  legend: {
    display: 'flex', gap: 16, padding: '6px 16px',
    borderBottom: '1px solid #18181b', flexShrink: 0,
  },
  legendItem: {
    display: 'flex', alignItems: 'center',
    gap: 5, fontSize: 10, color: '#71717a',
  },
  ldot: { width: 6, height: 6, borderRadius: '50%' },
  canvasWrap: {
    flex: 1,
    overflowY: 'auto',   // scroll vertically if video is taller than panel
    overflowX: 'hidden',
    width: '100%',
  },
  canvas: {
    display: 'block',
    width: '100%',
    height: 'auto',      // height driven by canvas.height attribute
  },
}