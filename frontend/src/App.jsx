import { useState, useEffect } from 'react'

const defaultGene = {
  vocab_dim: 4096,
  hidden_dim: 512,
  num_layers: 4,
  num_heads: 8,
  head_dim: 64,
  intermediate_size: 2048,
  max_position_embeddings: 2048,
  rope_theta: 10000.0,
  use_bias: true,
  attention_types: ['full'],
  hidden_act: 'gelu',
  pooling_type: 'cls',
  layer_norm_eps: 1e-5,
  rms_norm_eps: 1e-6,
  use_rms_norm: true,
  use_flash_attention: false,
  sliding_window: 4096,
  dropout: 0.0,
  use_rope: true,
  use_gated_activation: false,
}

function App() {
  const [gene, setGene] = useState(defaultGene)
  const [activeTab, setActiveTab] = useState('visualize')
  const [vizFormat, setVizFormat] = useState('ascii')
  const [vizOutput, setVizOutput] = useState('')
  const [validation, setValidation] = useState(null)
  const [stats, setStats] = useState(null)
  const [verified, setVerified] = useState(null)
  const [loading, setLoading] = useState(false)

  const updateField = (field, value) => {
    setGene(prev => ({ ...prev, [field]: value }))
  }

  const handleValidate = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/validate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(gene),
      })
      const data = await res.json()
      setValidation(data)
    } catch (e) {
      setValidation({ valid: false, errors: [e.message] })
    }
    setLoading(false)
  }

  const handleVerify = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/verify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(gene),
      })
      const data = await res.json()
      setVerified(data.verified)
    } catch (e) {
      setVerified(false)
    }
    setLoading(false)
  }

  const handleVisualize = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/visualize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...gene, format: vizFormat }),
      })
      const data = await res.json()
      setVizOutput(data.visualization || data.error)
    } catch (e) {
      setVizOutput(e.message)
    }
    setLoading(false)
  }

  const handleStats = async () => {
    setLoading(true)
    try {
      const res = await fetch('/api/stats', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(gene),
      })
      const data = await res.json()
      setStats(data)
    } catch (e) {
      setStats(null)
    }
    setLoading(false)
  }

  useEffect(() => {
    if (activeTab === 'visualize') handleVisualize()
    if (activeTab === 'validate') handleValidate()
    if (activeTab === 'stats') handleStats()
    if (activeTab === 'verify') handleVerify()
  }, [gene, vizFormat])

  return (
    <div className="app">
      <header>
        <h1>ArchGene</h1>
        <p>Neural Architecture Generator</p>
      </header>

      <div className="main">
        <aside className="sidebar">
          <h2>Parameters</h2>
          
          <div className="form-group">
            <label>vocab_dim</label>
            <input
              type="range" min="256" max="65536" step="256"
              value={gene.vocab_dim}
              onChange={e => updateField('vocab_dim', parseInt(e.target.value))}
            />
            <span>{gene.vocab_dim}</span>
          </div>

          <div className="form-group">
            <label>hidden_dim</label>
            <input
              type="range" min="64" max="2048" step="64"
              value={gene.hidden_dim}
              onChange={e => updateField('hidden_dim', parseInt(e.target.value))}
            />
            <span>{gene.hidden_dim}</span>
          </div>

          <div className="form-group">
            <label>num_layers</label>
            <input
              type="range" min="1" max="32"
              value={gene.num_layers}
              onChange={e => updateField('num_layers', parseInt(e.target.value))}
            />
            <span>{gene.num_layers}</span>
          </div>

          <div className="form-group">
            <label>num_heads</label>
            <input
              type="range" min="1" max="32"
              value={gene.num_heads}
              onChange={e => updateField('num_heads', parseInt(e.target.value))}
            />
            <span>{gene.num_heads}</span>
          </div>

          <div className="form-group">
            <label>head_dim</label>
            <input
              type="range" min="16" max="256" step="16"
              value={gene.head_dim}
              onChange={e => updateField('head_dim', parseInt(e.target.value))}
            />
            <span>{gene.head_dim}</span>
          </div>

          <div className="form-group">
            <label>intermediate_size</label>
            <input
              type="range" min="256" max="8192" step="256"
              value={gene.intermediate_size}
              onChange={e => updateField('intermediate_size', parseInt(e.target.value))}
            />
            <span>{gene.intermediate_size}</span>
          </div>

          <div className="form-group">
            <label>max_position_embeddings</label>
            <input
              type="range" min="256" max="8192" step="256"
              value={gene.max_position_embeddings}
              onChange={e => updateField('max_position_embeddings', parseInt(e.target.value))}
            />
            <span>{gene.max_position_embeddings}</span>
          </div>

          <div className="form-group">
            <label>rope_theta</label>
            <input
              type="number" step="1000"
              value={gene.rope_theta}
              onChange={e => updateField('rope_theta', parseFloat(e.target.value))}
            />
          </div>

          <div className="form-group">
            <label>hidden_act</label>
            <select
              value={gene.hidden_act}
              onChange={e => updateField('hidden_act', e.target.value)}
            >
              <option value="relu">relu</option>
              <option value="gelu">gelu</option>
              <option value="silu">silu</option>
              <option value="tanh">tanh</option>
              <option value="sigmoid">sigmoid</option>
            </select>
          </div>

          <div className="form-group">
            <label>pooling_type</label>
            <select
              value={gene.pooling_type}
              onChange={e => updateField('pooling_type', e.target.value)}
            >
              <option value="cls">cls</option>
              <option value="mean">mean</option>
              <option value="max">max</option>
            </select>
          </div>

          <div className="form-group">
            <label>use_bias</label>
            <input
              type="checkbox"
              checked={gene.use_bias}
              onChange={e => updateField('use_bias', e.target.checked)}
            />
          </div>

          <div className="form-group">
            <label>use_rms_norm</label>
            <input
              type="checkbox"
              checked={gene.use_rms_norm}
              onChange={e => updateField('use_rms_norm', e.target.checked)}
            />
          </div>

          <div className="form-group">
            <label>use_rope</label>
            <input
              type="checkbox"
              checked={gene.use_rope}
              onChange={e => updateField('use_rope', e.target.checked)}
            />
          </div>
        </aside>

        <main className="content">
          <nav className="tabs">
            <button
              className={activeTab === 'visualize' ? 'active' : ''}
              onClick={() => setActiveTab('visualize')}
            >
              Visualize
            </button>
            <button
              className={activeTab === 'validate' ? 'active' : ''}
              onClick={() => setActiveTab('validate')}
            >
              Validate
            </button>
            <button
              className={activeTab === 'stats' ? 'active' : ''}
              onClick={() => setActiveTab('stats')}
            >
              Stats
            </button>
            <button
              className={activeTab === 'verify' ? 'active' : ''}
              onClick={() => setActiveTab('verify')}
            >
              Z3 Verify
            </button>
          </nav>

          <div className="tab-content">
            {activeTab === 'visualize' && (
              <div>
                <div className="format-selector">
                  <label>Format:</label>
                  <select value={vizFormat} onChange={e => setVizFormat(e.target.value)}>
                    <option value="ascii">ASCII</option>
                    <option value="mermaid">Mermaid</option>
                    <option value="json">JSON</option>
                  </select>
                </div>
                <pre className="output">{vizOutput}</pre>
              </div>
            )}

            {activeTab === 'validate' && (
              <div>
                {validation && (
                  <div className={`status ${validation.valid ? 'valid' : 'invalid'}`}>
                    {validation.valid ? '✓ Gene is valid' : '✗ Gene is invalid'}
                  </div>
                )}
                {validation?.errors?.length > 0 && (
                  <ul className="errors">
                    {validation.errors.map((err, i) => (
                      <li key={i}>{err}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}

            {activeTab === 'stats' && (
              <div>
                {stats && (
                  <table className="stats-table">
                    <tbody>
                      <tr>
                        <td>Parameters</td>
                        <td>{stats.parameters.toLocaleString()}</td>
                      </tr>
                      <tr>
                        <td>Memory (MB)</td>
                        <td>{stats.memory_mb.toFixed(2)}</td>
                      </tr>
                    </tbody>
                  </table>
                )}
              </div>
            )}

            {activeTab === 'verify' && (
              <div>
                {verified !== null && (
                  <div className={`status ${verified ? 'valid' : 'invalid'}`}>
                    {verified
                      ? '✓ Z3 Verified: Architecture is well-formed'
                      : '✗ Z3 Verification Failed'}
                  </div>
                )}
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  )
}

export default App