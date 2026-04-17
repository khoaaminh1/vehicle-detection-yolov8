import React, { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Aperture, UploadCloud, RefreshCw, ChevronRight, PlayCircle, Image as ImageIcon, Download } from 'lucide-react';
import './styles.css';

const API_BASE = 'http://127.0.0.1:8000';

const defaultState = {
  files: [],
  previewUrls: [],
  models: [],
  selectedModel: '',
  results: [],
  loading: false,
  error: '',
};

export default function App() {
  const [currentPage, setCurrentPage] = useState('home'); // 'home' | 'detect'
  const [state, setState] = useState(defaultState);
  const fileInputRef = useRef(null);

  useEffect(() => {
    async function fetchModels() {
      try {
        const res = await fetch(`${API_BASE}/models/`);
        const data = await res.json();
        setState((prev) => ({
          ...prev,
          models: data,
          selectedModel: data[0] || '',
        }));
      } catch (err) {
        setState((prev) => ({
          ...prev,
          error: 'Cannot load models from backend.'
        }));
      }
    }
    fetchModels();
  }, []);

  const handleFileChange = (event) => {
    const selectedFiles = Array.from(event.target.files);
    if (!selectedFiles.length) return;

    state.previewUrls.forEach(url => URL.revokeObjectURL(url));

    const previewUrls = selectedFiles.map(f => URL.createObjectURL(f));
    setState((prev) => ({
      ...prev,
      files: selectedFiles,
      previewUrls,
      results: [],
      error: '',
    }));
  };

  const handleDetect = async () => {
    if (!state.files.length || !state.selectedModel) {
      setState((prev) => ({
        ...prev,
        error: 'Please select images and a model first.'
      }));
      return;
    }

    try {
      setState((prev) => ({ ...prev, loading: true, error: '', results: [] }));
      
      const newResults = [];
      for (let i = 0; i < state.files.length; i++) {
          const file = state.files[i];
          try {
              const formData = new FormData();
              formData.append('file', file);
              formData.append('model_name', state.selectedModel);

              const res = await fetch(`${API_BASE}/detect/`, {
                method: 'POST',
                body: formData,
              });

              if (!res.ok) {
                const errText = await res.text();
                let errMsg = `Detect failed for ${file.name}`;
                try {
                  const errJson = JSON.parse(errText);
                  errMsg = errJson.detail || errMsg;
                } catch(e) {}
                
                newResults.push({
                    name: file.name,
                    status: 'failed',
                    error: errMsg,
                    resultImage: URL.createObjectURL(file), // Fallback image
                    detections: []
                });
              } else {
                  const data = await res.json();
                  newResults.push({
                    name: file.name,
                    status: data.detections && data.detections.length > 0 ? 'success' : 'no_objects',
                    resultImage: `data:image/jpeg;base64,${data.image_b64}`,
                    detections: data.detections || [],
                  });
              }
          } catch(err) {
              newResults.push({
                  name: file.name,
                  status: 'failed',
                  error: err.message,
                  resultImage: URL.createObjectURL(file), // Fallback image
                  detections: []
              });
          }
          
          // Progressively update UI after each image
          setState((prev) => ({ ...prev, results: [...newResults] }));
      }

      setState((prev) => ({
        ...prev,
        loading: false,
      }));
    } catch (err) {
      setState((prev) => ({
        ...prev,
        loading: false,
        error: err.message || 'An error occurred during detection.'
      }));
    }
  };

  const handleReset = () => {
    state.previewUrls.forEach(url => URL.revokeObjectURL(url));
    setState((prev) => ({
      ...prev,
      files: [],
      previewUrls: [],
      results: [],
      error: '',
    }));
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const handleDownloadTxt = () => {
    if (!state.results.length) return;
    
    let txtLines = [];
    state.results.forEach((res, i) => {
        txtLines.push(`Image: ${res.name}`);
        txtLines.push(`Total detections: ${res.detections.length}`);
        txtLines.push('');
        
        if (res.detections.length === 0) {
            txtLines.push('* No reliable vehicle detected');
        } else {
            res.detections.forEach(obj => {
                const confFloat = parseFloat(obj.confidence);
                txtLines.push(`* ${obj.class_name} | ${confFloat.toFixed(2)}`);
            });
        }
        
        txtLines.push('');
        if (i !== state.results.length - 1) {
            txtLines.push('-'.repeat(40));
            txtLines.push('');
        }
    });
    
    const blob = new Blob([txtLines.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'batch_detection_report.txt';
    link.click();
    URL.revokeObjectURL(url);
  };

  const pageVariants = {
    initial: { opacity: 0, y: 20 },
    in: { opacity: 1, y: 0, transition: { duration: 0.6, ease: [0.22, 1, 0.36, 1] } },
    out: { opacity: 0, y: -20, transition: { duration: 0.4 } }
  };

  return (
    <>
      <nav>
        <div className="logo" onClick={() => setCurrentPage('home')}>
          <Aperture size={28} color="var(--accent-1)" />
          <span>Lumine</span>
        </div>
        {currentPage === 'detect' && (
          <button className="btn-secondary btn-small" onClick={() => setCurrentPage('home')}>
            Back to Home
          </button>
        )}
      </nav>

      <AnimatePresence mode="wait">
        {currentPage === 'home' && (
          <motion.div
            key="home"
            className="home-page"
            initial="initial"
            animate="in"
            exit="out"
            variants={pageVariants}
          >
            <div className="glow-bg" />
            <motion.div 
              className="badge"
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.2 }}
            >
              YOLOv8 Powered Engine
            </motion.div>
            <motion.h1
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.3 }}
            >
              Exquisite Vehicle Detection <br />
              <span className="gradient-text">Reimagined</span>
            </motion.h1>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
            >
              Experience real-time batch computer vision wrapped in an award-winning interface. Built for performance, designed for excellence.
            </motion.p>
            <motion.div 
              className="actions"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
            >
              <button className="btn-primary" onClick={() => setCurrentPage('detect')}>
                Get Started <ChevronRight size={18} />
              </button>
            </motion.div>
          </motion.div>
        )}

        {currentPage === 'detect' && (
          <motion.div
            key="detect"
            className="detect-page"
            initial="initial"
            animate="in"
            exit="out"
            variants={pageVariants}
          >
            <div className="detect-header">
              <h2>Batch Detection Sandbox</h2>
              <p>Upload multiple images to test our YOLOv8 models in real-time.</p>
            </div>

            <div className="detect-grid">
              {/* Left Panel: Settings */}
              <div className="panel">
                <h3><Aperture size={20} /> Configuration</h3>
                
                <div className="setup-group">
                  <label>Select Architecture</label>
                  <select
                    value={state.selectedModel}
                    onChange={(e) => setState((prev) => ({ ...prev, selectedModel: e.target.value }))}
                  >
                    {state.models.length === 0 && <option>No models available</option>}
                    {state.models.map((model) => (
                      <option key={model} value={model}>{model}</option>
                    ))}
                  </select>
                </div>

                <div className="setup-group">
                  <label>Media Source (Multiple)</label>
                  <div 
                    className="file-drop"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input 
                      type="file" 
                      accept="image/*" 
                      multiple
                      onChange={handleFileChange}
                      ref={fileInputRef}
                    />
                    <UploadCloud size={32} color="var(--text-muted)" style={{ marginBottom: '0.5rem' }} />
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-main)' }}>Click to browse multiple</p>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>{state.files.length > 0 ? `${state.files.length} selected` : 'JPG, PNG, WEBP'}</p>
                  </div>
                </div>

                {state.error && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="error-msg">
                    {state.error}
                  </motion.div>
                )}

                <div className="actions" style={{ marginTop: '2rem' }}>
                  <button 
                    className="btn-primary" 
                    style={{ flex: 1 }}
                    onClick={handleDetect}
                    disabled={state.loading || !state.files.length || !state.selectedModel}
                  >
                    {state.loading ? 'Processing...' : (
                      <>Run Inference <PlayCircle size={18} /></>
                    )}
                  </button>
                  <button 
                    className="btn-secondary"
                    onClick={handleReset}
                    title="Reset"
                  >
                    <RefreshCw size={18} />
                  </button>
                </div>
              </div>

              {/* Right Panel: Preview & Results */}
              <div className="panel" style={{ display: 'flex', flexDirection: 'column', maxHeight: '75vh', overflowY: 'auto' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                    <h3><ImageIcon size={20} /> Result Viewer</h3>
                    {state.results.length > 0 && (
                        <button className="btn-secondary btn-small" onClick={handleDownloadTxt} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                           <Download size={14} /> Download TXT
                        </button>
                    )}
                </div>
                
                <div className="preview-area" style={{ background: 'transparent', border: 'none', padding: 0 }}>
                  {state.loading && (
                    <div className="loading-overlay" style={{ minHeight: '100px', borderRadius: '12px', border: '1px solid var(--border)', marginBottom: '1rem' }}>
                      <div className="spinner" />
                      <span>Processing image {state.results.length + 1} of {state.files.length}...</span>
                    </div>
                  )}

                  {state.results.length > 0 ? (
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(280px, 1fr))', gap: '1.5rem', alignItems: 'start' }}>
                          {state.results.map((res, i) => (
                              <motion.div 
                                key={`res-${i}`} 
                                className="result-card" 
                                initial={{ opacity: 0, y: 15 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.05, ease: 'easeOut' }}
                                style={{ 
                                  display: 'flex',
                                  flexDirection: 'column',
                                  border: '1px solid rgba(255,255,255,0.08)', 
                                  borderRadius: '16px',
                                  background: 'rgba(20,20,25,0.6)',
                                  overflow: 'hidden',
                                  boxShadow: '0 4px 20px rgba(0,0,0,0.2)',
                                  backdropFilter: 'blur(10px)'
                                }}
                              >
                                  {/* 1. Image Area (Fixed Aspect Ratio) */}
                                  <div style={{ width: '100%', aspectRatio: '4/3', background: '#0a0a0c', position: 'relative', borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                                      <img 
                                        src={res.resultImage} 
                                        alt={res.name} 
                                        style={{ width: '100%', height: '100%', objectFit: 'contain', padding: '0.5rem' }} 
                                      />
                                      {/* Absolute Status Badge */}
                                      <div style={{ position: 'absolute', top: '0.75rem', right: '0.75rem' }}>
                                          {res.status === 'failed' ? (
                                              <span style={{ fontSize: '0.7rem', padding: '4px 10px', borderRadius: '20px', background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.3)', color: '#F87171', fontWeight: '600', backdropFilter: 'blur(4px)', letterSpacing: '0.02em', boxShadow: '0 2px 10px rgba(239,68,68,0.1)' }}>Failed</span>
                                          ) : res.detections.length === 0 ? (
                                              <span style={{ fontSize: '0.7rem', padding: '4px 10px', borderRadius: '20px', background: 'rgba(20,20,20,0.8)', border: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-muted)', backdropFilter: 'blur(4px)', letterSpacing: '0.02em' }}>No reliable vehicle detected</span>
                                          ) : (
                                              <span style={{ fontSize: '0.7rem', padding: '4px 10px', borderRadius: '20px', background: 'rgba(59,130,246,0.15)', border: '1px solid rgba(59,130,246,0.3)', color: '#60A5FA', fontWeight: '600', backdropFilter: 'blur(4px)', letterSpacing: '0.02em', boxShadow: '0 2px 10px rgba(59,130,246,0.1)' }}>Detected</span>
                                          )}
                                      </div>
                                  </div>

                                  {/* 2. Card Footer & Data */}
                                  <div style={{ padding: '1.25rem', display: 'flex', flexDirection: 'column', flex: 1 }}>
                                      {/* Header */}
                                      <div style={{ marginBottom: '1rem' }}>
                                          <h4 style={{ color: 'var(--text-main)', margin: 0, fontSize: '0.95rem', fontWeight: '500', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={res.name}>
                                              {res.name}
                                          </h4>
                                      </div>
                                      
                                      {/* Detections List (Compact Chips) */}
                                      <div style={{ flex: 1 }}>
                                          {res.status === 'failed' ? (
                                              <p style={{ margin: 0, color: '#F87171', fontSize: '0.85rem' }}>{res.error || 'Processing failed'}</p>
                                          ) : res.detections.length === 0 ? (
                                              <p style={{ margin: 0, color: 'var(--text-muted)', fontSize: '0.85rem' }}>No reliable detections found.</p>
                                          ) : (
                                              <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
                                                  {res.detections.map((d, j) => (
                                                      <div key={j} style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', padding: '4px 10px', background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '8px', fontSize: '0.8rem' }}>
                                                          <span style={{ color: 'var(--text-main)', textTransform: 'capitalize' }}>{d.class_name}</span>
                                                          <span style={{ color: '#60A5FA', opacity: 0.9, fontWeight: '500' }}>{(parseFloat(d.confidence)*100).toFixed(0)}%</span>
                                                      </div>
                                                  ))}
                                              </div>
                                          )}
                                      </div>
                                  </div>
                              </motion.div>
                          ))}
                      </div>
                  ) : !state.loading && state.previewUrls.length > 0 ? (
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: '1rem', alignItems: 'start' }}>
                         {state.previewUrls.map((url, i) => (
                             <img key={i} src={url} alt={`preview-${i}`} style={{ width: '100%', height: 'auto', borderRadius: '8px', border: '1px solid var(--border)', display: 'block' }} />
                         ))}
                      </div>
                  ) : !state.loading ? (
                    <div className="empty-state" style={{ minHeight: '300px', border: '1px dashed var(--border)', borderRadius: '12px' }}>
                      <ImageIcon size={48} opacity={0.2} />
                      <p>Awaiting visual input</p>
                    </div>
                  ) : null}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
