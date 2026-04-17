import React, { useEffect, useMemo, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Aperture, UploadCloud, RefreshCw, ChevronRight, CheckCircle2, PlayCircle, Image as ImageIcon } from 'lucide-react';
import './styles.css';

const API_BASE = 'http://localhost:8000';

const defaultState = {
  file: null,
  previewUrl: '',
  models: [],
  selectedModel: '',
  resultImage: '',
  detections: [],
  loading: false,
  error: '',
};

export default function App() {
  const [currentPage, setCurrentPage] = useState('home'); // 'home' | 'detect'
  const [state, setState] = useState(defaultState);
  const fileInputRef = useRef(null);

  const hasResult = useMemo(() => state.resultImage.length > 0, [state.resultImage]);

  const detectionSummary = useMemo(() => {
    const summary = {};
    state.detections.forEach(d => {
      const cls = d.class_name;
      if (!summary[cls]) summary[cls] = 0;
      summary[cls] += 1;
    });
    return Object.entries(summary).sort((a, b) => b[1] - a[1]);
  }, [state.detections]);

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
    const file = event.target.files?.[0];
    if (!file) return;

    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
    }

    const previewUrl = URL.createObjectURL(file);
    setState((prev) => ({
      ...prev,
      file,
      previewUrl,
      resultImage: '',
      detections: [],
      error: '',
    }));
  };

  const handleDetect = async () => {
    if (!state.file || !state.selectedModel) {
      setState((prev) => ({
        ...prev,
        error: 'Please select an image and a model first.'
      }));
      return;
    }

    try {
      setState((prev) => ({ ...prev, loading: true, error: '' }));
      const formData = new FormData();
      formData.append('file', state.file);
      formData.append('model_name', state.selectedModel);

      const res = await fetch(`${API_BASE}/detect/`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err?.detail || 'Detect failed');
      }

      const data = await res.json();
      setState((prev) => ({
        ...prev,
        resultImage: `data:image/jpeg;base64,${data.image_b64}`,
        detections: data.detections,
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
    if (state.previewUrl) {
      URL.revokeObjectURL(state.previewUrl);
    }
    setState((prev) => ({
      ...prev,
      file: null,
      previewUrl: '',
      resultImage: '',
      detections: [],
      error: '',
    }));
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
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
              Experience real-time, high-precision computer vision wrapped in an award-winning interface. Built for performance, designed for excellence.
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
              <h2>Detection Sandbox</h2>
              <p>Upload an image to test our YOLOv8 models in real-time.</p>
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
                  <label>Media Source</label>
                  <div 
                    className="file-drop"
                    onClick={() => fileInputRef.current?.click()}
                  >
                    <input 
                      type="file" 
                      accept="image/*" 
                      onChange={handleFileChange}
                      ref={fileInputRef}
                    />
                    <UploadCloud size={32} color="var(--text-muted)" style={{ marginBottom: '0.5rem' }} />
                    <p style={{ fontSize: '0.9rem', color: 'var(--text-main)' }}>Click to browse</p>
                    <p style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>JPG, PNG, WEBP</p>
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
                    disabled={state.loading || !state.file || !state.selectedModel}
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
              <div className="panel" style={{ display: 'flex', flexDirection: 'column' }}>
                <h3 style={{ marginBottom: '1rem' }}><ImageIcon size={20} /> Result Viewer</h3>
                
                <div className="preview-area">
                  {state.loading && (
                    <div className="loading-overlay">
                      <div className="spinner" />
                      <span>Running Tensor Engines...</span>
                    </div>
                  )}

                  {hasResult ? (
                    <motion.img 
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      src={state.resultImage} 
                      alt="Detection result" 
                    />
                  ) : state.previewUrl ? (
                    <motion.img 
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      src={state.previewUrl} 
                      alt="Preview" 
                      style={{ opacity: state.loading ? 0.3 : 1 }}
                    />
                  ) : (
                    <div className="empty-state">
                      <ImageIcon size={48} opacity={0.2} />
                      <p>Awaiting visual input</p>
                    </div>
                  )}
                </div>

                {/* Results List */}
                <div style={{ marginTop: '2rem', flex: 1 }}>
                  <h3><CheckCircle2 size={20} /> Analysis Summary</h3>
                  {state.detections.length === 0 && hasResult && (
                    <p style={{ color: 'var(--text-muted)', fontSize: '0.9rem' }}>No vehicles detected in this frame.</p>
                  )}
                  {state.detections.length > 0 && (
                    <ul className="results-list">
                      {detectionSummary.map(([className, count], index) => (
                        <motion.li 
                          key={className}
                          className="result-item"
                          initial={{ opacity: 0, x: -10 }}
                          animate={{ opacity: 1, x: 0 }}
                          transition={{ delay: index * 0.05 }}
                        >
                          <span>{className.charAt(0).toUpperCase() + className.slice(1)}</span>
                          <span className="conf">{count} {count > 1 ? 'objects' : 'object'}</span>
                        </motion.li>
                      ))}
                    </ul>
                  )}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
