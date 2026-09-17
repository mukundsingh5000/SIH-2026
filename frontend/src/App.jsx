import React, { useState, useRef } from 'react';
import { predictSonarImage } from './services/api';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import { 
  Upload, 
  Radar, 
  AlertTriangle, 
  CheckCircle, 
  RotateCcw, 
  Cpu, 
  FileImage,
  ShieldCheck,
  BarChart3,
  FileCheck
} from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState('detector'); // 'detector' | 'analytics'
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);

  const handleFileSelect = (file) => {
    if (!file) return;
    if (!file.type.startsWith('image/')) {
      setError('Please select a valid image file (PNG, JPG, BMP).');
      return;
    }

    setError(null);
    setSelectedFile(file);
    setResult(null);

    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewUrl(reader.result);
    };
    reader.readAsDataURL(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelect(e.dataTransfer.files[0]);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please upload a sonar image before running analysis.');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await predictSonarImage(selectedFile);
      setResult(data);
    } catch (err) {
      setError(err.message || 'An error occurred during inference.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setResult(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app-container">
      {/* Header */}
      <header className="header">
        <h1>TAR<span>ANG</span></h1>
        <p>AI-Powered Sonar Intelligence for Marine Debris Detection & Subsea Threat Classification</p>
      </header>

      {/* Navigation Tabs */}
      <nav className="nav-container">
        <button
          className={`nav-tab ${activeTab === 'detector' ? 'active' : ''}`}
          onClick={() => setActiveTab('detector')}
        >
          <Radar size={18} /> Detection
        </button>
        <button
          className={`nav-tab ${activeTab === 'analytics' ? 'active' : ''}`}
          onClick={() => setActiveTab('analytics')}
        >
          <BarChart3 size={18} /> Analytics
        </button>
      </nav>

      {/* Tab 1: Sonar Detector View */}
      {activeTab === 'detector' && (
        <>
          {/* Error Message Display */}
          {error && (
            <div className="error-banner">
              <AlertTriangle size={18} />
              <div>{error}</div>
            </div>
          )}

          {/* Main Grid: Upload & Predict vs Result */}
          <div className="main-grid">
            {/* Left Column: Image Input */}
            <div className="card">
              <h2 className="card-title">
                <FileImage size={18} color="var(--ocean-blue)" /> Sonar Image Panel
              </h2>

              <input
                type="file"
                ref={fileInputRef}
                onChange={(e) => handleFileSelect(e.target.files[0])}
                accept="image/*"
                style={{ display: 'none' }}
              />

              {!previewUrl ? (
                <div
                  className="drop-zone"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <Upload className="upload-icon" />
                  <p style={{ fontWeight: 700, color: 'var(--primary-navy)', marginBottom: '0.25rem' }}>
                    Click or Drag Sonar Image Here
                  </p>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    Supports PNG, JPG, BMP grayscale acoustic sonar frames
                  </p>
                </div>
              ) : (
                <div style={{ textAlign: 'center' }}>
                  <div className="image-preview-wrapper">
                    <img src={previewUrl} alt="Sonar Scan Preview" className="image-preview" />
                  </div>
                  {selectedFile && (
                    <div className="file-info-badge">
                      <FileCheck size={14} style={{ display: 'inline', marginRight: 4, verticalAlign: 'middle' }} />
                      Filename: {selectedFile.name}
                    </div>
                  )}
                </div>
              )}

              <div className="btn-group">
                <button
                  className="btn btn-primary"
                  onClick={handleAnalyze}
                  disabled={loading || !selectedFile}
                >
                  {loading ? (
                    <>
                      <div className="spinner"></div> Analyzing...
                    </>
                  ) : (
                    <>
                      <Cpu size={18} /> Analyze Sonar
                    </>
                  )}
                </button>

                <button
                  className="btn btn-secondary"
                  onClick={handleReset}
                  disabled={loading || (!selectedFile && !result)}
                >
                  <RotateCcw size={16} /> Reset
                </button>
              </div>
            </div>

            {/* Right Column: Prediction Result */}
            <div className="card">
              <h2 className="card-title">
                <ShieldCheck size={18} color="var(--ocean-blue)" /> Detection Result
              </h2>

              {result ? (
                <div className="result-box">
                  {/* Status Badge */}
                  <div className={`status-badge ${result.detected ? 'detected' : 'clear'}`}>
                    {result.detected ? <AlertTriangle size={16} /> : <CheckCircle size={16} />}
                    {result.detected ? `${result.class_name.toUpperCase()} DETECTED` : 'NO OBJECT DETECTED'}
                  </div>

                  {/* Class Name Header */}
                  <div>
                    <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                      Classification
                    </span>
                    <div className="class-title">
                      {result.detected ? `${result.class_name} Detected` : 'No Object Detected'}
                    </div>
                  </div>

                  {/* Metrics */}
                  <div className="metric-row">
                    <span className="metric-label">Confidence</span>
                    <span className="metric-value">{(result.confidence * 100).toFixed(1)}%</span>
                  </div>

                  <div className="metric-row">
                    <span className="metric-label">Class ID</span>
                    <span className="metric-value">{result.class_id}</span>
                  </div>

                  <div className="metric-row">
                    <span className="metric-label">Object Presence</span>
                    <span className="metric-value" style={{ color: result.detected ? 'var(--status-detected)' : 'var(--text-muted)' }}>
                      {result.detected ? 'Detected' : 'Not Detected'}
                    </span>
                  </div>
                </div>
              ) : (
                <div style={{ textAlign: 'center', padding: '3.5rem 1rem', color: 'var(--text-muted)' }}>
                  <Radar size={36} style={{ opacity: 0.35, color: 'var(--ocean-blue)', marginBottom: '0.75rem' }} />
                  <p style={{ fontSize: '0.9rem' }}>Upload a sonar frame and click <strong>Analyze Sonar</strong> to perform classification.</p>
                </div>
              )}
            </div>
          </div>
        </>
      )}

      {/* Tab 2: Analytics Dashboard View */}
      {activeTab === 'analytics' && <AnalyticsDashboard />}
    </div>
  );
}
