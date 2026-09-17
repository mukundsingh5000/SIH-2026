import React, { useState, useEffect } from 'react';
import { getAnalyticsData } from '../services/api';
import { 
  BarChart3, 
  PieChart, 
  TrendingUp, 
  Layers, 
  Activity, 
  ShieldAlert, 
  RefreshCw,
  Zap,
  Target
} from 'lucide-react';

const CLASS_COLORS = {
  background: '#64748b',
  aircraft: '#0284c7',
  bottle: '#06b6d4',
  cylinder: '#7c3aed',
  human: '#e11d48',
  net: '#dc2626',
  pipe: '#d97706',
  wreck: '#0d9488'
};

export default function AnalyticsDashboard() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadAnalytics();
  }, []);

  const loadAnalytics = async () => {
    setLoading(true);
    setError(null);
    try {
      const summary = await getAnalyticsData();
      setData(summary);
    } catch (err) {
      console.error('Analytics Error:', err);
      setError('Failed to load database analytics: ' + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div style={{ textAlign: 'center', padding: '4rem 1rem', color: 'var(--text-muted)' }}>
        <div className="spinner" style={{ margin: '0 auto 1rem', width: 24, height: 24, borderColor: 'var(--border-marine)', borderTopColor: 'var(--ocean-blue)' }}></div>
        <p style={{ fontSize: '0.9rem', fontWeight: 600 }}>Calculating Database Analytics...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="error-banner" style={{ margin: '1.5rem 0' }}>
        <ShieldAlert size={18} />
        <div>{error}</div>
      </div>
    );
  }

  const {
    total_scans,
    total_detected,
    average_confidence,
    most_detected_object,
    object_counts,
    confidence_by_class,
    recent_predictions
  } = data;

  const maxObjectCount = Math.max(...Object.values(object_counts), 1);
  const totalObjectSum = Object.values(object_counts).reduce((a, b) => a + b, 0);

  const renderConfidenceByClassChart = () => {
    if (!confidence_by_class) return null;
    const entries = Object.entries(confidence_by_class);

    return (
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', marginTop: '1rem' }}>
        {entries.map(([cls, avgConf]) => {
          const pct = avgConf * 100;
          const color = CLASS_COLORS[cls] || 'var(--ocean-blue)';
          return (
            <div key={cls}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.35rem' }}>
                <span style={{ textTransform: 'capitalize', fontWeight: 600, color: 'var(--primary-navy)' }}>
                  {cls} {cls === 'background' ? '(Background)' : ''}
                </span>
                <span style={{ color: color, fontWeight: 700 }}>
                  {pct > 0 ? `${pct.toFixed(1)}% Avg Confidence` : 'No Scans'}
                </span>
              </div>
              <div style={{ width: '100%', height: 10, background: '#e0f2fe', borderRadius: 4, overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${pct}%`,
                    height: '100%',
                    background: color,
                    borderRadius: 4,
                    transition: 'width 0.5s ease'
                  }}
                />
              </div>
            </div>
          );
        })}
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1.75rem' }}>
      {/* Header bar */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div>
          <h2 style={{ fontSize: '1.35rem', fontWeight: 800, color: 'var(--primary-navy)' }}>
            Analytics Dashboard
          </h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Real statistical metrics calculated directly from SQLite prediction logs
          </p>
        </div>
        <button className="btn btn-secondary" onClick={loadAnalytics} style={{ flex: 'none', padding: '0.45rem 0.9rem', fontSize: '0.85rem' }}>
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* KPI CARDS GRID */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(210px, 1fr))', gap: '1.25rem' }}>
        {/* KPI 1 */}
        <div className="card" style={{ padding: '1.15rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase' }}>Total Scans</span>
            <Activity size={18} color="var(--ocean-blue)" />
          </div>
          <div style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--primary-navy)' }}>
            {total_scans}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
            Total inferences executed
          </div>
        </div>

        {/* KPI 2 */}
        <div className="card" style={{ padding: '1.15rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase' }}>Objects Detected</span>
            <Target size={18} color="var(--status-detected)" />
          </div>
          <div style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--status-detected)' }}>
            {total_detected}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
            Excluding background
          </div>
        </div>

        {/* KPI 3 */}
        <div className="card" style={{ padding: '1.15rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase' }}>Avg Confidence</span>
            <Zap size={18} color="var(--ocean-blue)" />
          </div>
          <div style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--ocean-blue)' }}>
            {(average_confidence * 100).toFixed(1)}%
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
            Mean PyTorch score
          </div>
        </div>

        {/* KPI 4 */}
        <div className="card" style={{ padding: '1.15rem' }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.35rem' }}>
            <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)', fontWeight: 700, textTransform: 'uppercase' }}>Most Common</span>
            <Layers size={18} color="var(--turquoise)" />
          </div>
          <div style={{ fontSize: '1.85rem', fontWeight: 800, color: 'var(--turquoise)', textTransform: 'capitalize' }}>
            {most_detected_object}
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
            Top occurrence class
          </div>
        </div>
      </div>

      {/* CHARTS ROW 1 */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.3fr 1fr', gap: '1.5rem' }}>
        {/* Visual 1 */}
        <div className="card">
          <h3 className="card-title">
            <BarChart3 size={16} color="var(--ocean-blue)" /> Objects by Type
          </h3>
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', marginTop: '1rem' }}>
            {Object.entries(object_counts).map(([cls, count]) => {
              const pct = maxObjectCount > 0 ? (count / maxObjectCount) * 100 : 0;
              const color = CLASS_COLORS[cls] || 'var(--ocean-blue)';
              return (
                <div key={cls}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', marginBottom: '0.35rem' }}>
                    <span style={{ textTransform: 'capitalize', fontWeight: 600, color: 'var(--primary-navy)' }}>{cls}</span>
                    <span style={{ color: 'var(--text-muted)', fontWeight: 600 }}>{count}</span>
                  </div>
                  <div style={{ width: '100%', height: 10, background: '#e0f2fe', borderRadius: 4, overflow: 'hidden' }}>
                    <div
                      style={{
                        width: `${pct}%`,
                        height: '100%',
                        background: color,
                        borderRadius: 4,
                        transition: 'width 0.5s ease'
                      }}
                    />
                  </div>
                </div>
              );
            })}
          </div>
        </div>

        {/* Visual 2 */}
        <div className="card">
          <h3 className="card-title">
            <PieChart size={16} color="var(--ocean-blue)" /> Detection Breakdown
          </h3>
          {totalObjectSum > 0 ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', marginTop: '1rem' }}>
              {Object.entries(object_counts).map(([cls, count]) => {
                const share = totalObjectSum > 0 ? ((count / totalObjectSum) * 100).toFixed(1) : 0;
                const color = CLASS_COLORS[cls] || 'var(--ocean-blue)';
                return (
                  <div key={cls} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <span style={{ width: 10, height: 10, borderRadius: 2, backgroundColor: color }} />
                      <span style={{ textTransform: 'capitalize', fontWeight: 600, color: 'var(--primary-navy)' }}>{cls}</span>
                    </div>
                    <div style={{ fontWeight: 700, color: 'var(--primary-navy)' }}>
                      {share}% <span style={{ color: 'var(--text-muted)', fontWeight: 400 }}>({count})</span>
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '2.5rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              No debris objects detected yet in database logs.
            </div>
          )}
        </div>
      </div>

      {/* CHARTS ROW 2 */}
      <div className="card">
        <h3 className="card-title">
          <TrendingUp size={16} color="var(--ocean-blue)" /> Average Model Confidence vs Object Class
        </h3>
        {renderConfidenceByClassChart()}
      </div>

      {/* RECENT DETECTIONS TABLE */}
      <div className="card">
        <h3 className="card-title">
          <Activity size={16} color="var(--ocean-blue)" /> Prediction Log Records (SQLite)
        </h3>
        {recent_predictions && recent_predictions.length > 0 ? (
          <div style={{ overflowX: 'auto' }}>
            <table className="history-table">
              <thead>
                <tr>
                  <th>ID</th>
                  <th>Timestamp</th>
                  <th>Class / Object</th>
                  <th>Confidence</th>
                  <th>Status</th>
                </tr>
              </thead>
              <tbody>
                {recent_predictions.map((rec) => (
                  <tr key={rec.id}>
                    <td style={{ fontWeight: 600 }}>#{rec.id}</td>
                    <td style={{ color: 'var(--text-muted)' }}>{rec.timestamp}</td>
                    <td style={{ fontWeight: 700, textTransform: 'capitalize', color: 'var(--primary-navy)' }}>
                      {rec.class_name}
                    </td>
                    <td style={{ fontWeight: 600 }}>{(rec.confidence * 100).toFixed(1)}%</td>
                    <td>
                      <span
                        className={`status-badge ${rec.detected ? 'detected' : 'clear'}`}
                        style={{ padding: '0.15rem 0.5rem', fontSize: '0.75rem' }}
                      >
                        {rec.detected ? 'Detected' : 'Clear'}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>No prediction logs recorded yet.</p>
        )}
      </div>
    </div>
  );
}
