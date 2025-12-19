"""Anomaly detection for real-time monitoring."""
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loguru import logger
from collections import deque
import time


class AnomalyDetector:
    """Real-time anomaly detection for microplastic concentrations."""
    
    def __init__(self, threshold: float = 3.0, window_size: int = 100):
        """
        Initialize anomaly detector.
        
        Args:
            threshold: Number of standard deviations for anomaly detection
            window_size: Size of sliding window for baseline calculation
        """
        self.threshold = threshold
        self.window_size = window_size
        self.values = deque(maxlen=window_size)
        self.timestamps = deque(maxlen=window_size)
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def update(self, value: float, timestamp: Optional[float] = None):
        """Update detector with new value."""
        if timestamp is None:
            timestamp = time.time()
        
        self.values.append(value)
        self.timestamps.append(timestamp)
        
        if len(self.values) >= 10:  # Minimum samples for detection
            self.is_fitted = True
    
    def detect(self, value: float) -> Dict[str, Any]:
        """
        Detect if value is anomalous.
        
        Returns:
            Dictionary with 'is_anomaly', 'score', 'baseline_mean', 'baseline_std'
        """
        if not self.is_fitted or len(self.values) < 10:
            return {
                "is_anomaly": False,
                "score": 0.0,
                "baseline_mean": 0.0,
                "baseline_std": 0.0,
                "message": "Insufficient data for anomaly detection"
            }
        
        # Calculate baseline statistics
        baseline_values = np.array(list(self.values))
        baseline_mean = np.mean(baseline_values)
        baseline_std = np.std(baseline_values)
        
        # Z-score method
        if baseline_std > 0:
            z_score = abs((value - baseline_mean) / baseline_std)
            is_anomaly = z_score > self.threshold
        else:
            z_score = 0.0
            is_anomaly = False
        
        # Isolation Forest method (if enough data)
        if_anomaly = False
        if_score = 0.0
        if len(self.values) >= 20:
            try:
                X = np.array(list(self.values)).reshape(-1, 1)
                X_scaled = self.scaler.fit_transform(X)
                self.isolation_forest.fit(X_scaled)
                
                value_scaled = self.scaler.transform([[value]])
                if_prediction = self.isolation_forest.predict(value_scaled)[0]
                if_score = self.isolation_forest.score_samples(value_scaled)[0]
                if_anomaly = if_prediction == -1
            except Exception as e:
                logger.warning(f"Isolation Forest error: {e}")
        
        # Combine both methods
        final_anomaly = is_anomaly or if_anomaly
        
        result = {
            "is_anomaly": bool(final_anomaly),
            "score": float(z_score),
            "if_score": float(if_score),
            "baseline_mean": float(baseline_mean),
            "baseline_std": float(baseline_std),
            "value": float(value),
            "threshold": self.threshold
        }
        
        if final_anomaly:
            result["message"] = f"Anomaly detected: value {value:.2f} is {z_score:.2f} std devs from mean"
            logger.warning(result["message"])
        
        return result
    
    def detect_spike(self, value: float, spike_threshold: float = 2.0) -> Dict[str, Any]:
        """Detect concentration spikes (multiplicative threshold)."""
        if not self.is_fitted or len(self.values) < 10:
            return {
                "is_spike": False,
                "multiplier": 1.0,
                "message": "Insufficient data for spike detection"
            }
        
        baseline_values = np.array(list(self.values))
        baseline_mean = np.mean(baseline_values)
        
        if baseline_mean > 0:
            multiplier = value / baseline_mean
            is_spike = multiplier > spike_threshold
        else:
            multiplier = 1.0
            is_spike = False
        
        result = {
            "is_spike": bool(is_spike),
            "multiplier": float(multiplier),
            "value": float(value),
            "baseline_mean": float(baseline_mean),
            "spike_threshold": spike_threshold
        }
        
        if is_spike:
            result["message"] = f"Concentration spike detected: {multiplier:.2f}x baseline"
            logger.warning(result["message"])
        
        return result


class AlertManager:
    """Manage alerts for anomalies and spikes."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.email_enabled = config.get("email_enabled", False)
        self.webhook_url = config.get("webhook_url")
        self.alert_history = []
    
    def send_alert(self, alert_type: str, message: str, data: Dict[str, Any]):
        """Send alert via configured channels."""
        alert = {
            "type": alert_type,
            "message": message,
            "data": data,
            "timestamp": time.time()
        }
        
        self.alert_history.append(alert)
        
        # Log alert
        logger.warning(f"ALERT [{alert_type}]: {message}")
        
        # Send webhook if configured
        if self.webhook_url:
            try:
                import requests
                requests.post(self.webhook_url, json=alert, timeout=5)
            except Exception as e:
                logger.error(f"Failed to send webhook alert: {e}")
        
        # Send email if configured
        if self.email_enabled:
            # Email sending would be implemented here
            logger.info("Email alert (not implemented)")
    
    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return self.alert_history[-limit:]

