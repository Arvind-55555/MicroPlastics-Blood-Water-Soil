"""Monitoring service for real-time anomaly detection."""
import argparse
import time
from pathlib import Path
from loguru import logger
from typing import Dict, Any

from ..utils.config import load_config
from .anomaly_detector import AnomalyDetector, AlertManager


class MonitoringService:
    """Real-time monitoring service."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        monitoring_config = config.get("monitoring", {})
        
        # Initialize detectors for different media
        self.detectors = {
            "water": AnomalyDetector(
                threshold=monitoring_config.get("anomaly_threshold", 3.0),
                window_size=100
            ),
            "soil": AnomalyDetector(
                threshold=monitoring_config.get("anomaly_threshold", 3.0),
                window_size=100
            ),
            "blood": AnomalyDetector(
                threshold=monitoring_config.get("anomaly_threshold", 3.0),
                window_size=100
            )
        }
        
        # Alert manager
        self.alert_manager = AlertManager(monitoring_config.get("alerts", {}))
        
        self.spike_threshold = monitoring_config.get("concentration_spike_threshold", 2.0)
    
    def process_reading(self, media_type: str, concentration: float, 
                       location: str = "unknown", timestamp: float = None):
        """Process a new concentration reading."""
        if media_type not in self.detectors:
            logger.warning(f"Unknown media type: {media_type}")
            return
        
        detector = self.detectors[media_type]
        
        # Update detector
        detector.update(concentration, timestamp)
        
        # Check for anomalies
        anomaly_result = detector.detect(concentration)
        if anomaly_result["is_anomaly"]:
            self.alert_manager.send_alert(
                "anomaly",
                f"Anomalous {media_type} concentration at {location}",
                {
                    "media_type": media_type,
                    "location": location,
                    "concentration": concentration,
                    **anomaly_result
                }
            )
        
        # Check for spikes
        spike_result = detector.detect_spike(concentration, self.spike_threshold)
        if spike_result["is_spike"]:
            self.alert_manager.send_alert(
                "spike",
                f"Concentration spike in {media_type} at {location}",
                {
                    "media_type": media_type,
                    "location": location,
                    "concentration": concentration,
                    **spike_result
                }
            )
        
        return {
            "anomaly": anomaly_result,
            "spike": spike_result
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get monitoring status."""
        status = {
            "detectors": {},
            "recent_alerts": self.alert_manager.get_recent_alerts(10)
        }
        
        for media_type, detector in self.detectors.items():
            if detector.is_fitted:
                values = list(detector.values)
                status["detectors"][media_type] = {
                    "baseline_mean": float(np.mean(values)) if values else 0.0,
                    "baseline_std": float(np.std(values)) if values else 0.0,
                    "sample_count": len(values)
                }
            else:
                status["detectors"][media_type] = {
                    "status": "not_fitted",
                    "sample_count": len(detector.values)
                }
        
        return status


def main():
    """Main monitoring service."""
    import numpy as np  # Import here to avoid circular imports
    
    parser = argparse.ArgumentParser(description="Microplastic monitoring service")
    parser.add_argument("--config", type=str, default="config/config.yaml",
                       help="Path to configuration file")
    parser.add_argument("--test", action="store_true",
                       help="Run test monitoring with simulated data")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    logger.add(
        config.get("logging", {}).get("file", "logs/monitoring.log"),
        rotation="10 MB",
        retention="7 days"
    )
    
    # Initialize monitoring service
    service = MonitoringService(config)
    
    if args.test:
        # Test with simulated data
        logger.info("Running test monitoring with simulated data")
        
        # Simulate normal readings
        for i in range(50):
            concentration = np.random.normal(100, 10)
            service.process_reading("water", concentration, "test_location_1")
            time.sleep(0.1)
        
        # Simulate anomaly
        service.process_reading("water", 500, "test_location_1")
        
        # Simulate spike
        service.process_reading("water", 300, "test_location_1")
        
        # Print status
        status = service.get_status()
        logger.info(f"Monitoring status: {status}")
    else:
        # In production, this would connect to data sources
        logger.info("Monitoring service started (connect to data sources)")
        logger.info("Use --test flag to run with simulated data")


if __name__ == "__main__":
    main()

