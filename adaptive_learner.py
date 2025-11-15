# adaptive_learner.py
"""
Adaptive Threat Learning System for Autonomous Drone Swarm
Implements online incremental learning from user feedback

Features:
- Online learning from operator decisions
- Autonomous threat assessment for jammer mode
- Persistent model storage
- Rule-based fallback for untrained scenarios
"""

import numpy as np
import pickle
import os
import time
from datetime import datetime
from collections import deque
import logging

# Initialize logger
logger = logging.getLogger("LEARNER")
logger.setLevel(logging.INFO)

class AdaptiveThreatLearner:
    """
    Lightweight online learning system for threat pattern recognition.
    Learns from user confirmations and auto-adapts threat detection.
    
    Uses Gaussian Naive Bayes for fast incremental learning without
    requiring full dataset retraining.
    """
    
    def __init__(self, model_path='models/threat_model.pkl'):
        self.model_path = model_path
        
        # Create models directory if it doesn't exist
        model_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else 'models'
        os.makedirs(model_dir, exist_ok=True)
        
        # Incremental learner (updates with each example)
        try:
            from sklearn.naive_bayes import GaussianNB
            self.classifier = GaussianNB()
        except ImportError:
            logger.error("scikit-learn not installed! Run: pip install scikit-learn")
            self.classifier = None
            
        self.is_trained = False
        
        # Experience buffer (stores last 1000 examples)
        self.experience_buffer = deque(maxlen=1000)
        
        # Training history
        self.training_history = []
        
        # Load existing model if available
        self.load_model()
        
        # Basic threat rules (fallback when model not trained)
        self.threat_patterns = {
            'high_confidence_vehicle': {
                'min_conf': 0.75, 
                'classes': ['car', 'truck', 'bus'],
                'priority': 0.8
            },
            'person_detected': {
                'min_conf': 0.70, 
                'classes': ['person'],
                'priority': 0.85
            },
            'large_vehicle': {
                'min_conf': 0.65, 
                'classes': ['truck', 'bus'],
                'priority': 0.9
            }
        }
        
        logger.info(f"🧠 AdaptiveThreatLearner initialized")
        logger.info(f"   Model path: {model_path}")
        logger.info(f"   Experiences loaded: {len(self.experience_buffer)}")
        
    def extract_features(self, detection):
        """
        Extract machine learning features from a detection.
        
        Features extracted:
        1. Object class (encoded as integer)
        2. Detection confidence score
        3. X position (normalized)
        4. Y position (normalized)
        5. Bounding box area (normalized)
        6. Distance from patrol center
        7. Time of day (hour normalized)
        
        Args:
            detection: Dict containing detection information
            
        Returns:
            numpy array of shape (1, 7) containing features
        """
        features = []
        
        # Feature 1: Class encoding
        class_map = {
            'person': 0, 
            'car': 1, 
            'truck': 2, 
            'bus': 3, 
            'motorcycle': 4,
            'bicycle': 5
        }
        class_val = class_map.get(detection.get('class', ''), -1)
        features.append(class_val)
        
        # Feature 2: Confidence score
        confidence = float(detection.get('confidence', 0.5))
        features.append(confidence)
        
        # Features 3-4: Position (normalized to ±100m range)
        world_pos = detection.get('world_pos', (0, 0))
        x = float(world_pos[0]) / 100.0
        y = float(world_pos[1]) / 100.0
        features.append(x)
        features.append(y)
        
        # Feature 5: Bounding box area (normalized)
        bbox_area = detection.get('bbox_area', 1000.0)
        normalized_area = float(bbox_area) / 10000.0
        features.append(normalized_area)
        
        # Feature 6: Distance from patrol center (normalized)
        dist = np.sqrt(float(world_pos[0])**2 + float(world_pos[1])**2)
        normalized_dist = dist / 100.0
        features.append(normalized_dist)
        
        # Feature 7: Time of day (hour normalized to 0-1)
        hour = datetime.now().hour / 24.0
        features.append(hour)
        
        return np.array(features).reshape(1, -1)
    
    def predict_threat_level(self, detection):
        """
        Predict if detection is a real threat using learned patterns.
        
        Args:
            detection: Dict containing detection information
            
        Returns:
            tuple: (is_threat: bool, confidence: float)
                - is_threat: True if classified as threat
                - confidence: Prediction confidence (0.0-1.0)
        """
        features = self.extract_features(detection)
        
        # If model not trained yet, use rule-based fallback
        if not self.is_trained or len(self.experience_buffer) < 5:
            return self._rule_based_assessment(detection)
        
        # If sklearn not available, fallback to rules
        if self.classifier is None:
            return self._rule_based_assessment(detection)
        
        try:
            # Get ML prediction
            prediction = self.classifier.predict(features)[0]
            proba = self.classifier.predict_proba(features)[0]
            confidence = float(proba[int(prediction)])
            
            logger.debug(f"ML Prediction: threat={prediction}, conf={confidence:.2%}")
            return bool(prediction), confidence
            
        except Exception as e:
            logger.warning(f"Prediction error: {e}, falling back to rules")
            return self._rule_based_assessment(detection)
    
    def _rule_based_assessment(self, detection):
        """
        Fallback rule-based threat assessment.
        Used when ML model is not yet trained.
        
        Args:
            detection: Dict containing detection information
            
        Returns:
            tuple: (is_threat: bool, confidence: float)
        """
        obj_class = detection.get('class', '')
        conf = float(detection.get('confidence', 0.5))
        
        # Apply pattern rules
        for pattern_name, pattern in self.threat_patterns.items():
            if obj_class in pattern['classes']:
                if conf >= pattern['min_conf']:
                    # Threat detected by rules
                    rule_confidence = conf * pattern['priority']
                    logger.debug(f"Rule-based threat: {obj_class} matches {pattern_name}")
                    return True, rule_confidence
        
        # No threat detected
        return False, conf * 0.4
    
    def learn_from_feedback(self, detection, user_confirmed_threat):
        """
        Online learning: Update model based on user authorization decision.
        This is the core learning function that improves the model over time.
        
        Args:
            detection: The detection dict with all detection info
            user_confirmed_threat: True if user authorized strike, False if dismissed
        """
        if self.classifier is None:
            logger.warning("Cannot learn: scikit-learn not available")
            return
            
        features = self.extract_features(detection)
        label = 1 if user_confirmed_threat else 0
        
        # Add to experience buffer with timestamp
        self.experience_buffer.append({
            'features': features,
            'label': label,
            'timestamp': time.time(),
            'detection': {
                'class': detection.get('class'),
                'confidence': detection.get('confidence')
            }
        })
        
        # Incremental learning (partial_fit for online learning)
        try:
            # Partial fit updates the model with this single example
            # No need to retrain on entire dataset!
            self.classifier.partial_fit(
                features, 
                [label], 
                classes=np.array([0, 1])  # Binary: 0=safe, 1=threat
            )
            
            self.is_trained = True
            
            # Add to training history
            self.training_history.append({
                'timestamp': datetime.now().isoformat(),
                'label': label,
                'class': detection.get('class')
            })
            
            # Save model periodically (every 5 examples)
            if len(self.experience_buffer) % 5 == 0:
                self.save_model()
                
            # Log learning event
            action = "THREAT ✓" if label else "SAFE ✗"
            obj_class = detection.get('class', 'unknown')
            total_exp = len(self.experience_buffer)
            
            logger.info(f"✅ Learning: {obj_class} → {action} (Total: {total_exp} examples)")
                       
        except Exception as e:
            logger.error(f"Learning error: {e}")
    
    def autonomous_decision(self, detection):
        """
        Make autonomous decision when no human operator available (jammer mode).
        Uses learned patterns with a HIGH confidence threshold for safety.
        
        This is called when:
        - Connection lost (jammer scenario)
        - Operating in autonomous mode
        
        Args:
            detection: Dict containing detection information
            
        Returns:
            bool: True = authorize strike, False = hold fire
        """
        is_threat, confidence = self.predict_threat_level(detection)
        
        # Very conservative threshold for autonomous action
        # Only authorize strikes when VERY confident
        AUTONOMOUS_THRESHOLD = 0.80  # 80% confidence required
        
        obj_class = detection.get('class', 'unknown')
        
        if is_threat and confidence > AUTONOMOUS_THRESHOLD:
            logger.warning(
                f"🤖 AUTONOMOUS AUTHORIZATION: {obj_class} "
                f"at {confidence:.1%} confidence "
                f"(threshold: {AUTONOMOUS_THRESHOLD:.1%})"
            )
            return True
        else:
            logger.info(
                f"⏸️ Autonomous HOLD: {obj_class} at {confidence:.1%} "
                f"(below {AUTONOMOUS_THRESHOLD:.1%} threshold)"
            )
            return False
    
    def get_stats(self):
        """
        Return learning statistics for display in UI.
        
        Returns:
            dict: Statistics about the learning system
        """
        total = len(self.experience_buffer)
        
        if total == 0:
            return {
                'total_detections': 0,
                'confirmed_threats': 0,
                'denied_threats': 0,
                'accuracy_estimate': 0.0,
                'is_trained': False,
                'model_age_hours': 0.0
            }
        
        # Count confirmations vs denials
        confirmed = sum(1 for exp in self.experience_buffer if exp['label'] == 1)
        denied = total - confirmed
        
        # Estimate model age
        if self.training_history:
            first_training = datetime.fromisoformat(self.training_history[0]['timestamp'])
            age_hours = (datetime.now() - first_training).total_seconds() / 3600
        else:
            age_hours = 0.0
        
        return {
            'total_detections': total,
            'confirmed_threats': confirmed,
            'denied_threats': denied,
            'accuracy_estimate': confirmed / total if total > 0 else 0.0,
            'is_trained': self.is_trained,
            'model_age_hours': age_hours
        }
    
    def save_model(self):
        """
        Persist learned model to disk for recovery after restart.
        Saves classifier, experience buffer, and training history.
        """
        try:
            model_data = {
                'classifier': self.classifier,
                'experience': list(self.experience_buffer),
                'patterns': self.threat_patterns,
                'is_trained': self.is_trained,
                'training_history': self.training_history[-100:],  # Last 100 events
                'version': '1.0',
                'timestamp': datetime.now().isoformat()
            }
            
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
                
            logger.info(f"💾 Model saved: {len(self.experience_buffer)} experiences")
            
        except Exception as e:
            logger.error(f"Failed to save model: {e}")
    
    def load_model(self):
        """
        Load previously learned model from disk.
        Allows system to retain learning across restarts.
        """
        if not os.path.exists(self.model_path):
            logger.info("No existing model found - starting fresh")
            return
            
        try:
            with open(self.model_path, 'rb') as f:
                data = pickle.load(f)
                
            self.classifier = data.get('classifier', self.classifier)
            
            # Convert experience buffer back to deque
            exp_list = data.get('experience', [])
            self.experience_buffer = deque(exp_list, maxlen=1000)
            
            self.threat_patterns = data.get('patterns', self.threat_patterns)
            self.is_trained = data.get('is_trained', False)
            self.training_history = data.get('training_history', [])
            
            model_version = data.get('version', 'unknown')
            model_timestamp = data.get('timestamp', 'unknown')
            
            logger.info(f"📥 Model loaded successfully")
            logger.info(f"   Version: {model_version}")
            logger.info(f"   Last saved: {model_timestamp}")
            logger.info(f"   Experiences: {len(self.experience_buffer)}")
            logger.info(f"   Trained: {self.is_trained}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.info("Starting with fresh model")
    
    def reset_model(self):
        """
        Reset the learning model to initial state.
        Useful for testing or starting fresh.
        """
        try:
            from sklearn.naive_bayes import GaussianNB
            self.classifier = GaussianNB()
        except ImportError:
            self.classifier = None
            
        self.is_trained = False
        self.experience_buffer.clear()
        self.training_history.clear()
        
        logger.info("🔄 Model reset to initial state")