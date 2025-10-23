#!/usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ModelFactory:
    """Factory do tworzenia modeli ML"""
    
    @staticmethod
    def create_models(config: Dict[str, Any]) -> Dict[str, Any]:
        """Tworzenie słownika modeli na podstawie konfiguracji"""
        random_state = config.get('random_state', 42)
        n_estimators = config.get('n_estimators', 100)
        max_iter = config.get('max_iter', 1000)
        
        models = {
            'Random Forest': RandomForestClassifier(
                random_state=random_state, 
                n_estimators=n_estimators
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=random_state
            ),
            'Logistic Regression': LogisticRegression(
                random_state=random_state, 
                max_iter=max_iter
            ),
            'SVM': SVC(
                random_state=random_state, 
                probability=True
            )
        }
        
        logger.info(f"Utworzono {len(models)} modeli ML")
        return models
    
    @staticmethod
    def create_calibrated_model(base_model, method: str = 'isotonic', cv: int = 3):
        """Tworzenie kalibrowanego modelu"""
        return CalibratedClassifierCV(
            base_model,
            method=method,
            cv=cv
        )
