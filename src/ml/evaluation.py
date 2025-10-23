#!/usr/bin/env python3

import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_score, recall_score, f1_score, precision_recall_curve
)

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Klasa do oceny modeli ML"""
    
    def __init__(self):
        """Inicjalizacja ewaluatora"""
        self.evaluation_results = {}
    
    def evaluate_models(self, models: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Ocena wszystkich modeli"""
        logger.info("=== OCENA MODELI ===")
        
        evaluation_results = {}
        
        for name, model in models.items():
            logger.info(f"Ocena modelu: {name}")
            
            try:
                # Predykcje
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Metryki
                metrics = self._calculate_metrics(y_test, y_pred, y_pred_proba)
                
                evaluation_results[name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    **metrics
                }
                
                logger.info(f"Model {name}: AUC={metrics['auc_score']:.4f}, Accuracy={metrics['accuracy']:.4f}")
                
            except Exception as e:
                logger.error(f"Błąd oceny modelu {name}: {e}")
                raise
        
        self.evaluation_results = evaluation_results
        logger.info("Zakończono ocenę modeli")
        return evaluation_results
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Obliczenie metryk wydajności"""
        return {
            'accuracy': (y_pred == y_true).mean(),
            'auc_score': roc_auc_score(y_true, y_pred_proba),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
    
    def optimize_classification_threshold(self, X_test: np.ndarray, y_test: np.ndarray, 
                                        model, threshold_range: Tuple[float, float] = (0.1, 0.9)) -> Dict[str, Any]:
        """Optymalizacja progu klasyfikacji"""
        logger.info("=== OPTYMALIZACJA PROGU KLASYFIKACJI ===")
        
        # Predykcje prawdopodobieństw
        if model is not None:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError("Model nie został wytrenowany")
        
        # Optymalizacja progu na podstawie F1-score
        thresholds = np.linspace(threshold_range[0], threshold_range[1], 50)
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_pred_proba >= threshold).astype(int)
            f1 = f1_score(y_test, y_pred_thresh, average='weighted')
            f1_scores.append(f1)
        
        # Znalezienie optymalnego progu
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        # Obliczenie metryk dla optymalnego progu
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        optimal_metrics = self._calculate_metrics(y_test, y_pred_optimal, y_pred_proba)
        
        threshold_metrics = {
            'optimal_threshold': optimal_threshold,
            'optimal_f1_score': optimal_f1,
            'thresholds': thresholds.tolist(),
            'f1_scores': f1_scores,
            'metrics_at_optimal': optimal_metrics
        }
        
        logger.info(f"Optymalny próg: {optimal_threshold:.4f} (F1: {optimal_f1:.4f})")
        return threshold_metrics
