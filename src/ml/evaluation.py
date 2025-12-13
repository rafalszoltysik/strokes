#!/usr/bin/env python3

import numpy as np
import logging
from typing import Dict, Any, Tuple
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_score, recall_score, f1_score, precision_recall_curve
)

logger = logging.getLogger(__name__)

class EwaluatorModeli:
    
    def __init__(self):
        self.wyniki_oceny = {}
    
    def ocen_modele(self, modele: Dict[str, Any], X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        logger.info("=== OCENA MODELI ===")
        
        wyniki_oceny = {}
        
        for nazwa, model in modele.items():
            logger.info(f"Ocena modelu: {nazwa}")
            
            try:
                # Predykcje
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Metryki
                metryki = self._oblicz_metryki(y_test, y_pred, y_pred_proba)
                
                wyniki_oceny[nazwa] = {
                    'model': model,
                    'predykcje': y_pred,
                    'prawdopodobienstwa': y_pred_proba,
                    **metryki
                }
                
                logger.info(f"Model {nazwa}: AUC={metryki['wynik_auc']:.4f}, Dokladnosc={metryki['dokladnosc']:.4f}")
                
            except Exception as e:
                logger.error(f"Błąd oceny modelu {nazwa}: {e}")
                raise
        
        self.wyniki_oceny = wyniki_oceny
        logger.info("Zakończono ocenę modeli")
        return wyniki_oceny
    
    def _oblicz_metryki(self, y_true: np.ndarray, y_pred: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        return {
            'dokladnosc': (y_pred == y_true).mean(),
            'wynik_auc': roc_auc_score(y_true, y_pred_proba),
            'precyzja': precision_score(y_true, y_pred, average='weighted'),
            'czulosc': recall_score(y_true, y_pred, average='weighted'),
            'wynik_f1': f1_score(y_true, y_pred, average='weighted')
        }
    
    def optymalizuj_próg_klasyfikacji(self, X_test: np.ndarray, y_test: np.ndarray, 
                                        model, zakres_progu: Tuple[float, float] = (0.1, 0.9)) -> Dict[str, Any]:
        logger.info("=== OPTYMALIZACJA PROGU KLASYFIKACJI ===")
        
        # Predykcje prawdopodobieństw
        if model is not None:
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            raise ValueError("Model nie został wytrenowany")
        
        # Optymalizacja progu na podstawie F1-score
        progi = np.linspace(zakres_progu[0], zakres_progu[1], 50)
        wyniki_f1 = []
        
        for prog in progi:
            y_pred_prog = (y_pred_proba >= prog).astype(int)
            wynik_f1 = f1_score(y_test, y_pred_prog, average='weighted')
            wyniki_f1.append(wynik_f1)
        
        # Znalezienie optymalnego progu
        indeks_optymalny = np.argmax(wyniki_f1)
        prog_optymalny = progi[indeks_optymalny]
        wynik_f1_optymalny = wyniki_f1[indeks_optymalny]
        
        # Obliczenie metryk dla optymalnego progu
        y_pred_optimal = (y_pred_proba >= prog_optymalny).astype(int)
        metryki_optymalne = self._oblicz_metryki(y_test, y_pred_optimal, y_pred_proba)
        
        metryki_progu = {
            'prog_optymalny': prog_optymalny,
            'wynik_f1_optymalny': wynik_f1_optymalny,
            'progi': progi.tolist(),
            'wyniki_f1': wyniki_f1,
            'metryki_przy_optymalnym': metryki_optymalne
        }
        
        logger.info(f"Optymalny próg: {prog_optymalny:.4f} (F1: {wynik_f1_optymalny:.4f})")
        return metryki_progu
