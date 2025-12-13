#!/usr/bin/env python3

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class FabrykaModeli:
    
    @staticmethod
    def utworz_modele(konfiguracja: Dict[str, Any]) -> Dict[str, Any]:
        losowy_stan = konfiguracja.get('random_state', 42)
        liczba_estymatorow = konfiguracja.get('n_estimators', 100)
        maks_iteracji = konfiguracja.get('max_iter', 1000)
        
        modele = {
            'Random Forest': RandomForestClassifier(
                random_state=losowy_stan, 
                n_estimators=liczba_estymatorow
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                random_state=losowy_stan
            ),
            'Logistic Regression': LogisticRegression(
                random_state=losowy_stan, 
                max_iter=maks_iteracji
            ),
            'SVM': SVC(
                random_state=losowy_stan, 
                probability=True
            )
        }
        
        logger.info(f"Utworzono {len(modele)} modeli ML")
        return modele
    
    @staticmethod
    def utworz_model_kalibrowany(model_bazowy, metoda: str = 'isotonic', cv: int = 3):
        return CalibratedClassifierCV(
            model_bazowy,
            method=metoda,
            cv=cv
        )
