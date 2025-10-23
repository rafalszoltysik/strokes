#!/usr/bin/env python3
"""
Stroke Prediction System - Main Entry Point
"""

import sys
import logging
from pathlib import Path

# Dodanie ścieżki do modułów
sys.path.append('src')

from src.core.system import StrokePredictionSystem
from src.core.pipeline import PipelineOrchestrator
from src.utils.logging import setup_logging

# Konfiguracja logowania
logger = setup_logging(level="INFO")

def main():
    """Główna funkcja systemu"""
    print("STROKE PREDICTION SYSTEM")
    print("="*60)
    
    # Inicjalizacja systemu
    system = StrokePredictionSystem()
    orchestrator = PipelineOrchestrator(system)
    
    # Ścieżka do danych
    data_path = "data/raw/healthcare-dataset-stroke-data.csv"
    
    if not Path(data_path).exists():
        print(f"BŁĄD: Plik danych nie istnieje: {data_path}")
        print("Upewnij się, że plik healthcare-dataset-stroke-data.csv znajduje się w katalogu data/raw/")
        return
    
    print(f"Uruchamianie pipeline'u z danymi: {data_path}")
    print("="*60)
    
    # Uruchomienie pipeline'u
    results = orchestrator.run_pipeline(data_path)
    
    if results['success']:
        print("\n" + "="*60)
        print("PIPELINE ZAKOŃCZONY POMYŚLNIE!")
        print("="*60)
        print(f"Raport: {results['report_path']}")
        print(f"Wyniki: {len(results['results'])} komponentów")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("PIPELINE ZAKOŃCZONY BŁĘDEM!")
        print("="*60)
        print(f"Błąd: {results['error']}")
        print(f"Typ błędu: {results['error_type']}")
        print("="*60)

if __name__ == "__main__":
    main()