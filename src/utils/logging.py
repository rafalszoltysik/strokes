#!/usr/bin/env python3

import logging
import sys
from pathlib import Path
from typing import Optional

def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> logging.Logger:
    """Konfiguracja systemu logowania"""
    
    # Tworzenie katalogu logs jeśli nie istnieje
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Konfiguracja formatu logowania
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Konfiguracja handlerów
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Konfiguracja głównego loggera
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Wyłączenie niepotrzebnych logów
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('seaborn').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)

def get_logger(name: str) -> logging.Logger:
    """Pobieranie loggera dla konkretnego modułu"""
    return logging.getLogger(name)
