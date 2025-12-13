#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Konfiguracja logowania
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AnalizatorWizualizacji:
    
    def __init__(self, katalog_wyjscia: str = "results/plots"):
        self.katalog_wyjscia = Path(katalog_wyjscia)
        self.katalog_wyjscia.mkdir(parents=True, exist_ok=True)
        
        # Konfiguracja stylu wykresów
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Słownik do przechowywania wniosków
        self.wnioski = {}
        
    def utworz_podstawowe_rozkłady(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Tworzenie wykresów rozkładu podstawowych cech")
        
        fig, osie = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analiza rozkładu zmiennej docelowej i cech', fontsize=16, fontweight='bold')
        
        wnioski = {}
        
        # 1. Rozkład udarów
        liczniki_udarow = df['stroke'].value_counts()
        osie[0,0].pie(liczniki_udarow.values, labels=['Brak udaru', 'Udar'], 
                     autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
        osie[0,0].set_title('Rozkład udarów w zbiorze danych', fontweight='bold')
        
        # Wnioski z rozkładu udarów
        stopien_udarow = liczniki_udarow[1] / len(df) * 100
        wnioski['stroke_distribution'] = {
            'stroke_rate': stopien_udarow,
            'imbalance_level': 'high' if stopien_udarow < 10 else 'moderate',
            'conclusion': f'Wysoki poziom niezbalansowania klas ({stopien_udarow:.1f}% udarów)'
        }
        
        # 2. Rozkład wieku
        osie[0,1].hist(df['age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        osie[0,1].set_title('Rozkład wieku', fontweight='bold')
        osie[0,1].set_xlabel('Wiek (lata)')
        osie[0,1].set_ylabel('Liczba pacjentów')
        osie[0,1].grid(True, alpha=0.3)
        
        # Wnioski z rozkładu wieku
        statystyki_wieku = df['age'].describe()
        wnioski['age_distribution'] = {
            'mean_age': statystyki_wieku['mean'],
            'age_range': f"{statystyki_wieku['min']:.1f}-{statystyki_wieku['max']:.1f}",
            'conclusion': f'Średni wiek: {statystyki_wieku["mean"]:.1f} lat, zakres: {statystyki_wieku["min"]:.1f}-{statystyki_wieku["max"]:.1f}'
        }
        
        # 3. Rozkład BMI
        dane_bmi = df['bmi'].dropna()
        osie[1,0].hist(dane_bmi, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        osie[1,0].set_title('Rozkład BMI', fontweight='bold')
        osie[1,0].set_xlabel('BMI (kg/m²)')
        osie[1,0].set_ylabel('Liczba pacjentów')
        osie[1,0].grid(True, alpha=0.3)
        
        # Wnioski z rozkładu BMI
        statystyki_bmi = dane_bmi.describe()
        wnioski['bmi_distribution'] = {
            'mean_bmi': statystyki_bmi['mean'],
            'bmi_range': f"{statystyki_bmi['min']:.1f}-{statystyki_bmi['max']:.1f}",
            'conclusion': f'Średnie BMI: {statystyki_bmi["mean"]:.1f}, zakres: {statystyki_bmi["min"]:.1f}-{statystyki_bmi["max"]:.1f}'
        }
        
        # 4. Rozkład poziomu glukozy
        osie[1,1].hist(df['avg_glucose_level'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        osie[1,1].set_title('Rozkład średniego poziomu glukozy', fontweight='bold')
        osie[1,1].set_xlabel('Poziom glukozy (mg/dL)')
        osie[1,1].set_ylabel('Liczba pacjentów')
        osie[1,1].grid(True, alpha=0.3)
        
        # Wnioski z rozkładu glukozy
        statystyki_glukozy = df['avg_glucose_level'].describe()
        wnioski['glucose_distribution'] = {
            'mean_glucose': statystyki_glukozy['mean'],
            'glucose_range': f"{statystyki_glukozy['min']:.1f}-{statystyki_glukozy['max']:.1f}",
            'conclusion': f'Średni poziom glukozy: {statystyki_glukozy["mean"]:.1f} mg/dL'
        }
        
        plt.tight_layout()
        plt.savefig(self.katalog_wyjscia / 'eda_basic_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.wnioski['basic_distributions'] = wnioski
        logger.info("Zakończono analizę rozkładu podstawowych cech")
        return wnioski
    
    def analizuj_powiazania_z_udarem(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Tworzenie wykresów analizy powiązań z udarem")
        
        fig, osie = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analiza powiązań z udarem mózgu', fontsize=16, fontweight='bold')
        
        wnioski = {}
        
        # 1. Wiek vs udar
        df.boxplot(column='age', by='stroke', ax=osie[0,0])
        osie[0,0].set_title('Wiek vs udar', fontweight='bold')
        osie[0,0].set_xlabel('Udar (0=nie, 1=tak)')
        osie[0,0].set_ylabel('Wiek (lata)')
        osie[0,0].grid(True, alpha=0.3)
        
        # Analiza wieku
        wiek_udar = df.groupby('stroke')['age'].agg(['mean', 'median', 'std'])
        roznica_wieku = wiek_udar.loc[1, 'mean'] - wiek_udar.loc[0, 'mean']
        wnioski['age_stroke'] = {
            'mean_age_no_stroke': wiek_udar.loc[0, 'mean'],
            'mean_age_stroke': wiek_udar.loc[1, 'mean'],
            'age_difference': roznica_wieku,
            'conclusion': f'Pacjenci z udarem są średnio o {roznica_wieku:.1f} lat starsi'
        }
        
        # 2. BMI vs udar
        df.boxplot(column='bmi', by='stroke', ax=osie[0,1])
        osie[0,1].set_title('BMI vs udar', fontweight='bold')
        osie[0,1].set_xlabel('Udar (0=nie, 1=tak)')
        osie[0,1].set_ylabel('BMI (kg/m²)')
        osie[0,1].grid(True, alpha=0.3)
        
        # Analiza BMI
        bmi_udar = df.groupby('stroke')['bmi'].agg(['mean', 'median'])
        roznica_bmi = bmi_udar.loc[1, 'mean'] - bmi_udar.loc[0, 'mean']
        wnioski['bmi_stroke'] = {
            'mean_bmi_no_stroke': bmi_udar.loc[0, 'mean'],
            'mean_bmi_stroke': bmi_udar.loc[1, 'mean'],
            'bmi_difference': roznica_bmi,
            'conclusion': f'Pacjenci z udarem mają średnio o {roznica_bmi:.1f} wyższe BMI'
        }
        
        # 3. Poziom glukozy vs udar
        df.boxplot(column='avg_glucose_level', by='stroke', ax=osie[1,0])
        osie[1,0].set_title('Poziom glukozy vs udar', fontweight='bold')
        osie[1,0].set_xlabel('Udar (0=nie, 1=tak)')
        osie[1,0].set_ylabel('Poziom glukozy (mg/dL)')
        osie[1,0].grid(True, alpha=0.3)
        
        # Analiza glukozy
        glukoza_udar = df.groupby('stroke')['avg_glucose_level'].agg(['mean', 'median'])
        roznica_glukozy = glukoza_udar.loc[1, 'mean'] - glukoza_udar.loc[0, 'mean']
        wnioski['glucose_stroke'] = {
            'mean_glucose_no_stroke': glukoza_udar.loc[0, 'mean'],
            'mean_glucose_stroke': glukoza_udar.loc[1, 'mean'],
            'glucose_difference': roznica_glukozy,
            'conclusion': f'Pacjenci z udarem mają średnio o {roznica_glukozy:.1f} mg/dL wyższy poziom glukozy'
        }
        
        # 4. Nadciśnienie vs udar
        nadcisnienie_udar = pd.crosstab(df['hypertension'], df['stroke'])
        nadcisnienie_udar.plot(kind='bar', ax=osie[1,1], color=['lightblue', 'lightcoral'])
        osie[1,1].set_title('Nadciśnienie vs udar', fontweight='bold')
        osie[1,1].set_xlabel('Nadciśnienie (0=nie, 1=tak)')
        osie[1,1].set_ylabel('Liczba pacjentów')
        osie[1,1].legend(['Brak udaru', 'Udar'])
        osie[1,1].grid(True, alpha=0.3)
        
        # Analiza nadciśnienia
        stopien_nadcisnienia = nadcisnienie_udar.loc[1, 1] / nadcisnienie_udar.loc[1].sum() * 100
        wnioski['hypertension_stroke'] = {
            'hypertension_stroke_rate': stopien_nadcisnienia,
            'conclusion': f'Wśród pacjentów z nadciśnieniem {stopien_nadcisnienia:.1f}% miało udar'
        }
        
        plt.tight_layout()
        plt.savefig(self.katalog_wyjscia / 'eda_stroke_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.wnioski['stroke_relationships'] = wnioski
        logger.info("Zakończono analizę powiązań z udarem")
        return wnioski
    
    def utworz_analize_korelacji(self, df: pd.DataFrame) -> Dict[str, Any]:
        logger.info("Tworzenie analizy korelacji")
        
        # Wybór cech numerycznych
        kolumny_numeryczne = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'stroke']
        macierz_korelacji = df[kolumny_numeryczne].corr()
        
        plt.figure(figsize=(10, 8))
        maska = np.triu(np.ones_like(macierz_korelacji, dtype=bool))
        sns.heatmap(macierz_korelacji, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f', mask=maska, cbar_kws={"shrink": .8})
        plt.title('Macierz korelacji cech numerycznych', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.katalog_wyjscia / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analiza korelacji z udarem
        korelacje_udaru = macierz_korelacji['stroke'].sort_values(ascending=False)
        wnioski = {
            'top_correlations': korelacje_udaru.to_dict(),
            'strongest_predictor': korelacje_udaru.index[1],  # drugi po stroke (1.0)
            'strongest_correlation': korelacje_udaru.iloc[1],
            'conclusion': f'Najsilniejszy predyktor udaru: {korelacje_udaru.index[1]} (korelacja: {korelacje_udaru.iloc[1]:.3f})'
        }
        
        self.wnioski['correlation_analysis'] = wnioski
        logger.info("Zakończono analizę korelacji")
        return wnioski
    
    def utworz_analize_wydajnosci_modelu(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_pred_proba: np.ndarray, nazwa_modelu: str = "Model") -> Dict[str, Any]:
        logger.info(f"Tworzenie analizy wydajności modelu: {nazwa_modelu}")
        
        # Macierz pomyłek
        macierz_pomylek = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(macierz_pomylek, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Brak udaru', 'Udar'],
                    yticklabels=['Brak udaru', 'Udar'])
        plt.title(f'Macierz pomyłek - {nazwa_modelu}', fontsize=14, fontweight='bold')
        plt.ylabel('Rzeczywiste wartości')
        plt.xlabel('Przewidywane wartości')
        plt.tight_layout()
        plt.savefig(self.katalog_wyjscia / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Krzywa ROC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        wynik_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'Krzywa ROC (AUC = {wynik_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Wskaźnik Fałszywie Pozytywnych', fontsize=12)
        plt.ylabel('Wskaźnik Prawdziwie Pozytywnych', fontsize=12)
        plt.title(f'Krzywa ROC - {nazwa_modelu}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.katalog_wyjscia / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analiza wydajności
        tn, fp, fn, tp = macierz_pomylek.ravel()
        dokladnosc = (tp + tn) / (tp + tn + fp + fn)
        precyzja = tp / (tp + fp) if (tp + fp) > 0 else 0
        czulosc = tp / (tp + fn) if (tp + fn) > 0 else 0
        wynik_f1 = 2 * (precyzja * czulosc) / (precyzja + czulosc) if (precyzja + czulosc) > 0 else 0
        
        wnioski = {
            'confusion_matrix': macierz_pomylek.tolist(),
            'auc_score': wynik_auc,
            'accuracy': dokladnosc,
            'precision': precyzja,
            'recall': czulosc,
            'f1_score': wynik_f1,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'conclusion': f'Model osiągnął AUC: {wynik_auc:.3f}, Dokładność: {dokladnosc:.3f}, F1: {wynik_f1:.3f}'
        }
        
        self.wnioski['model_performance'] = wnioski
        logger.info("Zakończono analizę wydajności modelu")
        return wnioski
    
    def utworz_analize_waznosci_cech(self, model, nazwy_cech: List[str]) -> Dict[str, Any]:
        logger.info("Tworzenie analizy ważności cech")
        
        if hasattr(model, 'coef_'):
            # Dla Logistic Regression
            waznosc_cech = np.abs(model.coef_[0])
            df_waznosci = pd.DataFrame({
                'feature': nazwy_cech,
                'importance': waznosc_cech
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(12, 8))
            top_cechy = df_waznosci.tail(15)
            slupki = plt.barh(range(len(top_cechy)), top_cechy['importance'], 
                           color='steelblue', alpha=0.7)
            plt.yticks(range(len(top_cechy)), top_cechy['feature'])
            plt.xlabel('Ważność cechy (współczynniki)', fontsize=12)
            plt.title('Ważność cech w modelu', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')
            
            # Dodanie wartości na słupkach
            for i, slupek in enumerate(slupki):
                szerokosc = slupek.get_width()
                plt.text(szerokosc + 0.01, slupek.get_y() + slupek.get_height()/2, 
                       f'{szerokosc:.3f}', ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(self.katalog_wyjscia / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Wnioski
            top_3_cechy = df_waznosci.tail(3)
            wnioski = {
                'feature_importance': df_waznosci.to_dict('records'),
                'top_3_features': top_3_cechy.to_dict('records'),
                'most_important': top_3_cechy.iloc[-1]['feature'],
                'conclusion': f'Najważniejsza cecha: {top_3_cechy.iloc[-1]["feature"]} (ważność: {top_3_cechy.iloc[-1]["importance"]:.3f})'
            }
            
        else:
            wnioski = {'conclusion': 'Model nie obsługuje analizy ważności cech'}
        
        self.wnioski['feature_importance'] = wnioski
        logger.info("Zakończono analizę ważności cech")
        return wnioski
    
    def zapisz_raport_wnioskow(self) -> None:
        logger.info("Zapisywanie raportu z wnioskami")
        
        sciezka_raportu = self.katalog_wyjscia / 'insights_report.json'
        with open(sciezka_raportu, 'w', encoding='utf-8') as plik:
            json.dump(self.wnioski, plik, indent=2, ensure_ascii=False)
        
        # Zapisanie raportu tekstowego
        sciezka_raportu_tekstowego = self.katalog_wyjscia / 'insights_summary.txt'
        with open(sciezka_raportu_tekstowego, 'w', encoding='utf-8') as plik:
            plik.write("=== RAPORT WNIOSKÓW Z ANALIZY WIZUALIZACJI ===\n\n")
            
            for typ_analizy, wnioski in self.wnioski.items():
                plik.write(f"=== {typ_analizy.upper().replace('_', ' ')} ===\n")
                for klucz, wartosc in wnioski.items():
                    if isinstance(wartosc, dict) and 'conclusion' in wartosc:
                        plik.write(f"{klucz}: {wartosc['conclusion']}\n")
                    elif klucz == 'conclusion':
                        plik.write(f"{wartosc}\n")
                plik.write("\n")
        
        logger.info(f"Zapisano raport w: {sciezka_raportu}")

if __name__ == "__main__":
    # Przykład użycia
    analizator = AnalizatorWizualizacji()
    # analizator.utworz_podstawowe_rozkłady(df)
    # analizator.zapisz_raport_wnioskow()
