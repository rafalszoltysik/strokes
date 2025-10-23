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

class VisualizationAnalyzer:
    """
    Visualization and analysis for stroke prediction
    """
    
    def __init__(self, output_dir: str = "results/plots"):
        """Inicjalizacja analizatora wizualizacji"""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Konfiguracja stylu wykresów
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Słownik do przechowywania wniosków
        self.insights = {}
        
    def create_basic_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza rozkładu podstawowych cech"""
        logger.info("Tworzenie wykresów rozkładu podstawowych cech")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analiza rozkładu zmiennej docelowej i cech', fontsize=16, fontweight='bold')
        
        insights = {}
        
        # 1. Rozkład udarów
        stroke_counts = df['stroke'].value_counts()
        axes[0,0].pie(stroke_counts.values, labels=['Brak udaru', 'Udar'], 
                     autopct='%1.1f%%', startangle=90, colors=['lightblue', 'lightcoral'])
        axes[0,0].set_title('Rozkład udarów w zbiorze danych', fontweight='bold')
        
        # Wnioski z rozkładu udarów
        stroke_rate = stroke_counts[1] / len(df) * 100
        insights['stroke_distribution'] = {
            'stroke_rate': stroke_rate,
            'imbalance_level': 'high' if stroke_rate < 10 else 'moderate',
            'conclusion': f'Wysoki poziom niezbalansowania klas ({stroke_rate:.1f}% udarów)'
        }
        
        # 2. Rozkład wieku
        axes[0,1].hist(df['age'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0,1].set_title('Rozkład wieku', fontweight='bold')
        axes[0,1].set_xlabel('Wiek (lata)')
        axes[0,1].set_ylabel('Liczba pacjentów')
        axes[0,1].grid(True, alpha=0.3)
        
        # Wnioski z rozkładu wieku
        age_stats = df['age'].describe()
        insights['age_distribution'] = {
            'mean_age': age_stats['mean'],
            'age_range': f"{age_stats['min']:.1f}-{age_stats['max']:.1f}",
            'conclusion': f'Średni wiek: {age_stats["mean"]:.1f} lat, zakres: {age_stats["min"]:.1f}-{age_stats["max"]:.1f}'
        }
        
        # 3. Rozkład BMI
        bmi_data = df['bmi'].dropna()
        axes[1,0].hist(bmi_data, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
        axes[1,0].set_title('Rozkład BMI', fontweight='bold')
        axes[1,0].set_xlabel('BMI (kg/m²)')
        axes[1,0].set_ylabel('Liczba pacjentów')
        axes[1,0].grid(True, alpha=0.3)
        
        # Wnioski z rozkładu BMI
        bmi_stats = bmi_data.describe()
        insights['bmi_distribution'] = {
            'mean_bmi': bmi_stats['mean'],
            'bmi_range': f"{bmi_stats['min']:.1f}-{bmi_stats['max']:.1f}",
            'conclusion': f'Średnie BMI: {bmi_stats["mean"]:.1f}, zakres: {bmi_stats["min"]:.1f}-{bmi_stats["max"]:.1f}'
        }
        
        # 4. Rozkład poziomu glukozy
        axes[1,1].hist(df['avg_glucose_level'], bins=30, alpha=0.7, color='orange', edgecolor='black')
        axes[1,1].set_title('Rozkład średniego poziomu glukozy', fontweight='bold')
        axes[1,1].set_xlabel('Poziom glukozy (mg/dL)')
        axes[1,1].set_ylabel('Liczba pacjentów')
        axes[1,1].grid(True, alpha=0.3)
        
        # Wnioski z rozkładu glukozy
        glucose_stats = df['avg_glucose_level'].describe()
        insights['glucose_distribution'] = {
            'mean_glucose': glucose_stats['mean'],
            'glucose_range': f"{glucose_stats['min']:.1f}-{glucose_stats['max']:.1f}",
            'conclusion': f'Średni poziom glukozy: {glucose_stats["mean"]:.1f} mg/dL'
        }
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_basic_distributions.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.insights['basic_distributions'] = insights
        logger.info("Zakończono analizę rozkładu podstawowych cech")
        return insights
    
    def analyze_stroke_relationships(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza powiązań z udarem"""
        logger.info("Tworzenie wykresów analizy powiązań z udarem")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Analiza powiązań z udarem mózgu', fontsize=16, fontweight='bold')
        
        insights = {}
        
        # 1. Wiek vs udar
        df.boxplot(column='age', by='stroke', ax=axes[0,0])
        axes[0,0].set_title('Wiek vs udar', fontweight='bold')
        axes[0,0].set_xlabel('Udar (0=nie, 1=tak)')
        axes[0,0].set_ylabel('Wiek (lata)')
        axes[0,0].grid(True, alpha=0.3)
        
        # Analiza wieku
        age_stroke = df.groupby('stroke')['age'].agg(['mean', 'median', 'std'])
        age_diff = age_stroke.loc[1, 'mean'] - age_stroke.loc[0, 'mean']
        insights['age_stroke'] = {
            'mean_age_no_stroke': age_stroke.loc[0, 'mean'],
            'mean_age_stroke': age_stroke.loc[1, 'mean'],
            'age_difference': age_diff,
            'conclusion': f'Pacjenci z udarem są średnio o {age_diff:.1f} lat starsi'
        }
        
        # 2. BMI vs udar
        df.boxplot(column='bmi', by='stroke', ax=axes[0,1])
        axes[0,1].set_title('BMI vs udar', fontweight='bold')
        axes[0,1].set_xlabel('Udar (0=nie, 1=tak)')
        axes[0,1].set_ylabel('BMI (kg/m²)')
        axes[0,1].grid(True, alpha=0.3)
        
        # Analiza BMI
        bmi_stroke = df.groupby('stroke')['bmi'].agg(['mean', 'median'])
        bmi_diff = bmi_stroke.loc[1, 'mean'] - bmi_stroke.loc[0, 'mean']
        insights['bmi_stroke'] = {
            'mean_bmi_no_stroke': bmi_stroke.loc[0, 'mean'],
            'mean_bmi_stroke': bmi_stroke.loc[1, 'mean'],
            'bmi_difference': bmi_diff,
            'conclusion': f'Pacjenci z udarem mają średnio o {bmi_diff:.1f} wyższe BMI'
        }
        
        # 3. Poziom glukozy vs udar
        df.boxplot(column='avg_glucose_level', by='stroke', ax=axes[1,0])
        axes[1,0].set_title('Poziom glukozy vs udar', fontweight='bold')
        axes[1,0].set_xlabel('Udar (0=nie, 1=tak)')
        axes[1,0].set_ylabel('Poziom glukozy (mg/dL)')
        axes[1,0].grid(True, alpha=0.3)
        
        # Analiza glukozy
        glucose_stroke = df.groupby('stroke')['avg_glucose_level'].agg(['mean', 'median'])
        glucose_diff = glucose_stroke.loc[1, 'mean'] - glucose_stroke.loc[0, 'mean']
        insights['glucose_stroke'] = {
            'mean_glucose_no_stroke': glucose_stroke.loc[0, 'mean'],
            'mean_glucose_stroke': glucose_stroke.loc[1, 'mean'],
            'glucose_difference': glucose_diff,
            'conclusion': f'Pacjenci z udarem mają średnio o {glucose_diff:.1f} mg/dL wyższy poziom glukozy'
        }
        
        # 4. Nadciśnienie vs udar
        hypertension_stroke = pd.crosstab(df['hypertension'], df['stroke'])
        hypertension_stroke.plot(kind='bar', ax=axes[1,1], color=['lightblue', 'lightcoral'])
        axes[1,1].set_title('Nadciśnienie vs udar', fontweight='bold')
        axes[1,1].set_xlabel('Nadciśnienie (0=nie, 1=tak)')
        axes[1,1].set_ylabel('Liczba pacjentów')
        axes[1,1].legend(['Brak udaru', 'Udar'])
        axes[1,1].grid(True, alpha=0.3)
        
        # Analiza nadciśnienia
        hypertension_rate = hypertension_stroke.loc[1, 1] / hypertension_stroke.loc[1].sum() * 100
        insights['hypertension_stroke'] = {
            'hypertension_stroke_rate': hypertension_rate,
            'conclusion': f'Wśród pacjentów z nadciśnieniem {hypertension_rate:.1f}% miało udar'
        }
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'eda_stroke_relationships.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.insights['stroke_relationships'] = insights
        logger.info("Zakończono analizę powiązań z udarem")
        return insights
    
    def create_correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza korelacji między cechami"""
        logger.info("Tworzenie analizy korelacji")
        
        # Wybór cech numerycznych
        numeric_cols = ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease', 'stroke']
        correlation_matrix = df[numeric_cols].corr()
        
        plt.figure(figsize=(10, 8))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                    square=True, fmt='.2f', mask=mask, cbar_kws={"shrink": .8})
        plt.title('Macierz korelacji cech numerycznych', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analiza korelacji z udarem
        stroke_correlations = correlation_matrix['stroke'].sort_values(ascending=False)
        insights = {
            'top_correlations': stroke_correlations.to_dict(),
            'strongest_predictor': stroke_correlations.index[1],  # drugi po stroke (1.0)
            'strongest_correlation': stroke_correlations.iloc[1],
            'conclusion': f'Najsilniejszy predyktor udaru: {stroke_correlations.index[1]} (korelacja: {stroke_correlations.iloc[1]:.3f})'
        }
        
        self.insights['correlation_analysis'] = insights
        logger.info("Zakończono analizę korelacji")
        return insights
    
    def create_model_performance_analysis(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                        y_pred_proba: np.ndarray, model_name: str = "Model") -> Dict[str, Any]:
        """Analiza wydajności modelu"""
        logger.info(f"Tworzenie analizy wydajności modelu: {model_name}")
        
        # Macierz pomyłek
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Brak udaru', 'Udar'],
                    yticklabels=['Brak udaru', 'Udar'])
        plt.title(f'Macierz pomyłek - {model_name}', fontsize=14, fontweight='bold')
        plt.ylabel('Rzeczywiste wartości')
        plt.xlabel('Przewidywane wartości')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Krzywa ROC
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        auc_score = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=3, label=f'ROC curve (AUC = {auc_score:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.8)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f'Krzywa ROC - {model_name}', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Analiza wydajności
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        insights = {
            'confusion_matrix': cm.tolist(),
            'auc_score': auc_score,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'true_positives': int(tp),
            'false_positives': int(fp),
            'true_negatives': int(tn),
            'false_negatives': int(fn),
            'conclusion': f'Model osiągnął AUC: {auc_score:.3f}, Accuracy: {accuracy:.3f}, F1: {f1_score:.3f}'
        }
        
        self.insights['model_performance'] = insights
        logger.info("Zakończono analizę wydajności modelu")
        return insights
    
    def create_feature_importance_analysis(self, model, feature_names: List[str]) -> Dict[str, Any]:
        """Analiza ważności cech"""
        logger.info("Tworzenie analizy ważności cech")
        
        if hasattr(model, 'coef_'):
            # Dla Logistic Regression
            feature_importance = np.abs(model.coef_[0])
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': feature_importance
            }).sort_values('importance', ascending=True)
            
            plt.figure(figsize=(12, 8))
            top_features = importance_df.tail(15)
            bars = plt.barh(range(len(top_features)), top_features['importance'], 
                           color='steelblue', alpha=0.7)
            plt.yticks(range(len(top_features)), top_features['feature'])
            plt.xlabel('Ważność cechy (współczynniki)', fontsize=12)
            plt.title('Ważność cech w modelu', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3, axis='x')
            
            # Dodanie wartości na słupkach
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                       f'{width:.3f}', ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            plt.savefig(self.output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Wnioski
            top_3_features = importance_df.tail(3)
            insights = {
                'feature_importance': importance_df.to_dict('records'),
                'top_3_features': top_3_features.to_dict('records'),
                'most_important': top_3_features.iloc[-1]['feature'],
                'conclusion': f'Najważniejsza cecha: {top_3_features.iloc[-1]["feature"]} (ważność: {top_3_features.iloc[-1]["importance"]:.3f})'
            }
            
        else:
            insights = {'conclusion': 'Model nie obsługuje analizy ważności cech'}
        
        self.insights['feature_importance'] = insights
        logger.info("Zakończono analizę ważności cech")
        return insights
    
    def save_insights_report(self) -> None:
        """Zapisanie raportu z wnioskami"""
        logger.info("Zapisywanie raportu z wnioskami")
        
        report_path = self.output_dir / 'insights_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(self.insights, f, indent=2, ensure_ascii=False)
        
        # Zapisanie raportu tekstowego
        text_report_path = self.output_dir / 'insights_summary.txt'
        with open(text_report_path, 'w', encoding='utf-8') as f:
            f.write("=== RAPORT WNIOSKÓW Z ANALIZY WIZUALIZACJI ===\n\n")
            
            for analysis_type, insights in self.insights.items():
                f.write(f"=== {analysis_type.upper().replace('_', ' ')} ===\n")
                for key, value in insights.items():
                    if isinstance(value, dict) and 'conclusion' in value:
                        f.write(f"{key}: {value['conclusion']}\n")
                    elif key == 'conclusion':
                        f.write(f"{value}\n")
                f.write("\n")
        
        logger.info(f"Zapisano raport w: {report_path}")

if __name__ == "__main__":
    # Przykład użycia
    analyzer = VisualizationAnalyzer()
    # analyzer.create_basic_distributions(df)
    # analyzer.save_insights_report()
