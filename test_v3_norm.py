import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import pickle
import os
import time
import warnings
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import joblib
import itertools
from mpl_toolkits.mplot3d import Axes3D

# Отключение предупреждений для более чистого вывода
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

class NoiseInjector:
    """Класс для добавления различных типов шума в данные"""
    
    @staticmethod
    def add_gaussian_noise(X, intensity):
        """Добавляет гауссовский шум к данным
        
        Args:
            X: Исходные данные
            intensity: Интенсивность шума (стандартное отклонение)
            
        Returns:
            X_noisy: Данные с добавленным шумом
        """
        noise = np.random.normal(0, intensity, X.shape)
        X_noisy = X + noise
        return X_noisy
    
    @staticmethod
    def add_uniform_noise(X, intensity):
        """Добавляет равномерный шум к данным
        
        Args:
            X: Исходные данные
            intensity: Интенсивность шума (максимальная амплитуда)
            
        Returns:
            X_noisy: Данные с добавленным шумом
        """
        noise = np.random.uniform(-intensity, intensity, X.shape)
        X_noisy = X + noise
        return X_noisy
    
    @staticmethod
    def add_impulse_noise(X, intensity):
        """Добавляет импульсный шум к данным (случайные выбросы)
        
        Args:
            X: Исходные данные
            intensity: Интенсивность шума (вероятность выброса)
            
        Returns:
            X_noisy: Данные с добавленным шумом
        """
        X_noisy = X.copy()
        mask = np.random.random(X.shape) < intensity
        
        # Создаем импульсы с крайними значениями
        impulses = np.random.choice([-5, 5], size=X.shape)
        X_noisy[mask] = impulses[mask]
        
        return X_noisy
    
    @staticmethod
    def add_missing_values(X, intensity):
        """Добавляет пропущенные значения к данным
        
        Args:
            X: Исходные данные
            intensity: Интенсивность шума (вероятность пропуска)
            
        Returns:
            X_noisy: Данные с добавленным шумом
        """
        X_noisy = X.copy()
        mask = np.random.random(X.shape) < intensity
        X_noisy[mask] = np.nan
        
        return X_noisy

class ModelBuilder:
    """Класс для построения и оптимизации моделей классификации"""
    
    def __init__(self):
        """Инициализирует построитель моделей"""
        self.models = {}
        self.best_params = {}
        self.feature_scaler = StandardScaler()
        
    def build_main_neural_network(self, input_shape, num_classes, hyperparams=None):
        """Строит основную нейронную сеть с заданными гиперпараметрами
        
        Args:
            input_shape: Размерность входных данных
            num_classes: Количество классов
            hyperparams: Словарь с гиперпараметрами (если None, используются значения по умолчанию)
            
        Returns:
            model: Скомпилированная модель нейронной сети
        """
        if hyperparams is None:
            # Значения по умолчанию
            hyperparams = {
                'units_1': 128,
                'units_2': 64,
                'units_3': 32,
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'l2_reg': 0.001,
                'batch_size': 32
            }
        
        # Создаем модель с оптимизированной архитектурой
        inputs = Input(shape=input_shape)
        
        # Первый блок
        x = Dense(hyperparams['units_1'], activation='relu', 
                  kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(inputs)
        x = BatchNormalization()(x)
        x = Dropout(hyperparams['dropout_rate'])(x)
        
        # Второй блок
        x = Dense(hyperparams['units_2'], activation='relu', 
                  kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(x)
        x = BatchNormalization()(x)
        x = Dropout(hyperparams['dropout_rate'] * 0.8)(x)
        
        # Третий блок
        x = Dense(hyperparams['units_3'], activation='relu', 
                  kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(x)
        x = BatchNormalization()(x)
        x = Dropout(hyperparams['dropout_rate'] * 0.5)(x)
        
        # Выходной слой
        if num_classes == 2:
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(num_classes, activation='softmax')(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        # Компиляция модели
        optimizer = Adam(learning_rate=hyperparams['learning_rate'])
        
        if num_classes == 2:
            model.compile(optimizer=optimizer,
                          loss='binary_crossentropy',
                          metrics=['accuracy'])
        else:
            model.compile(optimizer=optimizer,
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
            
        return model
    
    def optimize_neural_network(self, X_train, y_train, X_val, y_val, input_shape, num_classes):
        """Оптимизирует гиперпараметры нейронной сети с помощью Optuna
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            X_val: Валидационные данные
            y_val: Валидационные метки
            input_shape: Размерность входных данных
            num_classes: Количество классов
            
        Returns:
            best_params: Лучшие найденные гиперпараметры
        """
        def objective(trial):
            # Определяем пространство поиска гиперпараметров
            hyperparams = {
                'units_1': trial.suggest_int('units_1', 32, 256),
                'units_2': trial.suggest_int('units_2', 16, 128),
                'units_3': trial.suggest_int('units_3', 8, 64),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'l2_reg': trial.suggest_float('l2_reg', 1e-4, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128])
            }
            
            # Подготовка данных
            if num_classes > 2:
                y_train_cat = to_categorical(y_train)
                y_val_cat = to_categorical(y_val)
            else:
                y_train_cat = y_train
                y_val_cat = y_val
            
            # Строим модель с текущими гиперпараметрами
            model = self.build_main_neural_network(input_shape, num_classes, hyperparams)
            
            # Обучаем модель
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
            
            history = model.fit(
                X_train, y_train_cat,
                epochs=50,
                batch_size=hyperparams['batch_size'],
                validation_data=(X_val, y_val_cat),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Оцениваем модель
            val_loss = min(history.history['val_loss'])
            
            return val_loss
        
        # Создаем исследование Optuna
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=10)  # Можно увеличить для лучших результатов
        
        print("Оптимизация нейронной сети завершена:")
        print(f"Лучшие гиперпараметры: {study.best_params}")
        print(f"Лучшее значение целевой функции: {study.best_value:.4f}")
        
        return study.best_params
    
    def optimize_support_models(self, X_train, y_train):
        """Оптимизирует гиперпараметры вспомогательных моделей
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            
        Returns:
            best_params: Словарь с лучшими параметрами для каждой модели
        """
        # Определяем пространства поиска для каждой модели
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10]
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7]
            },
            'svm': {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.1],
                'kernel': ['rbf', 'poly']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'p': [1, 2]
            }
        }
        
        # Модели для оптимизации
        base_models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier()
        }
        
        best_params = {}
        
        # Оптимизируем каждую модель
        for name, model in base_models.items():
            print(f"\nОптимизация модели {name}...")
            
            # Создаем объект GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=3,
                scoring='accuracy',
                n_jobs=-1,
                verbose=0
            )
            
            # Обучаем на данных
            grid_search.fit(X_train, y_train)
            
            # Сохраняем лучшие параметры
            best_params[name] = grid_search.best_params_
            print(f"Лучшие параметры для {name}: {grid_search.best_params_}")
            print(f"Лучшая точность при кросс-валидации: {grid_search.best_score_:.4f}")
        
        return best_params

    def build_ensemble_model(self, input_shape, num_classes, nn_params, support_params):
        """Строит ансамблевую модель с основной нейронной сетью и вспомогательными алгоритмами
        
        Args:
            input_shape: Размерность входных данных
            num_classes: Количество классов
            nn_params: Гиперпараметры нейронной сети
            support_params: Гиперпараметры вспомогательных моделей
            
        Returns:
            ensemble: Ансамблевая модель
        """
        # Создаем основную нейронную сеть
        main_nn = self.build_main_neural_network(input_shape, num_classes, nn_params)
        
        # Создаем вспомогательные модели с оптимизированными параметрами
        rf_params = support_params['random_forest']
        gb_params = support_params['gradient_boosting']
        svm_params = support_params['svm']
        knn_params = support_params['knn']
        
        rf_model = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_split=rf_params['min_samples_split'],
            random_state=42
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=gb_params['n_estimators'],
            learning_rate=gb_params['learning_rate'],
            max_depth=gb_params['max_depth'],
            random_state=42
        )
        
        svm_model = SVC(
            C=svm_params['C'],
            gamma=svm_params['gamma'],
            kernel=svm_params['kernel'],
            probability=True,
            random_state=42
        )
        
        knn_model = KNeighborsClassifier(
            n_neighbors=knn_params['n_neighbors'],
            weights=knn_params['weights'],
            p=knn_params['p']
        )
        
        # Сохраняем модели в словаре
        self.models = {
            'main_nn': main_nn,
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'svm': svm_model,
            'knn': knn_model
        }
        
        self.best_params = {
            'nn_params': nn_params,
            'support_params': support_params
        }
        
        return self.models

    class AdaptiveEnsemble:
        """Класс для адаптивного ансамбля моделей"""
        
        def __init__(self, models, confidence_threshold=0.7):
            """Инициализирует адаптивный ансамбль
            
            Args:
                models: Словарь с моделями
                confidence_threshold: Порог уверенности для основной модели
            """
            self.models = models
            self.confidence_threshold = confidence_threshold
            self.support_weights = {
                'random_forest': 0.3,
                'gradient_boosting': 0.3,
                'svm': 0.2,
                'knn': 0.2
            }
            
        def predict(self, X):
            """Делает предсказания с использованием адаптивного ансамбля
            
            Args:
                X: Данные для предсказания
                
            Returns:
                predictions: Предсказанные метки классов
            """
            # Получаем предсказания основной нейронной сети
            main_nn = self.models['main_nn']
            
            # Проверяем формат выхода (бинарная или многоклассовая классификация)
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                nn_probs = main_nn.predict(X)
                nn_conf = np.maximum(nn_probs, 1 - nn_probs)  # Уверенность
                nn_preds = (nn_probs > 0.5).astype(int)
            else:  # Многоклассовая классификация
                nn_probs = main_nn.predict(X)
                nn_conf = np.max(nn_probs, axis=1)  # Уверенность
                nn_preds = np.argmax(nn_probs, axis=1)
            
            # Находим примеры с низкой уверенностью
            low_conf_mask = nn_conf < self.confidence_threshold
            
            # Если все предсказания уверенные, возвращаем их
            if not np.any(low_conf_mask):
                return nn_preds
            
            # Для неуверенных примеров запускаем вспомогательные модели
            X_low_conf = X[low_conf_mask]
            
            # Получаем предсказания вспомогательных моделей
            support_preds = {}
            for name, model in self.models.items():
                if name == 'main_nn':
                    continue
                
                # Предсказания и вероятности
                if hasattr(model, 'predict_proba'):
                    probs = model.predict_proba(X_low_conf)
                    # Для бинарной классификации убедимся, что у нас двумерный массив
                    if probs.shape[1] == 2:
                        support_preds[name] = probs[:, 1]
                    else:
                        support_preds[name] = probs
                else:
                    # Для моделей без predict_proba используем решающую функцию
                    support_preds[name] = model.decision_function(X_low_conf)
            
            # Комбинируем предсказания вспомогательных моделей
            final_preds = nn_preds.copy()
            
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                weighted_support_preds = np.zeros(X_low_conf.shape[0])
                
                for name, preds in support_preds.items():
                    weighted_support_preds += self.support_weights[name] * preds
                
                ensemble_preds = (weighted_support_preds > 0.5).astype(int)
                final_preds[low_conf_mask] = ensemble_preds
                
            else:  # Многоклассовая классификация
                num_classes = main_nn.output_shape[-1]
                weighted_support_probs = np.zeros((X_low_conf.shape[0], num_classes))
                
                for name, probs in support_preds.items():
                    if len(probs.shape) == 1:
                        # Если модель выдала одномерный массив, преобразуем его
                        one_hot = np.zeros((probs.shape[0], num_classes))
                        one_hot[np.arange(probs.shape[0]), probs.astype(int)] = 1
                        weighted_support_probs += self.support_weights[name] * one_hot
                    else:
                        weighted_support_probs += self.support_weights[name] * probs
                
                ensemble_preds = np.argmax(weighted_support_probs, axis=1)
                final_preds[low_conf_mask] = ensemble_preds
            
            return final_preds
        
        def predict_proba(self, X):
            """Предсказывает вероятности классов
            
            Args:
                X: Данные для предсказания
                
            Returns:
                probabilities: Предсказанные вероятности классов
            """
            # Получаем вероятности от основной нейронной сети
            main_nn = self.models['main_nn']
            
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                nn_probs = main_nn.predict(X)
                # Преобразуем в формат [P(class=0), P(class=1)]
                return np.column_stack([1 - nn_probs, nn_probs])
            else:  # Многоклассовая классификация
                return main_nn.predict(X)
            
        def evaluate(self, X, y):
            """Оценивает производительность ансамбля
            
            Args:
                X: Тестовые данные
                y: Истинные метки
                
            Returns:
                metrics: Словарь с метриками производительности
            """
            # Делаем предсказания
            y_pred = self.predict(X)
            
            # Вычисляем метрики
            accuracy = accuracy_score(y, y_pred)
            report = classification_report(y, y_pred, output_dict=True)
            
            # Оцениваем производительность отдельных моделей
            models_accuracy = {}
            for name, model in self.models.items():
                if name == 'main_nn':
                    if model.output_shape[-1] == 1:  # Бинарная классификация
                        probs = model.predict(X)
                        preds = (probs > 0.5).astype(int)
                    else:  # Многоклассовая классификация
                        probs = model.predict(X)
                        preds = np.argmax(probs, axis=1)
                else:
                    preds = model.predict(X)
                
                models_accuracy[name] = accuracy_score(y, preds)
            
            # Возвращаем метрики
            return {
                'accuracy': accuracy,
                'report': report,
                'models_accuracy': models_accuracy
            }

class ExperimentRunner:
    """Класс для проведения экспериментов с моделями классификации на зашумленных данных"""
    
    def __init__(self, dataset_name=None, dataset_path=None):
        """Инициализирует средство проведения экспериментов
        
        Args:
            dataset_name: Название набора данных из sklearn (если используется встроенный набор)
            dataset_path: Путь к файлу с набором данных (если используется внешний набор)
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.noise_injector = NoiseInjector()
        self.model_builder = ModelBuilder()
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        self.scaler = StandardScaler()
        self.experiment_results = {}
        
    def load_dataset(self, dataset_name=None, dataset_path=None):
        """Загружает набор данных
        
        Args:
            dataset_name: Название набора данных из sklearn (если используется встроенный набор)
            dataset_path: Путь к файлу с набором данных (если используется внешний набор)
            
        Returns:
            X: Признаки
            y: Метки классов
        """
        if dataset_name is not None:
            self.dataset_name = dataset_name
        if dataset_path is not None:
            self.dataset_path = dataset_path
            
        # Загрузка встроенных наборов данных
        if self.dataset_name == 'iris':
            data = load_iris()
            self.X = data.data
            self.y = data.target
            self.feature_names = data.feature_names
            self.target_names = data.target_names
            print(f"Загружен набор данных Iris: {self.X.shape[0]} образцов, {self.X.shape[1]} признаков, {len(np.unique(self.y))} классов")
            
        elif self.dataset_name == 'wine':
            data = load_wine()
            self.X = data.data
            self.y = data.target
            self.feature_names = data.feature_names
            self.target_names = data.target_names
            print(f"Загружен набор данных Wine: {self.X.shape[0]} образцов, {self.X.shape[1]} признаков, {len(np.unique(self.y))} классов")
            
        elif self.dataset_name == 'breast_cancer':
            data = load_breast_cancer()
            self.X = data.data
            self.y = data.target
            self.feature_names = data.feature_names
            self.target_names = data.target_names
            print(f"Загружен набор данных Breast Cancer: {self.X.shape[0]} образцов, {self.X.shape[1]} признаков, {len(np.unique(self.y))} классов")
            
        elif self.dataset_name == 'digits':
            data = fetch_openml('mnist_784', version=1, parser='auto')
            # Для ускорения используем только часть набора данных MNIST
            n_samples = 5000
            self.X = data.data[:n_samples].astype(float).values
            self.y = data.target[:n_samples].astype(int).values
            self.feature_names = [f"pixel_{i}" for i in range(self.X.shape[1])]
            self.target_names = [str(i) for i in range(10)]
            print(f"Загружен набор данных MNIST (подвыборка): {self.X.shape[0]} образцов, {self.X.shape[1]} признаков, {len(np.unique(self.y))} классов")
            
        # Загрузка внешнего набора данных
        elif self.dataset_path is not None:
            if self.dataset_path.endswith('.csv'):
                df = pd.read_csv(self.dataset_path)
                
                # Предполагаем, что последний столбец - это метки классов
                X = df.iloc[:, :-1].values
                y = df.iloc[:, -1].values
                
                self.X = X
                self.y = y
                self.feature_names = df.columns[:-1].tolist()
                self.target_names = [str(label) for label in np.unique(y)]
                
                print(f"Загружен пользовательский набор данных: {self.X.shape[0]} образцов, {self.X.shape[1]} признаков, {len(np.unique(self.y))} классов")
            else:
                raise ValueError("Поддерживаются только файлы CSV")
                
        else:
            raise ValueError("Необходимо указать название набора данных или путь к файлу")
        
        return self.X, self.y
    
    def run_experiment(self, noise_type, noise_range, noise_step, n_experiments=3):
        """Проводит эксперимент с заданным типом и уровнем шума
        
        Args:
            noise_type: Тип шума ('gaussian', 'uniform', 'impulse', 'missing')
            noise_range: Диапазон уровня шума (min, max)
            noise_step: Шаг изменения уровня шума
            n_experiments: Количество экспериментов для усреднения результатов
            
        Returns:
            results: Словарь с результатами экспериментов
        """
        if self.X is None or self.y is None:
            raise ValueError("Набор данных не загружен")
        
        # Словарь для хранения результатов
        results = {
            'noise_levels': [],
            'ensemble_accuracy': [],
            'nn_accuracy': [],
            'rf_accuracy': [],
            'gb_accuracy': [],
            'svm_accuracy': [],
            'knn_accuracy': []
        }
        
        min_noise, max_noise = noise_range
        noise_levels = np.arange(min_noise, max_noise + noise_step, noise_step)
        
        # Предварительная обработка данных
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Количество классов
        num_classes = len(np.unique(self.y))
        input_shape = (self.X.shape[1],)
        
        # Разбиение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.25, random_state=42, stratify=y_train
        )
        
        print(f"\nПроводим эксперимент с шумом типа {noise_type}...")
        print(f"Диапазон шума: [{min_noise}, {max_noise}], шаг: {noise_step}")
        print(f"Количество экспериментов для усреднения: {n_experiments}")
        
        # Оптимизация гиперпараметров основной нейронной сети
        print("\nОптимизация гиперпараметров основной нейронной сети...")
        nn_params = self.model_builder.optimize_neural_network(
            X_train, y_train, X_val, y_val, input_shape, num_classes
        )
        
        # Оптимизация гиперпараметров вспомогательных моделей
        print("\nОптимизация гиперпараметров вспомогательных моделей...")
        support_params = self.model_builder.optimize_support_models(X_train, y_train)
        
        # Создаем ансамблевую модель
        print("\nСоздание ансамблевой модели...")
        models = self.model_builder.build_ensemble_model(
            input_shape, num_classes, nn_params, support_params
        )
        
        # Обучаем основную нейронную сеть
        print("\nОбучение основной нейронной сети...")
        if num_classes > 2:
            y_train_cat = to_categorical(y_train)
            y_val_cat = to_categorical(y_val)
        else:
            y_train_cat = y_train
            y_val_cat = y_val
            
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6
        )
        
        models['main_nn'].fit(
            X_train, y_train_cat,
            epochs=50,
            batch_size=nn_params['batch_size'],
            validation_data=(X_val, y_val_cat),
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Обучаем вспомогательные модели
        print("\nОбучение вспомогательных моделей...")
        for name, model in models.items():
            if name != 'main_nn':
                model.fit(X_train, y_train)
        
        # Создаем адаптивный ансамбль
        ensemble = self.model_builder.AdaptiveEnsemble(models)
        
        # Проводим эксперименты для каждого уровня шума
        for noise_level in noise_levels:
            print(f"\nТестирование с уровнем шума {noise_level:.3f}...")
            
            # Массивы для хранения результатов экспериментов
            ensemble_accs = []
            nn_accs = []
            rf_accs = []
            gb_accs = []
            svm_accs = []
            knn_accs = []
            
            for exp in range(n_experiments):
                print(f"Эксперимент {exp + 1}/{n_experiments}...")
                
                # Добавляем шум к тестовым данным
                if noise_type == 'gaussian':
                    X_test_noisy = self.noise_injector.add_gaussian_noise(X_test, noise_level)
                elif noise_type == 'uniform':
                    X_test_noisy = self.noise_injector.add_uniform_noise(X_test, noise_level)
                elif noise_type == 'impulse':
                    X_test_noisy = self.noise_injector.add_impulse_noise(X_test, noise_level)
                elif noise_type == 'missing':
                    X_test_noisy = self.noise_injector.add_missing_values(X_test, noise_level)
                    
                    # Для пропущенных значений используем простую стратегию заполнения
                    imputer = SimpleImputer(strategy='mean')
                    X_test_noisy = imputer.fit_transform(X_test_noisy)
                else:
                    raise ValueError(f"Неизвестный тип шума: {noise_type}")
                
                # Оцениваем ансамбль
                metrics = ensemble.evaluate(X_test_noisy, y_test)
                
                # Сохраняем результаты
                ensemble_accs.append(metrics['accuracy'])
                nn_accs.append(metrics['models_accuracy']['main_nn'])
                rf_accs.append(metrics['models_accuracy']['random_forest'])
                gb_accs.append(metrics['models_accuracy']['gradient_boosting'])
                svm_accs.append(metrics['models_accuracy']['svm'])
                knn_accs.append(metrics['models_accuracy']['knn'])
            
            # Вычисляем средние значения и стандартные отклонения
            results['noise_levels'].append(noise_level)
            results['ensemble_accuracy'].append((np.mean(ensemble_accs), np.std(ensemble_accs)))
            results['nn_accuracy'].append((np.mean(nn_accs), np.std(nn_accs)))
            results['rf_accuracy'].append((np.mean(rf_accs), np.std(rf_accs)))
            results['gb_accuracy'].append((np.mean(gb_accs), np.std(gb_accs)))
            results['svm_accuracy'].append((np.mean(svm_accs), np.std(svm_accs)))
            results['knn_accuracy'].append((np.mean(knn_accs), np.std(knn_accs)))
            
            print(f"Средняя точность ансамбля: {np.mean(ensemble_accs):.4f} ± {np.std(ensemble_accs):.4f}")
            print(f"Средняя точность нейронной сети: {np.mean(nn_accs):.4f} ± {np.std(nn_accs):.4f}")
            print(f"Средняя точность Random Forest: {np.mean(rf_accs):.4f} ± {np.std(rf_accs):.4f}")
            print(f"Средняя точность Gradient Boosting: {np.mean(gb_accs):.4f} ± {np.std(gb_accs):.4f}")
            print(f"Средняя точность SVM: {np.mean(svm_accs):.4f} ± {np.std(svm_accs):.4f}")
            print(f"Средняя точность KNN: {np.mean(knn_accs):.4f} ± {np.std(knn_accs):.4f}")
        
        # Сохраняем результаты эксперимента
        self.experiment_results[noise_type] = results
        
        return results
    
    def run_all_experiments(self, noise_range, noise_step, n_experiments=3):
        """Проводит все эксперименты с различными типами шума
        
        Args:
            noise_range: Диапазон уровня шума (min, max)
            noise_step: Шаг изменения уровня шума
            n_experiments: Количество экспериментов для усреднения результатов
            
        Returns:
            all_results: Словарь с результатами всех экспериментов
        """
        noise_types = ['gaussian', 'uniform', 'impulse', 'missing']
        all_results = {}
        
        for noise_type in noise_types:
            print(f"\n{'=' * 50}")
            print(f"Запуск экспериментов с шумом типа {noise_type}")
            print(f"{'=' * 50}")
            
            results = self.run_experiment(noise_type, noise_range, noise_step, n_experiments)
            all_results[noise_type] = results
        
        self.experiment_results = all_results
        return all_results
    
    def visualize_results(self, noise_type=None):
        """Визуализирует результаты экспериментов
        
        Args:
            noise_type: Тип шума для визуализации (если None, визуализируются все)
            
        Returns:
            fig: Объект фигуры matplotlib
        """
        if not self.experiment_results:
            raise ValueError("Нет результатов экспериментов для визуализации")
        
        if noise_type is not None:
            if noise_type not in self.experiment_results:
                raise ValueError(f"Нет результатов для шума типа {noise_type}")
            
            # Визуализация результатов для одного типа шума
            results = self.experiment_results[noise_type]
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            noise_levels = results['noise_levels']
            
            # Точность ансамбля
            ensemble_mean = [acc[0] for acc in results['ensemble_accuracy']]
            ensemble_std = [acc[1] for acc in results['ensemble_accuracy']]
            ax.plot(noise_levels, ensemble_mean, 'o-', linewidth=2, label='Ансамблевая модель')
            ax.fill_between(noise_levels, 
                            [m - s for m, s in zip(ensemble_mean, ensemble_std)],
                            [m + s for m, s in zip(ensemble_mean, ensemble_std)],
                            alpha=0.2)
            
            # Точность основной нейронной сети
            nn_mean = [acc[0] for acc in results['nn_accuracy']]
            nn_std = [acc[1] for acc in results['nn_accuracy']]
            ax.plot(noise_levels, nn_mean, 's-', linewidth=2, label='Нейронная сеть')
            ax.fill_between(noise_levels, 
                            [m - s for m, s in zip(nn_mean, nn_std)],
                            [m + s for m, s in zip(nn_mean, nn_std)],
                            alpha=0.2)
            
            # Точность остальных моделей
            rf_mean = [acc[0] for acc in results['rf_accuracy']]
            gb_mean = [acc[0] for acc in results['gb_accuracy']]
            svm_mean = [acc[0] for acc in results['svm_accuracy']]
            knn_mean = [acc[0] for acc in results['knn_accuracy']]
            
            ax.plot(noise_levels, rf_mean, '^-', linewidth=2, label='Random Forest')
            ax.plot(noise_levels, gb_mean, 'v-', linewidth=2, label='Gradient Boosting')
            ax.plot(noise_levels, svm_mean, 'D-', linewidth=2, label='SVM')
            ax.plot(noise_levels, knn_mean, 'p-', linewidth=2, label='K-NN')
            
            ax.set_xlabel('Уровень шума')
            ax.set_ylabel('Точность')
            ax.set_title(f'Зависимость точности от уровня шума типа {noise_type}')
            ax.legend()
            ax.grid(True)
            
            return fig
        else:
            # Визуализация сравнения результатов для всех типов шума
            fig, axes = plt.subplots(2, 2, figsize=(18, 12))
            axes = axes.flatten()
            
            for i, noise_type in enumerate(['gaussian', 'uniform', 'impulse', 'missing']):
                if noise_type not in self.experiment_results:
                    continue
                    
                results = self.experiment_results[noise_type]
                ax = axes[i]
                
                noise_levels = results['noise_levels']
                
                # Точность ансамбля
                ensemble_mean = [acc[0] for acc in results['ensemble_accuracy']]
                ensemble_std = [acc[1] for acc in results['ensemble_accuracy']]
                ax.plot(noise_levels, ensemble_mean, 'o-', linewidth=2, label='Ансамблевая модель')
                ax.fill_between(noise_levels, 
                                [m - s for m, s in zip(ensemble_mean, ensemble_std)],
                                [m + s for m, s in zip(ensemble_mean, ensemble_std)],
                                alpha=0.2)
                
                # Точность основной нейронной сети
                nn_mean = [acc[0] for acc in results['nn_accuracy']]
                ax.plot(noise_levels, nn_mean, 's-', linewidth=2, label='Нейронная сеть')
                
                # Точность остальных моделей
                rf_mean = [acc[0] for acc in results['rf_accuracy']]
                gb_mean = [acc[0] for acc in results['gb_accuracy']]
                svm_mean = [acc[0] for acc in results['svm_accuracy']]
                knn_mean = [acc[0] for acc in results['knn_accuracy']]
                
                ax.plot(noise_levels, rf_mean, '^-', linewidth=2, label='Random Forest')
                ax.plot(noise_levels, gb_mean, 'v-', linewidth=2, label='Gradient Boosting')
                ax.plot(noise_levels, svm_mean, 'D-', linewidth=2, label='SVM')
                ax.plot(noise_levels, knn_mean, 'p-', linewidth=2, label='K-NN')
                
                ax.set_xlabel('Уровень шума')
                ax.set_ylabel('Точность')
                ax.set_title(f'Шум типа {noise_type}')
                ax.legend()
                ax.grid(True)
            
            plt.tight_layout()
            return fig
    
    def generate_report(self):
        """Генерирует отчет о результатах экспериментов в виде таблицы
        
        Returns:
            report_df: DataFrame с результатами
        """
        if not self.experiment_results:
            raise ValueError("Нет результатов экспериментов для отчета")
        
        # Создаем список для хранения данных отчета
        report_data = []
        
        for noise_type, results in self.experiment_results.items():
            noise_levels = results['noise_levels']
            
            for i, level in enumerate(noise_levels):
                # Получаем средние значения и стандартные отклонения
                ensemble_acc = results['ensemble_accuracy'][i]
                nn_acc = results['nn_accuracy'][i]
                rf_acc = results['rf_accuracy'][i]
                gb_acc = results['gb_accuracy'][i]
                svm_acc = results['svm_accuracy'][i]
                knn_acc = results['knn_accuracy'][i]
                
                # Добавляем данные в отчет
                report_data.append({
                    'Тип шума': noise_type,
                    'Уровень шума': level,
                    'Ансамблевая модель': f"{ensemble_acc[0]:.4f} ± {ensemble_acc[1]:.4f}",
                    'Нейронная сеть': f"{nn_acc[0]:.4f} ± {nn_acc[1]:.4f}",
                    'Random Forest': f"{rf_acc[0]:.4f} ± {rf_acc[1]:.4f}",
                    'Gradient Boosting': f"{gb_acc[0]:.4f} ± {gb_acc[1]:.4f}",
                    'SVM': f"{svm_acc[0]:.4f} ± {svm_acc[1]:.4f}",
                    'K-NN': f"{knn_acc[0]:.4f} ± {knn_acc[1]:.4f}"
                })
        
        # Создаем DataFrame
        report_df = pd.DataFrame(report_data)
        
        return report_df
    
    def save_models(self, path='./models'):
        """Сохраняет обученные модели
        
        Args:
            path: Путь для сохранения моделей
        """
        if not os.path.exists(path):
            os.makedirs(path)
        
        # Сохраняем объект StandardScaler
        joblib.dump(self.scaler, os.path.join(path, 'scaler.pkl'))
        
        # Получаем модели из model_builder
        models = self.model_builder.models
        
        # Сохраняем модели
        if 'main_nn' in models:
            models['main_nn'].save(os.path.join(path, 'main_nn_model'))
        
        for name, model in models.items():
            if name != 'main_nn':
                joblib.dump(model, os.path.join(path, f'{name}_model.pkl'))
        
        # Сохраняем гиперпараметры
        with open(os.path.join(path, 'hyperparameters.pkl'), 'wb') as f:
            pickle.dump(self.model_builder.best_params, f)
        
        print(f"Модели успешно сохранены в директории {path}")
    
    def load_models(self, path='./models'):
        """Загружает обученные модели
        
        Args:
            path: Путь к сохраненным моделям
        """
        if not os.path.exists(path):
            raise ValueError(f"Директория {path} не существует")
        
        # Загружаем объект StandardScaler
        self.scaler = joblib.load(os.path.join(path, 'scaler.pkl'))
        
        # Загружаем модели
        models = {}
        
        # Загружаем основную нейронную сеть
        if os.path.exists(os.path.join(path, 'main_nn_model')):
            models['main_nn'] = keras.models.load_model(os.path.join(path, 'main_nn_model'))
        
        # Загружаем вспомогательные модели
        for name in ['random_forest', 'gradient_boosting', 'svm', 'knn']:
            model_path = os.path.join(path, f'{name}_model.pkl')
            if os.path.exists(model_path):
                models[name] = joblib.load(model_path)
        
        # Загружаем гиперпараметры
        with open(os.path.join(path, 'hyperparameters.pkl'), 'rb') as f:
            self.model_builder.best_params = pickle.load(f)
        
        # Устанавливаем загруженные модели
        self.model_builder.models = models
        
        print("Модели успешно загружены")
        
        return models

class NoisyDataClassificationApp:
    """Класс для создания графического интерфейса программного комплекса"""
    
    def __init__(self, root):
        """Инициализирует приложение
        
        Args:
            root: Корневой виджет Tkinter
        """
        self.root = root
        self.root.title("Классификация зашумленных данных")
        self.root.geometry("1200x800")
        
        # Создаем экземпляр ExperimentRunner
        self.experiment_runner = ExperimentRunner()
        
        # Создаем элементы интерфейса
        self.create_widgets()
    
    def create_widgets(self):
        """Создает виджеты интерфейса"""
        # Главный фрейм
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Фрейм для выбора набора данных
        dataset_frame = ttk.LabelFrame(main_frame, text="Выбор набора данных", padding="10")
        dataset_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Радиокнопки для выбора встроенного набора данных
        self.dataset_var = tk.StringVar(value="iris")
        
        ttk.Label(dataset_frame, text="Встроенные наборы данных:").grid(row=0, column=0, sticky=tk.W)
        
        ttk.Radiobutton(dataset_frame, text="Iris", variable=self.dataset_var, value="iris").grid(row=1, column=0, sticky=tk.W)
        ttk.Radiobutton(dataset_frame, text="Wine", variable=self.dataset_var, value="wine").grid(row=2, column=0, sticky=tk.W)
        ttk.Radiobutton(dataset_frame, text="Breast Cancer", variable=self.dataset_var, value="breast_cancer").grid(row=3, column=0, sticky=tk.W)
        ttk.Radiobutton(dataset_frame, text="MNIST (подвыборка)", variable=self.dataset_var, value="digits").grid(row=4, column=0, sticky=tk.W)
        
        # Кнопка для загрузки пользовательского набора данных
        ttk.Label(dataset_frame, text="Пользовательский набор данных:").grid(row=0, column=1, sticky=tk.W, padx=10)
        
        ttk.Button(dataset_frame, text="Загрузить CSV файл", command=self.load_custom_dataset).grid(row=1, column=1, sticky=tk.W, padx=10)
        
        self.custom_dataset_label = ttk.Label(dataset_frame, text="Файл не выбран")
        self.custom_dataset_label.grid(row=2, column=1, columnspan=2, sticky=tk.W, padx=10)
        
        # Фрейм для настройки параметров шума
        noise_frame = ttk.LabelFrame(main_frame, text="Параметры шума", padding="10")
        noise_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(noise_frame, text="Минимальное значение шума:").grid(row=0, column=0, sticky=tk.W)
        self.min_noise_var = tk.DoubleVar(value=0.0)
        ttk.Entry(noise_frame, textvariable=self.min_noise_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(noise_frame, text="Максимальное значение шума:").grid(row=1, column=0, sticky=tk.W)
        self.max_noise_var = tk.DoubleVar(value=0.5)
        ttk.Entry(noise_frame, textvariable=self.max_noise_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(noise_frame, text="Шаг изменения шума:").grid(row=2, column=0, sticky=tk.W)
        self.noise_step_var = tk.DoubleVar(value=0.1)
        ttk.Entry(noise_frame, textvariable=self.noise_step_var, width=10).grid(row=2, column=1, sticky=tk.W, padx=5)
        
        ttk.Label(noise_frame, text="Количество экспериментов:").grid(row=3, column=0, sticky=tk.W)
        self.n_experiments_var = tk.IntVar(value=3)
        ttk.Entry(noise_frame, textvariable=self.n_experiments_var, width=10).grid(row=3, column=1, sticky=tk.W, padx=5)

        # Флажки для выбора типов шума
        ttk.Label(noise_frame, text="Типы шума для эксперимента:").grid(row=0, column=2, sticky=tk.W, padx=10)
        
        self.noise_types = {
            'gaussian': tk.BooleanVar(value=True),
            'uniform': tk.BooleanVar(value=True),
            'impulse': tk.BooleanVar(value=True),
            'missing': tk.BooleanVar(value=True)
        }
        
        ttk.Checkbutton(noise_frame, text="Гауссовский шум", variable=self.noise_types['gaussian']).grid(row=1, column=2, sticky=tk.W, padx=10)
        ttk.Checkbutton(noise_frame, text="Равномерный шум", variable=self.noise_types['uniform']).grid(row=2, column=2, sticky=tk.W, padx=10)
        ttk.Checkbutton(noise_frame, text="Импульсный шум", variable=self.noise_types['impulse']).grid(row=1, column=3, sticky=tk.W)
        ttk.Checkbutton(noise_frame, text="Пропущенные значения", variable=self.noise_types['missing']).grid(row=2, column=3, sticky=tk.W)
        
        # Кнопки управления
        control_frame = ttk.Frame(main_frame, padding="10")
        control_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(control_frame, text="Запустить эксперименты", command=self.run_experiments).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Визуализировать результаты", command=self.visualize_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Сохранить модели", command=self.save_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Загрузить модели", command=self.load_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Сохранить отчет", command=self.save_report).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Очистить", command=self.clear_output).pack(side=tk.LEFT, padx=5)
        
        # Фрейм для вывода
        output_frame = ttk.LabelFrame(main_frame, text="Вывод", padding="10")
        output_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Создаем notebook для вкладок
        self.notebook = ttk.Notebook(output_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Вкладка для вывода текста
        self.text_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.text_frame, text="Журнал")
        
        # Текстовое поле с прокруткой
        text_scroll = ttk.Scrollbar(self.text_frame)
        text_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.text_output = tk.Text(self.text_frame, wrap=tk.WORD, yscrollcommand=text_scroll.set)
        self.text_output.pack(fill=tk.BOTH, expand=True)
        text_scroll.config(command=self.text_output.yview)
        
        # Вкладка для визуализации результатов
        self.plot_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.plot_frame, text="Графики")
        
        # Вкладка для таблицы с результатами
        self.table_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.table_frame, text="Таблица результатов")
        
        # Перенаправляем вывод в текстовое поле
        self.redirect_output()
    
    def redirect_output(self):
        """Перенаправляет стандартный вывод в текстовое поле"""
        class TextRedirector:
            def __init__(self, text_widget):
                self.text_widget = text_widget
            
            def write(self, string):
                self.text_widget.insert(tk.END, string)
                self.text_widget.see(tk.END)
            
            def flush(self):
                pass
        
        import sys
        sys.stdout = TextRedirector(self.text_output)
    
    def load_custom_dataset(self):
        """Загружает пользовательский набор данных"""
        file_path = filedialog.askopenfilename(
            title="Выберите файл CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            self.custom_dataset_label.config(text=f"Выбран файл: {os.path.basename(file_path)}")
            self.dataset_var.set("custom")
            self.custom_dataset_path = file_path
            print(f"Выбран пользовательский набор данных: {file_path}")
    
    def run_experiments(self):
        """Запускает эксперименты с выбранными параметрами"""
        try:
            # Получаем параметры
            dataset = self.dataset_var.get()
            min_noise = self.min_noise_var.get()
            max_noise = self.max_noise_var.get()
            noise_step = self.noise_step_var.get()
            n_experiments = self.n_experiments_var.get()
            
            # Проверяем параметры
            if min_noise < 0:
                raise ValueError("Минимальное значение шума должно быть неотрицательным")
            
            if max_noise <= min_noise:
                raise ValueError("Максимальное значение шума должно быть больше минимального")
            
            if noise_step <= 0:
                raise ValueError("Шаг изменения шума должен быть положительным")
            
            if n_experiments <= 0:
                raise ValueError("Количество экспериментов должно быть положительным")
            
            # Загружаем набор данных
            if dataset == "custom":
                if hasattr(self, 'custom_dataset_path'):
                    self.experiment_runner.load_dataset(dataset_path=self.custom_dataset_path)
                else:
                    raise ValueError("Пользовательский набор данных не выбран")
            else:
                self.experiment_runner.load_dataset(dataset_name=dataset)
            
            # Получаем выбранные типы шума
            selected_noise_types = [name for name, var in self.noise_types.items() if var.get()]
            
            if not selected_noise_types:
                raise ValueError("Необходимо выбрать хотя бы один тип шума")
            
            # Запускаем эксперименты
            for noise_type in selected_noise_types:
                print(f"\n{'=' * 50}")
                print(f"Запуск экспериментов с шумом типа {noise_type}")
                print(f"{'=' * 50}")
                
                self.experiment_runner.run_experiment(
                    noise_type, (min_noise, max_noise), noise_step, n_experiments
                )
            
            messagebox.showinfo("Информация", "Эксперименты успешно завершены")
            
            # Переключаемся на вкладку с таблицей результатов
            self.show_results_table()
            self.notebook.select(self.table_frame)
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка: {str(e)}")
    
    def visualize_results(self):
        """Визуализирует результаты экспериментов"""
        try:
            if not self.experiment_runner.experiment_results:
                raise ValueError("Нет результатов экспериментов для визуализации")
            
            # Очищаем фрейм с графиками
            for widget in self.plot_frame.winfo_children():
                widget.destroy()
            
            # Создаем фигуру с графиками
            fig = self.experiment_runner.visualize_results()
            
            # Встраиваем фигуру в интерфейс
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
            # Переключаемся на вкладку с графиками
            self.notebook.select(self.plot_frame)
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка: {str(e)}")
    
    def show_results_table(self):
        """Отображает таблицу с результатами экспериментов"""
        try:
            if not self.experiment_runner.experiment_results:
                raise ValueError("Нет результатов экспериментов для отображения")
            
            # Очищаем фрейм с таблицей
            for widget in self.table_frame.winfo_children():
                widget.destroy()
            
            # Получаем DataFrame с результатами
            report_df = self.experiment_runner.generate_report()
            
            # Создаем прокручиваемый фрейм
            table_scroll_frame = ttk.Frame(self.table_frame)
            table_scroll_frame.pack(fill=tk.BOTH, expand=True)
            
            # Добавляем прокрутку
            x_scroll = ttk.Scrollbar(table_scroll_frame, orient=tk.HORIZONTAL)
            y_scroll = ttk.Scrollbar(table_scroll_frame, orient=tk.VERTICAL)
            
            # Создаем таблицу
            table = ttk.Treeview(
                table_scroll_frame,
                columns=list(report_df.columns),
                show="headings",
                xscrollcommand=x_scroll.set,
                yscrollcommand=y_scroll.set
            )
            
            # Настраиваем прокрутку
            x_scroll.config(command=table.xview)
            y_scroll.config(command=table.yview)
            
            x_scroll.pack(side=tk.BOTTOM, fill=tk.X)
            y_scroll.pack(side=tk.RIGHT, fill=tk.Y)
            table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
            
            # Задаем заголовки столбцов
            for column in report_df.columns:
                table.heading(column, text=column)
                table.column(column, width=100, anchor=tk.CENTER)
            
            # Заполняем таблицу данными
            for i, row in report_df.iterrows():
                table.insert("", tk.END, values=list(row))
            
            # Сохраняем DataFrame для последующего использования
            self.report_df = report_df
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка: {str(e)}")
    
    def save_models(self):
        """Сохраняет обученные модели"""
        try:
            if not hasattr(self.experiment_runner.model_builder, 'models') or not self.experiment_runner.model_builder.models:
                raise ValueError("Нет обученных моделей для сохранения")
            
            # Запрашиваем директорию для сохранения
            save_dir = filedialog.askdirectory(title="Выберите директорию для сохранения моделей")
            
            if save_dir:
                self.experiment_runner.save_models(path=save_dir)
                messagebox.showinfo("Информация", f"Модели успешно сохранены в директории {save_dir}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка: {str(e)}")
    
    def load_models(self):
        """Загружает обученные модели"""
        try:
            # Запрашиваем директорию с моделями
            load_dir = filedialog.askdirectory(title="Выберите директорию с сохраненными моделями")
            
            if load_dir:
                self.experiment_runner.load_models(path=load_dir)
                messagebox.showinfo("Информация", "Модели успешно загружены")
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка: {str(e)}")
    
    def save_report(self):
        """Сохраняет отчет о результатах экспериментов"""
        try:
            if not hasattr(self, 'report_df') or self.report_df is None:
                raise ValueError("Нет данных для сохранения отчета")
            
            # Запрашиваем имя файла для сохранения
            file_path = filedialog.asksaveasfilename(
                title="Сохранить отчет",
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx"), ("All files", "*.*")]
            )
            
            if file_path:
                # Сохраняем отчет
                if file_path.endswith('.csv'):
                    self.report_df.to_csv(file_path, index=False)
                elif file_path.endswith('.xlsx'):
                    self.report_df.to_excel(file_path, index=False)
                else:
                    self.report_df.to_csv(file_path, index=False)
                
                messagebox.showinfo("Информация", f"Отчет успешно сохранен в файле {file_path}")
        
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))
            print(f"Ошибка: {str(e)}")
    
    def clear_output(self):
        """Очищает вывод"""
        # Очищаем текстовое поле
        self.text_output.delete(1.0, tk.END)
        
        # Очищаем графики
        for widget in self.plot_frame.winfo_children():
            widget.destroy()
        
        # Очищаем таблицу
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        
        # Сбрасываем данные эксперимента
        self.experiment_runner = ExperimentRunner()
        
        if hasattr(self, 'report_df'):
            del self.report_df
        
        print("Вывод очищен. Готов к новым экспериментам.")


# Запуск приложения
if __name__ == "__main__":
    root = tk.Tk()
    app = NoisyDataClassificationApp(root)
    root.mainloop()