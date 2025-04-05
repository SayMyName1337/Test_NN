import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, LeakyReLU, Activation, GaussianNoise
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from sklearn.datasets import load_iris, load_wine, load_breast_cancer, fetch_openml
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from scipy.stats import mode
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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import joblib
import itertools
from mpl_toolkits.mplot3d import Axes3D
import xgboost as xgb
import lightgbm as lgb
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.combine import SMOTETomek

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
    
    @staticmethod
    def add_salt_pepper_noise(X, intensity):
        """Добавляет шум типа "соль и перец" к данным
        
        Args:
            X: Исходные данные
            intensity: Интенсивность шума (вероятность искажения)
            
        Returns:
            X_noisy: Данные с добавленным шумом
        """
        X_noisy = X.copy()
        
        # Маска для "соли" (максимальные значения)
        salt_mask = np.random.random(X.shape) < intensity/2
        X_noisy[salt_mask] = np.max(X)
        
        # Маска для "перца" (минимальные значения)
        pepper_mask = np.random.random(X.shape) < intensity/2
        X_noisy[pepper_mask] = np.min(X)
        
        return X_noisy
    
    @staticmethod
    def add_multiplicative_noise(X, intensity):
        """Добавляет мультипликативный шум к данным
        
        Args:
            X: Исходные данные
            intensity: Интенсивность шума
            
        Returns:
            X_noisy: Данные с добавленным шумом
        """
        noise = 1 + np.random.normal(0, intensity, X.shape)
        X_noisy = X * noise
        return X_noisy

class NoisePreprocessor:
    """Класс для предобработки зашумленных данных в зависимости от типа шума"""
    
    def __init__(self):
        """Инициализирует препроцессор данных"""
        self.preprocessors = {}
        
    def preprocess_gaussian_noise(self, X):
        """Предобработка данных с гауссовским шумом
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Обработанные данные
        """
        # Для гауссовского шума эффективен медианный фильтр
        X_processed = X.copy()
        
        # Применяем скользящее окно для вычисления медианы
        # Для простоты реализации используем только по одному соседу с каждой стороны
        for i in range(1, X.shape[0]-1):
            # Берем текущую точку и соседние
            window = np.vstack((X_processed[i-1], X_processed[i], X_processed[i+1]))
            # Вычисляем медиану по каждой колонке (признаку)
            X_processed[i] = np.median(window, axis=0)
        
        return X_processed
    
    def preprocess_impulse_noise(self, X):
        """Предобработка данных с импульсным шумом
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Обработанные данные
        """
        # Для импульсного шума эффективен метод обнаружения и замены выбросов
        X_processed = X.copy()
        
        # Вычисляем z-оценки для обнаружения выбросов
        z_scores = stats.zscore(X_processed, axis=0, nan_policy='omit')
        
        # Заменяем выбросы (|z| > 3) медианными значениями
        abs_z_scores = np.abs(z_scores)
        filtered_entries = (abs_z_scores > 3)
        
        # Вычисляем медианы для каждого признака
        medians = np.nanmedian(X_processed, axis=0)
        
        # Заменяем выбросы
        for i in range(X_processed.shape[1]):
            column_outliers = filtered_entries[:, i]
            X_processed[column_outliers, i] = medians[i]
        
        return X_processed
    
    def preprocess_missing_values(self, X):
        """Предобработка данных с пропущенными значениями
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Обработанные данные
        """
        # Используем KNN для заполнения пропущенных значений
        imputer = KNNImputer(n_neighbors=5)
        X_processed = imputer.fit_transform(X)
        
        return X_processed
    
    def preprocess_uniform_noise(self, X):
        """Предобработка данных с равномерным шумом
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Обработанные данные
        """
        # Для равномерного шума применяем сглаживание
        X_processed = X.copy()
        
        # Простое сглаживание скользящим средним
        for i in range(1, X.shape[0]-1):
            # Берем текущую точку и соседние
            window = np.vstack((X_processed[i-1], X_processed[i], X_processed[i+1]))
            # Вычисляем среднее по каждой колонке (признаку)
            X_processed[i] = np.mean(window, axis=0)
        
        return X_processed
    
    def preprocess_salt_pepper_noise(self, X):
        """Предобработка данных с шумом типа "соль и перец"
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Обработанные данные
        """
        # Для шума типа "соль и перец" эффективен медианный фильтр
        return self.preprocess_gaussian_noise(X)
    
    def preprocess_multiplicative_noise(self, X):
        """Предобработка данных с мультипликативным шумом
        
        Args:
            X: Зашумленные данные
            
        Returns:
            X_processed: Обработанные данные
        """
        # Для мультипликативного шума применяем логарифмическое преобразование
        # и затем сглаживание
        X_processed = np.log1p(np.abs(X))  # log(1+x) для избежания log(0)
        
        # Применяем сглаживание
        return self.preprocess_uniform_noise(X_processed)
    
    def preprocess_data(self, X, noise_type):
        """Предобрабатывает данные в зависимости от типа шума
        
        Args:
            X: Зашумленные данные
            noise_type: Тип шума
            
        Returns:
            X_processed: Обработанные данные
        """
        preprocessing_methods = {
            'gaussian': self.preprocess_gaussian_noise,
            'uniform': self.preprocess_uniform_noise,
            'impulse': self.preprocess_impulse_noise,
            'missing': self.preprocess_missing_values,
            'salt_pepper': self.preprocess_salt_pepper_noise,
            'multiplicative': self.preprocess_multiplicative_noise
        }
        
        if noise_type in preprocessing_methods:
            return preprocessing_methods[noise_type](X)
        else:
            print(f"Предупреждение: Предобработка для шума типа '{noise_type}' не реализована.")
            return X

class ModelBuilder:
    """Класс для построения и оптимизации моделей классификации"""
    
    def __init__(self):
        """Инициализирует построитель моделей"""
        self.models = {}
        self.best_params = {}
        self.feature_scaler = RobustScaler()  # Более устойчив к выбросам
        self.feature_selector = None
        self.pca = None
        
    def build_main_neural_network(self, input_shape, num_classes, hyperparams=None):
        """Строит улучшенную нейронную сеть с заданными гиперпараметрами
        
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
                'units_1': 256,
                'units_2': 128,
                'units_3': 64,
                'units_4': 32,
                'dropout_rate': 0.4,
                'learning_rate': 0.001,
                'l2_reg': 0.001,
                'batch_size': 64,
                'activation': 'relu',
                'leaky_alpha': 0.2,
                'noise_stddev': 0.1,
                'use_bn': True
            }
        
        # Создаем модель с улучшенной архитектурой
        inputs = Input(shape=input_shape)
        
        # Добавляем слой шума для повышения устойчивости
        x = GaussianNoise(hyperparams.get('noise_stddev', 0.1))(inputs)
        
        # Первый блок с выбором функции активации
        x = Dense(hyperparams['units_1'], kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(x)
        if hyperparams.get('use_bn', True):
            x = BatchNormalization()(x)
        if hyperparams.get('activation', 'relu') == 'leaky_relu':
            x = LeakyReLU(alpha=hyperparams.get('leaky_alpha', 0.2))(x)
        else:
            x = Activation(hyperparams.get('activation', 'relu'))(x)
        x = Dropout(hyperparams['dropout_rate'])(x)
        
        # Второй блок
        x = Dense(hyperparams['units_2'], kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(x)
        if hyperparams.get('use_bn', True):
            x = BatchNormalization()(x)
        if hyperparams.get('activation', 'relu') == 'leaky_relu':
            x = LeakyReLU(alpha=hyperparams.get('leaky_alpha', 0.2))(x)
        else:
            x = Activation(hyperparams.get('activation', 'relu'))(x)
        x = Dropout(hyperparams['dropout_rate'] * 0.8)(x)
        
        # Третий блок
        x = Dense(hyperparams['units_3'], kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(x)
        if hyperparams.get('use_bn', True):
            x = BatchNormalization()(x)
        if hyperparams.get('activation', 'relu') == 'leaky_relu':
            x = LeakyReLU(alpha=hyperparams.get('leaky_alpha', 0.2))(x)
        else:
            x = Activation(hyperparams.get('activation', 'relu'))(x)
        x = Dropout(hyperparams['dropout_rate'] * 0.6)(x)
        
        # Четвертый блок (новый)
        x = Dense(hyperparams['units_4'], kernel_regularizer=regularizers.l2(hyperparams['l2_reg']))(x)
        if hyperparams.get('use_bn', True):
            x = BatchNormalization()(x)
        if hyperparams.get('activation', 'relu') == 'leaky_relu':
            x = LeakyReLU(alpha=hyperparams.get('leaky_alpha', 0.2))(x)
        else:
            x = Activation(hyperparams.get('activation', 'relu'))(x)
        x = Dropout(hyperparams['dropout_rate'] * 0.4)(x)
        
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
    
    def optimize_neural_network(self, X_train, y_train, X_val, y_val, input_shape, num_classes, n_trials=20):
        """Оптимизирует гиперпараметры нейронной сети с помощью Optuna
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            X_val: Валидационные данные
            y_val: Валидационные метки
            input_shape: Размерность входных данных
            num_classes: Количество классов
            n_trials: Количество испытаний оптимизации
            
        Returns:
            best_params: Лучшие найденные гиперпараметры
        """
        def objective(trial):
            # Определяем расширенное пространство поиска гиперпараметров
            hyperparams = {
                'units_1': trial.suggest_int('units_1', 64, 512),
                'units_2': trial.suggest_int('units_2', 32, 256),
                'units_3': trial.suggest_int('units_3', 16, 128),
                'units_4': trial.suggest_int('units_4', 8, 64),
                'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
                'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
                'l2_reg': trial.suggest_float('l2_reg', 1e-5, 1e-2, log=True),
                'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64, 128]),
                'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu', 'elu']),
                'leaky_alpha': trial.suggest_float('leaky_alpha', 0.01, 0.3),
                'noise_stddev': trial.suggest_float('noise_stddev', 0.01, 0.2),
                'use_bn': trial.suggest_categorical('use_bn', [True, False])
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
            early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
            reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6)
            
            history = model.fit(
                X_train, y_train_cat,
                epochs=100,  # Увеличиваем количество эпох
                batch_size=hyperparams['batch_size'],
                validation_data=(X_val, y_val_cat),
                callbacks=[early_stopping, reduce_lr],
                verbose=0
            )
            
            # Оцениваем модель
            val_loss = min(history.history['val_loss'])
            
            return val_loss
        
        # Создаем исследование Optuna с более эффективным сэмплером
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler())
        study.optimize(objective, n_trials=n_trials)
        
        print("Оптимизация нейронной сети завершена:")
        print(f"Лучшие гиперпараметры: {study.best_params}")
        print(f"Лучшее значение целевой функции: {study.best_value:.4f}")
        
        return study.best_params
    
    def optimize_support_models(self, X_train, y_train, n_jobs=-1):
        """Оптимизирует гиперпараметры вспомогательных моделей
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            n_jobs: Количество используемых процессов (-1 для использования всех)
            
        Returns:
            best_params: Словарь с лучшими параметрами для каждой модели
        """
        # Определяем расширенные пространства поиска для каждой модели
        param_grids = {
            'random_forest': {
                'n_estimators': [50, 100, 200, 300],
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'bootstrap': [True, False],
                'class_weight': [None, 'balanced']
            },
            'gradient_boosting': {
                'n_estimators': [50, 100, 200, 300],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'max_depth': [3, 5, 7, 9],
                'min_samples_split': [2, 5, 10],
                'subsample': [0.8, 0.9, 1.0]
            },
            'svm': {
                'C': [0.1, 1, 10, 100],
                'gamma': ['scale', 'auto', 0.01, 0.1, 1],
                'kernel': ['rbf', 'poly', 'sigmoid'],
                'class_weight': [None, 'balanced']
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9, 11, 15],
                'weights': ['uniform', 'distance'],
                'p': [1, 2],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
            },
            'xgboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_child_weight': [1, 3, 5],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'lightgbm': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.05, 0.1],
                'num_leaves': [31, 50, 100],
                'max_depth': [3, 5, 7, -1],
                'min_child_samples': [20, 30, 50]
            },
            'adaboost': {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 1.0],
                'algorithm': ['SAMME', 'SAMME.R']
            }
        }
        
        # Модели для оптимизации
        base_models = {
            'random_forest': RandomForestClassifier(random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier(),
            'xgboost': xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'lightgbm': lgb.LGBMClassifier(random_state=42, verbose=-1),
            'adaboost': AdaBoostClassifier(random_state=42)
        }
        
        best_params = {}
        
        # Оптимизируем каждую модель
        for name, model in base_models.items():
            print(f"\nОптимизация модели {name}...")
            
            # Создаем объект GridSearchCV
            grid_search = GridSearchCV(
                estimator=model,
                param_grid=param_grids[name],
                cv=5,  # Увеличиваем до 5-кратной кросс-валидации
                scoring='accuracy',
                n_jobs=n_jobs,
                verbose=0
            )
            
            # Обучаем на данных
            grid_search.fit(X_train, y_train)
            
            # Сохраняем лучшие параметры
            best_params[name] = grid_search.best_params_
            print(f"Лучшие параметры для {name}: {grid_search.best_params_}")
            print(f"Лучшая точность при кросс-валидации: {grid_search.best_score_:.4f}")
        
        return best_params
    
    def perform_feature_selection(self, X_train, y_train, n_features=None):
        """Выполняет отбор признаков для улучшения качества моделей
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            n_features: Количество признаков для отбора (если None, выбирается автоматически)
            
        Returns:
            X_train_selected: Преобразованные данные
        """
        if n_features is None:
            # Автоматически определяем оптимальное количество признаков (не менее 50%)
            n_features = max(int(X_train.shape[1] * 0.5), 2)
        
        # Используем ANOVA F-value для отбора признаков (для классификации)
        self.feature_selector = SelectKBest(f_classif, k=n_features)
        X_train_selected = self.feature_selector.fit_transform(X_train, y_train)
        
        print(f"Выполнен отбор признаков: из {X_train.shape[1]} оставлено {n_features}")
        
        # Можно применить PCA к отобранным признакам для дальнейшего улучшения
        # if X_train_selected.shape[1] > 10:
        #     self.pca = PCA(n_components=min(n_features, 10))
        #     X_train_selected = self.pca.fit_transform(X_train_selected)
        #     print(f"Применено PCA: финальная размерность {X_train_selected.shape[1]}")
        
        return X_train_selected
    
    def apply_feature_transformation(self, X):
        """Применяет преобразования признаков (отбор и PCA)
        
        Args:
            X: Исходные данные
            
        Returns:
            X_transformed: Преобразованные данные
        """
        if self.feature_selector is not None:
            X = self.feature_selector.transform(X)
        
        if self.pca is not None:
            X = self.pca.transform(X)
        
        return X

    def build_ensemble_model(self, input_shape, num_classes, nn_params, support_params):
        """Строит расширенную ансамблевую модель с основной нейронной сетью и вспомогательными алгоритмами
        
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
        xgb_params = support_params['xgboost']
        lgb_params = support_params['lightgbm']
        ada_params = support_params['adaboost']
        
        rf_model = RandomForestClassifier(
            n_estimators=rf_params['n_estimators'],
            max_depth=rf_params['max_depth'],
            min_samples_split=rf_params['min_samples_split'],
            min_samples_leaf=rf_params.get('min_samples_leaf', 1),
            bootstrap=rf_params.get('bootstrap', True),
            class_weight=rf_params.get('class_weight', None),
            random_state=42
        )
        
        gb_model = GradientBoostingClassifier(
            n_estimators=gb_params['n_estimators'],
            learning_rate=gb_params['learning_rate'],
            max_depth=gb_params['max_depth'],
            min_samples_split=gb_params.get('min_samples_split', 2),
            subsample=gb_params.get('subsample', 1.0),
            random_state=42
        )
        
        svm_model = SVC(
            C=svm_params['C'],
            gamma=svm_params['gamma'],
            kernel=svm_params['kernel'],
            class_weight=svm_params.get('class_weight', None),
            probability=True,
            random_state=42
        )
        
        knn_model = KNeighborsClassifier(
            n_neighbors=knn_params['n_neighbors'],
            weights=knn_params['weights'],
            p=knn_params['p'],
            algorithm=knn_params.get('algorithm', 'auto')
        )
        
        xgb_model = xgb.XGBClassifier(
            n_estimators=xgb_params['n_estimators'],
            learning_rate=xgb_params['learning_rate'],
            max_depth=xgb_params['max_depth'],
            min_child_weight=xgb_params.get('min_child_weight', 1),
            subsample=xgb_params.get('subsample', 1.0),
            colsample_bytree=xgb_params.get('colsample_bytree', 1.0),
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        lgb_model = lgb.LGBMClassifier(
            n_estimators=lgb_params['n_estimators'],
            learning_rate=lgb_params['learning_rate'],
            num_leaves=lgb_params.get('num_leaves', 31),
            max_depth=lgb_params['max_depth'],
            min_child_samples=lgb_params.get('min_child_samples', 20),
            random_state=42,
            verbose=-1
        )
        
        ada_model = AdaBoostClassifier(
            n_estimators=ada_params['n_estimators'],
            learning_rate=ada_params['learning_rate'],
            algorithm=ada_params.get('algorithm', 'SAMME.R'),
            random_state=42
        )
        
        # Также добавим дополнительную модель ExtraTrees
        et_model = ExtraTreesClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_split=2,
            random_state=42
        )
        
        # Сохраняем модели в словаре
        self.models = {
            'main_nn': main_nn,
            'random_forest': rf_model,
            'gradient_boosting': gb_model,
            'svm': svm_model,
            'knn': knn_model,
            'xgboost': xgb_model,
            'lightgbm': lgb_model,
            'adaboost': ada_model,
            'extra_trees': et_model
        }
        
        self.best_params = {
            'nn_params': nn_params,
            'support_params': support_params
        }
        
        return self.models
    
    def build_stacking_model(self, X_train, y_train, num_classes):
        """Строит модель стекинга на основе вспомогательных моделей
        
        Args:
            X_train: Обучающие данные
            y_train: Обучающие метки
            num_classes: Количество классов
            
        Returns:
            stacking_model: Модель стекинга
        """
        # Проверяем, есть ли уже вспомогательные модели
        if not self.models or len(self.models) <= 1:
            raise ValueError("Необходимо сначала создать вспомогательные модели с помощью build_ensemble_model")
        
        # Собираем базовые модели (исключая нейронную сеть)
        estimators = []
        for name, model in self.models.items():
            if name != 'main_nn':
                estimators.append((name, model))
        
        # Выбираем финальный классификатор в зависимости от количества классов
        if num_classes == 2:
            final_estimator = LogisticRegression(C=1.0, class_weight='balanced', random_state=42)
        else:
            final_estimator = LogisticRegression(C=1.0, class_weight='balanced', multi_class='multinomial', 
                                               solver='lbfgs', random_state=42)
        
        # Создаем модель стекинга
        stacking_model = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=5,
            stack_method='predict_proba',
            n_jobs=-1
        )
        
        # Обучаем модель стекинга
        stacking_model.fit(X_train, y_train)
        
        # Сохраняем модель стекинга
        self.models['stacking'] = stacking_model
        
        return stacking_model

    class ImprovedAdaptiveEnsemble:
        """Класс для улучшенного адаптивного ансамбля моделей"""
        
        def __init__(self, models, val_X=None, val_y=None, confidence_threshold=0.7):
            """Инициализирует адаптивный ансамбль
            
            Args:
                models: Словарь с моделями
                val_X: Валидационные данные для калибровки весов
                val_y: Валидационные метки для калибровки весов
                confidence_threshold: Порог уверенности для основной модели
            """
            self.models = models
            self.confidence_threshold = confidence_threshold
            
            # Динамические веса для моделей
            self.model_weights = self._calculate_model_weights(val_X, val_y) if val_X is not None and val_y is not None else {
                'random_forest': 0.15,
                'gradient_boosting': 0.15,
                'svm': 0.1,
                'knn': 0.05,
                'xgboost': 0.15,
                'lightgbm': 0.15,
                'adaboost': 0.1,
                'extra_trees': 0.1,
                'stacking': 0.05
            }
            
            print("Веса моделей в ансамбле:")
            for model_name, weight in self.model_weights.items():
                print(f"  - {model_name}: {weight:.3f}")
        
        def _calculate_model_weights(self, X, y):
            """Вычисляет веса моделей на основе их производительности на валидационном наборе
            
            Args:
                X: Валидационные данные
                y: Валидационные метки
                
            Returns:
                weights: Словарь с весами моделей
            """
            if X is None or y is None:
                return self.model_weights
            
            # Вычисляем точность каждой модели
            accuracies = {}
            
            # Оцениваем основную нейронную сеть
            main_nn = self.models['main_nn']
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                probs = main_nn.predict(X)
                preds = (probs > 0.5).astype(int).flatten()
            else:  # Многоклассовая классификация
                probs = main_nn.predict(X)
                preds = np.argmax(probs, axis=1)
            
            nn_accuracy = accuracy_score(y, preds)
            
            # Весовой коэффициент для нейронной сети (не используется напрямую в ансамбле,
            # но используется для масштабирования весов других моделей)
            nn_weight = max(0.5, nn_accuracy)
            
            # Оцениваем вспомогательные модели
            for name, model in self.models.items():
                if name == 'main_nn':
                    continue
                
                try:
                    preds = model.predict(X)
                    acc = accuracy_score(y, preds)
                    # Используем f1-score для лучшей оценки на несбалансированных данных
                    if len(np.unique(y)) == 2:  # Бинарная классификация
                        f1 = f1_score(y, preds, average='binary')
                    else:  # Многоклассовая классификация
                        f1 = f1_score(y, preds, average='weighted')
                    
                    # Комбинированный показатель
                    combined_score = 0.6 * acc + 0.4 * f1
                    accuracies[name] = combined_score
                except:
                    accuracies[name] = 0.5  # Если возникла ошибка, используем нейтральный вес
            
            # Нормализуем веса так, чтобы их сумма была равна 1 - nn_weight
            total = sum(accuracies.values())
            weights = {name: (acc / total) * (1 - nn_weight) for name, acc in accuracies.items()}
            
            return weights
        
        def predict(self, X, noise_type=None, noise_level=None):
            """Делает предсказания с использованием улучшенного адаптивного ансамбля
            
            Args:
                X: Данные для предсказания
                noise_type: Тип шума (если известен)
                noise_level: Уровень шума (если известен)
                
            Returns:
                predictions: Предсказанные метки классов
            """
            # Получаем предсказания основной нейронной сети
            main_nn = self.models['main_nn']
            
            # Проверяем формат выхода (бинарная или многоклассовая классификация)
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                nn_probs = main_nn.predict(X)
                nn_conf = np.maximum(nn_probs, 1 - nn_probs)  # Уверенность
                nn_preds = (nn_probs > 0.5).astype(int).flatten()
            else:  # Многоклассовая классификация
                nn_probs = main_nn.predict(X)
                nn_conf = np.max(nn_probs, axis=1)  # Уверенность
                nn_preds = np.argmax(nn_probs, axis=1)
            
            # Адаптируем порог уверенности в зависимости от типа и уровня шума
            adaptive_threshold = self.confidence_threshold
            if noise_type and noise_level:
                # Для сильного шума снижаем порог уверенности
                if noise_level > 0.3:
                    adaptive_threshold -= 0.1
                # Для специфических типов шума
                if noise_type in ['impulse', 'missing']:
                    adaptive_threshold -= 0.05
            
            # Находим примеры с низкой уверенностью
            low_conf_mask = nn_conf < adaptive_threshold
            
            # Если все предсказания уверенные, возвращаем их
            if not np.any(low_conf_mask):
                return nn_preds
            
            # Для неуверенных примеров запускаем вспомогательные модели
            X_low_conf = X[low_conf_mask]
            
            # Получаем предсказания вспомогательных моделей
            support_probs = {}
            for name, model in self.models.items():
                if name == 'main_nn':
                    continue
                
                try:
                    # Предсказания и вероятности
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X_low_conf)
                        support_probs[name] = probs
                    else:
                        # Пытаемся получить вероятности через решающую функцию
                        try:
                            decision_values = model.decision_function(X_low_conf)
                            # Преобразуем решающую функцию в вероятности с помощью softmax
                            if decision_values.ndim == 1:  # Бинарная классификация
                                probs = 1 / (1 + np.exp(-decision_values))
                                support_probs[name] = np.column_stack([1 - probs, probs])
                            else:  # Многоклассовая классификация
                                exp_decision = np.exp(decision_values - np.max(decision_values, axis=1, keepdims=True))
                                probs = exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
                                support_probs[name] = probs
                        except:
                            # Если ничего не работает, используем one-hot закодированные предсказания
                            preds = model.predict(X_low_conf)
                            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                                probs = np.zeros((len(preds), 2))
                                probs[np.arange(len(preds)), preds] = 1
                            else:  # Многоклассовая классификация
                                probs = np.zeros((len(preds), main_nn.output_shape[-1]))
                                probs[np.arange(len(preds)), preds] = 1
                            support_probs[name] = probs
                except Exception as e:
                    print(f"Ошибка при получении предсказаний от модели {name}: {e}")
                    # Создаем нейтральные вероятности
                    if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                        support_probs[name] = np.ones((X_low_conf.shape[0], 2)) * 0.5
                    else:  # Многоклассовая классификация
                        support_probs[name] = np.ones((X_low_conf.shape[0], main_nn.output_shape[-1])) / main_nn.output_shape[-1]
            
            # Комбинируем предсказания вспомогательных моделей
            final_preds = nn_preds.copy()
            
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                # Комбинируем вероятности с учетом весов
                weighted_probs = np.zeros((X_low_conf.shape[0], 2))
                
                for name, probs in support_probs.items():
                    if probs.shape[1] == 2:  # Убедимся, что у нас правильный формат
                        weighted_probs += self.model_weights.get(name, 0.1) * probs
                    else:
                        # Если вероятности в неправильном формате, создаем их из предсказаний
                        one_hot = np.zeros((probs.shape[0], 2))
                        preds = (probs > 0.5).astype(int) if probs.ndim == 1 else np.argmax(probs, axis=1)
                        one_hot[np.arange(probs.shape[0]), preds] = 1
                        weighted_probs += self.model_weights.get(name, 0.1) * one_hot
                
                # Нормализуем вероятности
                row_sums = weighted_probs.sum(axis=1)
                normalized_probs = weighted_probs / row_sums[:, np.newaxis]
                
                # Получаем финальные предсказания
                ensemble_preds = np.argmax(normalized_probs, axis=1)
                final_preds[low_conf_mask] = ensemble_preds
                
            else:  # Многоклассовая классификация
                num_classes = main_nn.output_shape[-1]
                weighted_probs = np.zeros((X_low_conf.shape[0], num_classes))
                
                for name, probs in support_probs.items():
                    if probs.shape[1] == num_classes:  # Убедимся, что у нас правильный формат
                        weighted_probs += self.model_weights.get(name, 0.1) * probs
                    else:
                        # Если вероятности в неправильном формате, создаем их из предсказаний
                        one_hot = np.zeros((probs.shape[0], num_classes))
                        preds = np.argmax(probs, axis=1) if probs.ndim > 1 else probs.astype(int)
                        one_hot[np.arange(probs.shape[0]), preds] = 1
                        weighted_probs += self.model_weights.get(name, 0.1) * one_hot
                
                # Нормализуем вероятности
                row_sums = weighted_probs.sum(axis=1)
                normalized_probs = weighted_probs / row_sums[:, np.newaxis]
                
                # Получаем финальные предсказания
                ensemble_preds = np.argmax(normalized_probs, axis=1)
                final_preds[low_conf_mask] = ensemble_preds
            
            return final_preds
        
        def predict_proba(self, X, noise_type=None, noise_level=None):
            """Предсказывает вероятности классов с учетом всех моделей в ансамбле
            
            Args:
                X: Данные для предсказания
                noise_type: Тип шума (если известен)
                noise_level: Уровень шума (если известен)
                
            Returns:
                probabilities: Предсказанные вероятности классов
            """
            # Получаем вероятности от основной нейронной сети
            main_nn = self.models['main_nn']
            
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                nn_probs_raw = main_nn.predict(X)
                nn_probs = np.column_stack([1 - nn_probs_raw, nn_probs_raw])
                nn_conf = np.max(nn_probs, axis=1)  # Уверенность
            else:  # Многоклассовая классификация
                nn_probs = main_nn.predict(X)
                nn_conf = np.max(nn_probs, axis=1)  # Уверенность
            
            # Адаптируем порог уверенности в зависимости от типа и уровня шума
            adaptive_threshold = self.confidence_threshold
            if noise_type and noise_level:
                if noise_level > 0.3:
                    adaptive_threshold -= 0.1
                if noise_type in ['impulse', 'missing']:
                    adaptive_threshold -= 0.05
            
            # Находим примеры с низкой уверенностью
            low_conf_mask = nn_conf < adaptive_threshold
            
            # Если все предсказания уверенные, возвращаем вероятности от основной модели
            if not np.any(low_conf_mask):
                return nn_probs
            
            # Для неуверенных примеров запускаем вспомогательные модели
            X_low_conf = X[low_conf_mask]
            
            # Получаем вероятности от вспомогательных моделей
            support_probs = {}
            for name, model in self.models.items():
                if name == 'main_nn':
                    continue
                
                try:
                    if hasattr(model, 'predict_proba'):
                        probs = model.predict_proba(X_low_conf)
                        support_probs[name] = probs
                    else:
                        try:
                            decision_values = model.decision_function(X_low_conf)
                            if decision_values.ndim == 1:  # Бинарная классификация
                                probs = 1 / (1 + np.exp(-decision_values))
                                support_probs[name] = np.column_stack([1 - probs, probs])
                            else:  # Многоклассовая классификация
                                exp_decision = np.exp(decision_values - np.max(decision_values, axis=1, keepdims=True))
                                probs = exp_decision / np.sum(exp_decision, axis=1, keepdims=True)
                                support_probs[name] = probs
                        except:
                            preds = model.predict(X_low_conf)
                            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                                probs = np.zeros((len(preds), 2))
                                probs[np.arange(len(preds)), preds] = 1
                            else:  # Многоклассовая классификация
                                probs = np.zeros((len(preds), main_nn.output_shape[-1]))
                                probs[np.arange(len(preds)), preds] = 1
                            support_probs[name] = probs
                except:
                    # Создаем нейтральные вероятности
                    if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                        support_probs[name] = np.ones((X_low_conf.shape[0], 2)) * 0.5
                    else:  # Многоклассовая классификация
                        support_probs[name] = np.ones((X_low_conf.shape[0], main_nn.output_shape[-1])) / main_nn.output_shape[-1]
            
            # Комбинируем вероятности с учетом весов
            final_probs = nn_probs.copy()
            
            if main_nn.output_shape[-1] == 1:  # Бинарная классификация
                weighted_probs = np.zeros((X_low_conf.shape[0], 2))
                
                for name, probs in support_probs.items():
                    if probs.shape[1] == 2:
                        weighted_probs += self.model_weights.get(name, 0.1) * probs
                    else:
                        one_hot = np.zeros((probs.shape[0], 2))
                        preds = (probs > 0.5).astype(int) if probs.ndim == 1 else np.argmax(probs, axis=1)
                        one_hot[np.arange(probs.shape[0]), preds] = 1
                        weighted_probs += self.model_weights.get(name, 0.1) * one_hot
                
                # Нормализуем вероятности
                row_sums = weighted_probs.sum(axis=1)
                normalized_probs = weighted_probs / row_sums[:, np.newaxis]
                
                # Обновляем финальные вероятности
                final_probs[low_conf_mask] = normalized_probs
                
            else:  # Многоклассовая классификация
                num_classes = main_nn.output_shape[-1]
                weighted_probs = np.zeros((X_low_conf.shape[0], num_classes))
                
                for name, probs in support_probs.items():
                    if probs.shape[1] == num_classes:
                        weighted_probs += self.model_weights.get(name, 0.1) * probs
                    else:
                        one_hot = np.zeros((probs.shape[0], num_classes))
                        preds = np.argmax(probs, axis=1) if probs.ndim > 1 else probs.astype(int)
                        one_hot[np.arange(probs.shape[0]), preds] = 1
                        weighted_probs += self.model_weights.get(name, 0.1) * one_hot
                
                # Нормализуем вероятности
                row_sums = weighted_probs.sum(axis=1)
                normalized_probs = weighted_probs / row_sums[:, np.newaxis]
                
                # Обновляем финальные вероятности
                final_probs[low_conf_mask] = normalized_probs
            
            return final_probs
            
        def evaluate(self, X, y, noise_type=None, noise_level=None):
            """Оценивает производительность ансамбля
            
            Args:
                X: Тестовые данные
                y: Истинные метки
                noise_type: Тип шума (если известен)
                noise_level: Уровень шума (если известен)
                
            Returns:
                metrics: Словарь с метриками производительности
            """
            # Делаем предсказания
            y_pred = self.predict(X, noise_type, noise_level)
            y_proba = self.predict_proba(X, noise_type, noise_level)
            
            # Вычисляем метрики
            accuracy = accuracy_score(y, y_pred)
            report = classification_report(y, y_pred, output_dict=True)
            
            # Дополнительные метрики
            if len(np.unique(y)) == 2:  # Бинарная классификация
                f1 = f1_score(y, y_pred, average='binary')
                precision = precision_score(y, y_pred, average='binary')
                recall = recall_score(y, y_pred, average='binary')
            else:  # Многоклассовая классификация
                f1 = f1_score(y, y_pred, average='weighted')
                precision = precision_score(y, y_pred, average='weighted')
                recall = recall_score(y, y_pred, average='weighted')
            
            # Оцениваем производительность отдельных моделей
            models_metrics = {}
            for name, model in self.models.items():
                try:
                    if name == 'main_nn':
                        if model.output_shape[-1] == 1:  # Бинарная классификация
                            probs = model.predict(X)
                            preds = (probs > 0.5).astype(int).flatten()
                        else:  # Многоклассовая классификация
                            probs = model.predict(X)
                            preds = np.argmax(probs, axis=1)
                    else:
                        preds = model.predict(X)
                    
                    model_acc = accuracy_score(y, preds)
                    if len(np.unique(y)) == 2:  # Бинарная классификация
                        model_f1 = f1_score(y, preds, average='binary')
                    else:  # Многоклассовая классификация
                        model_f1 = f1_score(y, preds, average='weighted')
                        
                    models_metrics[name] = {
                        'accuracy': model_acc,
                        'f1_score': model_f1
                    }
                except:
                    models_metrics[name] = {'accuracy': 0.0, 'f1_score': 0.0}
            
            # Возвращаем метрики
            return {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'report': report,
                'models_metrics': models_metrics
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
        self.noise_preprocessor = NoisePreprocessor()
        self.model_builder = ModelBuilder()
        self.X = None
        self.y = None
        self.feature_names = None
        self.target_names = None
        self.scaler = RobustScaler()  # Более устойчив к выбросам
        self.experiment_results = {}
        self.current_ensemble = None
        
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
    
    def run_experiment(self, noise_type, noise_range, noise_step, n_experiments=3, use_preprocessing=True):
        """Проводит эксперимент с заданным типом и уровнем шума
        
        Args:
            noise_type: Тип шума ('gaussian', 'uniform', 'impulse', 'missing', 'salt_pepper', 'multiplicative')
            noise_range: Диапазон уровня шума (min, max)
            noise_step: Шаг изменения уровня шума
            n_experiments: Количество экспериментов для усреднения результатов
            use_preprocessing: Применять ли предобработку данных в зависимости от типа шума
            
        Returns:
            results: Словарь с результатами экспериментов
        """
        if self.X is None or self.y is None:
            raise ValueError("Набор данных не загружен")
        
        # Словарь для хранения результатов
        results = {
            'noise_levels': [],
            'ensemble_accuracy': [],
            'ensemble_f1': [],
            'nn_accuracy': [],
            'rf_accuracy': [],
            'gb_accuracy': [],
            'svm_accuracy': [],
            'knn_accuracy': [],
            'xgb_accuracy': [],
            'lgb_accuracy': [],
            'preprocessing_impact': []  # Новый ключ для хранения влияния предобработки
        }
        
        min_noise, max_noise = noise_range
        noise_levels = np.arange(min_noise, max_noise + noise_step, noise_step)
        
        # Предварительная обработка данных
        X_scaled = self.scaler.fit_transform(self.X)
        
        # Проверяем на дисбаланс классов
        class_counts = np.bincount(self.y)
        min_class_count = np.min(class_counts)
        max_class_count = np.max(class_counts)
        class_imbalance_ratio = max_class_count / min_class_count
        
        # Если имеется сильный дисбаланс классов, применяем SMOTE
        use_smote = class_imbalance_ratio > 3
        if use_smote:
            print(f"\nОбнаружен дисбаланс классов (соотношение: {class_imbalance_ratio:.2f}). Применение SMOTE...")
        
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
        
        # Применяем SMOTE если необходимо
        if use_smote:
            smote = SMOTETomek(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"После SMOTE: {X_train.shape[0]} образцов, распределение классов: {np.bincount(y_train)}")
        
        print(f"\nПроводим эксперимент с шумом типа {noise_type}...")
        print(f"Диапазон шума: [{min_noise}, {max_noise}], шаг: {noise_step}")
        print(f"Количество экспериментов для усреднения: {n_experiments}")
        print(f"Применение предобработки шума: {use_preprocessing}")
        
        # Выполняем отбор признаков (если признаков много)
        if X_train.shape[1] > 10:
            print("\nВыполняем отбор признаков...")
            X_train_selected = self.model_builder.perform_feature_selection(X_train, y_train)
            X_val_selected = self.model_builder.apply_feature_transformation(X_val)
            X_test_selected = self.model_builder.apply_feature_transformation(X_test)
            
            # Обновляем размерность входных данных
            input_shape = (X_train_selected.shape[1],)
        else:
            X_train_selected = X_train
            X_val_selected = X_val
            X_test_selected = X_test
        
        # Оптимизация гиперпараметров основной нейронной сети
        print("\nОптимизация гиперпараметров основной нейронной сети...")
        nn_params = self.model_builder.optimize_neural_network(
            X_train_selected, y_train, X_val_selected, y_val, input_shape, num_classes, n_trials=15
        )
        
        # Оптимизация гиперпараметров вспомогательных моделей
        print("\nОптимизация гиперпараметров вспомогательных моделей...")
        support_params = self.model_builder.optimize_support_models(X_train_selected, y_train)
        
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
            
        # Callback для сохранения лучшей модели
        checkpoint = ModelCheckpoint(
            'best_nn_model',
            monitor='val_loss',
            save_best_only=True,
            mode='min',
            verbose=0
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=15, restore_best_weights=True
        )
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.2, patience=7, min_lr=1e-6
        )
        
        # Обучаем с увеличенным количеством эпох
        models['main_nn'].fit(
            X_train_selected, y_train_cat,
            epochs=100,
            batch_size=nn_params['batch_size'],
            validation_data=(X_val_selected, y_val_cat),
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1
        )
        
        # Загружаем лучшую модель (если сохранялась)
        if os.path.exists('best_nn_model'):
            models['main_nn'] = keras.models.load_model('best_nn_model')
            print("Загружена лучшая модель нейронной сети")
        
        # Обучаем вспомогательные модели
        print("\nОбучение вспомогательных моделей...")
        for name, model in models.items():
            if name != 'main_nn':
                model.fit(X_train_selected, y_train)
        
        # Создаем улучшенный адаптивный ансамбль с калибровкой весов
        print("\nСоздание улучшенного адаптивного ансамбля...")
        ensemble = self.model_builder.ImprovedAdaptiveEnsemble(models, X_val_selected, y_val)
        self.current_ensemble = ensemble
        
        # Проводим эксперименты для каждого уровня шума
        for noise_level in noise_levels:
            print(f"\nТестирование с уровнем шума {noise_level:.3f}...")
            
            # Массивы для хранения результатов экспериментов
            ensemble_accs = []
            ensemble_f1s = []
            nn_accs = []
            rf_accs = []
            gb_accs = []
            svm_accs = []
            knn_accs = []
            xgb_accs = []
            lgb_accs = []
            preprocessing_impacts = []  # Для измерения эффекта предобработки
            
            for exp in range(n_experiments):
                print(f"Эксперимент {exp + 1}/{n_experiments}...")
                
                # Добавляем шум к тестовым данным
                if noise_type == 'gaussian':
                    X_test_noisy = self.noise_injector.add_gaussian_noise(X_test_selected, noise_level)
                elif noise_type == 'uniform':
                    X_test_noisy = self.noise_injector.add_uniform_noise(X_test_selected, noise_level)
                elif noise_type == 'impulse':
                    X_test_noisy = self.noise_injector.add_impulse_noise(X_test_selected, noise_level)
                elif noise_type == 'missing':
                    X_test_noisy = self.noise_injector.add_missing_values(X_test_selected, noise_level)
                    # Для пропущенных значений используем KNN-импутацию
                    imputer = KNNImputer(n_neighbors=5)
                    X_test_noisy = imputer.fit_transform(X_test_noisy)
                elif noise_type == 'salt_pepper':
                    X_test_noisy = self.noise_injector.add_salt_pepper_noise(X_test_selected, noise_level)
                elif noise_type == 'multiplicative':
                    X_test_noisy = self.noise_injector.add_multiplicative_noise(X_test_selected, noise_level)
                else:
                    raise ValueError(f"Неизвестный тип шума: {noise_type}")
                
                # Делаем копию для оценки без предобработки
                X_test_raw = X_test_noisy.copy()
                
                # Применяем предобработку в зависимости от типа шума
                if use_preprocessing:
                    X_test_preprocessed = self.noise_preprocessor.preprocess_data(X_test_noisy, noise_type)
                    
                    # Оцениваем эффект предобработки
                    ensemble_metrics_raw = ensemble.evaluate(X_test_raw, y_test, noise_type, noise_level)
                    ensemble_metrics_preprocessed = ensemble.evaluate(X_test_preprocessed, y_test, noise_type, noise_level)
                    
                    # Сравниваем точность до и после предобработки
                    acc_raw = ensemble_metrics_raw['accuracy']
                    acc_preprocessed = ensemble_metrics_preprocessed['accuracy']
                    preprocessing_impact = acc_preprocessed - acc_raw
                    preprocessing_impacts.append(preprocessing_impact)
                    
                    # Используем предобработанные данные
                    X_test_final = X_test_preprocessed
                    print(f"  Влияние предобработки: {preprocessing_impact*100:.2f}% ({acc_raw:.4f} -> {acc_preprocessed:.4f})")
                else:
                    X_test_final = X_test_raw
                    preprocessing_impacts.append(0.0)
                
                # Оцениваем ансамбль
                metrics = ensemble.evaluate(X_test_final, y_test, noise_type, noise_level)
                
                # Сохраняем результаты
                ensemble_accs.append(metrics['accuracy'])
                ensemble_f1s.append(metrics['f1_score'])
                nn_accs.append(metrics['models_metrics']['main_nn']['accuracy'])
                rf_accs.append(metrics['models_metrics']['random_forest']['accuracy'])
                gb_accs.append(metrics['models_metrics']['gradient_boosting']['accuracy'])
                svm_accs.append(metrics['models_metrics']['svm']['accuracy'])
                knn_accs.append(metrics['models_metrics']['knn']['accuracy'])
                xgb_accs.append(metrics['models_metrics']['xgboost']['accuracy'])
                lgb_accs.append(metrics['models_metrics']['lightgbm']['accuracy'])
            
            # Вычисляем средние значения и стандартные отклонения
            results['noise_levels'].append(noise_level)
            results['ensemble_accuracy'].append((np.mean(ensemble_accs), np.std(ensemble_accs)))
            results['ensemble_f1'].append((np.mean(ensemble_f1s), np.std(ensemble_f1s)))
            results['nn_accuracy'].append((np.mean(nn_accs), np.std(nn_accs)))
            results['rf_accuracy'].append((np.mean(rf_accs), np.std(rf_accs)))
            results['gb_accuracy'].append((np.mean(gb_accs), np.std(gb_accs)))
            results['svm_accuracy'].append((np.mean(svm_accs), np.std(svm_accs)))
            results['knn_accuracy'].append((np.mean(knn_accs), np.std(knn_accs)))
            results['xgb_accuracy'].append((np.mean(xgb_accs), np.std(xgb_accs)))
            results['lgb_accuracy'].append((np.mean(lgb_accs), np.std(lgb_accs)))
            results['preprocessing_impact'].append((np.mean(preprocessing_impacts), np.std(preprocessing_impacts)))
            
            print(f"Средняя точность ансамбля: {np.mean(ensemble_accs):.4f} ± {np.std(ensemble_accs):.4f}")
            print(f"Средняя F1-мера ансамбля: {np.mean(ensemble_f1s):.4f} ± {np.std(ensemble_f1s):.4f}")
            print(f"Средняя точность нейронной сети: {np.mean(nn_accs):.4f} ± {np.std(nn_accs):.4f}")
            print(f"Средняя точность Random Forest: {np.mean(rf_accs):.4f} ± {np.std(rf_accs):.4f}")
            print(f"Средняя точность Gradient Boosting: {np.mean(gb_accs):.4f} ± {np.std(gb_accs):.4f}")
            print(f"Средняя точность SVM: {np.mean(svm_accs):.4f} ± {np.std(svm_accs):.4f}")
            print(f"Средняя точность KNN: {np.mean(knn_accs):.4f} ± {np.std(knn_accs):.4f}")
            print(f"Средняя точность XGBoost: {np.mean(xgb_accs):.4f} ± {np.std(xgb_accs):.4f}")
            print(f"Средняя точность LightGBM: {np.mean(lgb_accs):.4f} ± {np.std(lgb_accs):.4f}")
            if use_preprocessing:
                print(f"Среднее влияние предобработки: {np.mean(preprocessing_impacts)*100:.2f}% ± {np.std(preprocessing_impacts)*100:.2f}%")
        
        # Сохраняем результаты эксперимента
        self.experiment_results[noise_type] = results
        
        return results
    
    def run_all_experiments(self, noise_range, noise_step, n_experiments=3, use_preprocessing=True):
        """Проводит все эксперименты с различными типами шума
        
        Args:
            noise_range: Диапазон уровня шума (min, max)
            noise_step: Шаг изменения уровня шума
            n_experiments: Количество экспериментов для усреднения результатов
            use_preprocessing: Применять ли предобработку данных в зависимости от типа шума
            
        Returns:
            all_results: Словарь с результатами всех экспериментов
        """
        noise_types = ['gaussian', 'uniform', 'impulse', 'missing', 'salt_pepper', 'multiplicative']
        all_results = {}
        
        for noise_type in noise_types:
            print(f"\n{'=' * 50}")
            print(f"Запуск экспериментов с шумом типа {noise_type}")
            print(f"{'=' * 50}")
            
            results = self.run_experiment(noise_type, noise_range, noise_step, n_experiments, use_preprocessing)
            all_results[noise_type] = results
        
        self.experiment_results = all_results
        return all_results
    
    def visualize_results(self, noise_type=None, show_preprocessing=True, metric='accuracy', figsize=(12, 8)):
        """Визуализирует результаты экспериментов
        
        Args:
            noise_type: Тип шума для визуализации (если None, визуализируются все)
            show_preprocessing: Показывать ли влияние предобработки
            metric: Метрика для визуализации ('accuracy' или 'f1')
            figsize: Размер фигуры
            
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
            
            fig, ax = plt.subplots(figsize=figsize)
            
            noise_levels = results['noise_levels']
            
            # Настройка стилей
            try:
                plt.style.use('seaborn-v0_8')
            except:
                plt.style.use('seaborn')
            
            # Точность ансамбля
            if metric == 'accuracy':
                ensemble_mean = [acc[0] for acc in results['ensemble_accuracy']]
                ensemble_std = [acc[1] for acc in results['ensemble_accuracy']]
                metric_label = 'Точность'
            else:  # f1
                ensemble_mean = [f1[0] for f1 in results['ensemble_f1']]
                ensemble_std = [f1[1] for f1 in results['ensemble_f1']]
                metric_label = 'F1-мера'
            
            ax.plot(noise_levels, ensemble_mean, 'o-', linewidth=2, color='#1f77b4', label='Ансамблевая модель')
            ax.fill_between(noise_levels, 
                            [m - s for m, s in zip(ensemble_mean, ensemble_std)],
                            [m + s for m, s in zip(ensemble_mean, ensemble_std)],
                            alpha=0.2, color='#1f77b4')
            
            # Точность основной нейронной сети
            nn_mean = [acc[0] for acc in results['nn_accuracy']]
            nn_std = [acc[1] for acc in results['nn_accuracy']]
            ax.plot(noise_levels, nn_mean, 's-', linewidth=2, color='#d62728', label='Нейронная сеть')
            ax.fill_between(noise_levels, 
                            [m - s for m, s in zip(nn_mean, nn_std)],
                            [m + s for m, s in zip(nn_mean, nn_std)],
                            alpha=0.2, color='#d62728')
            
            # Точность остальных моделей
            rf_mean = [acc[0] for acc in results['rf_accuracy']]
            gb_mean = [acc[0] for acc in results['gb_accuracy']]
            svm_mean = [acc[0] for acc in results['svm_accuracy']]
            knn_mean = [acc[0] for acc in results['knn_accuracy']]
            xgb_mean = [acc[0] for acc in results['xgb_accuracy']]
            lgb_mean = [acc[0] for acc in results['lgb_accuracy']]
            
            # Используем более приятные цвета
            ax.plot(noise_levels, rf_mean, '^-', linewidth=2, color='#2ca02c', label='Random Forest')
            ax.plot(noise_levels, gb_mean, 'v-', linewidth=2, color='#ff7f0e', label='Gradient Boosting')
            ax.plot(noise_levels, svm_mean, 'D-', linewidth=2, color='#9467bd', label='SVM')
            ax.plot(noise_levels, knn_mean, 'p-', linewidth=2, color='#8c564b', label='K-NN')
            ax.plot(noise_levels, xgb_mean, '*-', linewidth=2, color='#e377c2', label='XGBoost')
            ax.plot(noise_levels, lgb_mean, 'X-', linewidth=2, color='#7f7f7f', label='LightGBM')
            
            # Если выбран показ влияния предобработки и оно есть в результатах
            if show_preprocessing and 'preprocessing_impact' in results:
                # Создаем вторую ось Y
                ax2 = ax.twinx()
                prep_mean = [impact[0] * 100 for impact in results['preprocessing_impact']]  # В процентах
                prep_std = [impact[1] * 100 for impact in results['preprocessing_impact']]
                
                ax2.plot(noise_levels, prep_mean, '--', linewidth=2, color='#17becf', label='Влияние предобработки')
                ax2.fill_between(noise_levels,
                                [m - s for m, s in zip(prep_mean, prep_std)],
                                [m + s for m, s in zip(prep_mean, prep_std)],
                                alpha=0.2, color='#17becf')
                
                # Настройки вторичной оси Y
                ax2.set_ylabel('Изменение точности после предобработки, %')
                ax2.spines['right'].set_color('#17becf')
                ax2.yaxis.label.set_color('#17becf')
                ax2.tick_params(axis='y', colors='#17becf')
                
                # Добавляем легенду для второй оси
                lines1, labels1 = ax.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, labels1 + labels2, loc='best')