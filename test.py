import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, ClassifierMixin
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.utils import to_categorical
from scikeras.wrappers import KerasClassifier
import tensorflow as tf
import optuna
from tqdm import tqdm
import warnings
import time
import os

# Отключаем предупреждения
warnings.filterwarnings('ignore')

# Устанавливаем стиль для графиков
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

# Для воспроизводимости результатов
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

class NoiseInjector:
    """Класс для добавления различных типов шума в данные."""
    
    @staticmethod
    def add_gaussian_noise(X, noise_level):
        """
        Добавление гауссовского шума.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Исходные данные
        noise_level : float
            Уровень шума (стандартное отклонение)
            
        Returns:
        --------
        numpy.ndarray
            Данные с добавленным шумом
        """
        noise = np.random.normal(0, noise_level, X.shape)
        return X + noise
    
    @staticmethod
    def add_uniform_noise(X, noise_level):
        """
        Добавление равномерного шума.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Исходные данные
        noise_level : float
            Уровень шума (максимальная амплитуда)
            
        Returns:
        --------
        numpy.ndarray
            Данные с добавленным шумом
        """
        noise = np.random.uniform(-noise_level, noise_level, X.shape)
        return X + noise
    
    @staticmethod
    def add_impulse_noise(X, noise_level):
        """
        Добавление импульсного шума (Salt and Pepper).
        
        Parameters:
        -----------
        X : numpy.ndarray
            Исходные данные
        noise_level : float
            Вероятность замены элемента шумом (от 0 до 1)
            
        Returns:
        --------
        numpy.ndarray
            Данные с добавленным шумом
        """
        X_noisy = X.copy()
        mask = np.random.random(X.shape) < noise_level
        
        # Создаем случайные значения для импульсного шума (минимальные и максимальные значения)
        X_min, X_max = np.min(X), np.max(X)
        noise_values = np.random.choice([X_min, X_max], size=np.sum(mask))
        
        # Заменяем значения в маске шумовыми значениями
        X_noisy[mask] = noise_values
        return X_noisy
    
    @staticmethod
    def add_missing_values(X, missing_rate):
        """
        Добавление пропущенных значений с последующей заменой на среднее.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Исходные данные
        missing_rate : float
            Доля пропущенных значений (от 0 до 1)
            
        Returns:
        --------
        numpy.ndarray
            Данные с замененными пропущенными значениями
        """
        X_missing = X.copy()
        mask = np.random.random(X.shape) < missing_rate
        
        # Заменяем значения в маске на NaN
        X_missing[mask] = np.nan
        
        # Заменяем пропущенные значения на среднее по столбцу
        imputer = SimpleImputer(strategy='mean')
        X_imputed = imputer.fit_transform(X_missing)
        
        return X_imputed
    
    @staticmethod
    def add_noise(X, noise_type, noise_level):
        """
        Добавление выбранного типа шума.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Исходные данные
        noise_type : str
            Тип шума ('gaussian', 'uniform', 'impulse', 'missing')
        noise_level : float
            Уровень шума
            
        Returns:
        --------
        numpy.ndarray
            Данные с добавленным шумом
        """
        if noise_type == 'gaussian':
            return NoiseInjector.add_gaussian_noise(X, noise_level)
        elif noise_type == 'uniform':
            return NoiseInjector.add_uniform_noise(X, noise_level)
        elif noise_type == 'impulse':
            return NoiseInjector.add_impulse_noise(X, noise_level)
        elif noise_type == 'missing':
            return NoiseInjector.add_missing_values(X, noise_level)
        else:
            raise ValueError(f"Неизвестный тип шума: {noise_type}")


class AdaptiveEnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Адаптивный ансамбль классификаторов, где главная роль отведена нейронной сети,
    а остальные классификаторы используются в случае неуверенности нейросети.
    """
    
    def __init__(self, main_classifier, helper_classifiers, confidence_threshold=0.7):
        """
        Parameters:
        -----------
        main_classifier : object
            Основной классификатор (нейронная сеть)
        helper_classifiers : list
            Список вспомогательных классификаторов
        confidence_threshold : float
            Порог уверенности, ниже которого используется голосование вспомогательных классификаторов
        """
        self.main_classifier = main_classifier
        self.helper_classifiers = helper_classifiers
        self.confidence_threshold = confidence_threshold
        self.is_fitted = False
        self.main_scaler = StandardScaler()
        self.helper_scalers = [StandardScaler() for _ in helper_classifiers]
    
    def fit(self, X, y):
        """
        Обучение ансамбля.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Обучающие данные
        y : numpy.ndarray
            Целевые метки
            
        Returns:
        --------
        self : object
            Возвращает экземпляр класса
        """
        # Преобразование y в one-hot кодирование для нейронной сети
        # и сохранение исходных меток для остальных классификаторов
        self.classes_ = np.unique(y)
        self.n_classes_ = len(self.classes_)
        
        # Масштабирование данных для основного классификатора
        X_main = self.main_scaler.fit_transform(X)
        
        # Обучение основного классификатора
        print("Обучение основного классификатора (нейронная сеть)...")
        self.main_classifier.fit(X_main, y)
        
        # Обучение вспомогательных классификаторов
        print("Обучение вспомогательных классификаторов...")
        for i, classifier in enumerate(self.helper_classifiers):
            X_helper = self.helper_scalers[i].fit_transform(X)
            classifier.fit(X_helper, y)
        
        self.is_fitted = True
        return self
    
    def predict_proba(self, X):
        """
        Предсказание вероятностей принадлежности к классам.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Тестовые данные
            
        Returns:
        --------
        numpy.ndarray
            Вероятности принадлежности к каждому классу
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите метод fit перед predict_proba.")
        
        # Масштабирование данных для основного классификатора
        X_main = self.main_scaler.transform(X)
        
        # Получение предсказаний от основного классификатора
        probas_main = self.main_classifier.predict_proba(X_main)
        
        # Определение максимальной вероятности для каждого образца
        max_probas = np.max(probas_main, axis=1)
        
        # Нахождение образцов, где уверенность ниже порога
        low_confidence_idx = max_probas < self.confidence_threshold
        
        # Если нет образцов с низкой уверенностью, возвращаем предсказания основного классификатора
        if not np.any(low_confidence_idx):
            return probas_main
        
        # Создаем копию предсказаний основного классификатора
        final_probas = probas_main.copy()
        
        # Для образцов с низкой уверенностью запрашиваем предсказания от вспомогательных классификаторов
        X_low_conf = X[low_confidence_idx]
        
        # Накапливаем вероятности от вспомогательных классификаторов
        helper_probas = np.zeros((X_low_conf.shape[0], self.n_classes_))
        
        for i, classifier in enumerate(self.helper_classifiers):
            X_helper = self.helper_scalers[i].transform(X_low_conf)
            helper_probas += classifier.predict_proba(X_helper)
        
        # Усредняем вероятности от вспомогательных классификаторов
        helper_probas /= len(self.helper_classifiers)
        
        # Заменяем предсказания для неуверенных образцов
        final_probas[low_confidence_idx] = helper_probas
        
        return final_probas
    
    def predict(self, X):
        """
        Предсказание меток классов.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Тестовые данные
            
        Returns:
        --------
        numpy.ndarray
            Предсказанные метки классов
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Вызовите метод fit перед predict.")
        
        # Получение вероятностей
        probas = self.predict_proba(X)
        
        # Возвращаем метки классов с наибольшей вероятностью
        return self.classes_[np.argmax(probas, axis=1)]


def create_nn_model(input_dim, num_classes):
    """
    Создание модели нейронной сети.
    
    Parameters:
    -----------
    input_dim : int
        Размерность входных данных
    num_classes : int
        Количество классов
        
    Returns:
    --------
    tensorflow.keras.models.Sequential
        Модель нейронной сети
    """
    model = Sequential([
        Dense(128, activation='relu', input_shape=(input_dim,)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        BatchNormalization(),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


def optimize_nn_hyperparameters(X_train, y_train, X_val, y_val, n_trials=50):
    """
    Оптимизация гиперпараметров нейронной сети с использованием Optuna.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Обучающие данные
    y_train : numpy.ndarray
        Целевые метки для обучения
    X_val : numpy.ndarray
        Валидационные данные
    y_val : numpy.ndarray
        Целевые метки для валидации
    n_trials : int
        Количество испытаний для оптимизации
        
    Returns:
    --------
    dict
        Оптимальные гиперпараметры
    """
    def objective(trial):
        # Определение гиперпараметров для оптимизации
        learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
        units1 = trial.suggest_categorical('units1', [64, 128, 256])
        units2 = trial.suggest_categorical('units2', [32, 64, 128])
        dropout1 = trial.suggest_float('dropout1', 0.1, 0.5)
        dropout2 = trial.suggest_float('dropout2', 0.1, 0.4)
        batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
        
        # Создание модели с выбранными гиперпараметрами
        model = Sequential([
            Dense(units1, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(dropout1),
            Dense(units2, activation='relu'),
            BatchNormalization(),
            Dropout(dropout2),
            Dense(len(np.unique(y_train)), activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Ранняя остановка для предотвращения переобучения
        early_stopping = EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        )
        
        # Обучение модели
        history = model.fit(
            X_train, y_train,
            epochs=100,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Оценка модели на валидационном наборе
        loss, accuracy = model.evaluate(X_val, y_val, verbose=0)
        
        return accuracy
    
    # Создание исследования Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    print(f"Лучшие гиперпараметры: {study.best_params}")
    print(f"Лучшая точность: {study.best_value:.4f}")
    
    return study.best_params


def optimize_classifier_hyperparameters(classifier, X_train, y_train, param_grid):
    """
    Оптимизация гиперпараметров классификатора с использованием GridSearchCV.
    
    Parameters:
    -----------
    classifier : object
        Классификатор
    X_train : numpy.ndarray
        Обучающие данные
    y_train : numpy.ndarray
        Целевые метки
    param_grid : dict
        Сетка гиперпараметров для поиска
        
    Returns:
    --------
    object
        Классификатор с оптимальными гиперпараметрами
    """
    grid_search = GridSearchCV(
        classifier,
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Лучшие параметры: {grid_search.best_params_}")
    print(f"Лучшая точность: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_


def create_and_optimize_classifiers(X_train, y_train, X_val, y_val):
    """
    Создание и оптимизация классификаторов.
    
    Parameters:
    -----------
    X_train : numpy.ndarray
        Обучающие данные
    y_train : numpy.ndarray
        Целевые метки для обучения
    X_val : numpy.ndarray
        Валидационные данные
    y_val : numpy.ndarray
        Целевые метки для валидации
        
    Returns:
    --------
    tuple
        Кортеж из основного и вспомогательных классификаторов
    """
    # Количество классов
    n_classes = len(np.unique(y_train))
    
    # Оптимизация гиперпараметров нейронной сети
    print("Оптимизация гиперпараметров нейронной сети...")
    nn_params = optimize_nn_hyperparameters(X_train, y_train, X_val, y_val, n_trials=20)
    
    # Создание модели нейронной сети с оптимальными гиперпараметрами
    def create_optimized_nn(input_dim=X_train.shape[1], num_classes=n_classes):
        model = Sequential([
            Dense(nn_params['units1'], activation='relu', input_shape=(input_dim,)),
            BatchNormalization(),
            Dropout(nn_params['dropout1']),
            Dense(nn_params['units2'], activation='relu'),
            BatchNormalization(),
            Dropout(nn_params['dropout2']),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=nn_params['learning_rate']),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    # Создание и обучение KerasClassifier
    main_classifier = KerasClassifier(
        model=create_optimized_nn,
        epochs=100,
        batch_size=nn_params['batch_size'],
        verbose=0,
        callbacks=[EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)]
    )
    
    # Оптимизация гиперпараметров для вспомогательных классификаторов
    print("\nОптимизация гиперпараметров для случайного леса...")
    rf_classifier = RandomForestClassifier(random_state=RANDOM_SEED)
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    rf_optimized = optimize_classifier_hyperparameters(rf_classifier, X_train, y_train, rf_param_grid)
    
    print("\nОптимизация гиперпараметров для SVM...")
    svm_classifier = SVC(probability=True, random_state=RANDOM_SEED)
    svm_param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 0.01],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }
    svm_optimized = optimize_classifier_hyperparameters(svm_classifier, X_train, y_train, svm_param_grid)
    
    print("\nОптимизация гиперпараметров для KNN...")
    knn_classifier = KNeighborsClassifier()
    knn_param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    knn_optimized = optimize_classifier_hyperparameters(knn_classifier, X_train, y_train, knn_param_grid)
    
    print("\nОптимизация гиперпараметров для Gradient Boosting...")
    gb_classifier = GradientBoostingClassifier(random_state=RANDOM_SEED)
    gb_param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7]
    }
    gb_optimized = optimize_classifier_hyperparameters(gb_classifier, X_train, y_train, gb_param_grid)
    
    # Создание списка оптимизированных вспомогательных классификаторов
    helper_classifiers = [rf_optimized, svm_optimized, knn_optimized, gb_optimized]
    
    return main_classifier, helper_classifiers


def run_experiment(X, y, noise_type, start_noise, end_noise, step_noise):
    """
    Проведение эксперимента с различными уровнями шума.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Исходные данные
    y : numpy.ndarray
        Целевые метки
    noise_type : str
        Тип шума ('gaussian', 'uniform', 'impulse', 'missing')
    start_noise : float
        Начальный уровень шума
    end_noise : float
        Конечный уровень шума
    step_noise : float
        Шаг изменения уровня шума
        
    Returns:
    --------
    pandas.DataFrame
        Результаты эксперимента
    """
    # Создание списка уровней шума
    noise_levels = np.arange(start_noise, end_noise + step_noise, step_noise)
    
    # Инициализация словаря для хранения результатов
    results = {
        'noise_level': [],
        'ensemble_accuracy': [],
        'rf_accuracy': [],
        'svm_accuracy': [],
        'knn_accuracy': [],
        'gb_accuracy': [],
        'nn_accuracy': []
    }
    
    # Для каждого уровня шума
    for noise_level in tqdm(noise_levels, desc=f"Эксперимент с {noise_type} шумом"):
        # Сброс счетчика случайных чисел для воспроизводимости
        np.random.seed(RANDOM_SEED)
        tf.random.set_seed(RANDOM_SEED)
        
        # Добавление шума в данные
        X_noisy = NoiseInjector.add_noise(X, noise_type, noise_level)
        
        # Разделение данных на обучающий, валидационный и тестовый наборы
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X_noisy, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=0.25, random_state=RANDOM_SEED, stratify=y_train_val
        )
        
        # Создание и оптимизация классификаторов
        main_classifier, helper_classifiers = create_and_optimize_classifiers(X_train, y_train, X_val, y_val)
        
        # Обучение и оценка ансамбля
        ensemble = AdaptiveEnsembleClassifier(main_classifier, helper_classifiers)
        ensemble.fit(X_train, y_train)
        ensemble_pred = ensemble.predict(X_test)
        ensemble_accuracy = accuracy_score(y_test, ensemble_pred)
        
        # Масштабирование данных для одиночных классификаторов
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Обучение и оценка нейронной сети отдельно
        main_classifier.fit(X_train_scaled, y_train, validation_data=(X_val, y_val))
        nn_pred = main_classifier.predict(X_test_scaled)
        nn_accuracy = accuracy_score(y_test, nn_pred)
        
        # Обучение и оценка других одиночных классификаторов
        accuracies = []
        
        for classifier in helper_classifiers:
            classifier.fit(X_train_scaled, y_train)
            y_pred = classifier.predict(X_test_scaled)
            accuracies.append(accuracy_score(y_test, y_pred))
        
        # Сохранение результатов
        results['noise_level'].append(noise_level)
        results['ensemble_accuracy'].append(ensemble_accuracy)
        results['rf_accuracy'].append(accuracies[0])
        results['svm_accuracy'].append(accuracies[1])
        results['knn_accuracy'].append(accuracies[2])
        results['gb_accuracy'].append(accuracies[3])
        results['nn_accuracy'].append(nn_accuracy)
    
    # Преобразование результатов в DataFrame
    results_df = pd.DataFrame(results)
    
    return results_df


def run_multiple_experiments(X, y, noise_type, start_noise, end_noise, step_noise, n_experiments=5):
    """
    Проведение нескольких экспериментов и усреднение результатов.
    
    Parameters:
    -----------
    X : numpy.ndarray
        Исходные данные
    y : numpy.ndarray
        Целевые метки
    noise_type : str
        Тип шума ('gaussian', 'uniform', 'impulse', 'missing')
    start_noise : float
        Начальный уровень шума
    end_noise : float
        Конечный уровень шума
    step_noise : float
        Шаг изменения уровня шума
    n_experiments : int
        Количество экспериментов
        
    Returns:
    --------
    pandas.DataFrame
        Усредненные результаты экспериментов
    """
    # Список для хранения результатов экспериментов
    experiment_results = []
    
    # Проведение n экспериментов
    for i in range(n_experiments):
        print(f"\nЭксперимент {i+1}/{n_experiments}")
        result = run_experiment(X, y, noise_type, start_noise, end_noise, step_noise)
        experiment_results.append(result)
    
    # Объединение всех результатов
    all_results = pd.concat(experiment_results)
    
    # Группировка по уровню шума и вычисление среднего и стандартного отклонения
    grouped_results = all_results.groupby('noise_level').agg(['mean', 'std'])
    
    # Преобразование формата результатов для удобства использования
    final_results = pd.DataFrame()
    final_results['noise_level'] = grouped_results.index
    
    for model in ['ensemble', 'rf', 'svm', 'knn', 'gb', 'nn']:
        final_results[f'{model}_accuracy'] = grouped_results[f'{model}_accuracy']['mean']
        final_results[f'{model}_std'] = grouped_results[f'{model}_accuracy']['std']
    
    return final_results


def visualize_noise_effect(results, noise_type):
    """
    Визуализация влияния шума на точность классификации для ансамбля.
    
    Parameters:
    -----------
    results : pandas.DataFrame
        Результаты экспериментов
    noise_type : str
        Тип шума
    """
    plt.figure(figsize=(12, 8))
    
    plt.errorbar(
        results['noise_level'],
        results['ensemble_accuracy'],
        yerr=results['ensemble_std'],
        fmt='-o',
        capsize=5,
        label='Адаптивный ансамбль',
        linewidth=2
    )
    
    plt.xlabel('Уровень шума')
    plt.ylabel('Точность классификации')
    plt.title(f'Влияние {noise_type} шума на точность классификации ансамбля')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Сохранение графика
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{noise_type}_noise_effect.png', dpi=300, bbox_inches='tight')
    plt.close()


def visualize_comparative_results(results, noise_type):
    """
    Визуализация сравнительных результатов для различных моделей.
    
    Parameters:
    -----------
    results : pandas.DataFrame
        Результаты экспериментов
    noise_type : str
        Тип шума
    """
    plt.figure(figsize=(12, 8))
    
    models = ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']
    model_names = ['Адаптивный ансамбль', 'Нейронная сеть', 'Случайный лес', 'SVM', 'KNN', 'Gradient Boosting']
    colors = sns.color_palette("Set2", len(models))
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        plt.errorbar(
            results['noise_level'],
            results[f'{model}_accuracy'],
            yerr=results[f'{model}_std'],
            fmt='-o',
            capsize=5,
            label=name,
            color=colors[i],
            linewidth=2 if model == 'ensemble' else 1.5,
            alpha=0.9 if model == 'ensemble' else 0.7
        )
    
    plt.xlabel('Уровень шума')
    plt.ylabel('Точность классификации')
    plt.title(f'Сравнение моделей при различных уровнях {noise_type} шума')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Сохранение графика
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/{noise_type}_comparative_results.png', dpi=300, bbox_inches='tight')
    plt.close()


def print_summary_results(all_results):
    """
    Вывод сводных результатов экспериментов.
    
    Parameters:
    -----------
    all_results : dict
        Словарь с результатами для разных типов шума
    """
    print("\n===== СВОДНЫЕ РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТОВ =====")
    
    # Создание таблицы для сравнения средней точности по всем уровням шума
    summary_data = {
        'Модель': ['Адаптивный ансамбль', 'Нейронная сеть', 'Случайный лес', 'SVM', 'KNN', 'Gradient Boosting'],
        'Модель (код)': ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']
    }
    
    for noise_type in all_results.keys():
        results = all_results[noise_type]
        
        # Вычисление средней точности для каждой модели по всем уровням шума
        for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']:
            mean_accuracy = results[f'{model}_accuracy'].mean()
            
            if noise_type not in summary_data:
                summary_data[noise_type] = []
            
            summary_data[noise_type].append(mean_accuracy)
    
    # Создание DataFrame
    summary_df = pd.DataFrame(summary_data)
    
    # Установка формата отображения чисел
    pd.set_option('display.float_format', '{:.4f}'.format)
    
    # Вывод таблицы
    print("\nСредняя точность по всем уровням шума:")
    print(summary_df[['Модель'] + list(all_results.keys())])
    
    # Вычисление и вывод рейтинга моделей
    ranking_df = summary_df.copy()
    model_codes = ranking_df['Модель (код)']
    ranking_df = ranking_df.drop(columns=['Модель', 'Модель (код)'])
    
    # Применение ранжирования для каждого типа шума
    for column in ranking_df.columns:
        ranking_df[f'{column}_rank'] = ranking_df[column].rank(ascending=False)
    
    # Вычисление среднего ранга
    rank_columns = [col for col in ranking_df.columns if col.endswith('_rank')]
    ranking_df['Средний ранг'] = ranking_df[rank_columns].mean(axis=1)
    
    # Сортировка по среднему рангу
    ranking_df = ranking_df.sort_values('Средний ранг')
    
    # Добавление названий моделей
    ranking_df['Модель'] = summary_df['Модель']
    ranking_df['Модель (код)'] = model_codes
    
    # Выбор и переупорядочивание столбцов для отображения
    columns_to_display = ['Модель', 'Средний ранг'] + rank_columns
    
    print("\nРейтинг моделей (меньше - лучше):")
    print(ranking_df[columns_to_display])
    
    # Дополнительная аналитика - улучшение точности ансамбля относительно других моделей
    print("\nУлучшение точности ансамбля относительно других моделей (в процентных пунктах):")
    
    for noise_type in all_results.keys():
        print(f"\n{noise_type.capitalize()} шум:")
        
        for model in ['nn', 'rf', 'svm', 'knn', 'gb']:
            model_name = {
                'nn': 'Нейронная сеть',
                'rf': 'Случайный лес',
                'svm': 'SVM',
                'knn': 'KNN',
                'gb': 'Gradient Boosting'
            }[model]
            
            # Вычисление разницы между ансамблем и моделью
            diff = summary_df[summary_df['Модель (код)'] == 'ensemble'][noise_type].values[0] - \
                   summary_df[summary_df['Модель (код)'] == model][noise_type].values[0]
            
            print(f"  vs {model_name}: {diff:.4f}")
    
    # Создание сводной таблицы со статистикой
    print("\n===== СТАТИСТИКА ПО ВСЕМ ЭКСПЕРИМЕНТАМ =====")
    
    for noise_type, results in all_results.items():
        print(f"\n{noise_type.capitalize()} шум:")
        
        # Максимальная точность для каждой модели
        max_accuracies = {model: results[f'{model}_accuracy'].max() for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']}
        
        # Минимальная точность для каждой модели
        min_accuracies = {model: results[f'{model}_accuracy'].min() for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']}
        
        # Относительное падение точности (от максимума к минимуму) в процентах
        rel_decrease = {model: (max_accuracies[model] - min_accuracies[model]) / max_accuracies[model] * 100 
                         for model in ['ensemble', 'nn', 'rf', 'svm', 'knn', 'gb']}
        
        # Вывод статистики
        print("  Максимальная точность:")
        for model, acc in max_accuracies.items():
            model_name = {
                'ensemble': 'Адаптивный ансамбль',
                'nn': 'Нейронная сеть',
                'rf': 'Случайный лес',
                'svm': 'SVM',
                'knn': 'KNN',
                'gb': 'Gradient Boosting'
            }[model]
            
            print(f"    {model_name}: {acc:.4f}")
        
        print("\n  Минимальная точность:")
        for model, acc in min_accuracies.items():
            model_name = {
                'ensemble': 'Адаптивный ансамбль',
                'nn': 'Нейронная сеть',
                'rf': 'Случайный лес',
                'svm': 'SVM',
                'knn': 'KNN',
                'gb': 'Gradient Boosting'
            }[model]
            
            print(f"    {model_name}: {acc:.4f}")
        
        print("\n  Относительное падение точности (%):")
        for model, dec in rel_decrease.items():
            model_name = {
                'ensemble': 'Адаптивный ансамбль',
                'nn': 'Нейронная сеть',
                'rf': 'Случайный лес',
                'svm': 'SVM',
                'knn': 'KNN',
                'gb': 'Gradient Boosting'
            }[model]
            
            print(f"    {model_name}: {dec:.2f}%")
        
        # Вычисление устойчивости моделей относительно друг друга
        ensemble_decrease = rel_decrease['ensemble']
        
        print("\n  Относительная устойчивость к шуму (по сравнению с ансамблем):")
        for model, dec in rel_decrease.items():
            if model == 'ensemble':
                continue
                
            model_name = {
                'nn': 'Нейронная сеть',
                'rf': 'Случайный лес',
                'svm': 'SVM',
                'knn': 'KNN',
                'gb': 'Gradient Boosting'
            }[model]
            
            # Отношение падения точности ансамбля к падению точности модели
            # Меньше 1 означает, что ансамбль более устойчив к шуму
            rel_stability = ensemble_decrease / dec if dec > 0 else float('inf')
            
            print(f"    {model_name}: {rel_stability:.4f}")


def main():
    """
    Основная функция программы.
    """
    print("===== ПРОГРАММНЫЙ КОМПЛЕКС ДЛЯ РЕШЕНИЯ ЗАДАЧИ КЛАССИФИКАЦИИ ЗАШУМЛЕННЫХ ДАННЫХ =====")
    print("Автор: [Ваше имя]")
    print("Дата: [Текущая дата]")
    print("\n")
    
    # Загрузка и подготовка данных
    print("Загрузка данных...")
    # В данном примере используем датасет Breast Cancer
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Стандартизация данных
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    print(f"Загружен датасет: {data.DESCR.split('==')[0].strip()}")
    print(f"Количество образцов: {X.shape[0]}, количество признаков: {X.shape[1]}")
    print(f"Количество классов: {len(np.unique(y))}")
    
    # Параметры шума для экспериментов
    noise_params = {
        'gaussian': {'start': 0.1, 'end': 1.0, 'step': 0.3},
        'uniform': {'start': 0.1, 'end': 1.0, 'step': 0.3},
        'impulse': {'start': 0.05, 'end': 0.4, 'step': 0.1},
        'missing': {'start': 0.05, 'end': 0.4, 'step': 0.1}
    }
    
    # Словарь для хранения результатов всех экспериментов
    all_results = {}
    
    # Проведение экспериментов для каждого типа шума
    for noise_type, params in noise_params.items():
        print(f"\n===== ЭКСПЕРИМЕНТЫ С {noise_type.upper()} ШУМОМ =====")
        
        results = run_multiple_experiments(
            X, y,
            noise_type=noise_type,
            start_noise=params['start'],
            end_noise=params['end'],
            step_noise=params['step'],
            n_experiments=5
        )
        
        all_results[noise_type] = results
        
        # Визуализация результатов
        visualize_noise_effect(results, noise_type)
        visualize_comparative_results(results, noise_type)
        
        print(f"\nРезультаты для {noise_type} шума:")
        print(results)
    
    # Вывод сводных результатов
    print_summary_results(all_results)
    
    print("\n===== ЭКСПЕРИМЕНТЫ ЗАВЕРШЕНЫ =====")
    print("Результаты и визуализации сохранены в директории 'results'")


if __name__ == "__main__":
    main()