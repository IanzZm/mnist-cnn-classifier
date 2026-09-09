# Classificador de Imagens (MNIST)

Rede neural pra reconhecer dígitos escritos à mão, comparando MLP com CNN

## O que faz
- Carrega o dataset MNIST (60k treino, 10k teste)
- Treina uma MLP simples como baseline
- Evolui pra uma CNN (Conv2D + MaxPooling)
- Usa EarlyStopping pra parar o treino sozinho, evitando overfitting
- CNN final: accuracy 0.9835 / loss 0.0557 (MLP: 0.9758 / 0.0824)

## Estrutura
- `src/modelo_cnn.py` — treino do modelo final (CNN)
- `notebooks/mlp_mnist.ipynb` — baseline com MLP
- `notebooks/cnn_mnist.ipynb` — construção e comparação da CNN

## Arquitetura
Input(28,28,1) → Conv2D(5, 3x3, relu) → MaxPooling2D(2x2) → Flatten → Dense(128, relu) → Dense(64, relu) → Dense(10, softmax)

## Instalação
Para rodar os notebooks com TensorFlow e MLflow:

```bash
pip install tensorflow mlflow tensorboard
```

O `tensorboard` é necessário quando o `mlflow.autolog()` registra métricas do TensorFlow.

## Como rodar
```python
import sys
sys.path.append('..')

import tensorflow as tf
from src.modelo_cnn import treinar_modelo_cnn

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28, 28, 1) / 255
x_test = x_test.reshape(-1, 28, 28, 1) / 255

modelo = treinar_modelo_cnn(x_train, y_train)

metricas = modelo.evaluate(x_test, y_test)
previsoes = modelo.predict(x_test)
```
