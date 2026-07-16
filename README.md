# Pipeline base de Deep Learning — Data Scientist III

Primer checkpoint del proyecto final. Implementa un pipeline reproducible de entrenamiento y validación en PyTorch utilizando el dataset Iris.

## Estructura

```text
ds3_pipeline_base/
├── data/
│   └── README.md
├── src/
│   └── train.py
├── results/              # se genera al ejecutar
├── .gitignore
├── README.md
└── requirements.txt
```

## Entorno y versiones

- Python 3.10 o superior.
- PyTorch 2.13.0.
- Dispositivo detectado automáticamente: CUDA, MPS o CPU.
- Semilla global: `42`.

## Instalación y ejecución

```bash
python -m venv .venv
```

En Windows:

```bash
.venv\Scripts\activate
pip install -r requirements.txt
python src/train.py
```

En Linux o macOS:

```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/train.py
```

## Dataset y preparación

Se usa Iris, incluido en scikit-learn: 150 observaciones, cuatro variables y tres clases. La división es estratificada: 80% entrenamiento y 20% validación. `StandardScaler` se ajusta únicamente con entrenamiento para evitar fuga de información.

## Arquitectura e hiperparámetros

- MLP: `Linear(4, 16) → ReLU → Linear(16, 3)`.
- Loss: `CrossEntropyLoss`.
- Optimizador: Adam.
- Learning rate: `0.001`.
- Batch size: `16`.
- Épocas: `100`.

El learning rate `0.001` es el valor inicial estándar de Adam y ofrece actualizaciones estables para esta red pequeña.

## Entrenamiento y validación

Cada época registra `loss` y `accuracy` para entrenamiento y validación. El ciclo aplica explícitamente:

1. Forward pass.
2. Cálculo de la pérdida.
3. `optimizer.zero_grad()`.
4. `loss.backward()`.
5. `optimizer.step()`.

La validación utiliza `torch.no_grad()` y datos no vistos durante el ajuste.

## Resultados e interpretación

La ejecución genera:

- `results/metrics.json`: versiones, configuración y métricas por época.
- `results/training_curves.png`: curvas de loss y accuracy.
- `results/iris_mlp.pt`: pesos entrenados.

Las métricas y la imagen de las curvas se versionan como evidencia del experimento. El archivo de pesos `.pt` queda excluido mediante `.gitignore`.

La interpretación esperada es que la pérdida de entrenamiento y validación disminuya durante las épocas y que el accuracy aumente. Si la pérdida de entrenamiento continúa bajando mientras la de validación sube de forma sostenida, existe evidencia de overfitting. Los valores concretos quedan registrados automáticamente en `metrics.json` para documentar el experimento real.

## Reproducibilidad

Se fijan las semillas de Python, NumPy y PyTorch. En CUDA también se activan opciones deterministas de cuDNN. Pueden existir pequeñas diferencias entre backends o versiones.

## Commits sugeridos

```text
chore: create project structure and dependencies
feat: add reproducible Iris data pipeline
feat: implement MLP training and validation loops
docs: document experiment setup and interpretation
```
