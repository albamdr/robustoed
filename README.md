# robustoed

Paquete Python para Diseño Óptimo de Experimentos (OED) robusto en modelos no lineales, con implementación actual orientada a criterio D-optimización y validación sobre el ejemplo de Baranyi del paper de referencia (https://doi.org/10.1016/j.chemolab.2024.105104).

## Estado actual

Versión funcional / prototipo técnico.

Actualmente incluye:

- construcción de modelos simbólicos con SymPy,
- cálculo de la matriz de información de Fisher (FIM),
- diseño D-óptimo local mediante algoritmo tipo Wynn–Fedorov,
- análisis de sensibilidad frente a incertidumbre paramétrica,
- screening de parámetros sensibles,
- augmentación robusta en dos pasos,
- replicación del caso Baranyi del paper.

## Estructura del proyecto

robust-oed/
├── pyproject.toml
├── README.md
├── examples/
│   ├── sensitivity_formal.py
│   ├── screen_params_demo.py
│   ├── manual_screen_then_augment.py
│   ├── baranyi_screening.py
│   └── baranyi_augment.py
├── src/
│   └── robustoed/
│       ├── __init__.py
│       ├── model_sympy.py
│       ├── fisher.py
│       ├── criterion_d.py
│       ├── regularize.py
│       ├── grid.py
│       ├── optim.py
│       ├── sensitivity.py
│       ├── screening.py
│       ├── augment.py
│       └── types.py
└── tests/

## Requisitos

- Python >= 3.9
- numpy
- scipy
- sympy
- matplotlib (solo para algunos ejemplos)

## Instalación

python -m venv .venv
source .venv/bin/activate
pip install -e .
pip install matplotlib

## Flujo de uso recomendado

1. definir el modelo no lineal
2. fijar valores nominales de parámetros
3. calcular el diseño D-óptimo local
4. hacer screening de parámetros
5. elegir manualmente parámetros sensibles
6. ejecutar augmentación robusta
7. evaluar robustez final

## Componentes principales

SympyModel  
Modelo simbólico con cálculo automático del jacobiano.

wynn_fedorov_d_opt  
Calcula diseño D-óptimo local.

sensitivity_report_d_vs_scenario_optimum  
Evalúa eficiencia frente a escenarios.

screen_uncertain_parameters_d  
Ranking de sensibilidad por parámetro.

robust_augment_two_step  
Algoritmo de augmentación robusta.

## Scripts de ejemplo

examples/sensitivity_formal.py  
Ejemplo básico de sensibilidad.

examples/screen_params_demo.py  
Ejemplo básico de screening.

examples/manual_screen_then_augment.py  
Flujo completo simple.

examples/baranyi_screening.py  
Screening del modelo Baranyi.

examples/baranyi_augment.py  
Augmentación robusta Baranyi.


## Entrada

- modelo en SymPy
- parámetros nominales (dict)
- grid de incertidumbre
- bounds

## Salida

- Design (puntos de soporte, pesos)
- informes de sensibilidad
- resultados de screening
- resultados de augmentación

## Limitaciones

- solo criterio D-opt
- tiempos altos en ejemplos complejos

## Testing

Validación mediante ejemplos:

- casos simples
- screening
- Baranyi screening
- Baranyi augmentation

## Referencia

Validado con el ejemplo Baranyi del paper de referencia.
