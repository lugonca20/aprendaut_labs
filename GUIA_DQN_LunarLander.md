# Guía para completar el agente DQN de `LunarLander_DQN_Tarea.ipynb`

> Este documento **no reemplaza** el trabajo de programar: resume qué pide el
> enunciado, qué conceptos hay que dominar y propone una idea de solución para
> cada uno de los 3 `TODO`, tomando como base la implementación de DQN ya
> resuelta en [FrozenGymQLAndDQN_Tarea.ipynb](/c:/Users/Pc/Desktop/Facu-Ingenieria/AprendAut/AprendAut26/Tarea2/FrozenGymQLAndDQN_Tarea.ipynb).

## 1. Qué pide el enunciado (resumen)

El cuaderno de LunarLander ya trae armado todo el andamiaje (entorno, interfaz
`Agente`, funciones `entrenar_agente` / `evaluar_agente`, gráficas y el GIF de
la política). Lo que falta son **3 bloques marcados con `TODO`**, que hoy
cortan la ejecución con `NotImplementedError`:

| # | Sección | Qué pide | Celda |
|---|---------|----------|-------|
| **TODO 1** | 6. El agente DQN | Implementar la clase `AgenteDQN` completa (constructor, `elegir_accion`, `aprender`, `fin_episodio`) | [celda 15](/c:/Users/Pc/Desktop/Facu-Ingenieria/AprendAut/AprendAut26/Tarea2/LunarLander_DQN_Tarea.ipynb) |
| **TODO 2** | 10. Qué aprendió la política | Analizar la política aprendida con "cortes": fijar algunas dimensiones del estado y variar otras | celda 26 |
| **TODO 3** | 11. Estudio de ablación | Comparar el DQN completo contra ≥3 variantes que le quitan un componente, con varias semillas y un test estadístico | celda 28 |

Requisito mínimo: **tasa de éxito ≥ 10 %** en evaluación (episodios con
recompensa ≥ 200, sin exploración). Además del cuaderno, se entrega un
**artículo IEEE de ≤5 páginas** con la discusión de resultados (eso no va en
el notebook).

## 2. Conceptos clave que hay que tener presentes

### 2.1 Por qué no alcanza con Q-Learning tabular
En FrozenLake el estado es un entero discreto (0 a 15 en el mapa 4x4) y la
Q-table tiene una fila por estado. En **LunarLander-v3** el estado es un
vector continuo de 8 componentes (`Box(8,)`): posición, velocidad, ángulo,
velocidad angular y contacto de las patas. No se puede indexar una tabla con
valores reales, por eso se usa una red neuronal como **aproximador de la
función Q**: `Q(s, a; θ) ≈ Q*(s, a)`.

### 2.2 La ecuación de Bellman y el objetivo de entrenamiento
El agente busca minimizar la diferencia entre:
- el valor estimado actual `Q(s, a; θ)`, y
- el **target TD**: `y = r + γ · max_a' Q(s', a'; θ⁻) · (1 - terminado)`

donde `θ⁻` son los pesos de una **red objetivo** (ver 2.3). Esto es
exactamente lo que ya hace `AgenteQLearning` en FrozenLake con su `td_target`,
solo que ahí `max_a' Q(s', a')` sale de la Q-table y acá sale de una red.

### 2.3 Los tres "trucos" que hacen estable a DQN
Son los mismos que ya están implementados en `AgenteDQN` de FrozenLake y hay
que replicar (adaptándolos) en LunarLander:

1. **Experience Replay** (`self.memoria = deque(maxlen=...)`): se guardan
   transiciones `(s, a, r, s', terminado)` y se entrena con minibatches
   aleatorios, no con la transición recién vivida. Rompe la correlación
   temporal entre muestras consecutivas y permite reusar experiencia.
2. **Red objetivo (target network)**: una copia congelada de la red principal
   que se actualiza cada *N* pasos (`actualizar_red_objetivo` en el ejemplo).
   Sin ella, el target se mueve a la vez que se actualiza `Q`, y el
   entrenamiento diverge (está "persiguiendo su propia cola").
3. **Exploración ε-greedy con decaimiento**: empezar con `epsilon=1`
   (totalmente aleatorio) y bajarlo con el entrenamiento hasta un mínimo
   (`epsilon_min`), para pasar de explorar a explotar la política aprendida.

Detalles adicionales que usa el ejemplo y conviene mantener:
- **`SmoothL1Loss` (Huber loss)** en lugar de MSE: penaliza menos los errores
  grandes (más robusto a rachas de transiciones con recompensa muy negativa,
  como un choque en LunarLander).
- **`torch.no_grad()`** al calcular el target con la red objetivo: no hay que
  propagar gradiente por ahí.
- **`.gather(1, acciones)`**: de los Q-valores de salida (uno por acción) nos
  quedamos solo con el de la acción que efectivamente se tomó.

### 2.4 Diferencias importantes entre el ejemplo (FrozenLake) y la tarea (LunarLander)

| | FrozenLake (ejemplo ya resuelto) | LunarLander (a implementar) |
|---|---|---|
| Estado | entero discreto (16 o 64 estados) | vector continuo `float32[8]` |
| Codificación de entrada | one-hot (`torch.eye(n)[estado]`) | el vector de estado tal cual (ya viene en un rango acotado, no hace falta one-hot) |
| Longitud de episodio | ≤100-150 pasos | hasta 1000 pasos (`TimeLimit`) |
| `terminado` vs `truncado` | se mezclan (`terminado or truncado`) porque no afecta mucho en una grilla chica | **hay que distinguirlos**: el enunciado ya lo resuelve en `entrenar_agente` pasándole `terminado` puro al `aprender()`. Si truncás por tiempo, el estado siguiente *sigue teniendo valor futuro* (no hay que cortar el bootstrap ahí) |
| Señal de recompensa | dispersa (0 en casi todos los pasos, 1 al llegar) | densa y con componentes negativas (combustible, choque) |
| Decaimiento de ε | por paso (`epsilon -= epsilon_decay` en cada `aprender`) | más cómodo por episodio, usando el *hook* `fin_episodio()` que ya expone la interfaz `Agente` |
| Firma de `elegir_accion` | `elegir_accion(estado)` | `elegir_accion(estado, explorar=True)` — la evaluación y el GIF llaman con `explorar=False`, hay que respetarlo devolviendo la acción greedy sin ruido |

Esto último es importante: la interfaz `Agente` en LunarLander agrega
`explorar` a `elegir_accion` y el método opcional `fin_episodio()`. Si no se
implementan con esa firma exacta, el resto del cuaderno (evaluación, GIF) no
va a funcionar aunque el agente "aprenda" bien.

## 3. TODO 1 — Implementar `AgenteDQN`: idea de solución

### 3.1 Arquitectura de red
Igual que en FrozenLake (una clase `DQN(nn.Module)` separada), pero con la
entrada/salida que corresponde a este entorno:

```python
class RedQ(nn.Module):
    def __init__(self, dim_entrada=8, dim_salida=4, ocultas=(128, 128)):
        super().__init__()
        capas = []
        anterior = dim_entrada
        for oculta in ocultas:
            capas += [nn.Linear(anterior, oculta), nn.ReLU()]
            anterior = oculta
        capas.append(nn.Linear(anterior, dim_salida))
        self.red = nn.Sequential(*capas)

    def forward(self, x):
        return self.red(x)
```

Con una sola capa oculta de 128 (como en el ejemplo de FrozenLake) alcanza
para pasar el umbral del 10 %; con dos capas ocultas (128, 128) se acerca
bastante más rápido a resolver el entorno (≥200 de recompensa media). No hace
falta nada más complejo (sin convoluciones: la entrada ya es un vector de 8
valores, no una imagen).

### 3.2 Hiperparámetros de partida sugeridos

| Hiperparámetro | Valor sugerido | Comentario |
|---|---|---|
| `gamma` | 0.99 | igual que en FrozenLake |
| `lr` | 5e-4 (o 1e-3) | Adam; si la curva oscila mucho, bajarlo |
| `epsilon` inicial | 1.0 | |
| `epsilon_decay` | multiplicativo, ~0.995 **por episodio** | ver nota de 2.4 |
| `epsilon_min` | 0.01 | |
| `memoria_max` | 50 000 – 100 000 | LunarLander necesita más buffer que FrozenLake (episodios más largos y ricos) |
| `batch_size` | 64 | |
| `contador_tope` (frecuencia de sync de la red objetivo) | cada 500-1000 *pasos* (no episodios) | |
| warm-up | no entrenar hasta tener al menos `batch_size` transiciones en memoria | igual que en el ejemplo |

Estos son puntos de partida razonables, no garantía: documentar en el
artículo IEEE qué se probó y por qué se eligió la configuración final (ahí
puede entrar el estudio de ablación del TODO 3 como evidencia).

### 3.3 Esqueleto de la clase (adaptado del ejemplo de FrozenLake)

```python
class AgenteDQN(Agente):
    def __init__(self, estado_space, accion_space, semilla=None,
                 ocultas=(128, 128), lr=5e-4, gamma=0.99,
                 epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01,
                 memoria_max=100_000, batch_size=64, pasos_sync_objetivo=1000):
        if semilla is not None:
            random.seed(semilla); np.random.seed(semilla)
            torch.manual_seed(semilla); accion_space.seed(semilla)

        self.dim_estado = estado_space.shape[0]      # 8, NO n (es continuo)
        self.n_acciones = accion_space.n              # 4

        self.gamma = gamma
        self.epsilon, self.epsilon_decay, self.epsilon_min = epsilon, epsilon_decay, epsilon_min
        self.batch_size = batch_size
        self.pasos_sync_objetivo = pasos_sync_objetivo
        self.pasos_dados = 0

        self.red_q = RedQ(self.dim_estado, self.n_acciones, ocultas)
        self.red_objetivo = RedQ(self.dim_estado, self.n_acciones, ocultas)
        self.red_objetivo.load_state_dict(self.red_q.state_dict())
        self.red_objetivo.eval()

        self.criterio = nn.SmoothL1Loss()
        self.optimizador = optim.Adam(self.red_q.parameters(), lr=lr)
        self.memoria = deque(maxlen=memoria_max)

    def elegir_accion(self, estado, explorar=True):
        if explorar and random.random() < self.epsilon:
            return random.randrange(self.n_acciones)
        estado_t = torch.as_tensor(estado, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_valores = self.red_q(estado_t)
        return int(torch.argmax(q_valores).item())

    def aprender(self, estado, accion, recompensa, siguiente_estado, terminado):
        self.memoria.append((estado, accion, recompensa, siguiente_estado, terminado))
        self.pasos_dados += 1

        if self.pasos_dados % self.pasos_sync_objetivo == 0:
            self.red_objetivo.load_state_dict(self.red_q.state_dict())

        if len(self.memoria) < self.batch_size:
            return

        batch = random.sample(self.memoria, self.batch_size)
        estados, acciones, recompensas, siguientes, terminados = zip(*batch)

        estados = torch.as_tensor(np.array(estados), dtype=torch.float32)
        acciones = torch.as_tensor(acciones, dtype=torch.long).unsqueeze(1)
        recompensas = torch.as_tensor(recompensas, dtype=torch.float32).unsqueeze(1)
        siguientes = torch.as_tensor(np.array(siguientes), dtype=torch.float32)
        terminados = torch.as_tensor(terminados, dtype=torch.float32).unsqueeze(1)

        q_valores = self.red_q(estados).gather(1, acciones)
        with torch.no_grad():
            q_siguiente_max = self.red_objetivo(siguientes).max(1, keepdim=True)[0]
            objetivo = recompensas + (1 - terminados) * self.gamma * q_siguiente_max

        perdida = self.criterio(q_valores, objetivo)
        self.optimizador.zero_grad()
        perdida.backward()
        # Opcional pero recomendado en LunarLander: evita explosiones de gradiente
        nn.utils.clip_grad_norm_(self.red_q.parameters(), max_norm=10)
        self.optimizador.step()

    def fin_episodio(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
```

Puntos a cuidar respecto del ejemplo de FrozenLake:
- **No hay one-hot**: el estado ya es un vector de floats; usar
  `torch.as_tensor(estado, dtype=torch.float32)` directamente (y
  `estado_space.shape[0]` en vez de `estado_space.n`, porque es un `Box`).
- El decaimiento de `epsilon` se mueve al `fin_episodio()` (por episodio), que
  es el *hook* que la función `entrenar_agente` del cuaderno ya invoca; en
  FrozenLake se hacía dentro de `aprender()` por paso.
- Respetar el parámetro `explorar` de `elegir_accion`: si es `False` nunca
  tirar el dado, aunque `epsilon > 0`.
- Como en el entorno hay episodios largos (hasta 1000 pasos), conviene medir
  `pasos_sync_objetivo` en **pasos de entorno**, no en episodios.
- `clip_grad_norm_` no está en el ejemplo de FrozenLake, pero en LunarLander
  (recompensas con magnitudes de ±100) ayuda a que no diverja el entrenamiento;
  queda como mejora opcional.

## 4. TODO 2 — "Cortes de la política": idea de solución

La idea es **fijar algunas de las 8 dimensiones del estado** (las que no se
quieren estudiar) y **barrer** una o dos dimensiones de interés, consultando
qué acción (o qué Q-valores) devuelve la red en cada punto de la grilla. Es el
análogo continuo de `mostrar_política()` en FrozenLake (ahí se podía mostrar
la política completa porque había 16 estados; acá hay infinitos, así que se
"cortan" rebanadas 2D).

Ejemplos concretos de cortes útiles para LunarLander:

1. **Posición vs. acción, nave quieta y nivelada.**
   Fijar `vx=vy=0`, `θ=0`, `ω=0`, `c_izq=c_der=0`; barrer `x ∈ [-1, 1]` e
   `y ∈ [0, 1.5]` en una grilla y graficar con `imshow`/`pcolormesh` la acción
   greedy (o el Q-valor máximo) en cada celda. Se espera ver que cerca del
   centro y bajo se prefiera "motor principal", y a los costados se usen los
   motores laterales para corregir `x`.

2. **Ángulo vs. velocidad angular, en el centro de la plataforma.**
   Fijar `x=y=0` (o `y` bajo, como si estuviera por tocar tierra),
   `vx=vy=0`, patas sin contacto; barrer `θ` y `ω`. Debería verse una
   política "simétrica": si `θ>0` (inclinado a la derecha) se espera que
   dispare el motor lateral que lo corrija, y lo contrario para `θ<0`.

3. **Efecto del contacto de las patas.**
   Repetir el corte 1 pero con `c_izq=c_der=1`: la política debería volverse
   mucho más pasiva (acción "no hacer nada" dominante), porque ya aterrizó.

Pseudocódigo general reutilizable:

```python
def cortar_politica(agente, fijos: dict, var1, rango1, var2, rango2, n=60):
    """fijos: dict índice->valor para las dimensiones que no varían.
    var1, var2: índices de las dos dimensiones a barrer."""
    acciones_mapa = np.zeros((len(rango2), len(rango1)), dtype=int)
    for i, v2 in enumerate(rango2):
        for j, v1 in enumerate(rango1):
            estado = np.zeros(8, dtype=np.float32)
            for k, val in fijos.items():
                estado[k] = val
            estado[var1], estado[var2] = v1, v2
            acciones_mapa[i, j] = agente.elegir_accion(estado, explorar=False)
    return acciones_mapa
```

Y graficar con `plt.imshow(acciones_mapa, extent=[...])` + una leyenda de
colores por acción (0 = nada, 1 = lateral izq., 2 = principal, 3 = lateral
der.). Para el informe, interpretar si la política tiene sentido físico
(simetrías esperadas, reacción monótona a la inclinación, etc.) — ese
análisis cualitativo es justamente lo que pide la sección 10.

## 5. TODO 3 — Estudio de ablación: idea de solución

### 5.1 Qué variantes comparar (mínimo 3, quitando una pieza de DQN cada vez)
Usando como base los "trucos" de la sección 2.3, variantes típicas y fáciles
de armar reusando la misma clase `AgenteDQN` con *flags*:

1. **Sin experience replay** (o buffer mínimo = batch): entrenar con la
   transición actual nomás (o un buffer de tamaño `batch_size`, sin
   verdadera diversidad de muestras). Se espera mayor varianza/inestabilidad.
2. **Sin red objetivo** (usar la misma red para elegir y para el target, o
   sincronizar en cada paso, `pasos_sync_objetivo=1`): se espera divergencia
   o entrenamiento mucho más ruidoso/lento en converger.
3. **Sin decaimiento de ε** (ε fijo, p. ej. 0.1 o 1.0 todo el entrenamiento):
   se espera o bien política que no termina de explotar lo aprendido (ε alto
   fijo) o poca exploración inicial (ε bajo fijo) y convergencia más pobre.
4. *(Opcional, cuarta variante)* **Huber vs. MSE** en la función de pérdida, o
   **sin `clip_grad_norm_`**, para mostrar sensibilidad a outliers de
   recompensa.

La forma más prolija de implementarlo es agregar *flags* booleanos al
constructor de `AgenteDQN` (p. ej. `usar_replay=True, usar_red_objetivo=True,
decae_epsilon=True`) en vez de duplicar la clase, así se reusa
`entrenar_agente` / `evaluar_agente` sin tocarlas.

### 5.2 Diseño experimental
- Entrenar **cada variante con varias semillas** (como mínimo 3-5) para tener
  una distribución de resultados, no un solo número.
- Evaluar todas con **las mismas semillas de evaluación** (p. ej.
  `semilla=10_000`, igual que ya hace el cuaderno para comparar DQN vs.
  aleatorio), sin exploración (`explorar=False`).
- Reportar, por variante, la recompensa de evaluación por semilla de
  entrenamiento (es decir, N resultados por variante, uno por semilla de
  entrenamiento), no solo el promedio.
- Comparar cada variante contra el DQN completo con un **test estadístico no
  paramétrico** (recomendado: **Mann-Whitney U**, porque no asume
  normalidad y la muestra de semillas suele ser chica) o un t-test si se
  justifica que las muestras son razonablemente normales. `scipy.stats`
  trae ambos (`mannwhitneyu`, `ttest_ind`).

Pseudocódigo del bucle de ablación:

```python
from scipy.stats import mannwhitneyu

variantes = {
    "completo": dict(),
    "sin_replay": dict(memoria_max=64),          # buffer == batch_size
    "sin_red_objetivo": dict(pasos_sync_objetivo=1),
    "epsilon_fijo": dict(epsilon_decay=1.0, epsilon=0.1),
}

semillas = [0, 1, 2, 3, 4]
resultados = {}

for nombre, overrides in variantes.items():
    recompensas_por_semilla = []
    for s in semillas:
        env = gym.make('LunarLander-v3')
        agente = AgenteDQN(env.observation_space, env.action_space, semilla=s, **overrides)
        entrenar_agente(agente, env, num_episodios=800, semilla=s, verbose=False)
        r = evaluar_agente(agente, env, num_episodios=100, semilla=10_000)
        recompensas_por_semilla.append(r['recompensa_promedio'])
        env.close()
    resultados[nombre] = recompensas_por_semilla

base = resultados["completo"]
for nombre, valores in resultados.items():
    if nombre == "completo":
        continue
    estadistico, p_valor = mannwhitneyu(base, valores, alternative='greater')
    print(f"{nombre}: media={np.mean(valores):.1f}  p-valor vs. completo={p_valor:.4f}")
```

Para el artículo IEEE: tabla con media ± desvío por variante, p-valor contra
el DQN completo, y 1-2 gráficas (curvas de aprendizaje superpuestas o boxplot
de recompensas de evaluación por variante).

## 6. Checklist / errores comunes a evitar

- [ ] `elegir_accion` debe aceptar y respetar `explorar=False` (lo usan
      `evaluar_agente` y `grabar_episodios`).
- [ ] `fin_episodio()` debe existir aunque no haga nada en algunas variantes
      (la clase base ya lo define como no-op, pero si se decae ε ahí hay que
      sobreescribirlo).
- [ ] No mezclar `terminado` y `truncado` dentro de `aprender`: usar el
      `terminado` que llega como parámetro tal cual, sin OR-earlo de nuevo
      (el cuaderno ya se encarga de pasar el valor correcto).
- [ ] Usar `estado_space.shape[0]` (no `.n`, que no existe en un `Box`).
- [ ] No actualizar la red hasta tener al menos `batch_size` transiciones en
      memoria (si no, el batch sería más chico de lo esperado o fallaría).
- [ ] Verificar que `red_objetivo` esté en modo `.eval()` y sus parámetros no
      se actualicen por backprop (usar `torch.no_grad()` al calcular el
      target, como en el ejemplo de FrozenLake).
- [ ] Fijar semillas (`random`, `numpy`, `torch`, y `accion_space.seed`) para
      que los experimentos de ablación sean reproducibles.
- [ ] Vigilar el tiempo de entrenamiento: 1200 episodios de hasta 1000 pasos
      puede tardar bastante en CPU; para la ablación (varias semillas ×
      varias variantes) conviene bajar `num_episodios` o usar
      `umbral_resolucion` para cortar antes si ya convergió.

## 7. Qué va en el artículo IEEE (no en el notebook)

Según la consigna, la discusión de resultados no va en el cuaderno. Sugerido
para el artículo:
1. Breve descripción del entorno y la arquitectura de red elegida.
2. Curvas de aprendizaje (recompensa media móvil) y tasa de éxito final.
3. Comparación cuantitativa DQN vs. agente aleatorio.
4. Interpretación de los cortes de política (TODO 2) con las figuras.
5. Resultados del estudio de ablación (TODO 3): tabla + test estadístico +
   discusión de qué componente resultó más crítico y por qué, en términos de
   la teoría de DQN (sección 2.3 de esta guía).

## 8. Referencias conceptuales

- Mnih et al., 2015, *"Human-level control through deep reinforcement
  learning"* (paper original de DQN, introduce replay buffer + red objetivo).
- Implementación ya resuelta de referencia:
  [FrozenGymQLAndDQN_Tarea.ipynb](/c:/Users/Pc/Desktop/Facu-Ingenieria/AprendAut/AprendAut26/Tarea2/FrozenGymQLAndDQN_Tarea.ipynb)
  (clases `DQN` y `AgenteDQN`, sección 2 del cuaderno).
- Documentación de Gymnasium para
  [`LunarLander-v3`](https://gymnasium.farama.org/environments/box2d/lunar_lander/).
