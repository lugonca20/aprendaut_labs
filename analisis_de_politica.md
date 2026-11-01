# Análisis de la política aprendida (DQN, LunarLander)

Se estudia la política greedy del agente entrenado (`agente_dqn`) mediante tres cortes 2D del espacio de estados de 8 dimensiones. En cada corte se barren dos variables en una grilla de 80×80 y el resto se fija. Se grafican la acción greedy y el Q-valor máximo.

Acciones: `0` Nada, `1` Motor izq., `2` Motor ppal., `3` Motor der.
Estado: `0=x, 1=y, 2=vx, 3=vy, 4=theta, 5=omega, 6=contacto pata izq., 7=contacto pata der.`

## Conclusión general

El agente aprendió una política estructurada y no aleatoria:

- Usa el motor principal para frenar la caída cuando está cerca del eje central.
- Corrige la inclinación con los motores laterales de forma casi simétrica.
- Reduce el uso del motor principal al apoyarse en el suelo.

El Q-valor máximo también tiene sentido: es alto cuando el aterrizaje es alcanzable y bajo cuando ya no hay nada por ganar.

| Corte | Q máx. medio |
|-------|--------------|
| 1 (cayendo, sin contacto) | **95.2** |
| 2 (ángulo vs. omega, bajo) | **17.7** |
| 3 (patas en contacto) | **−10.4** |

Hay una excepción a la hipótesis inicial: la asimetría izquierda/derecha del corte 1 (ver abajo).

---

## Corte 1: posición (x, y), cayendo nivelada

**Fijos:** `vx=0, vy=-1.0, theta=0, omega=0, sin contacto`. **Barrido:** `x∈[-1,1]`, `y∈[0,1.5]`.

| Región | Nada | Motor izq. | Motor ppal. | Motor der. |
|--------|------|-----------|-------------|-----------|
| Toda la grilla | 6.0 % | 0.05 % | 69.3 % | 24.7 % |
| Franja central `|x|≤0.2` | 0 % | 0 % | **94.0 %** | 6.0 % |
| Costados `|x|>0.2` | 7.6 % | 0.06 % | 63.1 % | 29.3 % |

**Interpretación**

- **Cerca del centro domina el motor principal (94 %).** Con `vy=-1` la nave cae rápido, y el agente frena con el motor principal. Nunca elige "Nada" en esa franja.
- **En los costados el motor principal baja a 63 % y aparece el motor derecho (29 %).** Con la nave quieta (`vx=0`) y lejos del centro, el agente corrige la posición con el motor lateral.
- **La política es asimétrica.** El motor izquierdo casi no se usa (0.05 %), mientras que el derecho se usa en un cuarto de la grilla. No se cumple la expectativa de correcciones simétricas a ambos lados.
  - Una causa posible, sin comprobar, es que el agente use el motor derecho como parte de su estrategia de control incluso con `theta=0`, o que haya poca cobertura de datos de un lado.
  - Esto es una limitación de la política aprendida, no un error del corte.
- **Q máx. medio = 95.2.** Es el valor más alto de los tres cortes, coherente con un estado donde el aterrizaje sigue siendo posible y se espera una buena recompensa futura.

## Corte 2: ángulo (theta) vs. velocidad angular (omega)

**Fijos:** `x=0, y=0.3, vx=0, vy=0, sin contacto`. **Barrido:** `theta∈[-0.5,0.5]`, `omega∈[-2,2]`.

| Región | Nada | Motor izq. | Motor ppal. | Motor der. |
|--------|------|-----------|-------------|-----------|
| Toda la grilla | 8.7 % | 28.3 % | 30.6 % | 32.3 % |
| `theta < 0` | 12.3 % | **45.8 %** | 31.1 % | 10.8 % |
| `theta > 0` | 5.1 % | 10.8 % | 30.2 % | **53.9 %** |

**Interpretación**

- **Es la evidencia más clara de que aprendió.** El motor elegido cambia de forma opuesta según el signo del ángulo: con `theta<0` domina el motor izquierdo (45.8 % contra 10.8 % del derecho), y con `theta>0` domina el derecho (53.9 % contra 10.8 % del izquierdo).
- **El comportamiento es casi simétrico respecto a theta=0.** Los porcentajes de uno y otro lado son parecidos y espejados, así que la política distingue el sentido de la inclinación y responde con el motor opuesto.
- **El motor principal se mantiene alrededor del 30 % en ambos lados.** No depende del ángulo; responde al hecho de que la nave está a baja altura (`y=0.3`) y sigue necesitando frenar.
- **"Nada" es poco frecuente (8.7 %).** Solo aparece en la zona estable (theta y omega chicos), donde no hay nada que corregir.
- **Q máx. medio = 17.7.** Es menor que en el corte 1 porque una nave muy inclinada o rotando rápido tiene peor perspectiva de aterrizar bien.

## Corte 3: posición (x, y) con ambas patas en contacto

**Fijos:** `vx=0, vy=0, theta=0, omega=0, contacto izq.=1, contacto der.=1`. **Barrido:** `x∈[-1,1]`, `y∈[-0.05,0.3]`.

| Acción | Fracción |
|--------|----------|
| Nada | 27.5 % |
| Motor izq. | 13.1 % |
| Motor ppal. | 14.3 % |
| Motor der. | 45.1 % |

**Interpretación**

- **El motor principal baja de 69 % (corte 1) a 14 %.** Con las patas apoyadas el agente deja de empujar hacia arriba, lo que confirma la hipótesis de que la política se vuelve más pasiva al aterrizar.
- **"Nada" sube de 6 % a 27.5 %.** Esto también va en la dirección esperada.
- **La política no es completamente pasiva.** El motor derecho se usa en 45 % de la grilla. Es la misma preferencia por el motor derecho que se vio en el corte 1, y no es un comportamiento deseable: en el simulador, disparar un motor lateral en el suelo consume combustible (recompensa negativa) y puede desestabilizar la nave.
- **Q máx. medio = −10.4.** Es el único valor negativo. Un aterrizaje apoyado ya no ofrece recompensa positiva adicional, y los estados lejos del centro (`|x|` grande) caen fuera de la zona de aterrizaje y tienen un valor peor. La interpretación exacta del signo depende de la forma del mapa de calor del gráfico.

---

## Resumen de la hipótesis de cada corte

| Corte | Hipótesis | Resultado |
|-------|-----------|-----------|
| 1 | El motor principal domina en el centro y los laterales corrigen a los costados | **Confirmada en parte.** Domina el principal (94 %) y los laterales aparecen a los costados, pero solo el derecho. |
| 2 | Corrección simétrica de la inclinación | **Confirmada.** Motor izq. para `theta<0` y motor der. para `theta>0`. |
| 3 | La política se vuelve más pasiva al aterrizar | **Confirmada en parte.** Baja el motor principal (14 %) y sube "Nada" (27.5 %), pero persiste el uso del motor derecho (45 %). |

## Limitaciones

- Los cortes dejan fijas 6 de las 8 dimensiones, así que describen la política solo en esas condiciones.
- Las zonas de la grilla que el agente rara vez visita durante el entrenamiento (por ejemplo, `|x|` grande) pueden mostrar acciones poco confiables.
- Las interpretaciones sobre la causa de la asimetría del motor derecho son hipótesis y no se verificaron.
