# Copa NAO CCM MX 2026: referencia de tiro penal

Fuente: `~/Downloads/Reglamento Copa NAO CCM MX 2026 V1.0.pdf`, versión 1.0, SHA-256 `e62ab72edce4f917fbee5da8ead090340128738c5f6990ac84c9846ce9583a83`. Secciones 2.4–2.10 (pp. 6–8) y 3.3 (pp. 18–21). El evento está previsto para el 3 de noviembre de 2026 en el Tecnológico de Monterrey, campus Ciudad de México (p. 4). Confirmar actualizaciones con la organización.

## Ronda y participación (§ 3.3)

- Dos equipos se enfrentan: un NAO tira y el adversario defiende. Cada equipo ejecuta hasta **cinco tiros consecutivos** mientras el otro defiende; luego intercambian roles. La serie puede terminar antes por diferencia insalvable. Si empatan, muerte súbita por gol de oro, que da un punto extra.
- Cada uno de **tres robots del equipo** debe ejecutar al menos un tiro; no hacerlo descalifica al equipo. El texto de descripción también dice que cada equipo está conformado por un robot: hay una contradicción que debe aclararse con los organizadores. Preparar la rotación de tres robots mientras se confirma.
- Un tiro es válido si el tirador toca la pelota, sin importar fuerza o precisión. Es gol si la pelota cruza completamente la línea de gol. Los jueces validan goles y faltas.
- El formato general (grupos, eliminatoria u otro) lo decide el comité. La orden de participación se anuncia el día del evento.

## Procedimiento por tiro (§§ 2.7, 3.3.5–3.3.6)

1. Hay **un minuto** para colocar ambos robots. El capitán coloca el tirador a **20 cm detrás de la marca penal**; el otro capitán coloca el portero sobre la línea de gol, entre los postes, bajo el travesaño y mirando al tirador.
2. El juez pita; cada equipo da la señal de activación a su robot dentro de los **10 segundos** posteriores al silbato. Ningún robot debe iniciar su rutina antes de la señal del juez.
3. El tiempo máximo de ejecución es **30 segundos desde el silbato**, no desde la activación. Ambos robots actúan autónomamente sin intervención posterior. El portero puede desplazarse únicamente dentro del área chica tras el silbato.
4. Exceder el minuto de preparación por el tirador o los 30 segundos de ejecución se marca como cero. Inicio anticipado de cualquiera de los robots o contacto/choque entre ellos causa repetición. Tres repeticiones en la ronda por estas faltas implican descalificación, según la tabla; aclarar con jueces cómo se contabilizan.
5. El gol es oro en caso de muerte súbita. Al finalizar, mantener el robot en el área hasta autorización del juez.

## Cancha y balón de referencia (§ 3.3.4, p. 19)

| Elemento | Medida o apariencia |
|---|---|
| Cancha | 240 × 240 cm, sobre duela; líneas negras de 1,9 cm |
| Marca penal | 10 cm de diámetro; a 130 cm de la línea de gol |
| Posición inicial del tirador | 20 cm detrás de la marca |
| Área chica | 60 × 220 cm |
| Portería | 150 cm de ancho × 80 cm de alto; postes azules y red blanca |
| Pelota oficial | Plástico naranja, diámetro 65 mm, peso 55 g |

La organización debe confirmar las dimensiones definitivas y enviar una fotografía de la duela antes del evento.

## Robot, despliegue y seguridad (§§ 2.7, 3.3.3)

- NAO de cualquier versión, sin modificaciones mecánicas externas. Se permiten sus sensores propios para localizar balón y portería.
- Antes de participar, cargar programas por **Ethernet**. Está prohibida la comunicación inalámbrica durante la competencia salvo excepción de categoría; no se permite control remoto ni comunicación externa que indique movimientos. El robot debe operar de forma autónoma.
- La programación de cada robot debe permitir una **desactivación manual** mediante un sensor que no esté en las piernas. La intervención durante una rutina sin autorización está prohibida; acordar con los jueces el uso de la parada en una emergencia.
- No modificar hardware ni software durante el bloque de competencia salvo reparaciones o ajustes menores autorizados por los jueces.
- El reglamento general menciona tanto un minuto para presentarse después de ser llamado (§ 2.4) como cinco minutos (§ 2.7); confirmar cuál aplica y estar listo para el plazo menor.

## Implicaciones para PenaltyStack

- La solución de competencia debe funcionar **sin GameController, Wi-Fi ni PC durante el tiro**, con activación física después del silbato y sin intervención posterior. El reloj del juez empieza al silbar: un límite interno de 28 s desde el botón puede terminar después de los 30 s oficiales si la activación se demora.
- Preparar ambos roles para cada robot seleccionado y una operación repetible de cinco tiros consecutivos, cambio de roles, rotación de los tres tiradores y muerte súbita; los jueces llevan el resultado y las repeticiones.
- La restricción del equipo de que el portero **no se lance** es adicional: el reglamento de tiro penal consultado no establece una prohibición explícita de lanzarse. Validar que la postura sentada y cualquier desplazamiento permanezcan dentro del área chica y que no haya contacto entre robots.
- El paquete NAOqi independiente todavía no ejecuta la patada real; el camino B-Human actual depende de estados de juego y necesita un flujo autónomo local por botón y parada manual verificable. Validar percepción del balón y portería, colocación, tiempo y recuperación con la duela y pelota reales antes de competir.
- El tiempo de calibración se limita a cinco minutos por prueba y puede no estar garantizado (§ 2.10); preparar ensayos fuera del evento.
