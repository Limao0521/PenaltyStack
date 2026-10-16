# Área 3 — Despliegue y operación en la Copa NAO CCM MX 2026

> Para alguien que despliega por primera vez: [guía paso a paso](../GUIA_DESPLIEGUE_ROBOT_FISICO.md).

## 1. Preparar el PC (una vez)

1. **WSL Ubuntu 24.04** (ya instalado en este PC). El toolchain del NAO se instala en el
   perfil del usuario, sin `sudo`:

   ```bash
   python tools/robot_build.py --install-toolchain
   ```

   Alternativa con permisos de administrador, dentro de WSL:
   `sudo apt install clang mold cmake ccache ninja-build rsync openssh-client`.
   `python tools/robot_build.py --check` confirma que no falta nada.
2. **Red cableada**: los robots tienen IP fija `192.168.49.<n>` en
   `robot/Config/Robots/<nombre>/network.cfg`. Configura el adaptador Ethernet del PC con
   `192.168.49.100/24`, sin puerta de enlace, y conecta los robots por cable o switch.
3. Entorno Python del repositorio (`.venv`). La ventana solo necesita Tkinter, que viene con
   Python.

## 2. Compilar

```bash
python tools/robot_build.py
```

Sincroniza `robot/` al árbol de WSL (`~/.cache/penaltystack/robot`) y compila `bhuman`
(Release). La primera vez tarda unos minutos; después solo recompila lo que cambió. Si se
cambian solo archivos de `Config/` (por ejemplo, los umbrales del balón), no hace falta
compilar: desmarca "Compilar antes de desplegar".

## 3. Ventana de despliegue

Doble clic en **`DeployCopaNAO.bat`** (raíz del repositorio), o
`python deploy/deploy_dialog.py`.

- **Configuración general**:
  - compilación Release/Develop y compilar antes de desplegar;
  - número de equipo y volumen;
  - modo de arranque: botón medio (recomendado), silbato, o botón o silbato;
  - "Apagar Wi-Fi" (activado por defecto, regla 2.7);
  - "Portero con estiradas laterales" (desmárcalo para usar guardias amplias sin caída);
  - "Iniciar bhuman al terminar" y "Simulacro" (muestra todo sin tocar los robots).
- **Tabla de robots**: marca los robots que vas a desplegar. Por robot eliges:
  - **Rol** inicial: Tirador o Portero;
  - **Política del tirador (RL)**: `striker_v2` (recomendado), `striker_preserving_v4`,
    `striker_visual_v3`, `striker_v1` o `scripted`;
  - **Política del portero**: `vision_v2_balanced` (recomendado), `vision_v2_original` o
    `center_guard`;
  - **número de jugador**.

  La columna Conexión muestra si el robot responde por Ethernet. Al elegir una política, debajo
  aparecen su estado y sus resultados de simulación.
- **Botones**:
  - "Comprobar conexión";
  - "Validar escenario" (grafo de módulos y configuraciones);
  - "Compilar";
  - **"Desplegar seleccionados"** (en paralelo, con confirmación);
  - "Estado" (servicio y manifiesto del robot);
  - "Detener" e "Iniciar" `bhuman`;
  - "Descargar logs" (a `runs/robot_logs/<robot>`).

Cada despliegue deja `/home/nao/Config/penaltystack_deploy.json` en el robot: rol, políticas,
hashes, modo de arranque, perfil Wi-Fi y fecha. Además verifica que el hash del binario en el
robot coincida con el compilado.

Equivalente por línea de comandos:

```bash
python deploy/robot_deploy.py --ping falcanao loki naonel
python deploy/robot_deploy.py --deploy falcanao:striker loki:keeper naonel:striker --team 49 --striker-policy striker_v2 --keeper-policy vision_v2_balanced
python deploy/robot_deploy.py --status falcanao
```

## 4. En la cancha (por cada tiro penal)

| Momento | Operador | Robot |
| :--- | :--- | :--- |
| Preparación (≤ 1 min) | Coloca el tirador 20 cm detrás de la marca, mirando la portería; el portero sobre la línea de gol, entre los postes, mirando al tirador. | Tras encenderlo: articulaciones sueltas. **Toque al pecho** → se pone de pie y dice "striker ready" o "keeper ready". |
| Rol | Si hace falta cambiarlo: **mantener el botón delantero de la cabeza 2 s** con el robot de pie esperando. | Dice "striker" o "keeper". |
| Silbato del juez | **Toque al botón medio de la cabeza** antes de 10 s (o silbato, si se desplegó ese modo). | Inicia la rutina; el reloj interno da 28 s. |
| Durante el tiro | No tocar el robot (regla 2.7). | Tirador: se acerca, patea una vez y queda quieto. Portero: sigue el balón y se defiende. |
| Desactivación manual | **Mantener el botón trasero de la cabeza 0,5 s.** | Se detiene de pie y dice "stop". |
| Fin del tiro | Esperar la autorización del juez. | Queda de pie e inmóvil (al acabar el tiro o a los 28 s). |
| Siguiente tiro | Recolocar y dar **un toque al botón delantero**. | Rearmado: dice "ready" y espera el inicio. |
| Apagar articulaciones | Mantener los tres botones de la cabeza 1 s. | Articulaciones sueltas; con el pecho vuelve a ponerse de pie. |

Los robots deben estar **sin Wi-Fi y sin cable** durante la prueba. El despliegue ya apaga el
Wi-Fi; desconecta el Ethernet antes de entrar al área.

## 5. Zona de calibraciones (regla 2.10)

1. **Detección del balón**: con el robot de pie y el balón a 0,2 m, 1,3 m y 2 m, inicia la
   rutina en modo simulacro de campo (sin portero) y descarga los logs:
   - un tirador que nunca libera la patada indica que no ve el balón;
   - si es así, ajusta `robot/Config/Scenarios/CopaNAOPenalty/copaOrangeBallPerceptor.cfg`
     (`minV`, `maxU`, `minChroma`, `minY`) y despliega sin compilar;
   - con mucha luz la madera clara puede acercarse al naranja: sube `minChroma`.
2. **Patada**: verifica que la patada se produce una sola vez y que el balón sale hacia la
   portería. El alcance pedido va de 1,5 a 2,9 m según la política.
3. **Portero**: prueba primero con "Portero con estiradas laterales" desmarcado (sin caídas) y
   activa las estiradas solo si el robot y la superficie lo permiten.
4. Revisa los logs (`[CopaPenalty]`): fases, `kick released`, `kick done` y `defense`.

## 6. Problemas frecuentes

| Síntoma | Causa probable | Qué hacer |
| :--- | :--- | :--- |
| "sin respuesta" en Conexión | PC fuera de `192.168.49.0/24`, cable o switch. | Revisar la IP del adaptador Ethernet. |
| `bhuman` no queda activo ("Estado") | Error de configuración al arrancar. | "Descargar logs"; "Validar escenario"; redesplegar. |
| El robot dice "striker policy missing" | Falta el `.cfg` de la política. | Ejecutar `python tools/export_robot_policies.py` y redesplegar. |
| El tirador no se mueve tras el inicio | No ve el balón. | Calibrar el color (sección 5). |
| El portero no se lanza | El balón no se rastrea entrando. | Revisar el color y la posición del portero sobre la línea; ver `defense` en los logs. |
| El robot arranca con el silbato de otra cancha | Modo silbato en un ambiente ruidoso. | Desplegar con "Botón medio de la cabeza" (por defecto). |

## 7. Lista de verificación antes de competir

- [ ] `python -m unittest tests.test_robot_policy_parity tests.test_robot_deploy` pasa.
- [ ] "Validar escenario" da 0 errores.
- [ ] Los tres robots desplegados, con el manifiesto correcto ("Estado").
- [ ] Wi-Fi apagado (perfil `NONE`) y Ethernet desconectado antes de la prueba.
- [ ] Prueba de inicio, desactivación (botón trasero) y rearme en cada robot.
- [ ] Balón detectado a 0,2–2 m con la luz de la sede.
