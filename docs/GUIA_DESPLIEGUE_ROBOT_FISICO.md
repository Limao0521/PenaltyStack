# Guía: desplegar el penal en un NAO físico

Esta guía es para una persona del equipo que **no participó en el desarrollo** y necesita
dejar uno o varios NAO V6 listos para la categoría **Tiro Penal** de la *Copa NAO CCM MX
2026*. Sigue los pasos en orden. Los detalles técnicos están en [docs/robot/](robot/).

> **Estado:** el software compila, sus decisiones coinciden con las de la simulación y el
> despliegue se probó en modo simulacro. **Todavía no se ha ejecutado en un robot físico.**
> Las primeras pruebas deben hacerse con el robot vigilado y en un área despejada.

---

## 1. Cómo funciona (en 2 minutos)

```
 PC con Windows                         Cable Ethernet                 NAO V6
┌──────────────────────────────┐                                  ┌───────────────────────────┐
│ Ventana de despliegue        │                                  │ /home/nao/Config/         │
│ (DeployCopaNAO.bat)          │   copia binario + configuración  │   bhuman  (el programa)   │
│        │                     │ ───────────────────────────────► │   Config  (rol, política, │
│        ▼                     │   ssh / rsync, red 192.168.49.x  │            modo arranque) │
│ WSL (Ubuntu): compila bhuman │                                  │ servicio bhuman: arranca  │
│ para el NAO y lo envía       │                                  │ solo al encender el robot │
└──────────────────────────────┘                                  └───────────────────────────┘
```

- **El robot decide solo.** Corre el sistema de B-Human (el que el equipo usa en RoboCup) con
  un escenario propio, `CopaNAOPenalty`, que contiene el tirador y el portero entrenados en
  simulación. Durante el penal no hay PC, red ni Wi-Fi.
- **El PC solo prepara al robot.** Compila el programa en WSL, que es Linux dentro de Windows,
  y lo copia por cable junto con la configuración de cada robot:
  - el rol inicial: tirador o portero;
  - la versión de la política RL;
  - el modo de arranque;
  - el número de jugador.

  Además apaga el Wi-Fi del robot, porque el reglamento exige solo cable (regla 2.7).
- **En la cancha el robot se maneja con sus botones.** El pecho lo pone de pie, el botón
  medio de la cabeza arranca la rutina y el botón trasero la detiene. La sección 7 tiene el
  protocolo completo.

---

## 2. Qué necesitas

**Hardware**
- PC con Windows 10 u 11 y puerto Ethernet (sirve un adaptador USB-Ethernet).
- Cable Ethernet. Si vas a conectar varios robots a la vez, un switch.
- NAO V6 con batería cargada, **ya preparado con el sistema B-Human del equipo**. Los robots
  de Sabana Herons ya lo están. Un NAO de fábrica (NAOqi) primero necesita la instalación de
  B-Human (`robot/Install/`, wiki de B-Human 2023), que esta guía no cubre.

**Software**
- Git.
- Python 3.11 o superior de [python.org](https://www.python.org/). Marca *Add to PATH*; el
  instalador ya incluye Tkinter, que usa la ventana. No hace falta instalar paquetes extra.
- WSL con Ubuntu 24.04 (se instala en el paso 3.2).

**Robots que reconoce la ventana**

Salen de `robot/Config/Robots/<nombre>/network.cfg`:

| Robot | IP por cable | Robot | IP por cable |
| :--- | :--- | :--- | :--- |
| ronaoldinho | 192.168.49.2 | ospinao | 192.168.49.7 |
| loki | 192.168.49.3 | Heron | 192.168.49.8 |
| falcanao | 192.168.49.4 | naonaldo | 192.168.49.9 |
| juan | 192.168.49.5 | naonel | 192.168.49.10 |
| iniesnao | 192.168.49.6 | | |

---

## 3. Preparar el PC (solo la primera vez)

### 3.1 Descargar el repositorio

```bash
git clone https://github.com/Limao0521/PenaltyStack.git
```

```bash
cd PenaltyStack
```

```bash
git checkout despliegue/copa-nao-ccm-2026
```

### 3.2 Instalar WSL con Ubuntu 24.04

En PowerShell **como administrador**:

```powershell
wsl --install -d Ubuntu-24.04
```

Reinicia el PC si lo pide. Al abrir *Ubuntu 24.04* por primera vez crea un usuario y una
contraseña de Linux, que se usarán para `sudo`.

### 3.3 Instalar las herramientas de compilación del NAO

Dentro de la terminal de Ubuntu:

```bash
sudo apt update
```

```bash
sudo apt install -y clang mold cmake ccache ninja-build rsync openssh-client
```

Si no tienes permisos de `sudo`, existe una alternativa que instala las herramientas en tu
usuario. Ejecútala desde la carpeta del repositorio, en Windows:

```bash
python tools/robot_build.py --install-toolchain
```

Comprueba que no falte nada; todas las líneas deben decir `ok`:

```bash
python tools/robot_build.py --check
```

Si tu distribución de WSL no se llama `Ubuntu-24.04`, define la variable de entorno
`PENALTYSTACK_WSL_DISTRO` con su nombre (`wsl -l -v` lo muestra).

### 3.4 Poner el PC en la red de los robots

El puerto Ethernet del PC debe tener una IP fija en la subred `192.168.49.x`. Usa
`192.168.49.100`, que no es de ningún robot.

- **Con la interfaz gráfica**: Configuración → Red e Internet → Ethernet → *Asignación de
  IP* → Editar → Manual → IPv4 activado. Dirección `192.168.49.100`, longitud de prefijo
  `24` (máscara `255.255.255.0`), puerta de enlace y DNS vacíos.
- **Con PowerShell como administrador**: primero mira el nombre del adaptador con
  `Get-NetAdapter`. Suele llamarse `Ethernet`.

```powershell
New-NetIPAddress -InterfaceAlias "Ethernet" -IPAddress 192.168.49.100 -PrefixLength 24
```

---

## 4. Compilar el programa del robot

Desde la carpeta del repositorio:

```bash
python tools/robot_build.py
```

La primera vez tarda entre 10 y 20 minutos; las siguientes, segundos o pocos minutos. Al final
debe aparecer `BHUMAN_BINARY=...` y una línea con el hash del binario. También puedes compilar
con el botón **Compilar** de la ventana.

Solo hace falta recompilar si cambia el código. Si solo cambias la configuración (sección 8),
basta con desplegar sin compilar.

---

## 5. Conectar y desplegar

### 5.1 Conectar el robot

1. Enciende el NAO con un toque corto al botón del pecho. El arranque tarda alrededor de un
   minuto. **No mantengas el pecho presionado**: más de 3 s apaga el robot.
2. Conecta el cable Ethernet al puerto de la parte trasera de la cabeza y al PC (o al switch).
3. **Seguridad**: durante el despliegue el programa del robot se detiene y el robot **pierde
   la rigidez**. Despliega con el robot sentado o sujetándolo.

### 5.2 Abrir la ventana

Doble clic en **`DeployCopaNAO.bat`**, en la raíz del repositorio. También se abre con:

```bash
python deploy/deploy_dialog.py
```

La ventana comprueba sola qué robots responden. La columna **Conexión** muestra
`● responde` en verde o `● sin respuesta` en rojo. Puedes repetir la prueba con
**Comprobar conexion**.

### 5.3 Configurar

**Configuración general** (vale para todos los robots que despliegues):

| Opción | Qué poner |
| :--- | :--- |
| Compilacion | `Release`. |
| Compilar antes de desplegar | Marcado la primera vez o si cambió el código. |
| Equipo # | El número de equipo asignado (0 = no cambiarlo). |
| Volumen % | 70 está bien; el robot habla al cambiar de estado. |
| Arranque | **Botón medio de la cabeza** (recomendado). Las opciones con silbato pueden arrancar con silbatos de otras canchas. |
| Apagar Wi-Fi | **Siempre marcado** en competencia (regla 2.7). |
| Portero con estiradas laterales | Desmarcado en las primeras pruebas: el portero hace guardias amplias sin tirarse al suelo. Márcalo cuando el robot y la superficie lo permitan. |
| Iniciar bhuman al terminar | Marcado. |
| Simulacro | Muestra lo que haría sin tocar los robots. Útil para practicar. |

**Tabla de robots** (una fila por robot):
1. Marca la casilla de cada robot que vas a desplegar.
2. **Rol**: el rol con el que arranca. Se puede cambiar después en el robot (sección 7).
3. **Política tirador (RL)**: `striker_v2` es la recomendada. Las demás (`striker_preserving_v4`,
   `striker_visual_v3`, `striker_v1`, `scripted`) son experimentales o de comparación. Al
   elegir una aparecen su estado y sus resultados debajo de la tabla.
4. **Política portero**: `vision_v2_balanced` es la recomendada. `vision_v2_original` es la
   alternativa y `center_guard` solo se agacha en el centro.
5. **Jugador**: un número distinto por robot.

Cada robot recibe **las dos políticas**. Así un mismo robot puede tirar y atajar sin
redesplegar.

### 5.4 Desplegar

1. Pulsa **Desplegar seleccionados** y confirma el resumen.
2. El registro de abajo muestra cada paso por robot:
   1. detiene `bhuman`;
   2. activa el perfil Wi-Fi `NONE` (Wi-Fi apagado);
   3. copia el programa y la configuración;
   4. ajusta el volumen;
   5. verifica el hash del binario en el robot;
   6. arranca `bhuman`.
3. Termina bien cuando la columna **Resultado** dice `OK` y el registro muestra
   `DEPLOY_OK <robot> ... sha256=...`.
4. Pulsa **Estado** con el robot seleccionado. Debe decir `bhuman: active` y mostrar el
   manifiesto del despliegue: rol, políticas, modo de arranque, Wi-Fi y fecha.

Lo mismo por línea de comandos (útil para scripts):

```bash
python deploy/robot_deploy.py --deploy falcanao:striker loki:keeper naonel:striker --team 49
```

---

## 6. Comprobar el robot antes de competir

Con el robot desplegado y `bhuman` activo:

1. Recién arrancado, el robot está **sin rigidez** (relajado o sentado). Es normal.
2. **Toque al pecho**: se pone de pie y dice "striker ready" o "keeper ready".
3. **Mantén el botón delantero de la cabeza 2 s**: cambia el rol y lo anuncia ("keeper" o
   "striker").
4. Coloca el balón y el robot como en un penal (sección 7) y da un **toque al botón medio**:
   - el tirador busca el balón, se acerca, patea **una sola vez** y se queda quieto;
   - el portero sigue el balón con la cabeza y se defiende cuando el balón va hacia él.
5. Durante la rutina, **mantén el botón trasero 0,5 s**: se detiene de pie y dice "stop".
6. **Toque al botón delantero**: se rearma ("ready") para el siguiente tiro.
7. **Mantén los tres botones de la cabeza 1 s**: pierde la rigidez. Con el pecho vuelve a
   ponerse de pie.

Para revisar lo que hizo, selecciona el robot y pulsa **Descargar logs**. Los archivos quedan
en `runs/robot_logs/<robot>/`. Busca las líneas `[CopaPenalty]`, por ejemplo:

```
[CopaPenalty] t=... start headButton attempt 1 role=striker
[CopaPenalty] t=... kick released policy=striker_v2 target_y=0.41 foot=right length=2650
[CopaPenalty] t=... kick done
[CopaPenalty] t=... defense left policy=vision_v2_balanced crossing_y=0.52 time_to_line=0.9
```

Si el tirador nunca llega a `kick released`, casi seguro **no ve el balón**: ajusta el color
(sección 8).

---

## 7. En la competencia: protocolo por tiro

Reglas clave: activación hasta 10 s después del silbato; 30 s para ejecutar el tiro;
**prohibido arrancar antes del silbato** (repetición; 3 repeticiones = descalificación);
prohibido tocar el robot una vez activado; los **tres robots del equipo deben tirar** al
menos una vez.

| Momento | Qué hace el operador | Qué hace el robot |
| :--- | :--- | :--- |
| Antes de entrar | Desconectar el cable Ethernet. El Wi-Fi ya quedó apagado. | — |
| Preparación (máx. 1 min) | **Tirador**: 20 cm detrás de la marca penal, mirando a la portería. **Portero**: sobre la línea de gol, entre los postes, mirando al tirador. Toque al pecho si está sin rigidez. | Se pone de pie y espera inmóvil ("ready"). |
| Cambio de rol (si hace falta) | Mantener el botón delantero de la cabeza 2 s. | Anuncia el nuevo rol. |
| Silbato del juez | **Toque al botón medio de la cabeza**, antes de 10 s. | Empieza la rutina; tiene 28 s. |
| Durante el tiro | **No tocar.** | Tirador: patea una vez y se queda quieto. Portero: se defiende sin salir del área chica. |
| Emergencia | Mantener el botón trasero de la cabeza 0,5 s. | Se detiene de pie ("stop"). |
| Fin del tiro | Esperar a que el juez autorice retirar el robot. | Queda de pie e inmóvil (al patear o a los 28 s). |
| Siguiente tiro | Recolocar y dar un toque al botón delantero. | Se rearma ("ready"). |

---

## 8. Ajustes en la sede, sin recompilar

Los archivos de configuración se cambian en el PC y se vuelven a desplegar con **Compilar
antes de desplegar desmarcado**.

- **Color del balón**: `robot/Config/Scenarios/CopaNAOPenalty/copaOrangeBallPerceptor.cfg`.
  - Si no ve el balón, baja `minV` o `minChroma`, o sube `maxU`, de a 10.
  - Si confunde la madera o la piel con el balón, sube `minChroma`.
  - Prueba con el balón a 0,2 m, 1,3 m y 2 m.
- **Tiempos y gestos**: `robot/Config/Scenarios/CopaNAOPenalty/copaPenaltyController.cfg`.
  Incluye la duración de la rutina (`activeDurationMs`, máximo 28 000) y los tiempos de los
  botones.
- **Medidas de la cancha**: `copaPenaltyTactics.cfg`, en la misma carpeta. Ajústalas si los
  organizadores confirman medidas distintas de la regla 3.3.4.

El rol, las políticas, el modo de arranque y las estiradas **no se tocan a mano**: se eligen
en la ventana.

---

## 9. Problemas frecuentes

| Síntoma | Causa probable | Solución |
| :--- | :--- | :--- |
| `● sin respuesta` | El robot no ha terminado de arrancar, el cable está suelto o el PC no tiene la IP `192.168.49.x`. | Esperar el arranque, revisar el cable y repetir el paso 3.4. |
| Responde en la ventana, pero el despliegue no conecta | WSL no alcanza la red cableada (VPN o firewall). | Desactivar la VPN. En PowerShell: `wsl --shutdown` y reintentar. Si persiste, activar en `%UserProfile%\.wslconfig` la sección `[wsl2]` con `networkingMode=mirrored`. |
| `Toolchain incompleto` al compilar | Faltan herramientas en WSL. | Repetir el paso 3.3 y `python tools/robot_build.py --check`. |
| `Estado` no muestra `bhuman: active` | El programa no arrancó. | Descargar logs, pulsar **Validar escenario** (debe dar 0 errores) y redesplegar. |
| El robot dice "striker policy missing" | Falta el archivo de la política. | Redesplegar desde la rama `despliegue/copa-nao-ccm-2026` actualizada. |
| No se pone de pie con el pecho | `bhuman` no está corriendo. | Pulsar **Estado** o **Iniciar bhuman**. |
| El tirador no se mueve tras el botón medio | No ve el balón. | Ajustar el color (sección 8) y revisar los logs. |
| El portero nunca se lanza | No detecta el balón entrando. | Revisar el color y que esté de frente al tirador sobre la línea. |
| Arrancó sin que nadie tocara el botón | Modo silbato con ruido de otra cancha. | Redesplegar con *Arranque: Botón medio de la cabeza*. |

---

## 10. Para saber más

- [docs/robot/01_AUDITORIA.md](robot/01_AUDITORIA.md): qué había, qué faltaba y cómo se cumple
  el reglamento.
- [docs/robot/02_PLAN_E_IMPLEMENTACION.md](robot/02_PLAN_E_IMPLEMENTACION.md): arquitectura en
  el robot y verificaciones.
- [docs/robot/03_DESPLIEGUE_Y_OPERACION.md](robot/03_DESPLIEGUE_Y_OPERACION.md): referencia
  rápida de operación.
- Código: `robot/Src/Modules/BehaviorControl/CopaPenalty*`, `deploy/` y `tools/robot_build.py`.
