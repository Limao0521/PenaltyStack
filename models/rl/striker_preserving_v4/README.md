# striker_preserving_v4

Checkpoint PPO de la actualización 64, elegido con validación contra nuestro portero v2 balanced y la FSM base. `model.zip`, `obs_norm.npz` y `config.json` se exportan byte a byte. La recomendación de esta prueba es `models/rl/striker_v2`; consultar los criterios y resultados completos en el [reporte](../../../docs/validation/rl/striker_preserving_v4/README.md).

Mantiene `rl_obs_v1` de 19 campos, cuatro acciones tácticas e historia 1. La inicialización `striker_policy_init_v3` conserva el actor de v2 y la normalización, comprime sólo las colas de sus medias gaussianas durante entrenamiento, reinicia Adam y pone en cero el crítico final. Recompensa nueva `striker_visual_reward_v3`, gamma 1 y retorno terminal ±1.

El exportador NumPy original es compatible sólo para acciones deterministas recortadas. El muestreo asíncrono gaussiano genérico no sirve para reentrenar esta clase; usar el runner dedicado `tools/retrain_striker_v2.py`. Sin validación en hardware ni convergencia demostrada.

Fingerprint: `3343ae9fee9a7ad2080bb26e9039fbc22958de62ebcadbaf60d322e9510256d7`.
