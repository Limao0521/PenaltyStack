# striker_visual_v3 — experimental

Checkpoint PPO de la actualización 16, elegido sólo con validación contra nuestro portero v2 visual + balanced y movimiento original. Se exportan `model.zip`, `obs_norm.npz` y `config.json` sin cambiar un byte; los pesos y el optimizador quedan en el modelo.

Compatible con observación `rl_obs_v1` de 19 campos, acción táctica de 4 componentes e historia 1. Conserva normalización congelada de striker_v2. Recompensa nueva: `striker_visual_reward_v2`; inicialización calibrada: `striker_policy_init_v2`.

Estado experimental, sin promoción del modelo original y sin validación física en NAO. [Resultados, limitaciones y evidencia](../../../docs/validation/rl/striker_visual_v3/README.md).

SHA256 del checkpoint seleccionado:

```json
{
  "model.zip": "6e5f921a581bd24f788286e60b0c94084b2a309cb9abf5c75cc8990091cd934d",
  "obs_norm.npz": "c90f56c39699d528e3b5fd76e5569840087024f438425ec048493656dd02ad60",
  "config.json": "0033aa2e9bb6918cd2178a667659dce16eee4cf99cb746ca01a19f254f2a8e8d"
}
```
