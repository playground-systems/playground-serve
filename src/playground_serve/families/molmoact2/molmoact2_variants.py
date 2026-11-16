"""The MolmoAct2 checkpoints Playground Serve registers, read by spec.py (in `pgs`) and serve.py (on the worker).
Standard library only.

Per entry: the Hugging Face repo and pinned revision; the norm tag the model card says to pass; the request keys of the
cameras in the order the model takes them; state and action sizes and the chunk length (the checkpoint's
norm_stats.json); the control rate. Sources: the model cards at the pinned revisions and the repo's servers
(examples/droid/host_server_droid.py, examples/yam/host_server_yam.py at COMMIT).
- DROID: the repo's server sends two images, [external, wrist] (the model card's example sends three, the first one
  twice); this follows the server, the deployment code.
- SO-101: the card says camera order does not matter; its sample is [top, side]. No official server.
- LIBERO: [agentview, wrist] as the card; delta end-effector actions (norm_stats.json "control_mode"). No official
  server.
"""

COMMIT = "6070080a20321b4f498ab30f28e1d09ac465edb7"  # github.com/allenai/molmoact2, 2026-10-01
DEFAULT_NUM_STEPS = 10  # the servers' default and the checkpoints' flow_matching_num_steps

VARIANTS = {
    "molmoact2_libero": dict(repo="allenai/MolmoAct2-LIBERO", revision="0d24a92bd1faf321ef497c3bbd5681af97c65aa2",
                             norm_tag="libero", cameras=("agentview", "wrist"), state_dim=8, action_dim=7, horizon=10,
                             hz=20, robot="libero"),
    "molmoact2_droid": dict(repo="allenai/MolmoAct2-DROID", revision="d8c1abd8a27d8e859455bbe514df2bcc617db0fb",
                            norm_tag="franka_droid", cameras=("external", "wrist"), state_dim=8, action_dim=8,
                            horizon=15, hz=15, robot="droid"),
    "molmoact2_so101": dict(repo="allenai/MolmoAct2-SO100_101", revision="152569fe57914d97be91055800035f54e250d009",
                            norm_tag="so100_so101_molmoact2", cameras=("top", "side"), state_dim=6, action_dim=6,
                            horizon=30, hz=30, robot="so101"),
    "molmoact2_yam": dict(repo="allenai/MolmoAct2-BimanualYAM", revision="8dcbed66f2380e4393189c303ea72488eb9e63c2",
                          norm_tag="yam_dual_molmoact2", cameras=("top", "left", "right"), state_dim=14, action_dim=14,
                          horizon=30, hz=30, robot="yam"),
}
