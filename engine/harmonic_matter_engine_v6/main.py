from __future__ import annotations

import argparse
import numpy as np

import jax.numpy as jnp

from .agents.architect import GenerativeArchitect
from .agents.luthier import MaterialLuthier
from .core.walrus import WalrusSurrogate
from .edge.litert_export import LiteRTCompiler
from .utils import clamp_demo_particle_count, load_config, package_root, save_json
from .core.jax_sph import LiquidPhysics


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="harmonic_matter_engine_v6")
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Override project_name prompt (e.g., 'Liquid Diamond').",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
        help="Number of simulation steps to run (default: 10).",
    )
    parser.add_argument(
        "--particle-cap",
        type=int,
        default=1024,
        help="Clamp particle_count for demo runtime (default: 1024).",
    )
    parser.add_argument(
        "--export-n",
        type=int,
        default=256,
        help="Particle count to compile into the .tflite (fixed shape).",
    )
    parser.add_argument(
        "--tflite-out",
        type=str,
        default="engine_v6_npu.tflite",
        help="Output path for the exported .tflite.",
    )
    parser.add_argument(
        "--tflite-quant",
        type=str,
        default="none",
        choices=["none", "dynamic", "float16", "int8"],
        help="Quantization mode for export: none|dynamic|float16|int8 (default: none).",
    )
    return parser.parse_args()


def main() -> None:
    print("==================================================")
    print("   HARMONIC MATTER ENGINE v6.0 (LIQUID INTEL)   ")
    print("==================================================")

    args = _parse_args()
    config = load_config()
    project_name = args.prompt or config.get("project_name", "Untitled")

    architect = GenerativeArchitect()
    physics_params = architect.design_simulation(project_name)

    # Apply architect-inferred params to the live config for this run.
    if "viscosity" in physics_params:
        config["physics"]["viscosity"] = float(physics_params["viscosity"])
    if "stiffness" in physics_params:
        config["physics"]["stiffness"] = float(physics_params["stiffness"])

    walrus = WalrusSurrogate()
    requested_N = int(config["physics"]["particle_count"])
    N = clamp_demo_particle_count(requested_N, cap=int(args.particle_cap))
    if N != requested_N:
        print(
            f">>> NOTE: Clamping particle_count {requested_N} -> {N} for demo runtime."
        )
    pos_np, vel_np = walrus.generate_initial_state(project_name, N)

    luthier = MaterialLuthier(num_splats=1024)
    scene = luthier.bake_scene()

    print(">>> SIMULATOR: Spinning up JAX-SPH (TVF + Riemann)...")
    solver = LiquidPhysics(config)

    print(f">>> RUNNING SIMULATION ({int(args.steps)} Steps)...")
    j_pos = jnp.asarray(pos_np)
    j_vel = jnp.asarray(vel_np)
    j_props = {"mass": jnp.ones((N,), dtype=jnp.float32)}

    last_rho = None
    for i in range(int(args.steps)):
        j_pos, j_vel, rho = solver.step(j_pos, j_vel, j_props)
        last_rho = rho
        if i % 2 == 0:
            print(f"    Step {i}: Max Density {float(jnp.max(rho)):.2f}")

    out_dir = package_root() / "output"
    save_json(
        out_dir / "particles.json",
        {
            "project": project_name,
            "positions": np.asarray(j_pos).tolist(),
            "velocities": np.asarray(j_vel).tolist(),
            "density": np.asarray(last_rho).tolist() if last_rho is not None else [],
        },
    )
    save_json(
        out_dir / "scene.json", {k: np.asarray(v).tolist() for k, v in scene.items()}
    )
    print(f">>> Wrote viewer artifacts to: {out_dir}")

    print(">>> EDGE: Compiling for NPU (LiteRT)...")
    compiler = LiteRTCompiler()

    export_n = int(args.export_n)
    export_n = min(export_n, int(j_pos.shape[0]))
    step_fn = solver.export_step_fn(dt=0.01)
    compiler.convert_jax_to_tflite(
        step_fn,
        [j_pos[:export_n], j_vel[:export_n], j_props["mass"][:export_n]],
        output_path=args.tflite_out,
        quantization=args.tflite_quant,
    )

    print(">>> v6.0 PIPELINE COMPLETE.")


if __name__ == "__main__":
    main()
