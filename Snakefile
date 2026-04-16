"""
Snakemake workflow for the Megatop CMB analysis pipeline.

Usage (local):
    snakemake --cores 4 --configfile paramfiles/e2e_check.yaml
    snakemake --cores 4 --configfile paramfiles/e2e_check.yaml --dry-run

Usage (SLURM cluster):
    snakemake --profile runfiles/snakemake_profiles/slurm --configfile paramfiles/e2e_check.yaml
"""

from itertools import product
from pathlib import Path

# ── Load megatop config ───────────────────────────────────────────────────────
# Pass the config path at invocation: --configfile path/to/config.yaml
MEGATOP_CONFIG = Path(workflow.configfiles[0])

from megatop import Config, DataManager

mt = Config.load_yaml(MEGATOP_CONFIG)
manager = DataManager(mt)

N_SKY = mt.map_sim_pars.n_sim
N_NOISE = mt.noise_sim_pars.n_sim
MAP_SETS = [ms.name for ms in mt.map_sets]


# ── Helper ────────────────────────────────────────────────────────────────────
def S(*path_lists):
    """Flatten DataManager Path lists to strings for Snakemake."""
    return [str(p) for lst in path_lists for p in lst]


# ── Target ────────────────────────────────────────────────────────────────────
rule all:
    input:
        S(*[manager.outputs_cl2r(i) for i in range(N_SKY)])
        if N_SKY > 0
        else S(manager.outputs_cl2r(None)),


# ── Global steps (no id_sim) ──────────────────────────────────────────────────
rule mask_handler:
    benchmark:
        "benchmarks/mask_handler.tsv"
    input:
        S(manager.inputs_mask())
    output:
        S(manager.outputs_mask())
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-mask-run --config {params.config}"


rule binner:
    benchmark:
        "benchmarks/binner.tsv"
    input:
        S(manager.inputs_binner())
    output:
        S(manager.outputs_binner())
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-binning-run --config {params.config}"


rule noisecov:
    benchmark:
        "benchmarks/noisecov.tsv"
    input:
        S(manager.inputs_noisecov())
    output:
        S(manager.outputs_noisecov())
    resources:
        mem_mb=32000,
        runtime=60,
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-noisecov-run --config {params.config}"


# ── Per-noise-sim rules ───────────────────────────────────────────────────────
for _i, _ms in product(range(N_NOISE), MAP_SETS):

    rule:
        benchmark:
            f"benchmarks/mock_noise_{_i:04d}_{_ms}.tsv"
        name: f"mock_noise_{_i:04d}_{_ms}"
        input:
            S(manager.inputs_mock_noise(_i))
        output:
            S(manager.outputs_mock_noise(_i, map_set=_ms))
        resources:
            mem_mb=16000,
            runtime=60,
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
            map_set=_ms,
        shell:
            "megatop-mock-noise-run --config {params.config} --sim {params.sim} --map-set {params.map_set}"


# ── Per-sky-sim rules ─────────────────────────────────────────────────────────
for _i, _ms in product(range(N_SKY), MAP_SETS):

    rule:
        benchmark:
            f"benchmarks/mock_signal_{_i:04d}_{_ms}.tsv"
        name: f"mock_signal_{_i:04d}_{_ms}"
        input:
            S(manager.inputs_mock_signal(_i))
        output:
            S(manager.outputs_mock_signal(_i, map_set=_ms))
        resources:
            mem_mb=16000,
            runtime=60,
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
            map_set=_ms,
        shell:
            "megatop-mock-signal-run --config {params.config} --sim {params.sim} --map-set {params.map_set}"

for _i in range(N_SKY):

    rule:
        benchmark:
            f"benchmarks/preproc_{_i:04d}.tsv"
        name: f"preproc_{_i:04d}"
        input:
            S(manager.inputs_preproc(_i))
        output:
            S(manager.outputs_preproc(_i))
        resources:
            mem_mb=16000,
            runtime=30,
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
        shell:
            "megatop-preproc-run --config {params.config} --sim {params.sim}"

    rule:
        benchmark:
            f"benchmarks/compsep_{_i:04d}.tsv"
        name: f"compsep_{_i:04d}"
        input:
            S(manager.inputs_compsep(_i))
        output:
            S(manager.outputs_compsep(_i))
        resources:
            mem_mb=32000,
            runtime=120,
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
        shell:
            "megatop-compsep-run --config {params.config} --sim {params.sim}"

    rule:
        benchmark:
            f"benchmarks/map2cl_{_i:04d}.tsv"
        name: f"map2cl_{_i:04d}"
        input:
            S(manager.inputs_map2cl(_i))
        output:
            S(manager.outputs_map2cl(_i))
        resources:
            mem_mb=16000,
            runtime=30,
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
        shell:
            "megatop-map2cl-run --config {params.config} --sim {params.sim}"

    rule:
        benchmark:
            f"benchmarks/noisespectra_{_i:04d}.tsv"
        name: f"noisespectra_{_i:04d}"
        input:
            S(manager.inputs_noisespectra(_i))
        output:
            S(manager.outputs_noisespectra(_i))
        resources:
            mem_mb=16000,
            runtime=30,
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
        shell:
            "megatop-noisespectra-run --config {params.config} --sim {params.sim}"

    rule:
        benchmark:
            f"benchmarks/cl2r_{_i:04d}.tsv"
        name: f"cl2r_{_i:04d}"
        input:
            S(manager.inputs_cl2r(_i))
        output:
            S(manager.outputs_cl2r(_i))
        resources:
            mem_mb=16000,
            runtime=120,
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
        shell:
            "megatop-cl2r-run --config {params.config} --sim {params.sim}"
