"""
Snakemake workflow for the Megatop CMB analysis pipeline.

Usage (local):
    snakemake --cores 4 --configfile paramfiles/e2e_check.yaml
    snakemake --cores 4 --configfile paramfiles/e2e_check.yaml --dry-run

Usage (on a SLURM allocation):
    snakemake --cores $SLURM_CPUS_PER_TASK --configfile paramfiles/e2e_check.yaml
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
LOGS = manager.path_to_output / "logs"


rule mask_handler:
    input:
        S(manager.inputs_mask())
    output:
        S(manager.outputs_mask())
    log:
        str(LOGS / "mask_handler.log")
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-mask-run --config {params.config} > {log} 2>&1"


rule binner:
    input:
        S(manager.inputs_binner())
    output:
        S(manager.outputs_binner())
    log:
        str(LOGS / "binner.log")
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-binning-run --config {params.config} > {log} 2>&1"


rule noisecov:
    input:
        S(manager.inputs_noisecov())
    output:
        S(manager.outputs_noisecov())
    log:
        str(LOGS / "noisecov.log")
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-noisecov-run --config {params.config} > {log} 2>&1"


# ── Per-noise-sim rules ───────────────────────────────────────────────────────
for _i, _ms in product(range(N_NOISE), MAP_SETS):

    rule:
        name: f"mock_noise_{_i:04d}_{_ms}"
        input:
            S(manager.inputs_mock_noise(_i))
        output:
            S(manager.outputs_mock_noise(_i, map_set=_ms))
        log:
            str(LOGS / f"mock_noise_{_i:04d}_{_ms}.log")
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
            map_set=_ms,
        shell:
            "megatop-mock-noise-run --config {params.config} --sim {params.sim} --map-set {params.map_set} > {log} 2>&1"

for _i in range(N_NOISE):

    rule:
        name: f"noise_preproc_{_i:04d}"
        input:
            S(manager.inputs_noise_preproc(_i))
        output:
            S(manager.outputs_noise_preproc(_i))
        log:
            str(LOGS / f"noise_preproc_{_i:04d}.log")
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
        shell:
            "megatop-noise-preproc-run --config {params.config} --sim {params.sim} > {log} 2>&1"


# ── Per-sky-sim rules ─────────────────────────────────────────────────────────
for _i, _ms in product(range(N_SKY), MAP_SETS):

    rule:
        name: f"mock_signal_{_i:04d}_{_ms}"
        input:
            S(manager.inputs_mock_signal(_i))
        output:
            S(manager.outputs_mock_signal(_i, map_set=_ms))
        log:
            str(LOGS / f"mock_signal_{_i:04d}_{_ms}.log")
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
            map_set=_ms,
        shell:
            "megatop-mock-signal-run --config {params.config} --sim {params.sim} --map-set {params.map_set} > {log} 2>&1"

for _i in range(N_SKY):

    rule:
        name: f"preproc_{_i:04d}"
        input:
            S(manager.inputs_preproc(_i))
        output:
            S(manager.outputs_preproc(_i))
        log:
            str(LOGS / f"preproc_{_i:04d}.log")
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
        shell:
            "megatop-preproc-run --config {params.config} --sim {params.sim} > {log} 2>&1"

    rule:
        name: f"compsep_{_i:04d}"
        input:
            S(manager.inputs_compsep(_i))
        output:
            S(manager.outputs_compsep(_i))
        log:
            str(LOGS / f"compsep_{_i:04d}.log")
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
        shell:
            "megatop-compsep-run --config {params.config} --sim {params.sim} > {log} 2>&1"

    rule:
        name: f"map2cl_{_i:04d}"
        input:
            S(manager.inputs_map2cl(_i))
        output:
            S(manager.outputs_map2cl(_i))
        log:
            str(LOGS / f"map2cl_{_i:04d}.log")
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
        shell:
            "megatop-map2cl-run --config {params.config} --sim {params.sim} > {log} 2>&1"

    rule:
        name: f"noisespectra_{_i:04d}"
        input:
            S(manager.inputs_noisespectra(_i))
        output:
            S(manager.outputs_noisespectra(_i))
        log:
            str(LOGS / f"noisespectra_{_i:04d}.log")
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
        shell:
            "megatop-noisespectra-run --config {params.config} --sim {params.sim} > {log} 2>&1"

    rule:
        name: f"cl2r_{_i:04d}"
        input:
            S(manager.inputs_cl2r(_i))
        output:
            S(manager.outputs_cl2r(_i))
        log:
            str(LOGS / f"cl2r_{_i:04d}.log")
        params:
            config=MEGATOP_CONFIG,
            sim=_i,
        shell:
            "megatop-cl2r-run --config {params.config} --sim {params.sim} > {log} 2>&1"


# ── Plot rules ────────────────────────────────────────────────────────────────
# All plotters take only --config; per-sim plotters internally use sim 0 (or
# None for real data).  Outputs are touch-sentinels because some filenames are
# config-conditional and nothing in the pipeline reads plot files.

_id_repr = 0 if N_SKY > 0 else None

_all_cl2r = (
    S(*[manager.outputs_cl2r(i) for i in range(N_SKY)])
    if N_SKY > 0
    else S(manager.outputs_cl2r(None))
)


rule plots:
    input:
        str(manager.path_to_masks_plots / ".done"),
        str(manager.path_to_mock_plots / ".done"),
        str(manager.path_to_covar_plots / ".done"),
        str(manager.path_to_preproc_plots / ".done"),
        str(manager.path_to_components_plots / ".done"),
        str(manager.path_to_spectra_plots / ".done_map2cl"),
        str(manager.path_to_spectra_plots / ".done_noisespectra"),
        str(manager.path_to_mcmc_plots / ".done_cl2r"),
        str(manager.path_to_mcmc_plots / ".done_mcmc"),


rule plot_mask:
    input:
        S(manager.outputs_mask())
    output:
        touch(str(manager.path_to_masks_plots / ".done"))
    log:
        str(LOGS / "plot_mask.log")
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-mask-plot --config {params.config} > {log} 2>&1"


rule plot_mock:
    input:
        S(manager.outputs_binner(), manager.outputs_mask(),
          *[manager.outputs_mock_signal(0, map_set=ms) for ms in MAP_SETS])
        if N_SKY > 0
        else S(manager.outputs_binner(), manager.outputs_mask())
    output:
        touch(str(manager.path_to_mock_plots / ".done"))
    log:
        str(LOGS / "plot_mock.log")
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-mock-plot --config {params.config} > {log} 2>&1"


rule plot_noisecov:
    input:
        S(manager.outputs_noisecov())
    output:
        touch(str(manager.path_to_covar_plots / ".done"))
    log:
        str(LOGS / "plot_noisecov.log")
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-noisecov-plot --config {params.config} > {log} 2>&1"


rule plot_preproc:
    input:
        S(manager.outputs_preproc(_id_repr))
    output:
        touch(str(manager.path_to_preproc_plots / ".done"))
    log:
        str(LOGS / "plot_preproc.log")
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-preproc-plot --config {params.config} > {log} 2>&1"


rule plot_compsep:
    input:
        S(manager.outputs_compsep(_id_repr))
    output:
        touch(str(manager.path_to_components_plots / ".done"))
    log:
        str(LOGS / "plot_compsep.log")
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-compsep-plot --config {params.config} > {log} 2>&1"


rule plot_map2cl:
    input:
        S(manager.outputs_map2cl(_id_repr))
    output:
        touch(str(manager.path_to_spectra_plots / ".done_map2cl"))
    log:
        str(LOGS / "plot_map2cl.log")
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-map2cl-plot --config {params.config} > {log} 2>&1"


rule plot_noisespectra:
    input:
        S(*[manager.outputs_noisespectra(i) for i in range(N_SKY)],
          *[manager.outputs_map2cl(i) for i in range(N_SKY)])
        if N_SKY > 0
        else S(manager.outputs_noisespectra(None), manager.outputs_map2cl(None))
    output:
        touch(str(manager.path_to_spectra_plots / ".done_noisespectra"))
    log:
        str(LOGS / "plot_noisespectra.log")
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-noisespectra-plot --config {params.config} > {log} 2>&1"


rule plot_cl2r:
    input:
        _all_cl2r
    output:
        touch(str(manager.path_to_mcmc_plots / ".done_cl2r"))
    log:
        str(LOGS / "plot_cl2r.log")
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-cl2r-plot --configs {params.config} > {log} 2>&1"


rule plot_mcmc:
    input:
        _all_cl2r
    output:
        touch(str(manager.path_to_mcmc_plots / ".done_mcmc"))
    log:
        str(LOGS / "plot_mcmc.log")
    params:
        config=MEGATOP_CONFIG
    shell:
        "megatop-cl2r_mcmc-plot --config {params.config} > {log} 2>&1"
