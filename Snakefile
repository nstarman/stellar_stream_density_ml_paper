# ==============================================================================
# Isochrone files

rule download_brutus_mist_file:
    output:
        "src/data/brutus/MIST_1.2_iso_vvcrit0.0.h5"
    conda:
        "environment.yml"
    params:
        load_from_static=True,  # set to False to train the model
        save_to_static=False,
    cache:
        True
    script:
        "src/scripts/brutus/download_mist_file.py"


rule download_brutus_nn_file:
    output:
        "src/data/brutus/nn_c3k.h5"
    params:
        load_from_static=True,  # set to False to train the model
        save_to_static=False,
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/brutus/download_nn_file.py"


# ==============================================================================
# Mock data

rule mock_make_data:
    output:
        "src/data/mock/data.asdf"
    input:
        "src/data/brutus/MIST_1.2_iso_vvcrit0.0.h5",
        "src/data/brutus/nn_c3k.h5",
    params:
        seed=10,
        diagnostic_plots=True
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/mock/data/make.py"


rule mock_nstream_variable:
    output:
        "src/tex/output/mock/nstream_variable.txt"
    input:
        "src/data/mock/data.asdf"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/mock/data/variable_nstream.py"


rule mock_nbackround_variable:
    output:
        "src/tex/output/mock/nbackground_variable.txt"
    input:
        "src/data/mock/data.asdf"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/mock/data/variable_nbackground.py"


rule mock_isochrone_age_variable:
    output:
        "src/tex/output/mock/isochrone_age_variable.txt"
    input:
        "src/data/mock/data.asdf"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/mock/data/variable_isochrone_age.py"


rule mock_isochrone_feh_variable:
    output:
        "src/tex/output/mock/isochrone_feh_variable.txt"
    input:
        "src/data/mock/data.asdf"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/mock/data/variable_isochrone_feh.py"


rule mock_train_flow:
    output:
        "src/data/mock/flow_model.pt"
    input:
        "src/data/mock/data.asdf"
    params:
        load_from_static=True,  # set to False to train the model
        save_to_static=False,
        diagnostic_plots=True,
        epochs=400,
        batch_size=500,
        lr=1e-3
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/mock/model/1-train_flow.py"


rule mock_train_model:
    output:
        "src/data/mock/model.pt"
    input:
        "src/data/mock/data.asdf",
        "src/data/mock/flow_model.pt",
    params:
        load_from_static=True,  # set to False to train the model
        save_to_static=False,
        diagnostic_plots=True,
        # epoch milestones
        init_T=500,
        T_0=500,
        n_T=3,
        final_T=600,
        eta_min=1e-4,
        lr=5e-3
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/mock/model/2-train_model.py"


rule mock_ncorrect_variable:
    output:
        "src/tex/output/mock/ncorrect_variable.txt"
    input:
        "src/data/mock/data.asdf",
        "src/data/mock/model.pt",
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/mock/plot/variable_ncorrect.py"


rule mock_falseident_variable:
    output:
        "src/tex/output/mock/falseident_variable.txt"
    input:
        "src/data/mock/data.asdf",
        "src/data/mock/model.pt",
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/mock/plot/variable_falseident.py"



# ==============================================================================
# Dustmaps

rule download_dustmaps:
    output:
        "src/data/dustmaps/bayestar/bayestar2019.h5"
    params:
        load_from_static=True,  # set to False to train the model
        save_to_static=False,
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/dustmaps/download_dustmaps.py"


# ==============================================================================
# GD-1

rule gd1_query_data:
    output:
        protected("src/data/gd1/gaia_ps1_xm_polygons.asdf")
    params:
        load_from_static=True,  # set to False to redownload
        save_to_static=False,
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/gd1/data/1-query_data.py"


rule gd1_combine_data:
    output:
        protected("src/data/gd1/gaia_ps1_xm.asdf")
    input:
        "src/data/dustmaps/bayestar/bayestar2019.h5",
        "src/data/gd1/gaia_ps1_xm_polygons.asdf",
    params:
        load_from_static=True,  # set to False to recompute
        save_to_static=False,
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/gd1/data/2-combine_data.py"


rule gd1_variable_ndata:
    output:
        "src/tex/output/gd1/ndata_variable.txt",
    input:
        "src/data/gd1/gaia_ps1_xm.asdf",
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/gd1/data/2.5-variable_ndata.py"


rule gd1_masks_pm:
    output:
        "src/data/gd1/pm_edges.ecsv"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/gd1/data/3.1-pm_edges.py"


rule gd1_masks_iso:
    output:
        "src/data/gd1/isochrone.asdf"
    input:
        "src/data/brutus/MIST_1.2_iso_vvcrit0.0.h5",
        "src/data/brutus/nn_c3k.h5",
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/gd1/data/3.2-phot_edges.py"


rule gd1_masks:
    output:
        "src/data/gd1/masks.asdf"
    input:
        "src/data/gd1/gaia_ps1_xm.asdf",
        "src/data/gd1/pm_edges.ecsv",
        "src/data/gd1/isochrone.asdf",
    params:
        load_from_static=True,  # set to False to recompute
        save_to_static=False,
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/gd1/data/3.3-masks.py"


rule gd1_stream_control_points:
    output:
        "src/data/gd1/stream_control_points.ecsv"
    cache:
        True
    script:
        "src/scripts/gd1/model/0-stream_control_points.py"


rule gd1_spur_control_points:
    output:
        "src/data/gd1/spur_control_points.ecsv"
    cache:
        True
    script:
        "src/scripts/gd1/model/0-spur_control_points.py"


rule gd1_info:
    output:
        "src/data/gd1/info.asdf"
    input:
        "src/data/gd1/gaia_ps1_xm.asdf",
        "src/data/gd1/masks.asdf",
    params:
        pm_mask="pm_tight",
        phot_mask="cmd_medium",
    cache:
        True
    script:
        "src/scripts/gd1/model/1-info.py"


# NOTE: this is a hacky way to aggregate the dependencies of the data script
rule gd1_data_script:
    output:
        temp("src/data/gd1/data.tmp")
    input:
        "src/data/gd1/gaia_ps1_xm.asdf",
        "src/data/gd1/info.asdf",
    cache:
        False
    script:
        "src/scripts/gd1/datasets.py"


# NOTE: this is a hacky way to aggregate the dependencies of the model script
rule gd1_model_script:
    output:
        temp("src/data/gd1/model.tmp")
    input:
        "src/data/gd1/info.asdf",
        "src/data/gd1/stream_control_points.ecsv",
        "src/data/gd1/spur_control_points.ecsv",
    cache:
        False
    script:
        "src/scripts/gd1/define_model.py"


rule gd1_train_flow:
    output:
        "src/data/gd1/flow_model.pt"
    input:
        "src/data/gd1/data.tmp",
        "src/data/gd1/model.tmp",
    params:
        load_from_static=True,  # set to False to recompute
        save_to_static=False,
        diagnostic_plots=True,
        epochs=1_000,
    cache:
        True
    script:
        "src/scripts/gd1/model/2-train_flow.py"


# TODO: rerun from scratch
rule gd1_train_model:
    output:
        "src/data/gd1/model.pt"
    input:
        "src/data/gd1/data.tmp",
        "src/data/gd1/model.tmp",
        "src/data/gd1/flow_model.pt",
    params:
        load_from_static=True,  # set to False to recompute
        save_to_static=False,
        diagnostic_plots=True,
        # epoch milestones
        epochs=1_250 * 10,
        lr=1e-3,
        weight_decay=1e-8,
    cache:
        True
    script:
        "src/scripts/gd1/model/3-train_model.py"


rule gd1_member_likelihoods:
    output:
        "src/data/gd1/membership_likelhoods.ecsv"
    input:
        "src/data/gd1/data.tmp",
        "src/data/gd1/model.tmp",
    cache:
        True
    script:
        "src/scripts/gd1/model/4-likelihoods.py"


rule gd1_member_table_select:
    output:
        "src/tex/output/gd1/member_table_select.tex"
    input:
        "src/data/gd1/data.tmp",
        "src/data/gd1/model.tmp",
        "src/data/gd1/membership_likelhoods.ecsv"
    cache:
        True
    script:
        "src/scripts/gd1/table/member_table_select.py"


rule gd1_member_table_full:
    output:
        "src/tex/output/gd1/member_table_full.tex"
    input:
        "src/data/gd1/data.tmp",
        "src/data/gd1/model.tmp",
        "src/data/gd1/membership_likelhoods.ecsv"
    cache:
        True
    script:
        "src/scripts/gd1/table/member_table_full.py"


# ==============================================================================
# Palomar 5

rule pal5_query_data:
    output:
        protected("src/data/pal5/gaia_ps1_xm_polygons.asdf")
    params:
        load_from_static=True,  # set to False to redownload
        save_to_static=False,
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/pal5/data/1-query_data.py"


rule pal5_combine_data:
    output:
        protected("src/data/pal5/gaia_ps1_xm.asdf")
    input:
        "src/data/dustmaps/bayestar/bayestar2019.h5",
        "src/data/pal5/gaia_ps1_xm_polygons.asdf",
    params:
        load_from_static=True,  # set to False to recompute
        save_to_static=False,
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/pal5/data/2-combine_data.py"


rule pal5_variable_ndata:
    output:
        "src/tex/output/pal5/ndata_variable.txt",
    input:
        "src/data/pal5/gaia_ps1_xm.asdf",
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/pal5/data/2.5-variable_ndata.py"


rule pal5_masks:
    output:
        "src/data/pal5/masks.asdf"
    input:
        "src/data/pal5/gaia_ps1_xm.asdf",
    params:
        load_from_static=True,  # set to False to recompute
        save_to_static=False,
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/pal5/data/3.3-masks.py"


rule pal5_stream_control_points:
    output:
        "src/data/pal5/stream_control_points.ecsv"
    cache:
        True
    script:
        "src/scripts/pal5/model/0-stream_control_points.py"


rule pal5_info:
    output:
        "src/data/pal5/info.asdf"
    input:
        "src/data/pal5/gaia_ps1_xm.asdf",
        "src/data/pal5/masks.asdf",
    params:
        pm_mask="pm_tight_icrs",
    cache:
        True
    script:
        "src/scripts/pal5/model/1-info.py"


# NOTE: this is a hacky way to aggregate the dependencies of the data script
rule pal5_data_script:
    output:
        temp("src/data/pal5/data.tmp")
    input:
        "src/data/pal5/gaia_ps1_xm.asdf",
        "src/data/pal5/info.asdf",
    cache:
        False
    script:
        "src/scripts/pal5/datasets.py"
