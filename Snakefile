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
        "src/data/mock/background_photometry_model.pt"
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
        "src/data/mock/background_photometry_model.pt",
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


rule panstarrs1_corrections:
    output:
        "src/data/dustmaps/ps1_corrections.json"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/dustmaps/ps1_corrections.py"


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
        "src/data/dustmaps/ps1_corrections.json",
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


rule gd1_control_points_distance:
    output:
        "src/data/gd1/control_points_distance.ecsv"
    cache:
        True
    script:
        "src/scripts/gd1/model/0-control_points_distance.py"


rule gd1_control_points_stream:
    output:
        "src/data/gd1/control_points_stream.ecsv"
    cache:
        True
    script:
        "src/scripts/gd1/model/0-control_points_stream.py"


rule gd1_control_points_spur:
    output:
        "src/data/gd1/control_points_spur.ecsv"
    cache:
        True
    script:
        "src/scripts/gd1/model/0-control_points_spur.py"


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
        "src/data/gd1/control_points_stream.ecsv",
        "src/data/gd1/control_points_spur.ecsv",
    cache:
        False
    script:
        "src/scripts/gd1/define_model.py"


rule gd1_train_background_photometry_flow:
    output:
        "src/data/gd1/background_photometry_model.pt"
    input:
        "src/data/gd1/data.tmp",
        "src/data/gd1/model.tmp",
    params:
        load_from_static=True,  # set to False to recompute
        save_to_static=False,
        diagnostic_plots=True,
        epochs=2_000,
    cache:
        True
    script:
        "src/scripts/gd1/model/2-train_background_photometry_flow.py"


rule gd1_train_background_parallax_flow:
    output:
        "src/data/gd1/background_parallax_model.pt"
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
        "src/scripts/gd1/model/2-train_background_parallax_flow.py"


# TODO: rerun from scratch
rule gd1_train_model:
    output:
        "src/data/gd1/model.pt"
    input:
        "src/data/gd1/data.tmp",
        "src/data/gd1/model.tmp",
        "src/data/gd1/background_photometry_model.pt",
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
        "src/data/dustmaps/ps1_corrections.json",
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


rule pal5_masks_pm:
    output:
        "src/data/pal5/pm_edges.ecsv"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/pal5/data/3.1-pm_edges.py"


rule pal5_masks_off_field:
    output:
        "src/data/pal5/footprint.npz"
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/pal5/data/3.3-off_field.py"


rule pal5_masks:
    output:
        "src/data/pal5/masks.asdf"
    input:
        "src/data/pal5/gaia_ps1_xm.asdf",
        "src/data/pal5/pm_edges.ecsv",
        "src/data/pal5/footprint.npz",
    params:
        load_from_static=True,  # set to False to recompute
        save_to_static=False,
    conda:
        "environment.yml"
    cache:
        True
    script:
        "src/scripts/pal5/data/4-masks.py"


rule pal5_control_points_stream:
    output:
        "src/data/pal5/control_points_stream.ecsv"
    cache:
        True
    script:
        "src/scripts/pal5/model/0-control_points_stream.py"


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


# NOTE: this is a hacky way to aggregate the dependencies of the model script
rule pal5_model_script:
    output:
        temp("src/data/pal5/model.tmp")
    input:
        "src/data/pal5/info.asdf",
        "src/data/pal5/control_points_stream.ecsv",
        "src/data/pal5/control_points_spur.ecsv",
    cache:
        False
    script:
        "src/scripts/pal5/define_model.py"


rule pal5_train_model:
    output:
        "src/data/pal5/model.pt"
    input:
        "src/data/pal5/data.tmp",
        "src/data/pal5/model.tmp",
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
        "src/scripts/pal5/model/3-train_model.py"


rule pal5_member_likelihoods:
    output:
        "src/data/gd1/membership_likelhoods.ecsv"
    input:
        "src/data/gd1/data.tmp",
        "src/data/gd1/model.tmp",
    cache:
        True
    script:
        "src/scripts/gd1/model/4-likelihoods.py"


rule pal5_member_table_select:
    output:
        "src/tex/output/pal5/member_table_select.tex"
    input:
        "src/data/pal5/data.tmp",
        "src/data/pal5/model.tmp",
        "src/data/pal5/membership_likelhoods.ecsv"
    cache:
        True
    script:
        "src/scripts/pal5/table/member_table_select.py"


rule pal5_member_table_full:
    output:
        "src/tex/output/pal5/member_table_full.tex"
    input:
        "src/data/pal5/data.tmp",
        "src/data/pal5/model.tmp",
        "src/data/pal5/membership_likelhoods.ecsv"
    cache:
        True
    script:
        "src/scripts/pal5/table/member_table_full.py"
