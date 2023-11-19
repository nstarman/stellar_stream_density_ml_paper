################################################################################
# PGM

rule pgm:
    output:
        "src/tex/output/pgm.pdf"
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/pgm.py"


################################################################################
# Isochrone files

rule download_brutus_mist_file:
    output:
        "src/data/brutus/MIST_1.2_iso_vvcrit0.0.h5"
    conda:
        "environment.yml"
    params:
        load_from_static=True,  # set to False to train the model
        save_to_static=False,
    cache: True
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
    cache: True
    script:
        "src/scripts/brutus/download_nn_file.py"


################################################################################
# Mock data

rule mock_make_data:
    output:
        "src/data/mock/data.asdf"
    input:
        "src/data/brutus/MIST_1.2_iso_vvcrit0.0.h5",
        "src/data/brutus/nn_c3k.h5",
    params:
        seed=10,
        diagnostic_plots=True,
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/mock/data/1-make.py"


rule mock_nstream_variable:
    output:
        "src/tex/output/mock/nstream_variable.txt"
    input:
        "src/data/mock/data.asdf"
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/mock/data/2-variable_nstream.py"


rule mock_nbackround_variable:
    output:
        "src/tex/output/mock/nbackground_variable.txt"
    input:
        "src/data/mock/data.asdf"
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/mock/data/2-variable_nbackground.py"


rule mock_isochrone_age_variable:
    output:
        "src/tex/output/mock/isochrone_age_variable.txt"
    input:
        "src/data/mock/data.asdf"
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/mock/data/2-variable_isochrone_age.py"


rule mock_isochrone_feh_variable:
    output:
        "src/tex/output/mock/isochrone_feh_variable.txt"
    input:
        "src/data/mock/data.asdf"
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/mock/data/2-variable_isochrone_feh.py"


rule gd1_model_script:
    output: touch("src/data/mock/model.done")
    input:
        "src/data/mock/data.asdf",
    conda:
        "environment.yml"
    cache: False
    script:
        "src/scripts/mock/model.py"


rule mock_train_flow:
    output:
        "src/data/mock/background_photometry_model.pt"
    input:
        "src/data/mock/data.asdf",
        "model.done",
    params:
        load_from_static=True,  # set to False to train the model
        save_to_static=False,
        diagnostic_plots=True,
        epochs=400,
        batch_size=500,
        lr=1e-3,
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/mock/models/1-train_flow.py"


rule mock_train_model:
    output:
        "src/data/mock/model.pt"
    input:
        "src/data/mock/data.asdf",
        "model.done",
        "src/data/mock/background_photometry_model.pt",
    params:
        load_from_static=True,  # set to False to train the model
        save_to_static=False,
        diagnostic_plots=True,
        # epoch milestones
        lr=5e-3,
        epochs=3_000
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/mock/models/2-train_model.py"


rule mock_ncorrect_variable:
    output:
        "src/tex/output/mock/ncorrect_variable.txt"
    input:
        "src/data/mock/data.asdf",
        "src/data/mock/model.pt",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/mock/stats/variable_ncorrect.py"


rule mock_falseident_variable:
    output:
        "src/tex/output/mock/falseident_variable.txt"
    input:
        "src/data/mock/data.asdf",
        "src/data/mock/model.pt",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/mock/stats/variable_falseident.py"



################################################################################
# Dustmaps

rule download_dustmaps:
    output:
        "src/data/dustmaps/bayestar/bayestar2019.h5"
    params:
        load_from_static=True,  # set to False to train the model
        save_to_static=False,
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/dustmaps/download_dustmaps.py"


rule panstarrs1_corrections:
    output:
        "src/data/dustmaps/ps1_corrections.json"
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/dustmaps/ps1_corrections.py"


###############################################################################
# GD-1

# ---------------------------------------------------------
# Data

rule gd1_query_data:
    output:
        temp("src/data/gd1/gaia_ps1_xm_polygons.asdf")
    params:
        load_from_static=True,  # set to False to redownload
        save_to_static=True,
    conda:
        "environment.yml"
    cache: True
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
    cache: True
    script:
        "src/scripts/gd1/data/2-combine_data.py"


rule gd1_variable_ndata:
    output:
        "src/tex/output/gd1/ndata_variable.txt",
    input:
        "src/data/gd1/gaia_ps1_xm.asdf",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/data/2.5-variable_ndata.py"


rule gd1_masks_pm:
    output:
        "src/data/gd1/pm_edges.ecsv"
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/data/3-pm_edges.py"


rule gd1_masks_iso:
    output:
        "src/data/gd1/isochrone.asdf"
    input:
        "src/data/brutus/MIST_1.2_iso_vvcrit0.0.h5",
        "src/data/brutus/nn_c3k.h5",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/data/3-phot_edges.py"


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
    cache: True
    script:
        "src/scripts/gd1/data/4-masks.py"


# ---------------------------------------------------------
# Model

rule gd1_info:
    output:
        "src/data/gd1/info.asdf"
    input:
        "src/data/gd1/gaia_ps1_xm.asdf",
        "src/data/gd1/masks.asdf",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/models/0-info.py"


# NOTE: this is a hacky way to aggregate the dependencies of the data script
rule gd1_dataset_script:
    output: touch("src/data/gd1/subset/data.done")
    input:
        "src/data/gd1/gaia_ps1_xm.asdf",
        "src/data/gd1/info.asdf",
    conda:
        "environment.yml"
    cache: False
    script:
        "src/scripts/gd1/datasets.py"


rule gd1_control_points_distance:
    output:
        "src/data/gd1/control_points_distance.ecsv"
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/model/1-control_points_distance.py"


rule gd1_control_points_stream:
    output:
        "src/data/gd1/control_points_stream.ecsv"
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/model/1-control_points_stream.py"


rule gd1_control_points_spur:
    output:
        "src/data/gd1/control_points_spur.ecsv"
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/model/1-control_points_spur.py"


rule gd1_model_script:
    output: touch("src/data/gd1/model.done")
    input:
        "src/data/gd1/info.asdf",
        "src/data/gd1/control_points_stream.ecsv",
        "src/data/gd1/control_points_spur.ecsv",
        "src/data/gd1/control_points_distance.ecsv",
    conda:
        "environment.yml"
    cache: False
    script:
        "src/scripts/gd1/model.py"


rule gd1_train_background_parallax_flow:
    output:
        "src/data/gd1/subset/background_parallax_model.pt"
    input:
        "src/data/gd1/subset/data.done",
        "src/data/gd1/model.done",
    params:
        load_from_static=True,  # set to False to recompute
        save_to_static=False,
        diagnostic_plots=True,
        epochs=1_000,
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/model/subset/1-train_background_parallax_flow.py"


rule gd1_train_background_photometry_flow:
    output:
        "src/data/gd1/subset/background_photometry_model.pt"
    input:
        "src/data/gd1/subset/data.done",
        "src/data/gd1/model.done",
    params:
        load_from_static=True,  # set to False to recompute
        save_to_static=False,
        diagnostic_plots=True,
        epochs=2_000,
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/model/subset/2-train_background_photometry_flow.py"


rule gd1_train_model:
    output:
        "src/data/gd1/subset/model.pt"
    input:
        "src/data/gd1/subset/data.done",
        "src/data/gd1/model.done",
        "src/data/gd1/subset/background_photometry_model.pt",
        "src/data/gd1/subset/background_parallax_model.pt",
    params:
        load_from_static=True,  # set to False to recompute
        save_to_static=False,
        diagnostic_plots=True,
        # epoch milestones
        epochs=1_250 * 10,
        lr=1e-3,
        weight_decay=1e-8,
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/model/subset/3-train_model.py"


rule gd1_member_likelihoods:
    output:
        "src/data/gd1/membership_likelhoods.ecsv"
    input:
        "src/data/gd1/data.done",
        "src/data/gd1/model.done",
        "src/data/gd1/model.pt",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/model/4-likelihoods.py"


rule gd1_member_table_select:
    output:
        "src/tex/output/gd1/member_table_select.tex"
    input:
        "src/data/gd1/data.done",
        "src/data/gd1/model.done",
        "src/data/gd1/membership_likelhoods.ecsv",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/table/member_table_select.py"


rule gd1_member_table_full:
    output:
        "src/tex/output/gd1/member_table_full.tex"
    input:
        "src/data/gd1/data.done",
        "src/data/gd1/model.done",
        "src/data/gd1/membership_likelhoods.ecsv",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/table/member_table_full.py"


rule gd1_variable_nstream:
    output:
        "src/tex/output/gd1/nstream/nstream_variable.txt"
    input:
        "src/data/gd1/membership_likelhoods.ecsv",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/model/5-variable_nstream.py"


rule gd1_variable_nspur:
    output:
        "src/tex/output/gd1/nspur/nspur_variable.txt"
    input:
        "src/data/gd1/membership_likelhoods.ecsv",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/model/5-variable_nspur.py"


# ---------------------------------------------------------
# Table

rule gd1_control_points_table:
    output:
        "src/tex/output/gd1/control_points.tex"
    input:
        "src/data/gd1/control_points_stream.ecsv",
        "src/data/gd1/control_points_spur.ecsv",
        "src/data/gd1/control_points_distance.ecsv",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/gd1/table/control_points.py"



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
    cache: True
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
    cache: True
    script:
        "src/scripts/pal5/data/2-combine_data.py"


rule pal5_variable_ndata:
    output:
        "src/tex/output/pal5/ndata_variable.txt",
    input:
        "src/data/pal5/gaia_ps1_xm.asdf",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/pal5/data/2.5-variable_ndata.py"


rule pal5_masks_pm:
    output:
        "src/data/pal5/pm_edges.ecsv"
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/pal5/data/3-pm_edges.py"


rule pal5_masks_iso:
    output:
        "src/data/pal5/isochrone.asdf"
    params:
        diagnostic_plots=True,
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/pal5/data/3-phot_edges.py"


rule pal5_masks_off_field:
    output:
        "src/data/pal5/footprint.npz"
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/pal5/data/4-off_field.py"


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
        diagnostic_plots=True,
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/pal5/data/5-masks.py"

rule pal5_info:
    output:
        "src/data/pal5/info.asdf"
    input:
        "src/data/pal5/gaia_ps1_xm.asdf",
        "src/data/pal5/masks.asdf",
    params:
        pm_mask="pm_med_icrs",
        phot_mask="phot_15",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/pal5/models/0-info.py"


rule pal5_control_points_stream:
    output:
        "src/data/pal5/control_points_stream.ecsv"
    conda:
        "environment.yml"
    params:
        diagnostic_plots=True,
    cache: True
    script:
        "src/scripts/pal5/models/1-control_points_stream.py"



# NOTE: this is a hacky way to aggregate the dependencies of the data script
rule pal5_data_script:
    output: touch("src/data/pal5/data.done")
    input:
        "src/data/pal5/gaia_ps1_xm.asdf",
        "src/data/pal5/info.asdf",
    conda:
        "environment.yml"
    cache: False
    script:
        "src/scripts/pal5/datasets.py"


# NOTE: this is a hacky way to aggregate the dependencies of the model script
rule pal5_model_script:
    output: touch("src/data/pal5/model.done")
    input:
        "src/data/pal5/info.asdf",
        "src/data/pal5/control_points_stream.ecsv",
    conda:
        "environment.yml"
    cache: False
    script:
        "src/scripts/pal5/model.py"


rule pal5_train_model:
    output:
        "src/data/pal5/model.pt"
    input:
        "src/data/pal5/data.done",
        "src/data/pal5/model.done",
    params:
        load_from_static=True,  # set to False to recompute
        save_to_static=False,
        diagnostic_plots=True,
        # epoch milestones
        epochs=1_250 * 10,
        lr=1e-4,
        weight_decay=1e-8,
        # end point
        early_stopping = -1,
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/pal5/models/3-train_model.py"


rule pal5_member_likelihoods:
    output:
        "src/data/pal5/membership_likelhoods.ecsv"
    input:
        "src/data/pal5/data.done",
        "src/data/pal5/model.done",
        "src/data/pal5/model.pt",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/pal5/models/4-likelihoods.py"


rule pal5_member_table_select:
    output:
        "src/tex/output/pal5/member_table_select.tex"
    input:
        "src/data/pal5/data.done",
        "src/data/pal5/model.done",
        "src/data/pal5/membership_likelhoods.ecsv",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/pal5/table/member_table_select.py"


rule pal5_member_table_full:
    output:
        "src/tex/output/pal5/member_table_full.tex"
    input:
        "src/data/pal5/data.done",
        "src/data/pal5/model.done",
        "src/data/pal5/membership_likelhoods.ecsv",
    conda:
        "environment.yml"
    cache: True
    script:
        "src/scripts/pal5/table/member_table_full.py"
