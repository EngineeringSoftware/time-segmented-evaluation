#!/bin/bash

# This script documents the exact procedures we use to get the
# dataset, get the models running, and collect the results.

# Each function is a group of commands and a later function usually
# requires the execution of all the proceeding functions.

# The commands within each function should always be executed one
# after one sequentially unless only partial functionality is wanted.


_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )


# ========== Metrics, Tables, and Plots

function metric_dataset() {
        python -m tseval.main collect_metrics --which=raw-data-filtered
        python -m tseval.main collect_metrics --which=split-dataset --split=Full

        for task in CG MN; do
                for setup in CP MP T; do
                        python -m tseval.main collect_metrics --which=setup-dataset --task=$task --setup=$setup
                done
        done
        
        for task in CG MN; do
                for setup in CP MP T; do
                        python -m tseval.main collect_metrics --which=setup-dataset-leak --task=$task --setup=$setup
                done
        done
}


function tables_dataset() {
        python -m tseval.main make_tables --which=dataset-metrics
}


function plots_dataset() {
        for task in CG MN; do
                python -m tseval.main make_plots --which=dataset-metrics-dist --task=$task
        done
}


function analyze_exps() {
        for exps in cg mn; do
                python -m tseval.main analyze_extract_metrics --exps=./exps/${exps}.yaml
                python -m tseval.main analyze_sign_test --exps=./exps/${exps}.yaml
        done
}


function tables_exps() {
        for exps in cg mn; do
                python -m tseval.main analyze_make_tables --exps=./exps/${exps}.yaml
        done
}


function plots_exps() {
        for exps in cg mn; do
                python -m tseval.main analyze_make_plots --exps=./exps/${exps}.yaml
        done
}


function extract_data_sim() {
        for exps in cg mn; do
                python -m tseval.main analyze_extract_data_similarities --exps=./exps/${exps}.yaml
        done
}


function analyze_near_duplicates() {
        for exps in cg mn; do
                python -m tseval.main analyze_near_duplicates --exps=./exps/${exps}.yaml \
                        --config=same_code \
                        --code_sim=1 --nl_sim=1.1

                python -m tseval.main analyze_near_duplicates --exps=./exps/${exps}.yaml \
                        --config=same_nl \
                        --code_sim=1.1 --nl_sim=1

                python -m tseval.main analyze_near_duplicates --exps=./exps/${exps}.yaml \
                        --config=sim_90 \
                        --code_sim=0.9 --nl_sim=0.9
        done
}


function analyze_near_duplicates_only_tables_plots() {
        for exps in cg mn; do
                python -m tseval.main analyze_near_duplicates --exps=./exps/${exps}.yaml \
                        --config=same_code \
                        --code_sim=1 --nl_sim=1.1 --only_tables_plots

                python -m tseval.main analyze_near_duplicates --exps=./exps/${exps}.yaml \
                        --config=same_nl \
                        --code_sim=1.1 --nl_sim=1 --only_tables_plots

                python -m tseval.main analyze_near_duplicates --exps=./exps/${exps}.yaml \
                        --config=sim_90 \
                        --code_sim=0.9 --nl_sim=0.9 --only_tables_plots
        done
}


# ========== Data collection

function collect_repos() {
        python -m tseval.main collect_repos
        python -m tseval.main filter_repos
}


function collect_raw_data() {
        python -m tseval.main collect_raw_data
        python -m tseval.main process_raw_data
}


# ========== Eval preparation

function prepare_envs() {
        # Require tseval conda env first, prepared using prepare_conda_env
        python -m tseval.main prepare_envs
}


function prepare_splits() {
        python -m tseval.main get_splits --seed=7 --split=Debug --debug
        python -m tseval.main get_splits --seed=7 --split=Full
}


function cg_debug_workflow() {
        python -m tseval.main exp_prepare --task=CG --setup=StandardSetup --setup_name=Debug --split_name=Debug --split_type=MP
        python -m tseval.main exp_train --task=CG --setup_name=Debug --model_name=TransformerACL20 --exp_name=TransformerACL20
        for action in test_common val test_standard; do
                python -m tseval.main exp_eval --task=CG --setup_name=Debug --exp_name=TransformerACL20 --action=$action
        done
        for action in test_common val test_standard; do
                python -m tseval.main exp_compute_metrics --task=CG --setup_name=Debug --exp_name=TransformerACL20 --action=$action
        done
}


function cg_prepare_setups() {
        for split_type in MP CP T; do
                python -m tseval.main exp_prepare --task=CG --setup=StandardSetup --setup_name=$split_type --split_name=Full --split_type=$split_type
        done
}


function cg_debug_model() {
        local model=$1; shift
        local args="$@"; shift
        echo "Arguments to model $model: $args"

        set -e
        set -x
        python -m tseval.main exp_train --task=CG --setup_name=Debug --exp_name=$model\
               --model_name=$model $args
        for action in test_common val test_standard; do
                python -m tseval.main exp_eval --task=CG --setup_name=Debug --exp_name=$model --action=$action
                python -m tseval.main exp_compute_metrics --task=CG --setup_name=Debug --exp_name=$model --action=$action
        done
}


function cg_run_model() {
        local model=$1; shift
        local setup=$1; shift
        local args="$@"; shift
        echo "Arguments to model $model: $args"

        set -e
        set -x
        python -m tseval.main exp_train --task=CG --setup_name=$setup --exp_name=$model\
               --model_name=$model $args
        for action in test_common val test_standard; do
                python -m tseval.main exp_eval --task=CG --setup_name=$setup --exp_name=$model --action=$action
                python -m tseval.main exp_compute_metrics --task=CG --setup_name=$setup --exp_name=$model --action=$action
        done
}


function mn_debug_workflow() {
        python -m tseval.main exp_prepare --task=MN --setup=StandardSetup --setup_name=Debug --split_name=Debug --split_type=MP
        python -m tseval.main exp_train --task=MN --setup_name=Debug --model_name=Code2SeqICLR19 --exp_name=Code2SeqICLR19
        for action in test_common val test_standard; do
                python -m tseval.main exp_eval --task=MN --setup_name=Debug --exp_name=Code2SeqICLR19 --action=$action
        done
        for action in test_common val test_standard; do
                python -m tseval.main exp_compute_metrics --task=MN --setup_name=Debug --exp_name=Code2SeqICLR19 --action=$action
        done
}


function mn_prepare_setups() {
        for split_type in MP CP T; do
                python -m tseval.main exp_prepare --task=MN --setup=StandardSetup --setup_name=$split_type --split_name=Full --split_type=$split_type
        done
}


function mn_debug_model() {
        local model=$1; shift
        local args="$@"; shift
        echo "Arguments to model $model: $args"

        set -e
        set -x
        python -m tseval.main exp_train --task=MN --setup_name=Debug --exp_name=$model\
               --model_name=$model $args
        for action in test_common val test_standard; do
                python -m tseval.main exp_eval --task=MN --setup_name=Debug --exp_name=$model --action=$action
                python -m tseval.main exp_compute_metrics --task=MN --setup_name=Debug --exp_name=$model --action=$action
        done
}


function mn_run_model() {
        local model=$1; shift
        local setup=$1; shift
        local args="$@"; shift
        echo "Arguments to model $model: $args"

        set -e
        set -x
        python -m tseval.main exp_train --task=MN --setup_name=$setup --exp_name=$model\
               --model_name=$model $args
        for action in test_common val test_standard; do
                python -m tseval.main exp_eval --task=MN --setup_name=$setup --exp_name=$model --action=$action
                python -m tseval.main exp_compute_metrics --task=MN --setup_name=$setup --exp_name=$model --action=$action
        done
}




# ==========
# Main function -- program entry point
# This script can be executed as ./run.sh the_function_to_run

function main() {
        local action=${1:?Need Argument}; shift

        ( cd ${_DIR}
          $action "$@"
        )
}

main "$@"
