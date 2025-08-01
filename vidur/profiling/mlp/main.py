import argparse
import datetime
import itertools
import os
from typing import Any, List

import pandas as pd
import ray
import yaml
from tqdm import tqdm

from vidur.profiling.common.model_config import ModelConfig
from vidur.profiling.mlp.mlp_wrapper import MlpWrapper
from vidur.profiling.utils import ProfileMethod, get_num_tokens_to_profile


def parse_args():
    parser = argparse.ArgumentParser(description="MLP Profiling")
    parser.add_argument(
        "--disable_ray",
        action="store_true",
        help="Disable Ray",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=8,
        help="Number of GPUs to use for profiling",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/home/ubuntu/vidur/data/profiling/compute",
        help="Output directory for profiling results",
    )
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="T4",
        help="GPU type for organizing results",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=[
            "meta-llama/Meta-Llama-3-8B",
        ],
        help="Models to profile",
    )
    parser.add_argument(
        "--num_tensor_parallel_workers",
        type=int,
        nargs="+",
        default=[8],
        help="Number of tensor parallel workers to profile",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=8192,
        help="Maximum number of tokens to profile",
    )
    parser.add_argument(
        "--profile_method",
        default="record_function",
        choices=[e.value for e in ProfileMethod],
        help="Method to use for measuring time taken by operations (default: %(default)s)",
    )
    args = parser.parse_args()

    # Create GPU-specific directory structure
    args.output_dir = f"{args.output_dir}/{args.gpu_type}/mlp"
    os.makedirs(args.output_dir, exist_ok=True)

    return args


def profile_model_for_tp(
    args: argparse.Namespace, model: str, num_tensor_parallel_workers: int, num_tokens_to_profile: List[int], pbar: Any
):
    model_config = ModelConfig.from_model_name(model)

    if model_config.no_tensor_parallel and num_tensor_parallel_workers > 1:
        pbar.update(len(num_tokens_to_profile))
        return pd.DataFrame()

    promises = []
    all_results = []

    model_wrapper_actor = ray.remote(
        num_cpus=1,
        num_gpus=1,
    )(
        MlpWrapper,
    ).options(runtime_env={"env_vars": {"KINETO_LOG_LEVEL": "5"}})

    model_wrappers = [
        model_wrapper_actor.remote(
            model_config,
            num_tensor_parallel_workers,
            args.profile_method,
            rank,
            args.output_dir,
        )
        for rank in range(args.num_gpus)
    ]
    
    for num_tokens in num_tokens_to_profile:
        worker_id = len(promises)
        promise = model_wrappers[worker_id].profile.remote(
            num_tokens,
        )
        promises.append(promise)

        if len(promises) >= args.num_gpus:
            results = ray.get(promises)
            all_results.extend(results)
            promises = []

        pbar.update(1)

    results = ray.get(promises)
    all_results.extend(results)

    df = pd.DataFrame(all_results)
    if not df.empty:
        # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix
        df = (
            pd.json_normalize(df["time_stats"])
            .add_prefix("time_stats.")
            .join(df.drop(columns=["time_stats"]))
        )

    return df


def profile_model_for_tp_no_ray(
    args: argparse.Namespace, model: str, num_tensor_parallel_workers: int, num_tokens_to_profile: List[int], pbar: Any
):
    model_config = ModelConfig.from_model_name(model)

    if model_config.no_tensor_parallel and num_tensor_parallel_workers > 1:
        pbar.update(len(num_tokens_to_profile))
        return pd.DataFrame()

    all_results = []

    # Create a single model wrapper without Ray
    model_wrapper = MlpWrapper(
        model_config,
        num_tensor_parallel_workers,
        args.profile_method,
        0,  # rank 0
        args.output_dir,
    )
    
    for num_tokens in num_tokens_to_profile:
        result = model_wrapper.profile(num_tokens)
        all_results.append(result)
        pbar.update(1)

    df = pd.DataFrame(all_results)
    if not df.empty:
        # the time_stats column is a dict, so we need to expand it into columns recursively and add prefix
        df = (
            pd.json_normalize(df["time_stats"])
            .add_prefix("time_stats.")
            .join(df.drop(columns=["time_stats"]))
        )

    return df


def main():
    args = parse_args()
    yaml.dump(vars(args), open(f"{args.output_dir}/config.yaml", "w"))

    if not args.disable_ray:
        ray.init()

    num_tokens_to_profile = get_num_tokens_to_profile(args.max_tokens)

    total_combos = itertools.product(
        args.models,
        num_tokens_to_profile,
        args.num_tensor_parallel_workers,
    )

    pbar = tqdm(total=len(list(total_combos)))

    for model in args.models:
        for num_tensor_parallel_workers in args.num_tensor_parallel_workers:
            if args.disable_ray:
                result_df = profile_model_for_tp_no_ray(
                    args,
                    model,
                    num_tensor_parallel_workers,
                    num_tokens_to_profile,
                    pbar,
                )
            else:
                result_df = profile_model_for_tp(
                    args,
                    model,
                    num_tensor_parallel_workers,
                    num_tokens_to_profile,
                    pbar,
                )
            # Create directory structure: model/tp_workers/
            model_dir = f"{args.output_dir}/{model}/tp_{num_tensor_parallel_workers}"
            os.makedirs(model_dir, exist_ok=True)
            result_df.to_csv(f"{model_dir}/mlp.csv", index=False)


if __name__ == "__main__":
    main()
