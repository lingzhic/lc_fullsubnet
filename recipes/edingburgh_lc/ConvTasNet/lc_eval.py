import os
import random
import soundfile as sf
import torch
import yaml
import json
import argparse
import pandas as pd
from tqdm import tqdm
from pprint import pprint
from pathlib import Path
import sys

sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "..", "..")))  # without installation, add /path/to/Audio-ZEN
from from_asteroid.metrics import get_metrics
from audio_zen.dataset.edinburgh_dataset import Edingburgh
from from_asteroid.losses import PITLossWrapper, pairwise_neg_sisdr
from from_asteroid.models import ConvTasNet
from from_asteroid.models import save_publishable
from from_asteroid.utils import tensors_to_device
from from_asteroid.dsp.normalization import normalize_estimates
from from_asteroid.metrics import WERTracker, MockWERTracker


parser = argparse.ArgumentParser()
parser.add_argument("--test_path", type=str, required=True, help="Test directory including the csv files")
parser.add_argument("--out_dir",   type=str, required=True, help="Directory in exp_dir where the eval results" " will be stored")
parser.add_argument("--use_gpu",   type=int, default=0, help="Whether to use the GPU for model execution")
parser.add_argument("--exp_dir",   default="exp/tmp", help="Experiment root")
parser.add_argument("--n_save_ex", type=int, default=10, help="Number of audio examples to save, -1 means all")

COMPUTE_METRICS = ["si_sdr", "sdr", "sir", "sar", "stoi"]


def main(conf):
    compute_metrics = COMPUTE_METRICS
    # anno_df = pd.read_csv(Path(conf["test_path"]).parent.parent.parent / "test_annotations.csv")
    wer_tracker = (MockWERTracker())
    model_path = os.path.join(conf["exp_dir"], "best_model.pth")
    model = ConvTasNet.from_pretrained(model_path)
    # Handle device placement
    if conf["use_gpu"]:
        model.cuda()
    model_device = next(model.parameters()).device
    test_set = Edingburgh(
        csv_path=conf["test_path"],
        target_sr=conf["target_sr"],
        segment=None,
        return_id=True,
    )  # Uses all segment length
    # Used to reorder sources only
    loss_func = PITLossWrapper(pairwise_neg_sisdr, pit_from="pw_mtx")

    # Randomly choose the indexes of sentences to save.
    eval_save_dir = os.path.join(conf["exp_dir"], conf["out_dir"])
    ex_save_dir = os.path.join(eval_save_dir, "examples/")
    if conf["n_save_ex"] == -1:
        conf["n_save_ex"] = len(test_set)
    save_idx = random.sample(range(len(test_set)), conf["n_save_ex"])
    series_list = []
    torch.no_grad().__enter__()
    for idx in tqdm(range(len(test_set))):
        # Forward the network on the mixture.
        mix, sources, ids = test_set[idx]
        mix, sources = tensors_to_device([mix, sources], device=model_device)
        est_sources = model(mix.unsqueeze(0))       # NOTE Do the inferencing job
        loss, reordered_sources = loss_func(est_sources, sources[None], return_est=True)
        mix_np = mix.cpu().data.numpy()
        sources_np = sources.cpu().data.numpy()
        est_sources_np = reordered_sources.squeeze(0).cpu().data.numpy()
        # For each utterance, we get a dictionary with the mixture path,
        # the input and output metrics
        utt_metrics = get_metrics(
            mix_np,
            sources_np,
            est_sources_np,
            sample_rate=conf["target_sr"],
            metrics_list=COMPUTE_METRICS,
        )
        utt_metrics["mix_path"] = test_set.noisy_path
        est_sources_np_normalized = normalize_estimates(est_sources_np, mix_np)
        utt_metrics.update(
            **wer_tracker(
                mix=mix_np,
                clean=sources_np,
                estimate=est_sources_np_normalized,
                wav_id=ids,
                sample_rate=conf["target_sr"],
            )
        )
        series_list.append(pd.Series(utt_metrics))              # This is the eval score list

        # Save some examples in a folder. Wav files and metrics as text.
        if idx in save_idx:
            local_save_dir = os.path.join(ex_save_dir, "ex_{}/".format(idx))
            os.makedirs(local_save_dir, exist_ok=True)
            sf.write(local_save_dir + "mixture.wav", mix_np, conf["target_sr"])
            # Loop over the sources and estimates
            for src_idx, src in enumerate(sources_np):
                sf.write(local_save_dir + "s{}.wav".format(src_idx), src, conf["target_sr"])
            for src_idx, est_src in enumerate(est_sources_np_normalized):
                sf.write(
                    local_save_dir + "s{}_estimate.wav".format(src_idx),
                    est_src,
                    conf["target_sr"],
                )
            # Write local metrics to the example folder.
            with open(local_save_dir + "metrics.json", "w") as f:
                json.dump(utt_metrics, f, indent=0)

    # Save all metrics to the experiment folder.
    all_metrics_df = pd.DataFrame(series_list)
    all_metrics_df.to_csv(os.path.join(eval_save_dir, "all_metrics.csv"))

    # Print and save summary metrics
    final_results = {}
    for metric_name in compute_metrics:
        input_metric_name = "input_" + metric_name
        ldf = all_metrics_df[metric_name] - all_metrics_df[input_metric_name]
        final_results[metric_name] = all_metrics_df[metric_name].mean()
        final_results[metric_name + "_imp"] = ldf.mean()

    print("Overall metrics :")
    pprint(final_results)

    with open(os.path.join(eval_save_dir, "final_metrics.json"), "w") as f:
        json.dump(final_results, f, indent=0)

    model_dict = torch.load(model_path, map_location="cpu")


if __name__ == "__main__":
    args = parser.parse_args()
    arg_dic = dict(vars(args))
    # Load training config
    conf_path = os.path.join(args.exp_dir, "conf.yml")
    with open(conf_path) as f:
        train_conf = yaml.safe_load(f)
    arg_dic["target_sr"] = 16000
    arg_dic["train_conf"] = None

    pprint(arg_dic)
    main(arg_dic)
