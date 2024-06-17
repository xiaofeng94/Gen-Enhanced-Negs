# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

import argparse
import os
import functools
import io
import datetime
import itertools
import json
from tqdm import tqdm

import numpy as np
import torch
import torch.distributed as dist

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.engine.inference import inference, create_positive_dict, clean_name
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, is_main_process, all_gather
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir
from maskrcnn_benchmark.utils.stats import get_model_complexity_info

from transformers import AutoTokenizer, CLIPTokenizerFast
import omnilabeltools as olt
from omnilabeltools import OmniLabel, OmniLabelEval


def init_distributed_mode(args):
    """Initialize distributed training, if appropriate"""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.gpu = int(os.environ["LOCAL_RANK"])
    elif "SLURM_PROCID" in os.environ:
        args.rank = int(os.environ["SLURM_PROCID"])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print("Not using distributed mode")
        args.distributed = False
        return

    # args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = "nccl"
    print("| distributed init (rank {}): {}".format(args.rank, args.dist_url), flush=True)

    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=args.rank,
        timeout=datetime.timedelta(0, 7200),
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def remove_full_stop(description_list):
    ret_list = []
    for descript in description_list:
        if descript[-1] == '.':
            descript = descript[:-1]    # remove '.'
        ret_list.append(descript)
    return ret_list

def num_of_words(text):
    return len(text.split(' '))

def create_queries_and_maps(labels, label_list, tokenizer, additional_labels=None, cfg=None):

    # Clean label list
    label_list = [clean_name(i) for i in label_list]
    # Form the query and get the mapping
    tokens_positive = []
    start_i = 0
    end_i = 0
    objects_query = ""

    # sep between tokens, follow training
    separation_tokens = cfg.DATASETS.SEPARATION_TOKENS

    caption_prompt = cfg.DATASETS.CAPTION_PROMPT
    use_caption_prompt = cfg.DATASETS.USE_CAPTION_PROMPT and caption_prompt is not None
    for _index, label in enumerate(label_list):
        if use_caption_prompt:
            objects_query += caption_prompt[_index]["prefix"]

        start_i = len(objects_query)

        if use_caption_prompt:
            objects_query += caption_prompt[_index]["name"]
        else:
            objects_query += label

        end_i = len(objects_query)
        tokens_positive.append([(start_i, end_i)])  # Every label has a [(start, end)]

        if use_caption_prompt:
            objects_query += caption_prompt[_index]["suffix"]

        if _index != len(label_list) - 1:
            objects_query += separation_tokens

    if additional_labels is not None:
        objects_query += separation_tokens
        for _index, label in enumerate(additional_labels):
            objects_query += label
            if _index != len(additional_labels) - 1:
                objects_query += separation_tokens

    # print(objects_query)

    if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
        tokenized = tokenizer(objects_query, return_tensors="pt")
    elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "roberta-base":
        tokenized = tokenizer(objects_query, return_tensors="pt")
    elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
        tokenized = tokenizer(
            objects_query, max_length=cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN, truncation=True, return_tensors="pt"
        )
    else:
        raise NotImplementedError

    # Create the mapping between tokenized sentence and the original label
    # if one_hot:
    #     positive_map_token_to_label, positive_map_label_to_token = create_one_hot_dict(labels, no_minus_one_for_one_hot=cfg.DATASETS.NO_MINUS_ONE_FOR_ONE_HOT)
    # else:
    positive_map_token_to_label, positive_map_label_to_token = create_positive_dict(
        tokenized, tokens_positive, labels=labels
    )  # from token position to original label
    return objects_query, positive_map_label_to_token

def main():
    parser = argparse.ArgumentParser(description="PyTorch Detection to Grounding Inference")
    parser.add_argument(
        "--config-file",
        default="configs/pretrain/glip_Swin_T_O365_GoldG.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument(
        "--weight",
        default=None,
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "opts", help="Modify config options using the command-line", default=None, nargs=argparse.REMAINDER
    )
    parser.add_argument("--world-size", default=1, type=int, help="number of distributed processes")
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")

    parser.add_argument("--task_config", default=None)
    parser.add_argument("--chunk_size", default=20, type=int, help="number of descriptions each time")
    parser.add_argument("--threshold", default=None, type=float, help="number of boxes stored in each run")
    parser.add_argument("--topk_per_eval", default=None, type=int, help="number of boxes stored in each run")

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    if distributed:
        # torch.cuda.set_device(args.local_rank)
        # torch.distributed.init_process_group(
        #     backend="nccl", init_method="env://"
        # )
        init_distributed_mode(args)
        print("Passed distributed init")

    cfg.local_rank = args.local_rank
    cfg.num_gpus = num_gpus

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    log_dir = cfg.OUTPUT_DIR
    if args.weight:
        log_dir = os.path.join(log_dir, "eval", os.path.splitext(os.path.basename(args.weight))[0])
    if log_dir:
        mkdir(log_dir)

    logger = setup_logger("maskrcnn_benchmark", log_dir, get_rank())
    logger.info(args)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(cfg)

    # logger.info("Collecting env info (might take some time)")
    # logger.info("\n" + collect_env_info())

    device = cfg.MODEL.DEVICE
    cpu_device = torch.device("cpu")

    model = build_detection_model(cfg)
    model.to(device)

    # we currently disable this
    # params, flops = get_model_complexity_info(model,
    #                                           (3, cfg.INPUT.MAX_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST),
    #                                           input_constructor=lambda x: {'images': [torch.rand(x).cuda()]})
    # print("FLOPs: {}, #Parameter: {}".format(params, flops))

    checkpointer = DetectronCheckpointer(cfg, model, save_dir=cfg.OUTPUT_DIR)
    if args.weight:
        _ = checkpointer.load(args.weight, force=True)
    else:
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

    # build tokenizer to process data
    # tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "roberta-base":
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            tokenizer = CLIPTokenizerFast.from_pretrained(
                "openai/clip-vit-base-patch32", from_slow=True, mask_token="ðŁĴĳ</w>"
            )
        else:
            tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32", from_slow=True)
    else:
        tokenizer = None
        raise NotImplementedError

    ### inference & evaluation
    topk_per_eval = args.topk_per_eval
    threshold = args.threshold

    model.eval()

    chunk_size = args.chunk_size    # num of texts each time
    if cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
        class_plus = 1
    else:
        class_plus = 0

    task_config = args.task_config
    assert task_config is not None, "task_config should be assigned"
    cfg_ = cfg.clone()
    cfg_.defrost()
    cfg_.merge_from_file(task_config)
    cfg_.merge_from_list(args.opts)

    dataset_name = cfg_.DATASETS.TEST[0]
    output_folder = os.path.join(log_dir, "inference", dataset_name)
    if not os.path.exists(output_folder):
        mkdir(output_folder)

    data_loaders_val = make_data_loader(cfg_, is_train=False, is_distributed=distributed)
    _iterator = tqdm(data_loaders_val[0])   # only for the first test set

    predictions = []
    for iidx, batch in enumerate(_iterator):
        images, targets, image_ids, *_ = batch

        # import ipdb
        # ipdb.set_trace()
        images = images.to(device)
        text_queries = targets[0].get_field('inference_obj_descriptions')
        text_queries_ids = targets[0].get_field("inference_obj_description_ids")
        image_size = targets[0].size
        image_id = image_ids[0]

        des_id_start = 0
        while des_id_start < len(text_queries_ids):
            # sinlge descriptions each time
            if num_of_words(text_queries[des_id_start]) > 2:
                description_list = remove_full_stop([text_queries[des_id_start]])
                description_id_list = [text_queries_ids[des_id_start]]
                des_id_start += 1
            else:
                description_list = remove_full_stop(text_queries[des_id_start:des_id_start+chunk_size])
                description_id_list = text_queries_ids[des_id_start:des_id_start+chunk_size]
                des_id_start += chunk_size
            
            # create postive map, always use continuous labels starting from 1
            continue_labels = np.arange(0, chunk_size) + class_plus
            cur_queries, positive_map_label_to_token = create_queries_and_maps(continue_labels, description_list, tokenizer, cfg=cfg)
            
            with torch.no_grad():
                output = model(images, captions=[cur_queries], positive_map=positive_map_label_to_token)
                output = output[0].to(cpu_device).convert(mode="xywh")
                output = output.resize(image_size)  # to the oringinal scale

            # import ipdb
            # ipdb.set_trace()
            # thresolding
            if threshold is not None:
                scores = output.get_field('scores')
                output = output[scores > threshold]
            # sorted by scores
            if topk_per_eval is not None:
                scores = output.get_field('scores')
                _, sortIndices = scores.sort(descending=True)
                output = output[sortIndices]
                # topk 
                output = output[:topk_per_eval]

            # map continuous id to description id
            cont_ids_2_descript_ids = {i:v for i, v in enumerate(description_id_list)}
            pred_boxes = output.bbox
            pred_labels = output.get_field('labels') - class_plus   # continuous ids, starting from 0
            pred_scores = output.get_field('scores')

            # convert continuous id to description id
            for box_idx, box in enumerate(pred_boxes):
                predictions.append({
                    "image_id": image_id,
                    "bbox": box.cpu().tolist(),
                    "description_ids": [cont_ids_2_descript_ids[pred_labels[box_idx].item()]],
                    "scores": [pred_scores[box_idx].item()],
                })

    # collect predictions from all GPUs
    synchronize()
    all_predictions = all_gather(predictions)
    all_predictions = list(itertools.chain(*all_predictions))
    if not is_main_process():
        return

    # import ipdb
    # ipdb.set_trace()
    # save results to .json
    result_save_json = "%s_results.json"%(dataset_name)
    results_path = os.path.join(output_folder, result_save_json)
    print('Saving to', results_path)
    json.dump(all_predictions, open(results_path, 'w'))

    from maskrcnn_benchmark.config.paths_catalog import DatasetCatalog
    datasetMeta = DatasetCatalog.get(dataset_name)
    gt_path_json = datasetMeta['args']['ann_file']
    # import ipdb
    # ipdb.set_trace()
    # evaluation
    gt = OmniLabel(gt_path_json)              # load ground truth dataset
    dt = gt.load_res(results_path)         # load prediction results
    ole = OmniLabelEval(gt, dt)
    # ole.params.resThrs = ...                    # set evaluation parameters as desired
    ole.evaluate()
    ole.accumulate()
    ole.summarize()

if __name__ == "__main__":
    main()
