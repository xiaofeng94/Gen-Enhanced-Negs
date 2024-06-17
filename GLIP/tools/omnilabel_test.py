import os
import argparse
import json
from tqdm import tqdm

import numpy as np
import cv2
import omnilabeltools as olt
from omnilabeltools import OmniLabel, OmniLabelEval

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo


def load_omnilabel_json(path_json: str, path_imgs: str):
    assert isinstance(path_json, str)

    ol = olt.OmniLabel(path_json)
    dataset_dicts = []
    for img_id in ol.image_ids:
        img_sample = ol.get_image_sample(img_id)
        dataset_dicts.append({
            "image_id": img_sample["id"],
            "file_name": os.path.join(path_imgs, img_sample["file_name"]),
            "inference_obj_descriptions": [od["text"] for od in img_sample["labelspace"]],
            "inference_obj_description_ids": [od["id"] for od in img_sample["labelspace"]],
        })

    return dataset_dicts


def create_tokens_positive(descript_list):
    cat_descript = ''
    tokens_positive = []

    for des_idx, descript in enumerate(descript_list):
        if descript[-1] == '.':
            descript = descript[:-1]    # remove '.'

        cur_token_pos = len(cat_descript)
        tokens_positive.append([[cur_token_pos, cur_token_pos + len(descript)]])
        cat_descript = cat_descript + descript

        # add seperator if not the last description
        if des_idx + 1 < len(descript_list):
            cat_descript = cat_descript + '. '

    return cat_descript, tokens_positive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="PyTorch Detection to Grounding Inference")
    parser.add_argument(
        "--config_file",
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

    parser.add_argument("--image_path", default=None, help="")
    parser.add_argument("--gt_json", default=None, help="")
    parser.add_argument("--chunk_size", default=20, type=int, help="number of descriptions each time")
    parser.add_argument("--threshold", default=0.05, type=float, help="")

    parser.add_argument("--save_root", default=None, help="folder to save result .json")
    parser.add_argument("--save_json", default=None, help="result .json file name")

    args = parser.parse_args()

    ###################################################################################
    path_imgs = args.image_path    # 'DATASET/omnilabel/imgs'
    gt_path_json = args.gt_json    # 'DATASET/omnilabel/2023-02-09-v0.1.1/dataset_all_clean_val_v0.1.1.json'

    # use_cat_descriptions = False
    num_descript_each_block = args.chunk_size   # 20
    score_thres = args.threshold    # 0.05

    result_save_root = args.save_root   # 'OUTPUT/omnilabel_eval'
    result_save_json = args.save_json   # 'glip_T_val_v0.1.1_results.json'
    
    # model config
    config_file = args.config_file  # 'configs/pretrain/glip_Swin_T_O365_GoldG.yaml'
    weight_file = args.weight   # 'MODEL/glip_tiny_model_o365_goldg_cc_sbu.pth'

    ###################################################################################

    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    if cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
        class_plus = 1
    else:
        class_plus = 0

    if not os.path.exists(result_save_root):
        os.makedirs(result_save_root)

    glip_demo = GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=score_thres,
        show_mask_heatmaps=False
    )

    dataset_dicts = load_omnilabel_json(gt_path_json, path_imgs)
    
    # Format of result_data:
    # [
    #     {
    #         image_id        ... the image id this predicted box belongs to
    #         bbox            ... the bounding box coordinates of the object (x,y,w,h)
    #         description_ids ... list of description IDs that refer to this object
    #         scores          ... list of confidences, one for each description
    #     },
    #     ...
    # ]
    result_data = []
 
    for iidx, data_dict in enumerate(tqdm(dataset_dicts)):
        # if iidx > 10:
        #     break
        # import ipdb
        # ipdb.set_trace()
        imgPath = data_dict["file_name"]
        image = cv2.imread(imgPath)

        image_id = data_dict["image_id"]

        obj_descriptions = data_dict["inference_obj_descriptions"]
        obj_description_ids = data_dict["inference_obj_description_ids"]

        des_id_start = 0
        while des_id_start < len(obj_description_ids):
            description_list = obj_descriptions[des_id_start:des_id_start+num_descript_each_block]
            description_id_list = obj_description_ids[des_id_start:des_id_start+num_descript_each_block]
            # update des_id_start, it may exceed len(obj_description_ids)
            des_id_start += num_descript_each_block

            cont_ids_2_descript_ids = {i:v for i, v in enumerate(description_id_list)}
            in_caption, in_tokens_positive = create_tokens_positive(description_list)

            predictions = glip_demo.inference(image, in_caption, custom_tokens_positive=in_tokens_positive)   # BoxList(), box mode: xyxy
            predictions = predictions.convert(mode="xywh")  # xyxy --> xywh
            pred_boxes = predictions.bbox
            pred_labels = predictions.get_field('labels') - class_plus   # continuous ids, starting from 0
            pred_scores = predictions.get_field('scores')

            # convert continuous id to description id
            for box_idx, box in enumerate(pred_boxes):
                result_data.append({
                    "image_id": image_id,
                    "bbox": box.cpu().tolist(),
                    "description_ids": [cont_ids_2_descript_ids[pred_labels[box_idx].item()]],
                    "scores": [pred_scores[box_idx].item()],
                })
        # else:
        #     for des_idx, descript in enumerate(obj_descriptions):
        #         description_list = [descript]
        #         description_id_list = [obj_description_ids[des_idx]]

        #         cont_ids_2_descript_ids = {i:v for i, v in enumerate(description_id_list)}
        #         in_caption, in_tokens_positive = create_tokens_positive(description_list)

        #         predictions = glip_demo.inference(image, in_caption, custom_tokens_positive=in_tokens_positive)   # BoxList(), box mode: xyxy
        #         predictions = predictions.convert(mode="xywh")  # xyxy --> xywh
        #         pred_boxes = predictions.bbox
        #         pred_labels = predictions.get_field('labels') - class_plus   # continuous ids, starting from 0
        #         pred_scores = predictions.get_field('scores')

        #         # convert continuous id to description id
        #         for box_idx, box in enumerate(pred_boxes):
        #             result_data.append({
        #                 "image_id": image_id,
        #                 "bbox": box.cpu().tolist(),
        #                 "description_ids": cont_ids_2_descript_ids[pred_labels[box_idx].item()],
        #                 "scores": pred_scores[box_idx],
        #             })

    # import ipdb
    # ipdb.set_trace()
    # write results json
    results_path = os.path.join(result_save_root, result_save_json)
    print('Saving to', results_path)
    json.dump(result_data, open(results_path, 'w'))

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