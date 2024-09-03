import os

from PIL import Image
from tqdm import tqdm

from deeplab import DeeplabV3
from utils.utils_metrics import compute_mIoU, show_results, compute_Dice


if __name__ == "__main__":
    # ---------------------------------------------------------------------------#
    #   miou_mode=0 predict and compute mIoU
    #   miou_mode=1 predict only
    #   miou_mode=2 compute mIoU only
    # ---------------------------------------------------------------------------#
    miou_mode = 0

    num_classes = 19

    name_classes = ["unlabelled", "asphalt", "dirt", "mud", "water", "gravel", "other-terrain", "tree-trunk",
                    "tree-foliage", "bush", "fence", "structure", "pole", "vehicle", "rock", "log", "other-object",
                    "sky", "grass"]

    data_path = 'datasets'

    image_ids = open(os.path.join(data_path, "lists/test.txt"), 'r').read().splitlines()
    gt_dir = os.path.join(data_path, "labels/")
    miou_out_path = "miou_out"
    pred_dir = os.path.join(miou_out_path, 'detection-results')

    if miou_mode == 0 or miou_mode == 1:
        if not os.path.exists(pred_dir):
            os.makedirs(pred_dir)

        print("Load model.")
        deeplab = DeeplabV3()
        print("Load model done.")

        print("Get predict result.")
        for image_id in tqdm(image_ids):
            image_path_1 = os.path.join(data_path, "images/" + image_id + ".jpg")
            image_path_2 = os.path.join(data_path, "images/" + image_id + ".png")
            if os.path.exists(image_path_1):
                image_path = image_path_1
            elif os.path.exists(image_path_2):
                image_path = image_path_2
            else:
                raise ValueError(f"Unsupported image format or file not found for {image_id}")
            image = Image.open(image_path)
            image = deeplab.get_miou_png(image)
            image.save(os.path.join(pred_dir, image_id + ".png"))
        print("Get predict result done.")

    if miou_mode == 0 or miou_mode == 2:
        print("Get miou.")
        hist, IoUs, PA_Recall, Precision = compute_mIoU(gt_dir, pred_dir, image_ids, num_classes,
                                                        name_classes)
        dices, mean_dice = compute_Dice(gt_dir, pred_dir, image_ids, num_classes, name_classes)
        print("Get miou done.")
        show_results(miou_out_path, hist, IoUs, PA_Recall, Precision, name_classes, dices)
