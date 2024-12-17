from multiprocessing import Pool
from pathlib import Path

import hydra
import seaborn as sns
import torch.multiprocessing
from omegaconf import DictConfig

from crf import dense_crf
from data import *
from modules import *
from train_segmentation_eigen import LitUnsupervisedSegmenter, get_class_labels

torch.multiprocessing.set_sharing_strategy('file_system')


def plot_cm(histogram, label_cmap, cfg):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    hist = histogram.detach().cpu().to(torch.float32)
    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues", cbar=False)
    ax.set_title('Predicted labels', fontsize=28)
    ax.set_ylabel('True labels', fontsize=28)
    names = get_class_labels(cfg.dataset_name)
    if cfg.extra_clusters:
        names = names + ["Extra"]
    ax.set_xticks(np.arange(0, len(names)) + .5)
    ax.set_yticks(np.arange(0, len(names)) + .5)
    ax.xaxis.tick_top()
    ax.xaxis.set_ticklabels(names, fontsize=18)
    ax.yaxis.set_ticklabels(names, fontsize=18)
    colors = [label_cmap[i] / 255.0 for i in range(len(names))]
    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
    plt.tight_layout()


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])


def batched_crf(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)


@hydra.main(config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = join("src_EAGLE", cfg.pytorch_data_dir)

    for model_path in cfg.model_paths:
        print(str(model_path))
        # path_ = str(model_path)

        """
        Since LitUnsupervisedSegmenter extend pl.LightningModule, we can call load_from_checkpoint(...)
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#id3
        """
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        loader_crop = "center"

        test_dataset = ContrastiveSegDataset(
            pytorch_data_dir=pytorch_data_dir,
            dataset_name=model.cfg.dataset_name,  # cocostuff, cityscape etc
            crop_type=None,
            image_set="val",
            transform=get_transform(cfg.res, False, loader_crop),
            target_transform=get_transform(cfg.res, True, loader_crop),
            mask=True,
            cfg=model.cfg,
        )

        test_loader = DataLoader(test_dataset, cfg.batch_size * 2,
                                 shuffle=False, num_workers=cfg.num_workers,
                                 pin_memory=True, collate_fn=flexible_collate)

        model.eval().cuda()

        if cfg.use_ddp:
            par_model = torch.nn.DataParallel(model.net)
        else:
            par_model = model.net

        """
        Note: 27 classes of COCO
        """

        # saved_data = defaultdict(list)
        with (Pool(cfg.num_workers + 5) as pool):
            for i, batch in enumerate(tqdm(test_loader)):

                with torch.no_grad():
                    img = batch["img"].cuda()
                    label = batch["label"].cuda()

                    """
                    Feature extraction phase, in particular we are interested in code_kk, code2_kk which are the projection
                    linear / non linear of standard image-features and qkv image-features.
                    Look at vision_transformer.py - [get_intermediate_feat] for explanation of feature extraction 
                    """
                    _, _, _, code_kk = par_model(img)  # feats, feats_kk, code1, code_kk
                    _, _, _, code2_kk = par_model(img.flip(dims=[3]))  # feats, feats2_kk, code2, code2_kk

                    # [12, 512, 40, 40] -> [12, 512, 40, 40] | mean of code_kk and code2_kk
                    code = (code_kk + code2_kk.flip(dims=[3])) / 2

                    # [12, 512, 40, 40] -> [12, 512, 320, 320] | (320, 320) is the shape of the labels
                    code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

                    # [12, 512, 320, 320] -> [12, 27, 320, 320] | src_EAGLE/train_segmentation_eigen.py:93
                    linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)

                    # [12, 512, 320, 320] -> [12, 32, 320, 320] | src_EAGLE.modules.ClusterLookup
                    _, cluster_probs = model.cluster_probe(code, 4, log_probs=True)

                    if cfg.run_crf:  # last step before results computation
                        # [12, 27, 320, 320] -> [12, 320, 320] argmax along the first axis to get the class
                        linear_preds = batched_crf(pool, img, linear_probs).argmax(1).cuda()
                        # [12, 32, 320, 320] -> [12, 320, 320] argmax along the first axis to get the class
                        cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()
                    else:
                        linear_preds = linear_probs.argmax(1)
                        cluster_preds = cluster_probs.argmax(1)

                    if cfg.export_predictions_and_labels:
                        for idx in range(label.shape[0]):

                            src_img = unnorm(img[idx, ...])
                            src_img = src_img.permute(1, 2, 0)
                            src_img = src_img.cpu().numpy()

                            data = label[idx, ...].cpu().to(torch.uint8).numpy()
                            label_img = label_to_color_image(data, num_classes=test_dataset.n_classes).astype(np.uint8)

                            data = linear_preds[idx, ...].cpu().to(torch.uint8).numpy()
                            linear_preds_img = label_to_color_image(data, num_classes=test_dataset.n_classes).astype(np.uint8)

                            data = cluster_preds[idx, ...].cpu().to(torch.uint8).numpy()
                            cluster_preds_img = label_to_color_image(data, num_classes=test_dataset.n_classes).astype(np.uint8)

                            """ Save the LABEL-LINEARPROBS-CLUSTERPROBS plot """
                            fig, axis = plt.subplots(1, 4)
                            fig.set_size_inches(10, 5)
                            axis[0].set_title("Source Image")
                            axis[0].imshow(src_img)
                            axis[1].set_title("Label Image")
                            axis[1].imshow(label_img)
                            axis[2].set_title("Linear preds")
                            axis[2].imshow(linear_preds_img)
                            axis[3].set_title("Cluster preds")
                            axis[3].imshow(cluster_preds_img)

                            output_path = Path(cfg.export_predictions_and_labels_path) / f'{idx:02d}.png'

                            plt.savefig(output_path)
                            plt.close()

                    model.test_linear_metrics.update(linear_preds, label)
                    model.test_cluster_metrics.update(cluster_preds, label)

        tb_metrics = {
            **model.test_linear_metrics.compute(training=False),
            **model.test_cluster_metrics.compute(training=False),
        }

        tb_metrics['assignments'] = tb_metrics['assignments'][-27:]

        print("")
        print(model_path)
        print(tb_metrics)


if __name__ == "__main__":
    prep_args()
    my_app()
