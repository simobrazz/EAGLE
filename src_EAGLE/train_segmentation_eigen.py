import sys
import warnings
from datetime import datetime
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import pytorch_lightning as pl
import pytz
import torch.multiprocessing
import torch.optim as optim
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from data import *
from eigen_modules import *
from modules import *

warnings.filterwarnings(action='ignore')
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["WANDB__SERVICE_WAIT"] = "300"


def get_class_labels(dataset_name):
    if dataset_name.startswith("cityscapes"):
        return [
            'road', 'sidewalk', 'parking', 'rail track', 'building',
            'wall', 'fence', 'guard rail', 'bridge', 'tunnel',
            'pole', 'polegroup', 'traffic light', 'traffic sign', 'vegetation',
            'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'caravan', 'trailer', 'train',
            'motorcycle', 'bicycle']
    elif dataset_name == "cocostuff27":
        return [
            "electronic", "appliance", "food", "furniture", "indoor",
            "kitchen", "accessory", "animal", "outdoor", "person",
            "sports", "vehicle", "ceiling", "floor", "food",
            "furniture", "rawmaterial", "textile", "wall", "window",
            "building", "ground", "plant", "sky", "solid",
            "structural", "water"]
    elif dataset_name == "voc":
        return [
            'background',
            'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
            'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
    elif dataset_name == "potsdam":
        return [
            'roads and cars',
            'buildings and clutter',
            'trees and vegetation']
    else:
        raise ValueError("Unknown Dataset {}".format(dataset_name))


def scheduler(cfg, step):
    if step > cfg.step_schedulers:
        return 1
    else:
        return 0


def get_current_date():
    tz = pytz.timezone('Europe/Rome')
    return datetime.now(tz).strftime('%b%d_%H-%M-%S')


class LitUnsupervisedSegmenter(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        # data_dir = join(cfg.output_root, "data")

        if cfg.arch == "dino":
            self.net = DinoFeaturizer(dim, cfg)
        else:
            raise ValueError("Unknown arch {}".format(cfg.arch))

        self.train_cluster_probe_eigen = ClusterLookup(cfg.eigen_cluster - 1, cfg.eigen_cluster_out)
        self.train_cluster_probe_eigen_aug = ClusterLookup(cfg.eigen_cluster - 1, cfg.eigen_cluster_out)

        self.train_cluster_probe = ClusterLookup(dim, n_classes)
        self.cluster_probe = ClusterLookup(dim, n_classes + cfg.extra_clusters)
        self.linear_probe = nn.Conv2d(dim, n_classes, (1, 1))
        self.decoder = nn.Conv2d(dim, self.net.n_feats, (1, 1))

        if self.cfg.use_head:
            self.project_head = nn.Linear(cfg.dim, cfg.dim)

        self.cluster_metrics = UnsupervisedMetrics("test/cluster/", n_classes, cfg.extra_clusters, True)
        self.linear_metrics = UnsupervisedMetrics("test/linear/", n_classes, 0, False)

        self.test_cluster_metrics = UnsupervisedMetrics("final/cluster/", n_classes, cfg.extra_clusters, True)
        self.test_linear_metrics = UnsupervisedMetrics("final/linear/", n_classes, 0, False)

        self.CELoss = newLocalGlobalInfoNCE(cfg, n_classes)
        self.eigen_loss_fn = EigenLoss(cfg)
        # self.eigen_new_loss_fn = new_EigenLoss(cfg)
        self.linear_probe_loss_fn = torch.nn.CrossEntropyLoss()

        self.contrastive_corr_loss_fn = CorrespondenceLoss(cfg)
        for p in self.contrastive_corr_loss_fn.parameters():
            p.requires_grad = False

        self.automatic_optimization = False

        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

        self.val_steps = 0

        self.validation_step_outputs = []

        self.update_prams = 0.0
        self.save_hyperparameters()

    def forward(self, x):
        # in lightning, forward defines the prediction/inference actions
        return self.net(x)[1]

    def __show_img(self, img):
        if not img.is_cpu:
            img = img.to("cpu")
        plt.imshow(img)
        plt.show()

    def __logs_config(self):
        standard = dict(sync_dist=False, rank_zero_only=True)
        extended = dict(sync_dist=False, rank_zero_only=True, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return standard, extended

    def __z_projection(self, code_pos_z, code_pos_aug_z):
        # projection [25088, 512] -> [25088, 512]
        code_pos_z = self.project_head(code_pos_z)
        # projection [25088, 512] -> [25088, 512]
        code_pos_aug_z = self.project_head(code_pos_aug_z)

        # normalization along rows (dim=1) that preserves the dimensions
        # [25088, 512] -> [25088, 512]
        code_pos_z = F.normalize(code_pos_z, dim=1)
        # [25088, 512] -> [25088, 512]
        code_pos_aug_z = F.normalize(code_pos_aug_z, dim=1)

        return code_pos_z, code_pos_aug_z

    def __k_sim(self, feats_pos_kk, feats_pos):
        # [32, 1152, 28, 28] -> [32, 1152, 784]
        feats_pos_reshaped = feats_pos_kk.view(feats_pos.shape[0], feats_pos.shape[1], -1)
        # [32, 784, 1152] * [32, 1152, 784]
        corr_feats_pos = torch.matmul(feats_pos_reshaped.transpose(2, 1), feats_pos_reshaped)
        # [32, 784, 784] -> [32, 784, 784]
        corr_feats_pos = F.normalize(corr_feats_pos, dim=1)

        return corr_feats_pos

    def __k_sim_aug(self, feats_pos_aug_kk, feats_pos_aug):
        # [32, 1152, 28, 28] -> [32, 1152, 784]
        feats_pos_aug_reshaped = feats_pos_aug_kk.view(feats_pos_aug.shape[0], feats_pos_aug.shape[1], -1)
        # [32, 784, 1152] * [32, 1152, 784]
        corr_feats_pos_aug = torch.matmul(feats_pos_aug_reshaped.transpose(2, 1), feats_pos_aug_reshaped)
        # [32, 784, 784] -> [32, 784, 784]
        corr_feats_pos_aug = F.normalize(corr_feats_pos_aug, dim=1)

        return corr_feats_pos_aug

    def __draw_meicue(self, argmax_clases):
        pass  # TODO ...

    def training_step(self, batch, batch_idx):

        if self.cfg.centroid_mode == 'learned' or self.cfg.centroid_mode == 'prototype':
            net_optim, linear_probe_optim, cluster_probe_optim, project_head_optim, centroid_optim, cluster_eigen_optim, cluster_eigen_optim_aug = self.optimizers()
        else:
            net_optim, linear_probe_optim, cluster_probe_optim, project_head_optim, cluster_eigen_optim, cluster_eigen_optim_aug = self.optimizers()

        # zeroed the gradients
        net_optim.zero_grad()
        linear_probe_optim.zero_grad()
        cluster_probe_optim.zero_grad()
        project_head_optim.zero_grad()
        cluster_eigen_optim.zero_grad()
        cluster_eigen_optim_aug.zero_grad()

        scheduler_cluster = optim.lr_scheduler.StepLR(cluster_probe_optim, step_size=50, gamma=0.1)
        if self.cfg.centroid_mode == 'learned' or self.cfg.centroid_mode == 'prototype':
            centroid_optim.zero_grad()

        with torch.no_grad():
            img = batch["img"]
            img_pos = batch["img_pos"]
            label = batch["label"]
            img_pos_aug = batch["img_pos_aug"]

            """ UNUSED STUFF
            ind = batch["ind"]
            label_pos = batch["label_pos"]
            """

        # self.__show_img(prep_for_plot(img[0]))
        # self.__show_img(prep_for_plot(img_pos[0]))
        # self.__show_img(prep_for_plot(img_pos_aug[0]))

        # [32, 3, 224, 224] -> [32, 1152, 28, 28] (feats, feats_kk) or [32, 512, 28, 28] (code, code_kk)
        feats, feats_kk, code, code_kk = self.net(img)
        # [32, 3, 224, 224] -> [32, 1152, 28, 28] (feats, feats_kk) or [32, 512, 28, 28] (code, code_kk)
        feats_pos, feats_pos_kk, code_pos, code_pos_kk = self.net(img_pos)
        # [32, 3, 224, 224] -> [32, 1152, 28, 28] (feats, feats_kk) or [32, 512, 28, 28] (code, code_kk)
        feats_pos_aug, feats_pos_aug_kk, code_pos_aug, code_pos_aug_kk = self.net(img_pos_aug)

        # log declaration
        log_args_standard, log_args_extended = self.__logs_config()

        # [32, 512, 28, 28] -> [32, 28, 28, 512] -> [25088, 512]
        code_pos_z = code_pos_kk.permute(0, 2, 3, 1).reshape(-1, self.cfg.dim)
        # [32, 512, 28, 28] -> [32, 28, 28, 512] -> [25088, 512]
        code_pos_aug_z = code_pos_aug_kk.permute(0, 2, 3, 1).reshape(-1, self.cfg.dim)

        """
        Here we use the projection head to create the feature vector Z and ~Z
        """
        if self.cfg.use_head:
            # [25088, 512] -> [25088, 512]
            code_pos_z, code_pos_aug_z = self.__z_projection(code_pos_z, code_pos_aug_z)

        # [32, 1152, 28, 28] ->  [32, 784, 784]
        corr_feats_pos = self.__k_sim(feats_pos_kk, feats_pos)
        # [32, 1152, 28, 28] -> [32, 784, 784]
        corr_feats_pos_aug = self.__k_sim_aug(feats_pos_aug_kk, feats_pos_aug)

        loss = 0

        if self.cfg.neg_samples == 0:
            (
                pos_inter_loss, pos_inter_cd, feat_neg_torch,
                feat_neg_torch, code_neg_torch

            ) = self.contrastive_corr_loss_fn(
                feats, feats_pos,
                code, code_pos
            )

            pos_inter_loss = pos_inter_loss.mean()
            self.log('loss/pos_inter', pos_inter_loss, **log_args_standard)
            self.log('cd/pos_inter', pos_inter_cd.mean(), **log_args_standard)
            loss += (self.cfg.pos_inter_weight * pos_inter_loss) * self.cfg.correspondence_weight

        elif self.cfg.neg_samples > 0:
            update_params = scheduler(self.cfg, self.global_step)
            update_params = min(update_params, self.cfg.momentum_limit)

            (
                pos_inter_loss, pos_inter_cd,
                neg_inter_loss, neg_inter_cd
            ) = self.contrastive_corr_loss_fn(
                feats_kk, feats_pos_kk, feats_pos_aug_kk,
                code_kk, code_pos_kk, code_pos_aug_kk
            )
            neg_inter_loss = neg_inter_loss.mean()
            pos_inter_loss = pos_inter_loss.mean()

            # 2. EIGENLOSS

            # **2.1 pos

            # [32, 1152, 28, 28] -> [32, 784, 1152]
            feats_pos_reshaped = feats_pos_kk.reshape(feats_pos.shape[0], feats_pos.shape[1], -1).permute(0, 2, 1)
            # [32, 512, 28, 28] -> [32, 784, 512]
            code_pos_reshaped = code_pos_kk.reshape(code_pos.shape[0], code_pos.shape[1], -1).permute(0, 2, 1)

            # img.shape = [32, 3, 224, 224]
            # corr_feats_pos.shape = [32, 784, 784]
            # -> [32, 784, 4]
            eigenvectors = self.eigen_loss_fn(img, feats_pos_reshaped, code_pos_reshaped,
                                              corr_feats_pos, None, neg_sample=5)

            # we skip the first eigenvector in the batch, it represents a constant vector with eigenvalue 0 (always?)
            # if the graph is connected the molteplicity of the zeroed eigenvalues is 1,
            # otherwise equal to the number of connected subgraphs
            # [32, 784, 4] -> [32, 784, 3]
            eigenvectors = eigenvectors[:, :, 1:]
            # [32, 784, 3] -> [32, 28, 28, 3]
            eigenvectors = eigenvectors.reshape(eigenvectors.shape[0], feats_pos.shape[-1], feats_pos.shape[-1], -1)
            # [32, 28, 28, 3] -> [32, 3, 28, 28]
            eigenvectors = eigenvectors.permute(0, 3, 1, 2)

            """ Meicue Calculation """
            cluster_eigen_loss, cluster_eigen_probs = self.train_cluster_probe_eigen(eigenvectors,
                                                                                     self.cfg.eigen_cluster_alpha,
                                                                                     log_probs=True,
                                                                                     verbosity=True)
            # [32, 32, 28, 28] -> [32, 28, 28] for each pixel position (28x28) get the cluster with higher confidence!
            # NOTE: this implementation is different from that in the paper
            cluster_eigen_probs = cluster_eigen_probs.argmax(1)  # this give us Meicue as argmax per classes
            # self.__draw_meicue(cluster_eigen_probs)
            """ ------------------ """

            # **2.2 pos_aug

            # [32, 1152, 28, 28] -> [32, 1152, 784] (reshape) -> [32, 784, 1152] (permutation)
            feats_pos_aug_re = feats_pos_aug_kk.reshape(feats_pos_aug.shape[0], feats_pos_aug.shape[1], -1)
            # [32, 1152, 784] -> [32, 784, 1152]
            feats_pos_aug_re = feats_pos_aug_re.permute(0, 2, 1)

            # [32, 512, 28, 28] -> [32, 512, 784] (reshape) -> [32, 784, 512]
            code_pos_aug_re = code_pos_aug_kk.reshape(code_pos_aug.shape[0], code_pos_aug.shape[1], -1)
            # [32, 512, 784] -> [32, 784, 512]
            code_pos_aug_re = code_pos_aug_re.permute(0, 2, 1)

            # [32, 784, 4]
            eigenvectors_aug = self.eigen_loss_fn(img, feats_pos_aug_re, code_pos_aug_re, corr_feats_pos, None,
                                                  neg_sample=5)
            # [32, 784, 4] -> [32, 784, 3]
            eigenvectors_aug = eigenvectors_aug[:, :, 1:]
            # [32, 784, 3] -> [32, 28, 28, 3]
            eigenvectors_aug = eigenvectors_aug.reshape(eigenvectors_aug.shape[0], feats_pos.shape[-1],
                                                        feats_pos.shape[-1], -1)
            # [32, 28, 28, 3] -> [32, 3, 28, 28]
            eigenvectors_aug = eigenvectors_aug.permute(0, 3, 1, 2)

            """ Meicue Calculation """
            cluster_eigen_aug_loss, cluster_eigen_aug_probs = self.train_cluster_probe_eigen_aug(eigenvectors_aug,
                                                                                                 self.cfg.eigen_cluster_alpha,
                                                                                                 log_probs=True)
            # [32, 32, 28, 28] -> [32, 28, 28] for each pixel position (28x28) get the cluster with higher confidence!
            cluster_eigen_aug_probs = cluster_eigen_aug_probs.argmax(1)  # this give us Meicue as argmax per classes
            """ ------------------ """

            # end 2.2

            """ NCE Loss computation """
            local_pos_loss = self.CELoss(code_pos_z, code_pos_aug_z, cluster_eigen_probs, corr_feats_pos)
            local_pos_aug_loss = self.CELoss(code_pos_aug_z, code_pos_z, cluster_eigen_aug_probs, corr_feats_pos_aug)
            """ -------------------- """

            self.log('loss/pos_inter', pos_inter_loss, **log_args_standard)
            self.log('loss/neg_inter', neg_inter_loss, **log_args_standard)
            self.log('loss/local_pos_loss', local_pos_loss, **log_args_standard)
            self.log('loss/local_pos_aug_loss', local_pos_aug_loss, **log_args_standard)
            self.log('loss/cluster_eigen_loss', cluster_eigen_loss, **log_args_standard)

            self.log('cd/pos_inter', pos_inter_cd.mean(), **log_args_standard)
            self.log('cd/neg_inter', neg_inter_cd.mean(), **log_args_standard)

            # Eigen Loss (mean of normal and augmented)
            loss += (cluster_eigen_aug_loss + cluster_eigen_loss) / 2

            # Correspondence Loss
            loss += (self.cfg.pos_inter_weight * pos_inter_loss + self.cfg.neg_inter_weight * neg_inter_loss) * self.cfg.correspondence_weight * (1.0 - update_params)

            # NCE Loss
            loss += (self.cfg.local_pos_weight * local_pos_loss + self.cfg.local_pos_aug_weight * local_pos_aug_loss) * update_params

            self.log('cd/update_params', update_params, **log_args_standard)

        # [32, 224, 224] -> [32*224*224]
        flat_label = label.reshape(-1)
        mask = (flat_label >= 0) & (flat_label < self.n_classes)  # we can have values of -1

        # detached_code = torch.clone(code.detach())
        # [32, 512, 28, 28] -> [32, 512, 28, 28] Tensor.detach() disconnect the tensor from the current graph
        detached_code_kk = torch.clone(code_kk.detach())
        # [32, 512, 28, 28] -> [32, 27, 28, 28]
        linear_logits = self.linear_probe(detached_code_kk)
        # [32, 27, 28, 28] -> [32, 27, 224, 224]
        linear_logits = F.interpolate(linear_logits, label.shape[-2:], mode='bilinear', align_corners=False)
        # [32, 27, 224, 224] -> [32, 224, 224, 27] -> [32*224*224, 27]
        linear_logits = linear_logits.permute(0, 2, 3, 1).reshape(-1, self.n_classes)
        # Apply a CrossEntropy loss (apparently the mean is not used)
        linear_loss = self.linear_probe_loss_fn(linear_logits[mask], flat_label[mask]).mean()

        cluster_loss, cluster_probs = self.cluster_probe(detached_code_kk, None)

        loss += linear_loss
        loss += cluster_loss

        self.log('loss/linear', linear_loss, **log_args_extended)
        self.log('loss/cluster', cluster_loss, **log_args_extended)
        self.log('loss/total', loss, **log_args_extended)

        """ All the optimizer steps and loss backward """

        scheduler_cluster.step()  # not clear why this step is called here before manual_backward()

        # mandatory when multiple optimizers are in place
        # https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        self.manual_backward(loss)

        net_optim.step()
        cluster_probe_optim.step()
        linear_probe_optim.step()
        cluster_eigen_optim.step()
        cluster_eigen_optim_aug.step()

        if self.cfg.use_head:
            project_head_optim.step()

        if self.cfg.centroid_mode == 'learned' or self.cfg.centroid_mode == 'prototype':
            centroid_optim.step()

        return loss

    def on_train_start(self):
        tb_metrics = {
            **self.linear_metrics.compute(training=True),
            **self.cluster_metrics.compute(training=True)
        }
        self.log_dict(tb_metrics)
        self.logger.log_hyperparams(self.cfg)

    def validation_step(self, batch, batch_idx):
        img = batch["img"]
        label = batch["label"]
        self.net.eval()

        with torch.no_grad():
            feats, feats_kk, code, code_kk = self.net(img)
            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)
            code_kk = F.interpolate(code_kk, label.shape[-2:], mode='bilinear', align_corners=False)

            linear_preds = self.linear_probe(code_kk)
            linear_preds = linear_preds.argmax(1)
            self.linear_metrics.update(linear_preds, label)

            cluster_loss, cluster_preds = self.cluster_probe(code_kk, None)
            cluster_preds = cluster_preds.argmax(1)
            self.cluster_metrics.update(cluster_preds, label)

            output = {
                'img': img[:self.cfg.n_images].detach().cpu(),
                'linear_preds': linear_preds[:self.cfg.n_images].detach().cpu(),
                "cluster_preds": cluster_preds[:self.cfg.n_images].detach().cpu(),
                "label": label[:self.cfg.n_images].detach().cpu()
            }

            self.validation_step_outputs.append(output)

            return output

    def __on_validation_epoch_end_save_fig(self, outputs):
        """
        In order to make this plot works it must be output.shape[0] == self.cfg.n_images.
        output.shape[0] is the validation batch size `src_EAGLE/train_segmentation_eigen.py:538`
        """
        if self.cfg.val_sample == -1:
            random_sample = random.randint(0, len(outputs) - 1)
        else:
            random_sample = self.cfg.val_sample

        output = {k: v.detach().cpu() for k, v in outputs[random_sample].items()}

        fig, ax = plt.subplots(4, self.cfg.n_images, figsize=(self.cfg.n_images * 3, 4 * 3))
        for i in range(self.cfg.n_images):
            ax[0, i].imshow(prep_for_plot(output["img"][i]))
            ax[1, i].imshow(self.label_cmap[output["label"][i]])
            ax[2, i].imshow(self.label_cmap[output["linear_preds"][i]])
            ax[3, i].imshow(self.label_cmap[self.cluster_metrics.map_clusters(output["cluster_preds"][i])])
        ax[0, 0].set_ylabel("Image", fontsize=16)
        ax[1, 0].set_ylabel("Label", fontsize=16)
        ax[2, 0].set_ylabel("Linear Probe", fontsize=16)
        ax[3, 0].set_ylabel("Cluster Probe", fontsize=16)

        figure_path = Path(self.cfg.validation_figure_path)
        filename = f"{get_current_date()}.png"
        figure_path = figure_path / filename

        plt.savefig(figure_path)
        # plt.show()  otherwise the main UI process is blocked
        plt.close()

    def on_validation_epoch_end(self) -> None:
        super().on_validation_epoch_end()
        # outputs = torch.stack(self.validation_step_outputs)
        outputs = self.validation_step_outputs
        with torch.no_grad():
            tb_metrics = {
                **self.linear_metrics.compute(training=True),
                **self.cluster_metrics.compute(training=True),
            }

            if self.trainer.is_global_zero and not self.cfg.submitting_to_aml:

                self.__on_validation_epoch_end_save_fig(outputs)

                # add_plot(self.logger.experiment.log, "plot_labels", self.global_step)

                if self.cfg.has_labels:
                    fig = plt.figure(figsize=(13, 10))
                    ax = fig.gca()
                    hist = self.cluster_metrics.histogram.detach().cpu().to(torch.float32)
                    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
                    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues")
                    ax.set_xlabel('Predicted labels')
                    ax.set_ylabel('True labels')
                    names = get_class_labels(self.cfg.dataset_name)
                    if self.cfg.extra_clusters:
                        names = names + ["Extra"]
                    ax.set_xticks(np.arange(0, len(names)) + .5)
                    ax.set_yticks(np.arange(0, len(names)) + .5)
                    ax.xaxis.tick_top()
                    ax.xaxis.set_ticklabels(names, fontsize=14)
                    ax.yaxis.set_ticklabels(names, fontsize=14)
                    colors = [self.label_cmap[i] / 255.0 for i in range(len(names))]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
                    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
                    # ax.yaxis.get_ticklabels()[-1].set_color(self.label_cmap[0] / 255.0)
                    # ax.xaxis.get_ticklabels()[-1].set_color(self.label_cmap[0] / 255.0)
                    plt.xticks(rotation=90)
                    plt.yticks(rotation=0)
                    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
                    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
                    plt.tight_layout()

                    # add_plot(self.logger.experiment.log, "conf_matrix", self.global_step)

                    all_bars = torch.cat([
                        self.cluster_metrics.histogram.sum(0).cpu(),
                        self.cluster_metrics.histogram.sum(1).cpu()
                    ], axis=0)
                    ymin = max(all_bars.min() * .8, 1)
                    ymax = all_bars.max() * 1.2

                    fig, ax = plt.subplots(1, 2, figsize=(2 * 5, 1 * 4))
                    ax[0].bar(range(self.n_classes + self.cfg.extra_clusters),
                              self.cluster_metrics.histogram.sum(0).cpu(),
                              tick_label=names,
                              color=colors)
                    ax[0].set_ylim(ymin, ymax)
                    ax[0].set_title("Label Frequency")
                    ax[0].set_yscale('log')
                    ax[0].tick_params(axis='x', labelrotation=90)

                    ax[1].bar(range(self.n_classes + self.cfg.extra_clusters),
                              self.cluster_metrics.histogram.sum(1).cpu(),
                              tick_label=names,
                              color=colors)
                    ax[1].set_ylim(ymin, ymax)
                    ax[1].set_title("Cluster Frequency")
                    ax[1].set_yscale('log')
                    ax[1].tick_params(axis='x', labelrotation=90)

                    plt.tight_layout()
                    # add_plot(tb_logger.log, "label frequency", self.global_step)

            if self.global_step > 2:
                self.log_dict(tb_metrics)

                # if self.trainer.is_global_zero and self.cfg.azureml_logging:
                #     from azureml.core.run import Run
                #     run_logger = Run.get_context()
                #     for metric, value in tb_metrics.items():
                #         run_logger.log(metric, value)

            self.linear_metrics.reset()
            self.cluster_metrics.reset()

        self.validation_step_outputs.clear()

    def configure_optimizers(self):  # project_head_cluster_optim
        main_params = list(self.net.parameters())

        if self.cfg.rec_weight > 0:
            main_params.extend(self.decoder.parameters())

        net_optim = torch.optim.Adam(main_params, lr=self.cfg.lr)
        linear_probe_optim = torch.optim.Adam(list(self.linear_probe.parameters()), lr=self.cfg.lr_linear)
        cluster_probe_optim = torch.optim.Adam(list(self.cluster_probe.parameters()), lr=self.cfg.lr_cluster)
        cluster_eigen_optim = torch.optim.Adam(list(self.train_cluster_probe_eigen.parameters()),
                                               lr=self.cfg.lr_cluster_eigen)
        cluster_eigen_optim_aug = torch.optim.Adam(list(self.train_cluster_probe_eigen_aug.parameters()),
                                                   lr=self.cfg.lr_cluster_eigen)

        if self.cfg.use_head and (self.cfg.centroid_mode == 'learned' or self.cfg.centroid_mode == 'prototype'):
            project_head_optim = torch.optim.Adam(self.project_head.parameters(), lr=self.cfg.lr)
            centroid_optim = torch.optim.Adam(self.CELoss.parameters(), lr=self.cfg.lr)
            return net_optim, linear_probe_optim, cluster_probe_optim, project_head_optim, centroid_optim, cluster_eigen_optim, cluster_eigen_optim_aug

        elif self.cfg.use_head and (self.cfg.centroid_mode == 'mean' or self.cfg.centroid_mode == 'medoid'):
            project_head_optim = torch.optim.Adam(self.project_head.parameters(), lr=self.cfg.lr)
            return net_optim, linear_probe_optim, cluster_probe_optim, project_head_optim, cluster_eigen_optim, cluster_eigen_optim_aug
        else:
            return net_optim, linear_probe_optim, cluster_probe_optim, cluster_eigen_optim, cluster_eigen_optim_aug


@hydra.main(config_path="configs", config_name="train_config_cocostuff.yml")
def my_app(cfg: DictConfig) -> None:
    OmegaConf.set_struct(cfg, False)
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = join("src_EAGLE", cfg.pytorch_data_dir)
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")
    exp_name = f"{cfg.experiment_name}"

    prefix = "{}/{}_{}_{}_{}".format(cfg.dataset_name, cfg.log_dir, get_current_date(), cfg.model_type, exp_name)

    cfg.full_name = prefix

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(data_dir)
    print(cfg.output_root)

    geometric_transforms = T.Compose([
        T.RandomHorizontalFlip(),
        T.RandomResizedCrop(size=cfg.res, scale=(0.8, 1.0))
    ])
    photometric_transforms = T.Compose([
        # T.ColorJitter(brightness=.3, contrast=.3, saturation=.3, hue=.1),
        # T.RandomGrayscale(.2),
        T.RandomApply([T.GaussianBlur((3, 3))])
    ])

    sys.stdout.flush()

    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=geometric_transforms,
        aug_photometric_transform=photometric_transforms,
        mask=True,
    )

    if cfg.dataset_name == "voc":
        val_loader_crop = None
    else:
        val_loader_crop = "center"

    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(320, False, val_loader_crop),
        target_transform=get_transform(320, True, val_loader_crop),
        mask=True,
        cfg=cfg,
    )

    train_loader = DataLoader(train_dataset,
                              cfg.batch_size,
                              shuffle=True,
                              num_workers=cfg.num_workers,
                              pin_memory=True)

    if cfg.submitting_to_aml:
        val_batch_size = 16
    else:
        # needed to fit the GiB of 16 given by GPU0
        val_batch_size = cfg.batch_size // 8

    val_loader = DataLoader(val_dataset,
                            val_batch_size,
                            shuffle=False,
                            num_workers=cfg.num_workers,
                            pin_memory=True)

    model = LitUnsupervisedSegmenter(train_dataset.n_classes, cfg)

    # tb_logger = WandbLogger(
    #     name=cfg.log_dir + "_" + exp_name, project=cfg.project_name, entity=cfg.entity
    # )

    if cfg.dataset_name == 'cocostuff27':
        gpu_args = dict(accelerator='gpu', devices=1, val_check_interval=cfg.val_freq)
    else:
        gpu_args = dict(accelerator='gpu', devices=1, val_check_interval=cfg.val_freq)

    if gpu_args["val_check_interval"] > len(train_loader) // 4:
        gpu_args.pop("val_check_interval")

    model_checkpoint = ModelCheckpoint(
        verbose=True,
        every_n_train_steps=5,
        # every_n_epochs=1,
        save_top_k=1,
        monitor="loss/total",
        mode="max",
        dirpath=join(checkpoint_dir, prefix),
        filename='{epoch:02d}-{step:08d}-{test/total:.2f}'
    )

    trainer = Trainer(
        log_every_n_steps=cfg.scalar_log_freq,
        logger=True,
        # logger=tb_logger,
        max_steps=cfg.max_steps,
        # max_epochs=1,
        callbacks=[model_checkpoint],
        num_sanity_val_steps=0,
        **gpu_args
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    prep_args()
    my_app()
