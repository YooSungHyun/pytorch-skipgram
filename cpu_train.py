import logging
import math
import os
from logging import StreamHandler
from typing import Optional, Union

import json
import pandas as pd
import torch
import wandb
from arguments.training_args import TrainingArguments
from networks.models import Net
from setproctitle import setproctitle
from simple_parsing import ArgumentParser
from torch.utils.data import RandomSampler, SequentialSampler, random_split
from trainer.cpu import Trainer
from utils.comfy import (
    apply_to_collection,
    dataclass_to_namespace,
    seed_everything,
    tensor_dict_to_device,
    web_log_every_n,
)
from utils.data.custom_dataloader import CustomDataLoader
from utils.data.custom_sampler import LengthGroupedSampler
from utils.data.nlp_dataset import CBOWDataset
import random
from collections import defaultdict, deque

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(levelname)8s] %(message)s")

timeFileHandler = StreamHandler()
timeFileHandler.setFormatter(formatter)

logger.addHandler(timeFileHandler)


# TODO(User): override training_step and eval_loop for your style
class CPUTrainer(Trainer):
    def __init__(
        self,
        criterion,
        eval_metric=None,
        precision="fp32",
        cmd_logger=None,
        web_logger=None,
        max_epochs: Optional[int] = 1000,
        max_steps: Optional[int] = None,
        grad_accum_steps: int = 1,
        limit_train_batches: Union[int, float] = float("inf"),
        limit_val_batches: Union[int, float] = float("inf"),
        validation_frequency: int = 1,
        checkpoint_dir: str = "./checkpoints",
        checkpoint_frequency: int = 1,
        chk_addr_dict: dict = None,
        non_blocking: bool = True,
        log_every_n: int = 1,
    ):
        super().__init__(
            criterion,
            eval_metric,
            precision,
            cmd_logger,
            web_logger,
            max_epochs,
            max_steps,
            grad_accum_steps,
            limit_train_batches,
            limit_val_batches,
            validation_frequency,
            checkpoint_dir,
            checkpoint_frequency,
            chk_addr_dict,
            non_blocking,
            log_every_n,
        )

    def training_step(self, model, batch, batch_idx) -> torch.Tensor:
        """A single training step, running forward and backward. The optimizer step is called separately, as this is
        given as a closure to the optimizer step.

        Args:
            model: model to train
            batch: the batch to run the forward on
            batch_idx: index of the current batch w.r.t the current epoch

        """
        positive_score, negative_score = model(**batch)
        positive_pred = self.criterion(positive_score)
        negative_size = negative_score.size(1)
        negative_score = torch.sum(negative_score, dim=1) / negative_size
        negative_pred = self.criterion(negative_score)

        eps = torch.tensor(1e-08)
        # positive의 sigmoid를 최대화 하려면 음의 로그를 작게 만들면 됨
        positive_loss = -torch.log(positive_pred + eps)

        # 확률 문제에서 정답에 대한 반대(1-정답), negative_sampling 개수를 고려해야 하니까 sum
        negative_loss = -torch.log((1 - negative_pred) + eps)

        loss = torch.mean(positive_loss + negative_loss)
        # loss = torch.mean(loss)

        def on_before_backward(loss):
            pass

        on_before_backward(loss)
        loss.backward()

        def on_after_backward():
            pass

        on_after_backward()

        outputs = {"loss": loss}
        # avoid gradients in stored/accumulated values -> prevents potential OOM
        self._current_train_return = apply_to_collection(outputs, dtype=torch.Tensor, function=lambda x: x.detach())

        web_log_every_n(
            self.web_logger,
            {
                "train/loss": self._current_train_return["loss"],
                "train/step": self.step,
                "train/global_step": self.global_step,
                "train/epoch": self.current_epoch,
            },
            self.step,
            self.log_every_n,
        )
        return loss

    def eval_loop(
        self,
        model,
        val_loader: Optional[torch.utils.data.DataLoader],
        limit_batches: Union[int, float] = float("inf"),
    ):
        """The validation loop ruunning a single validation epoch.

        Args:
            model: model
            val_loader: The dataloader yielding the validation batches.
            limit_batches: Limits the batches during this validation epoch.
                If greater than the number of batches in the ``val_loader``, this has no effect.

        """
        # no validation if val_loader wasn't passed
        if val_loader is None:
            return

        def on_start_eval(model):
            model.eval()
            # requires_grad = True, but loss.backward() raised error
            # because grad_fn is None
            torch.set_grad_enabled(False)

        on_start_eval(model)

        def on_validation_epoch_start():
            pass

        iterable = self.progbar_wrapper(val_loader, total=min(len(val_loader), limit_batches), desc="Validation")
        eval_step = 0
        tot_batch_losses = list()
        for batch_idx, batch in enumerate(iterable):
            tensor_dict_to_device(batch, "cpu", non_blocking=self.non_blocking)
            # end epoch if stopping training completely or max batches for this epoch reached
            if self.should_stop or batch_idx >= limit_batches:
                break

            def on_validation_batch_start(batch, batch_idx):
                pass

            on_validation_batch_start(batch, batch_idx)

            positive_score, negative_score = model(**batch)

            positive_pred = self.criterion(positive_score)
            negative_size = negative_score.size(1)
            negative_score = torch.sum(negative_score, dim=1) / negative_size
            negative_pred = self.criterion(negative_score)

            eps = torch.tensor(1e-08)
            # positive의 sigmoid를 최대화 하려면 음의 로그를 작게 만들면 됨
            positive_loss = -torch.log(positive_pred + eps)

            # 확률 문제에서 정답에 대한 반대(1-정답), negative_sampling 개수를 고려해야 하니까 sum
            negative_loss = -torch.log((1 - negative_pred) + eps)

            loss = torch.mean(positive_loss + negative_loss)

            log_output = {"loss": loss}
            # avoid gradients in stored/accumulated values -> prevents potential OOM
            self._current_val_return = apply_to_collection(log_output, torch.Tensor, lambda x: x.detach())

            tot_batch_losses.append(self._current_val_return["loss"])
            web_log_every_n(
                self.web_logger,
                {
                    "eval_step/loss": self._current_val_return["loss"],
                    "eval_step/step": eval_step,
                    "eval_step/global_step": self.global_step,
                    "eval_step/epoch": self.current_epoch,
                },
                eval_step,
                self.log_every_n,
            )
            self._format_iterable(iterable, self._current_val_return, "val")
            eval_step += 1

        # TODO(User): Create any form you want to output to wandb!
        def on_validation_epoch_end(tot_batch_losses):
            tot_batch_losses = torch.stack(tot_batch_losses, dim=0)
            epoch_loss = torch.mean(tot_batch_losses)
            # epoch monitoring is must doing every epoch
            web_log_every_n(
                self.web_logger, {"eval/loss": epoch_loss, "eval/epoch": self.current_epoch}, self.current_epoch, 1
            )

        on_validation_epoch_end(tot_batch_losses)

        def on_validation_model_train(model):
            torch.set_grad_enabled(True)
            model.train()

        on_validation_model_train(model)


def main(hparams: TrainingArguments):
    # reference: https://www.kaggle.com/code/anitarostami/lstm-multivariate-forecasting
    setproctitle(os.environ.get("WANDB_PROJECT", "torch-trainer"))
    web_logger = wandb.init(config=hparams)
    seed_everything(hparams.seed)

    tot_dataset = pd.read_csv(hparams.train_datasets_path, header=0, encoding="utf-8")
    train_dataset = tot_dataset[tot_dataset["split"] == "train"]
    eval_dataset = tot_dataset[tot_dataset["split"] == "val"]
    logger.info(tot_dataset.head())

    with open("raw_data/500_vocab.json", "r") as st_json:
        tokenizer = json.load(st_json)

    with open("raw_data/500_train_cnt.json", "r") as st_json:
        train_counter = json.load(st_json)
    with open("raw_data/500_eval_cnt.json", "r") as st_json:
        eval_counter = json.load(st_json)

    negative_sample_n = hparams.negative_sample_n
    window_size = hparams.window_size

    def preprocess(example, subsample_freq, negative_list):
        context = example["context"].split()
        input_ids = list()
        # q = deque(context)
        # while len(input_ids) < window_size * 2 - 1:
        # token = q.popleft()
        # if random.uniform(0, 1) >= subsample_freq[token]:
        input_ids = [tokenizer[token] for token in context]
        # else:
        # q.append(token)
        if negative_list and negative_sample_n > 0:
            negative_tokens = list()
            weights = list()
            for token, weight in negative_list:
                negative_tokens.append(token)
                weights.append(weight)
            negative_samples = list()
            visited = defaultdict(lambda: False)
            while True:
                if len(negative_samples) == negative_sample_n:
                    break
                else:
                    random_token = random.choices(negative_tokens, weights=weights, k=1)[0]
                    if (
                        example["context"].find(random_token) == -1
                        and example["target"].find(random_token) == -1
                        and not visited[random_token]
                    ):
                        negative_samples.append(random_token)
                        visited[random_token] = True
            example["negative_samples"] = [tokenizer[token] for token in negative_samples]
        labels = [tokenizer[token] for token in example["target"].split()]
        example["input_ids"] = input_ids
        example["labels"] = labels
        return example

    train_dataset = CBOWDataset(train_dataset, train_counter, transform=preprocess)
    eval_dataset = CBOWDataset(eval_dataset, eval_counter, transform=preprocess)

    # Instantiate objects
    model = Net(vocab_size=len(tokenizer.keys()), embedding_size=512)
    web_logger.watch(model, log_freq=hparams.log_every_n)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hparams.learning_rate,
        eps=hparams.optim_eps,
        betas=(hparams.optim_beta1, hparams.optim_beta2),
        weight_decay=hparams.weight_decay,
    )

    generator = None
    custom_train_sampler = RandomSampler(train_dataset, generator=generator)
    custom_eval_sampler = SequentialSampler(eval_dataset)

    # If 1 device for training, sampler suffle True and dataloader shuffle True is same meaning
    train_dataloader = CustomDataLoader(
        dataset=train_dataset,
        batch_size=hparams.per_device_train_batch_size,
        sampler=custom_train_sampler,
        num_workers=hparams.num_workers,
        drop_last=hparams.dataloader_drop_last,
    )

    eval_dataloader = CustomDataLoader(
        dataset=eval_dataset,
        batch_size=hparams.per_device_eval_batch_size,
        sampler=custom_eval_sampler,
        num_workers=hparams.num_workers,
        drop_last=hparams.dataloader_drop_last,
    )

    # dataloader already calculate total_data / batch_size
    # accumulation is always floor
    train_steps_per_epoch = math.floor(len(train_dataloader) / (hparams.accumulate_grad_batches))

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=hparams.learning_rate,
        pct_start=hparams.warmup_ratio,
        epochs=hparams.max_epochs,
        final_div_factor=hparams.final_div_factor,
        steps_per_epoch=train_steps_per_epoch,
    )

    # monitor: ReduceLROnPlateau scheduler is stepped using loss, so monitor input train or val loss
    lr_scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1, "monitor": None}
    assert id(scheduler) == id(lr_scheduler["scheduler"])
    criterion = torch.nn.Sigmoid()
    trainable_loss = None

    # I think some addr is same into trainer init&fit respectfully
    chk_addr_dict = {
        "train_dataloader": id(train_dataloader),
        "eval_dataloader": id(eval_dataloader),
        "model": id(model),
        "optimizer": id(optimizer),
        "criterion": id(criterion),
        "scheduler_cfg": id(lr_scheduler),
        "scheduler_cfg[scheduler]": id(lr_scheduler["scheduler"]),
        "trainable_loss": id(trainable_loss),
    }

    log_str = f"""\n##########################################
    train_dataloader addr: {chk_addr_dict["train_dataloader"]}
    eval_dataloader addr: {chk_addr_dict["eval_dataloader"]}
    model addr: {chk_addr_dict["model"]}
    optimizer addr: {chk_addr_dict["optimizer"]}
    criterion addr: {chk_addr_dict["criterion"]}
    scheduler_cfg addr: {chk_addr_dict["scheduler_cfg"]}
    scheduler addr: {chk_addr_dict["scheduler_cfg[scheduler]"]}
    ##########################################
    """
    logger.debug(log_str)
    # TODO(User): input your eval_metric
    eval_metric = None
    trainer = CPUTrainer(
        criterion=criterion,
        eval_metric=eval_metric,
        precision=hparams.model_dtype,
        cmd_logger=logger,
        web_logger=web_logger,
        max_epochs=hparams.max_epochs,
        grad_accum_steps=hparams.accumulate_grad_batches,
        chk_addr_dict=chk_addr_dict,
        checkpoint_dir=hparams.output_dir,
        log_every_n=hparams.log_every_n,
    )

    trainer.fit(
        model=model,
        optimizer=optimizer,
        scheduler_cfg=lr_scheduler,
        train_loader=train_dataloader,
        val_loader=eval_dataloader,
        ckpt_path=hparams.output_dir,
        trainable_loss=trainable_loss,
    )

    web_logger.finish(exit_code=0)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_arguments(TrainingArguments, dest="training_args")
    args = parser.parse_args()
    args = dataclass_to_namespace(args, "training_args")

    main(args)
