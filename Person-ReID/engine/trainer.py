# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage

from utils.reid_metric import R1_mAP


def create_supervised_trainer(cfg, model, optimizer, loss_fn,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        if cfg.MODEL.METRIC_LOSS_TYPE == 'ours' or cfg.MODEL.METRIC_LOSS_TYPE == 'ours_triplet':
            loss, acc, proxypos, proxyneg, possim, negsim = loss_fn(score, feat, target)
            loss.backward()
            optimizer.step()
            return loss.item(), acc, proxypos.item(), proxyneg.item(), possim.item(), negsim.item()
        else:
            loss = loss_fn(score, feat, target)
            loss.backward()
            optimizer.step()
            acc = (score.max(1)[1] == target).float().mean()
            return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cetner_loss_weight,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = loss_fn(score, feat, target)
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def adjust_learning_rate_auto(Adjust_LR, optimizer):
    if Adjust_LR == 'on':
        if optimizer.param_groups[0]['lr'] == 3.5e-5:
            for param_group in optimizer.param_groups:
                param_group['lr'] = 7.0e-5
        elif optimizer.param_groups[0]['lr'] == 3.5e-4:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 10.0
        else:
            for param_group in optimizer.param_groups:
                param_group['lr'] = param_group['lr'] / 5.0
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 10.0


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch, 
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(cfg, model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
                                                                     'optimizer': optimizer.state_dict()})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)


    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')
    if cfg.MODEL.METRIC_LOSS_TYPE == 'ours' or cfg.MODEL.METRIC_LOSS_TYPE == 'ours_triplet':
        RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'avg_proxypos')
        RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'avg_proxyneg')
        RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'avg_possim')
        RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'avg_negsim')
    map_list = []

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        if cfg.MODEL.METRIC_LOSS_TYPE == 'ours' or cfg.MODEL.METRIC_LOSS_TYPE == 'triplets':
            pass
        else:
            scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1
        if iter % log_period == 0:
            if cfg.MODEL.METRIC_LOSS_TYPE == 'ours' or cfg.MODEL.METRIC_LOSS_TYPE == 'ours_triplet':
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}\tAcc: {:.3f}\nProxyPos: {:.3f}\tProxyNeg: {:.3f}\tPosSim {:.3f}\tNegSim {:.3f}\tBase Lr: {:.2e}"
                            .format(engine.state.epoch, iter, len(train_loader),
                                    engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                    engine.state.metrics['avg_proxypos'], engine.state.metrics['avg_proxyneg'],
                                    engine.state.metrics['avg_possim'], engine.state.metrics['avg_negsim'],
                                    optimizer.param_groups[0]['lr']))                
            else:    
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(engine.state.epoch, iter, len(train_loader),
                                    engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                    scheduler.get_lr()[0]))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    
    
    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        #import pdb; pdb.set_trace()
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
            map_list.append(mAP)
            if cfg.MODEL.METRIC_LOSS_TYPE == 'ours' or cfg.MODEL.METRIC_LOSS_TYPE == 'triplets':
                if optimizer.param_groups[0]['lr'] == 3.5e-4 or optimizer.param_groups[0]['lr'] == 1e-4:
                    tolenrance = 3
                    #if engine.state.epoch > 20:
                    #    tolenrance = 1
                elif optimizer.param_groups[0]['lr'] == 7.0e-5:
                    tolenrance = 3
                elif optimizer.param_groups[0]['lr'] == 1.4e-5:
                    tolenrance = 3
                elif optimizer.param_groups[0]['lr'] == 3.5e-5:
                    tolenrance = 6
                else:
                    tolenrance = 1000
                #map_list.append(mAP)
                if len(map_list) > tolenrance and max(map_list[-tolenrance:]) < max(map_list[:-tolenrance]):
                    adjust_learning_rate_auto(cfg.MODEL.ADJUST_LR, optimizer)

            #logger.info(map_list)
            logger.info('The max mAP is {:.1%}'.format(max(map_list)))
            logger.info('The max mAP is Epoch {}'.format(map_list.index(max(map_list))))

    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model.state_dict(),
                                                                     'optimizer': optimizer.state_dict(),
                                                                     'optimizer_center': optimizer_center.state_dict()})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, iter, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)