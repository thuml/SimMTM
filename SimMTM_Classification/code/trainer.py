import os
import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, precision_score, f1_score, recall_score
from utils.augmentations import data_transform_masked4cl
from model import TFC, target_classifier
import warnings
warnings.filterwarnings("ignore")

def build_model(args, lr, configs, device, chkpoint=None):
    # Model Backbone
    model = TFC(configs, args).to(device)
    if chkpoint:
        pretrained_dict = chkpoint["model_state_dict"]
        model_dict = model.state_dict()
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)

    # Classification Head
    classifier = target_classifier(configs).to(device)

    # Optimizer
    model_optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(configs.beta1, configs.beta2), weight_decay=0)
    classifier_optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, betas=(configs.beta1, configs.beta2),
                                            weight_decay=0)
    model_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=model_optimizer, T_max=args.finetune_epoch)

    return model, classifier, model_optimizer, classifier_optimizer, model_scheduler


def Trainer(model,
            model_optimizer,
            model_scheduler,
            train_dl,
            valid_dl,
            test_dl,
            device,
            logger,
            args,
            configs,
            experiment_log_dir,
            seed):
    logger.debug("Pre-training started ....")
    os.makedirs(os.path.join(experiment_log_dir, f"saved_models"), exist_ok=True)

    # Pre-training
    best_performance = None
    for epoch in range(1, args.pretrain_epoch + 1):

        total_loss, total_cl_loss, total_rb_loss = model_pretrain(model, model_optimizer, model_scheduler, train_dl,
                                                                  configs, args, device)

        logger.debug(
            f'Pre-training Epoch: {epoch}\t Train Loss: {total_loss:.4f}\t CL Loss: {total_cl_loss:.4f}\t RB Loss: {total_rb_loss:.4f}\n')

        chkpoint = {'seed': seed, 'epoch': epoch, 'train_loss': total_loss, 'model_state_dict': model.state_dict()}
        torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'ckp_ep{epoch}.pt'))

        if epoch % 2 == 0:

            # Fine-tuning
            logger.debug("Fine-tuning started ....")
            ft_model, ft_classifier, ft_model_optimizer, ft_classifier_optimizer, ft_scheduler = build_model(args,
                                                                                                             args.lr,
                                                                                                             configs,
                                                                                                             device,
                                                                                                             chkpoint)

            for ep in range(1, args.finetune_epoch + 1):

                valid_loss, valid_acc, valid_auc, valid_prc, emb_finetune, label_finetune, F1 = model_finetune(ft_model,
                                                                                                               valid_dl,
                                                                                                               device,
                                                                                                               ft_model_optimizer,
                                                                                                               ft_scheduler,
                                                                                                               classifier=ft_classifier,
                                                                                                               classifier_optimizer=ft_classifier_optimizer)

                if ep % args.log_epoch == 0:
                    # Test
                    logger.debug(
                        f'\nEpoch : {ep}\t | \t  finetune Loss: {valid_loss:.4f}\t | \tAcc: {valid_acc:2.4f}\t | \tF1: {F1:0.4f}')
                    test_loss, test_acc, test_auc, test_prc, emb_test, label_test, performance = model_test(ft_model,
                                                                                                            test_dl,
                                                                                                            device,
                                                                                                            classifier=ft_classifier)

                    if best_performance is None:
                        best_performance = performance
                    else:
                        if performance[0] > best_performance[0]:
                            best_performance = performance
                            logger.debug(
                                'EP%s - Better Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f' % (
                                ep, performance[0], performance[1], performance[2], performance[3]))

                            chkpoint = {'seed': seed, 'epoch': epoch, 'train_loss': total_loss,
                                        'model_state_dict': model.state_dict()}
                            torch.save(chkpoint, os.path.join(experiment_log_dir, f"saved_models/", f'ckp_best.pt'))
            logger.debug("Fine-tuning ended ....")
            logger.debug("=" * 100)
            logger.debug('EP%s - Best Testing: Acc=%.4f| Precision = %.4f | Recall = %.4f | F1 = %.4f' % (
            epoch, best_performance[0], best_performance[1], best_performance[2], best_performance[3]))
            logger.debug("=" * 100)

    return best_performance


def model_pretrain(model, model_optimizer, model_scheduler, train_loader, configs, args, device):
    total_loss = []
    total_cl_loss = []
    total_rb_loss = []

    model.train()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data_masked_m, mask = data_transform_masked4cl(data, args.masking_ratio, args.lm, args.positive_nums)
        data_masked_om = torch.cat([data, data_masked_m], 0)

        data, labels, data_masked_om = data.float().to(device), labels.float().to(device), data_masked_om.float().to(
            device)

        # Produce embeddings of original and masked samples
        loss, loss_cl, loss_rb = model(data_masked_om, pretrain=True)

        model_optimizer.zero_grad()
        loss.backward()
        model_optimizer.step()

        total_loss.append(loss.item())
        total_cl_loss.append(loss_cl.item())
        total_rb_loss.append(loss_rb.item())

    total_loss = torch.tensor(total_loss).mean()
    total_cl_loss = torch.tensor(total_cl_loss).mean()
    total_rb_loss = torch.tensor(total_rb_loss).mean()

    model_scheduler.step()

    return total_loss, total_cl_loss, total_rb_loss


def model_finetune(model, val_dl, device, model_optimizer, model_scheduler, classifier=None, classifier_optimizer=None):
    model.train()
    classifier.train()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])

    for data, labels in val_dl:
        data, labels = data.float().to(device), labels.long().to(device)

        # Produce embeddings
        h, z = model(data)

        # Add supervised classifier: 1) it's unique to finetuning. 2) this classifier will also be used in test
        fea_concat = h

        predictions = classifier(fea_concat)
        fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
        loss = criterion(predictions, labels)

        acc_bs = labels.eq(predictions.detach().argmax(dim=1)).float().mean()
        onehot_label = F.one_hot(labels)
        pred_numpy = predictions.detach().cpu().numpy()

        try:
            auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro", multi_class="ovr")
        except:
            auc_bs = 0.0

        try:
            prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy)
        except:
            prc_bs = 0.0

        total_acc.append(acc_bs)

        if auc_bs != 0:
            total_auc.append(auc_bs)
        if prc_bs != 0:
            total_prc.append(prc_bs)
        total_loss.append(loss.item())

        loss.backward()
        model_optimizer.step()
        classifier_optimizer.step()

        pred = predictions.max(1, keepdim=True)[1]
        outs = np.append(outs, pred.cpu().numpy())
        trgs = np.append(trgs, labels.data.cpu().numpy())

    labels_numpy = labels.detach().cpu().numpy()
    pred_numpy = np.argmax(pred_numpy, axis=1)
    F1 = f1_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))

    total_loss = torch.tensor(total_loss).mean()  # average loss
    total_acc = torch.tensor(total_acc).mean()  # average acc
    total_auc = torch.tensor(total_auc).mean()  # average auc
    total_prc = torch.tensor(total_prc).mean()

    model_scheduler.step(total_loss)

    return total_loss, total_acc, total_auc, total_prc, fea_concat_flat, trgs, F1


def model_test(model, test_dl, device, classifier=None):
    model.eval()
    classifier.eval()

    total_loss = []
    total_acc = []
    total_auc = []
    total_prc = []
    total_precision, total_recall, total_f1 = [], [], []

    criterion = nn.CrossEntropyLoss()
    outs = np.array([])
    trgs = np.array([])
    emb_test_all = []

    with torch.no_grad():
        labels_numpy_all, pred_numpy_all = np.zeros(1), np.zeros(1)
        for data, labels in test_dl:

            data, labels = data.float().to(device), labels.long().to(device)

            # Add supervised classifier: 1) it's unique to fine-tuning. 2) this classifier will also be used in test
            h, z = model(data)

            fea_concat = h
            predictions_test = classifier(fea_concat)
            fea_concat_flat = fea_concat.reshape(fea_concat.shape[0], -1)
            emb_test_all.append(fea_concat_flat)

            loss = criterion(predictions_test, labels)
            acc_bs = labels.eq(predictions_test.detach().argmax(dim=1)).float().mean()
            onehot_label = F.one_hot(labels)
            pred_numpy = predictions_test.detach().cpu().numpy()
            labels_numpy = labels.detach().cpu().numpy()

            try:
                auc_bs = roc_auc_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro",
                                       multi_class="ovr")
            except:
                auc_bs = 0.0

            try:
                prc_bs = average_precision_score(onehot_label.detach().cpu().numpy(), pred_numpy, average="macro")
            except:
                prc_bs = 0.0

            pred_numpy = np.argmax(pred_numpy, axis=1)
            precision = precision_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))
            recall = recall_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))
            F1 = f1_score(labels_numpy, pred_numpy, average='macro', )  # labels=np.unique(ypred))

            total_acc.append(acc_bs)

            if auc_bs != 0:
                total_auc.append(auc_bs)
            if prc_bs != 0:
                total_prc.append(prc_bs)
            total_precision.append(precision)
            total_recall.append(recall)
            total_f1.append(F1)

            total_loss.append(loss.item())

            pred = predictions_test.max(1, keepdim=True)[1]  # get the index of the max log-probability
            outs = np.append(outs, pred.cpu().numpy())
            trgs = np.append(trgs, labels.data.cpu().numpy())

            labels_numpy_all = np.concatenate((labels_numpy_all, labels_numpy))
            pred_numpy_all = np.concatenate((pred_numpy_all, pred_numpy))

    labels_numpy_all = labels_numpy_all[1:]
    pred_numpy_all = pred_numpy_all[1:]

    precision = precision_score(labels_numpy_all, pred_numpy_all, average='macro', )
    recall = recall_score(labels_numpy_all, pred_numpy_all, average='macro', )
    F1 = f1_score(labels_numpy_all, pred_numpy_all, average='macro', )
    acc = accuracy_score(labels_numpy_all, pred_numpy_all, )

    total_loss = torch.tensor(total_loss).mean()
    total_acc = torch.tensor(total_acc).mean()
    total_auc = torch.tensor(total_auc).mean()
    total_prc = torch.tensor(total_prc).mean()

    performance = [acc * 100, precision * 100, recall * 100, F1 * 100, total_auc * 100, total_prc * 100]

    emb_test_all = torch.cat(tuple(emb_test_all))
    return total_loss, total_acc, total_auc, total_prc, emb_test_all, trgs, performance