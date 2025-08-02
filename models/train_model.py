import os, sys
import torch
import torch.nn as nn
from torch.nn.functional import one_hot, binary_cross_entropy, cross_entropy
from torch.nn.utils.clip_grad import clip_grad_norm_
import numpy as np
from .evaluate_model import evaluate
from torch.autograd import Variable, grad
from .atkt import _l2_normalize_adv
from ..utils.utils import debug_print
from pykt.config import que_type_models
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cal_loss(model, ys, r, rshft, sm, preloss=[], auxloss=None ):
    model_name = model.model_name

    if model_name in ["simplekt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        # print(f"loss1: {y.shape}")
        loss1 = binary_cross_entropy(y.double(), t.double())
        loss = loss1

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss = model.l1*loss1+model.l2*ys[1]+model.l3*ys[2]
            else:
                loss = model.l1*loss1+model.l2*ys[1]
        elif model.emb_type.find("predhis") != -1:
            loss = model.l1*loss1+model.l2*ys[1]
        else:
            loss = loss1
    elif model_name in ["ddsimplekt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())

        if model.emb_type.find("predcurc") != -1:
            if model.emb_type.find("his") != -1:
                loss = model.l1*loss+model.l2*ys[1]+model.l3*ys[2]+auxloss
            else:
                loss = model.l1*loss+model.l2*ys[1]+auxloss
        elif model.emb_type.find("predhis") != -1:
            loss = model.l1*loss+model.l2*ys[1]+auxloss
        else:
            loss = loss+auxloss

    elif model_name in ["sakt", "saint"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())
    elif model_name in ["ddsaint","desakt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double())
        loss = loss + auxloss

    elif model_name in ["akt"]:
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
    elif model_name == "ddakt":
        y = torch.masked_select(ys[0], sm)
        t = torch.masked_select(rshft, sm)
        loss = binary_cross_entropy(y.double(), t.double()) + preloss[0]
        loss = loss + auxloss
    return loss


def model_forward(model, data, rel=None):
    model_name = model.model_name
    # if model_name in ["dkt_forget", "lpkt"]:
    #     q, c, r, qshft, cshft, rshft, m, sm, d, dshft = data

    dcur = data
    q, c, r, t = dcur["qseqs"].to(device), dcur["cseqs"].to(device), dcur["rseqs"].to(device), dcur["tseqs"].to(device)
    qshft, cshft, rshft, tshft = dcur["shft_qseqs"].to(device), dcur["shft_cseqs"].to(device), dcur["shft_rseqs"].to(device), dcur["shft_tseqs"].to(device)
    m, sm = dcur["masks"].to(device), dcur["smasks"].to(device)

    ys, preloss = [], []
    cq = torch.cat((q[:,0:1], qshft), dim=1)
    cc = torch.cat((c[:,0:1], cshft), dim=1)
    cr = torch.cat((r[:,0:1], rshft), dim=1)

    if model_name in ["simplekt"]:
        y, y2, y3 = model(dcur, train=True)
        ys = [y[:,1:], y2, y3]
    elif model_name in ["ddsimplekt"]:
        y, y2, y3, auxloss = model(dcur, train=True)
        ys = [y[:,1:], y2, y3]

    if model_name in ["sakt"]:
        y = model(c.long(), r.long(), cshft.long())
        ys.append(y)
    elif model_name in ["desakt"]:
        y, auxloss = model(c.long(), r.long(), cshft.long())
        ys.append(y)
        # preloss.append(auxloss)
    elif model_name in ["saint"]:
        y = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
    elif model_name in ["ddsaint"]:
        y, auxloss = model(cq.long(), cc.long(), r.long())
        ys.append(y[:, 1:])
        # preloss.append(auxloss)
    elif model_name in ["akt"]:               
        y, reg_loss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(reg_loss)
    elif model_name in ["ddakt"]:               
        y, c_reg_loss, auxloss = model(cc.long(), cr.long(), cq.long())
        ys.append(y[:,1:])
        preloss.append(c_reg_loss)
        # auxloss = reg_loss

    if model_name in ["ddsaint","desakt","ddakt","ddsimplekt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss, auxloss)
    elif model_name in ["simplekt","akt"]:
        loss = cal_loss(model, ys, r, rshft, sm, preloss)
    elif model_name in ["sakt", "saint"]:
        loss = cal_loss(model, ys, r, rshft, sm)
    else:
        loss = cal_loss(model, ys, r, rshft, sm)

    return loss
    

def train_model(model, train_loader, valid_loader, num_epochs, opt, ckpt_path, test_loader=None, test_window_loader=None, save_model=False, data_config=None, fold=None):
    max_auc, best_epoch = 0, -1
    train_step = 0

    rel = None

    for i in range(1, num_epochs + 1):
        loss_mean = []
        for data in train_loader:
            train_step+=1
            if model.model_name in que_type_models and model.model_name not in ["lpkt", "rkt"]:
                model.model.train()
            else:
                model.train()
            if model.model_name in ['rkt']:
                loss = model_forward(model, data, rel)
            else:
                loss = model_forward(model, data)
            opt.zero_grad()
            loss.backward()#compute gradients
            if model.model_name == "rkt":
                clip_grad_norm_(model.parameters(), model.grad_clip)
            if model.model_name == "dtransformer":
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()#update model’s parameters
                
            loss_mean.append(loss.detach().cpu().numpy())
            if model.model_name == "gkt" and train_step%10==0:
                text = f"Total train step is {train_step}, the loss is {loss.item():.5}"
                debug_print(text = text,fuc_name="train_model")

        loss_mean = np.mean(loss_mean)
        
        auc, acc = evaluate(model, valid_loader, model.model_name)
        ### auc, acc = round(auc, 4), round(acc, 4)

        if auc > max_auc+1e-3:
            if save_model:
                torch.save(model.state_dict(), os.path.join(ckpt_path, model.emb_type+"_model.ckpt"))
            max_auc = auc
            best_epoch = i
            testauc, testacc = -1, -1
            window_testauc, window_testacc = -1, -1
            if not save_model:
                if test_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_predictions.txt")
                    testauc, testacc = evaluate(model, test_loader, model.model_name, save_test_path)
                if test_window_loader != None:
                    save_test_path = os.path.join(ckpt_path, model.emb_type+"_test_window_predictions.txt")
                    window_testauc, window_testacc = evaluate(model, test_window_loader, model.model_name, save_test_path)
            validauc, validacc = auc, acc
        print(f"Epoch: {i}, validauc: {validauc:.4}, validacc: {validacc:.4}, best epoch: {best_epoch}, best auc: {max_auc:.4}, train loss: {loss_mean}, emb_type: {model.emb_type}, model: {model.model_name}, save_dir: {ckpt_path}")
        print(f"            testauc: {round(testauc,4)}, testacc: {round(testacc,4)}, window_testauc: {round(window_testauc,4)}, window_testacc: {round(window_testacc,4)}")


        if i - best_epoch >= 10:
            break
    return testauc, testacc, window_testauc, window_testacc, validauc, validacc, best_epoch
