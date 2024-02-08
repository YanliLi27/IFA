from torch.nn import init
import numpy as np


def define_model_destroyer(model, topstack_layer):
    if topstack_layer > 1:
            for name, param in model.linear2.named_parameters():
                if name.startswith("weight"):
                    init.xavier_normal_(param)
                else:
                    init.zeros_(param)
    if topstack_layer > 2:
        for name, param in model.linear1.named_parameters():
            if name.startswith("weight"):
                init.xavier_normal_(param)
            else:
                init.zeros_(param)
    if topstack_layer > 3:
        try:
            for name, param in model.conv3.named_parameters():
                if name.startswith("weight"):
                    init.xavier_normal_(param)
                else:
                    init.zeros_(param)
        except:
            for name, param in model.conv2.named_parameters():
                if name.startswith("weight"):
                    init.xavier_normal_(param)
                else:
                    init.zeros_(param)
    if topstack_layer > 4:
        for name, param in model.conv1.named_parameters():
            if name.startswith("weight"):
                init.xavier_normal_(param)
            else:
                init.zeros_(param)         
    return model


def vgg_destroyer(model, topstack_layer):
    if topstack_layer > 0:    
        for name, param in model.classifier[-1].named_parameters():
            if name.startswith("weight"):
                init.xavier_normal_(param)
            else:
                init.zeros_(param)
    if topstack_layer > 1:
            for name, param in model.classifier[-2].named_parameters():
                if name.startswith("weight"):
                    init.xavier_normal_(param)
                else:
                    init.zeros_(param)
    if topstack_layer > 2:
            for name, param in model.classifier[-3].named_parameters():
                if name.startswith("weight"):
                    init.xavier_normal_(param)
                else:
                    init.zeros_(param)
    if topstack_layer > 3:
        for name, param in model.features[-3:].named_parameters():
            if name.startswith("weight"):
                init.xavier_normal_(param)
            # else:
            #     init.zeros_(param)
    if topstack_layer > 4:
        for name, param in model.features[-6:-3].named_parameters():
            if name.startswith("weight"):
                init.xavier_normal_(param)
            # else:
            #     init.zeros_(param)
    if topstack_layer > 5:
        for name, param in model.features.named_parameters():
            if name.startswith("weight"):
                init.xavier_normal_(param)
            # else:
            #     init.zeros_(param)         
    return model


def resnet_destroyer(model, topstack_layer):
    if topstack_layer > 0:    
        for name, param in model.fc.named_parameters():
            if name.startswith("weight"):
                init.xavier_normal_(param)
            else:
                init.zeros_(param)
    if topstack_layer > 1:
            for name, param in model.layer4[-4:].named_parameters():
                if name.startswith("weight"):
                    init.xavier_normal_(param)
                # else:
                #     init.zeros_(param)
    if topstack_layer > 2:
            for name, param in model.layer4.named_parameters():
                if name.startswith("weight"):
                    init.xavier_normal_(param)
                # else:
                #     init.zeros_(param)
    if topstack_layer > 3:
        for name, param in model.layer2.named_parameters():
            if name.startswith("weight"):
                init.xavier_normal_(param)
            # else:
            #     init.zeros_(param)
    if topstack_layer > 4:
        for name, param in model.layer1.named_parameters():
            if name.startswith("weight"):
                init.xavier_normal_(param)
            # else:
            #     init.zeros_(param)      
    return model



def model_destroyer(model, model_flag, topstack_layer:int=1):
    if model_flag=='scratch':
        model = define_model_destroyer(model, topstack_layer)
    elif model_flag=='vgg':
        model = vgg_destroyer(model, topstack_layer)
    elif model_flag=='resnet':
        model = resnet_destroyer(model, topstack_layer)
    else:
        raise ValueError('Not valid model flag')
    return model
    
        