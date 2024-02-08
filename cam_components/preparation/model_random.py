from torch.nn import init


def model_randomizer(model, severity:float=0):
    module_list = []
    for name, module in model.named_modules():
        module_list.append(module)
    length_module = len(module_list) * severity
    select_module_list = module_list[int(-length_module):]
    if len(select_module_list)>0:
        for select_module in select_module_list:
            for name, param in select_module.named_parameters():
                if name.startswith("weight"):
                    init.xavier_normal_(param)
                else:
                    init.zeros_(param)
    else:
        print('no module randomized')
    return model
    
        