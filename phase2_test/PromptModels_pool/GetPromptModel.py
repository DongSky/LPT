"""
build_promptmodel script  ver： Mar 25th 19:20

"""
import timm
import torch
from .structure import *


def build_promptmodel(num_classes=2, img_size=224, model_idx='ViT', patch_size=16, base_model='vit_base_patch16_224_in21k',
                      Prompt_Token_num=10, VPT_type="Deep", pool_size=20):
    # VPT_type = "Deep" / "Shallow"

    if model_idx[0:3] == 'ViT':
        # ViT_Prompt
        import timm

        basic_model = timm.create_model(base_model,
                                        pretrained=True)
        base_state_dict = basic_model.state_dict()
        del base_state_dict['head.weight']
        del base_state_dict['head.bias']
        model = VPT_ViT(num_classes=num_classes, img_size=img_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                        VPT_type=VPT_type, pool_size=pool_size)

        model.load_state_dict(base_state_dict, False)
        #model.New_CLS_head(num_classes)
        model.Freeze()
    else:
        print("The model is not difined in the Prompt script！！")
        return -1

    #try:
    #    img = torch.randn(1, 3, img_size, img_size)
    #    preds = model(img)  # (1, class_number)
    #    print('test model output：', preds)
    #except:
    #    print("Problem exist in the model defining process！！")
    #    return -1
    #else:
    #    print('model is ready now!')
    return model
