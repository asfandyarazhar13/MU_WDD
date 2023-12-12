import copy
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import arg_parser
import evaluation
import pruner
import unlearn
import utils
from trainer import validate


def main():
    args = arg_parser.parse_args()

    if torch.cuda.is_available():
        torch.cuda.set_device(int(args.gpu))
        device = torch.device(f"cuda:{int(args.gpu)}")
    else:
        device = torch.device("cpu")

    os.makedirs(args.save_dir, exist_ok=True)
    if args.seed:
        utils.setup_seed(args.seed)
    seed = args.seed
    # prepare dataset
    (model, 
     val_loader, 
     test_loader, 
     marked_loader_forget, 
     marked_loader_retain,
     forget_eval_loader,
    ) = utils.setup_model_dataset(args)
    model.cuda()

    def replace_loader_dataset(
        dataset, batch_size=args.batch_size, seed=1, shuffle=True
    ):
        utils.setup_seed(seed)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=True,
            shuffle=shuffle,
        )
    # ADD custom_replace_loader_dataset()
    
    forget_dataset_distill = copy.deepcopy(marked_loader_forget.dataset)
    forget_dataset_whole = copy.deepcopy(forget_eval_loader.dataset)
    # forget_dataset = copy.deepcopy(marked_loader.dataset)

    # THE ABOVE DOESN'T WORK, WE NEED TO GET IMGS AND TARGETS INTO A LIST COMPLREHENSION BY ITERATING OVER THE LOADER
    # WHY TF IS THIS HAPPENING IN THE ARGS_DISTILL PORTION OF THE CODE??
    
    # forget_dataset_whole_targets = [targets for imgs, targets in forget_dataset_whole]
    # forget_dataset_whole_imgs = [imgs for imgs, targets in forget_dataset_whole]
    
    if args.dataset == "custom":
        # try:
        #     marked = forget_dataset.targets < 0
        #     forget_dataset.data = forget_dataset.data[marked]
        #     forget_dataset.targets = -forget_dataset.targets[marked] - 1
        #     forget_loader = replace_loader_dataset(
        #         forget_dataset, seed=seed, shuffle=True
        #     )
        #     print(len(forget_dataset))
        #     retain_dataset = copy.deepcopy(marked_loader.dataset)
        #     marked = retain_dataset.targets >= 0
        #     retain_dataset.data = retain_dataset.data[marked]
        #     retain_dataset.targets = retain_dataset.targets[marked]
        #     retain_loader = replace_loader_dataset(
        #         retain_dataset, seed=seed, shuffle=True
        #     )
        #     print(len(retain_dataset))
        #     print('hello yo Milky way')
        #     # assert len(forget_dataset) + len(retain_dataset) == len(
        #     #     train_loader_full.dataset
        #     # )
        # except:
        if args.all_distill:
            print('forget_dataset_distill:', forget_dataset_distill.targets)
            marked = forget_dataset_distill.targets == args.class_to_replace
            print('class to forget, index:', marked)
            forget_dataset_distill.imgs = forget_dataset_distill.imgs[marked]
            forget_dataset_distill.targets = forget_dataset_distill.targets[marked]
            forget_loader_distill = replace_loader_dataset(
                forget_dataset_distill, seed=seed, shuffle=True
            )
            print('forget_dataset_distill_len:', len(forget_dataset_distill))
            
            print('forget_dataset_whole:', forget_dataset_whole.targets[0])
            marked = forget_dataset_whole.targets == args.class_to_replace
            print('class to forget, index:', marked)
            forget_dataset_whole.imgs = forget_dataset_whole.imgs[marked]
            forget_dataset_whole.targets = forget_dataset_whole.targets[marked]            
            forget_loader_whole = replace_loader_dataset(
                forget_dataset_whole, seed=seed, shuffle=True
            )
            print('forget_dataset_whole_len:', len(forget_dataset_whole))
            
            retain_dataset = copy.deepcopy(marked_loader_retain.dataset)
            marked = retain_dataset.targets != args.class_to_replace
            retain_dataset.imgs = retain_dataset.imgs[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print('retain_dataset_len:', len(retain_dataset))  
        else:
            print('forget_dataset_distill:', forget_dataset_distill.targets)
            marked = forget_dataset_distill.targets == args.class_to_replace
            print('class to forget, index:', marked)
            forget_dataset_distill.imgs = forget_dataset_distill.imgs[marked]
            forget_dataset_distill.targets = forget_dataset_distill.targets[marked]
            forget_loader_distill = replace_loader_dataset(
                forget_dataset_distill, seed=seed, shuffle=True
            )
            print('forget_dataset_distill_len:', len(forget_dataset_distill))
            
            marked = forget_dataset_whole.targets < 0
            forget_dataset_whole.data = forget_dataset_whole.data[marked]
            forget_dataset_whole.targets = -forget_dataset_whole.targets[marked] - 1
            forget_loader = replace_loader_dataset(
                forget_dataset_whole, seed=seed, shuffle=True
            )
            print('forget_dataset_len:', len(forget_dataset_whole))
            retain_dataset = copy.deepcopy(marked_loader_retain.dataset)
            marked = retain_dataset.targets >= 0
            retain_dataset.data = retain_dataset.data[marked]
            retain_dataset.targets = retain_dataset.targets[marked]
            retain_loader = replace_loader_dataset(
                retain_dataset, seed=seed, shuffle=True
            )
            print('retain_dataset_len:', len(retain_dataset))
 
            # print('FIRST --> forget_dataset_whole_len:', len(forget_dataset_whole))
            # print('forget_dataset_targets:', forget_dataset_whole.targets)
            # print('MAX, MIN: ', max(forget_dataset_whole.targets), min(forget_dataset_whole.targets))
            # # marked = forget_dataset.targets < 0
            # # marked = args.class_to_replace
            # marked = forget_dataset_whole.targets == args.class_to_replace
            # print('class to forget, index:', marked)
            # forget_dataset_whole.data = forget_dataset_whole.data[marked]
            # # forget_dataset.targets = -forget_dataset.targets[marked] - 1
            # forget_dataset_whole.targets = forget_dataset_whole.targets[marked]
            # print(forget_dataset_whole.targets)
            # print('SECOND --> forget_dataset_whole_len:', len(forget_dataset_whole))
            # forget_loader = replace_loader_dataset(
            #     forget_dataset_whole, seed=seed, shuffle=True
            # )
            # print('forget_dataset_len:', len(forget_dataset))
            # retain_dataset = copy.deepcopy(marked_loader_retain.dataset)
            # # # marked = retain_dataset.targets >= 0
            # # marked = retain_dataset.targets != args.class_to_replace
            # # retain_dataset.data = retain_dataset.data[marked]
            # # retain_dataset.targets = retain_dataset.targets[marked]
            # # retain_loader = replace_loader_dataset(
            # #     retain_dataset, seed=seed, shuffle=True
            # # )
            # marked = retain_dataset.targets > 0
            # retain_dataset.data = retain_dataset.data[marked]
            # retain_dataset.targets = retain_dataset.targets[marked]
            # retain_loader = replace_loader_dataset(
            #     retain_dataset, seed=seed, shuffle=True
            # )
            # print('retain_dataset_len:', len(retain_dataset))
            
            # marked = retain_dataset.targets == 0
            # forget_eval_dataset.data = retain_dataset.data[marked]
            # forget_eval_dataset.targets = retain_dataset.targets[marked]
            # forget_eval_loader = replace_loader_dataset(
            #     forget_eval_dataset, seed=seed, shuffle=True
            # )
            # print('forget_eval_dataset_len:', len(forget_eval_dataset))                # assert len(forget_dataset) + len(retain_dataset) == len(
                #     train_loader_full.dataset
                # )
            
    unlearn_data_loaders = OrderedDict(
        retain=retain_loader, forget=forget_loader_distill, val=val_loader, test=test_loader, forget_eval=forget_loader
    )

    criterion = nn.CrossEntropyLoss()

    evaluation_result = None

    if args.resume:
        checkpoint = unlearn.load_unlearn_checkpoint(model, device, args)

    if args.resume and checkpoint is not None:
        model, evaluation_result = checkpoint
    else:
        checkpoint = torch.load(args.mask, map_location=device)
        if "state_dict" in checkpoint.keys():
            checkpoint = checkpoint["state_dict"]
        current_mask = pruner.extract_mask(checkpoint)
        pruner.prune_model_custom(model, current_mask)
        pruner.check_sparsity(model)

        if (
            args.unlearn != "retrain"
            and args.unlearn != "retrain_sam"
            and args.unlearn != "retrain_ls"
        ):
            model.load_state_dict(checkpoint, strict=False)

        unlearn_method = unlearn.get_unlearn_method(args.unlearn)
        print('UNLEARNING METHOD IS:', args.unlearn)

        unlearn_method(unlearn_data_loaders, model, criterion, args)
        unlearn.save_unlearn_checkpoint(model, None, args)

    if evaluation_result is None:
        evaluation_result = {}

    if "new_accuracy" not in evaluation_result:
        accuracy = {}
        for name, loader in unlearn_data_loaders.items():
            utils.dataset_convert_to_test(loader.dataset, args)
            val_acc = validate(loader, model, criterion, args)
            accuracy[name] = val_acc
            print(f"{name} acc: {val_acc}")

        evaluation_result["accuracy"] = accuracy
        unlearn.save_unlearn_checkpoint(model, evaluation_result, args)
            
if __name__ == "__main__":
    main()
