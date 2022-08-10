import torch


def put_default_config(config):
    def set_default(key, value):
        if key not in config:
            config[key] = value

    set_default('master_addr', '127.0.0.1')
    import random
    set_default('master_port', random.randint(30000, 40000))  # This random goes before the pseudo random
    set_default('seed', 100)
    set_default('root_dir', '/home/share/KGTransformer/data')
    set_default('data_dir', f'{config["root_dir"]}/FB15k-237-q2b')
    # Training related
    set_default('num_epoch', 100)
    set_default('optimizer', 'AdamW')
    set_default('cache_ttl', 5)
    set_default('is_resume', False)
    set_default('upstream_task_name', None)
    set_default('downstream_task_name', 'default')
    set_default('num_workers', 10)
    set_default('batch_size', 64)
    set_default('epoch_choosing', False)
    set_default('grad_accum', 1)
    set_default('pre_norm', True)
    set_default('save_interval', 50)
    set_default('test_interval', 20)
    # Hyper parameters
    set_default('hidden_size', 1024)
    set_default('num_heads', 8)
    set_default('num_layers', 6)
    set_default('dim_feedforward', 2048)
    set_default('lr', 1e-4)
    set_default('scheduler', 'exp')
    set_default('exponential_lr_rate', 0.99)
    set_default('loss', 'CE')
    set_default('smoothing', 0)
    set_default('eta_min', 0)
    set_default('mask_ratio', 0.8)
    set_default('p_mask_ratio', 1.0)
    set_default('dropout', 0.1)
    set_default('attention_dropout', 0.1)
    # Pretrain sampling
    set_default('pretrain_mask_ratio', [0.2, 0.4])  # BERT [mask] token ratio range
    set_default('pretrain_mask_type_ratio', [1, 0])  # Ratio of entity : relation
    set_default('pretrain_dataset_source', 'relation')  # 'relation' or 'entity'
    set_default('edge_drop_out_rate', 0)
    set_default('sample_retries', 5)
    set_default('ladies_size', 8)
    set_default('pretrain_sampler_ratio', {
        '1p': 0,
        '2p': 0,
        '3p': 0,
        '2i': 0,
        '3i': 0,
        'meta_tree': 5,
        'ladies': 5,
    })
    set_default('induced_edge_prob', 0.8)
    # set moe
    set_default('moe', True)
    set_default('moe_num_expert', 4)
    set_default('moe_top_k', 2)
    # reasoning options
    set_default('reasoning_train_modes', ['1p', '2p', '3p', '2i', '3i'])
    set_default('reasoning_test_modes', ['1p', '2p', '3p', '2i', '3i', 'ip', 'pi', '2u', 'up'])
    # Epoch choosing
    set_default('from_best', False)
    set_default('save_best', False)
    return config


def run_pretrain(config):
    from tasks.betae_pretrain import FB15K237
    task = FB15K237(config)

    from train import main_mp
    main_mp(
        config,
        task.num_nodes,
        task.relation_cnt,
        task.dataloader_train(config),
        task.dataloader_test(),
    )


def run_reasoning(config):
    from tasks.betae_pretrain import FB15K237
    pretrain_task = FB15K237(config)
    from tasks.fb15k237_reasoning import FB15K237_reasoning
    downstream_task = FB15K237_reasoning(
        pretrain_task.betae,
        relation_cnt=pretrain_task.relation_cnt,
        train_mode=config['reasoning_train_modes'],
        test_mode=config['reasoning_test_modes'][:1],  # 2u up does not support mixing
    )
    if config['from_best']:
        best_input = config['upstream_task_name']
        best_result = 0
        valid_dl = downstream_task.dataloader_valid(config)
        for i in range(1, 10000000):
            from pathlib import Path
            task_name = f'{config["upstream_task_name"]}{i}'
            load_path = Path(config['root_dir']) / 'chkpt' / f'{task_name}.pt'
            if not load_path.exists():
                break
            from train import ft_test
            cur_val = ft_test(
                valid_dl,
                num_nodes=pretrain_task.num_nodes,
                relation_cnt=pretrain_task.relation_cnt,
                config=config,
                task_name=task_name,
                quiet=True,
            )
            if cur_val > best_result:
                best_input = task_name
                best_result = cur_val
        del valid_dl
        config['upstream_task_name'] = best_input
        torch.cuda.empty_cache()

    # Finetune training
    print('Finetune training time!')
    from train import main_mp
    test_modes = config['reasoning_test_modes']
    if test_modes is None or test_modes == []:
        test_loader = None
    else:
        test_loader = downstream_task.dataloader_test(config)
    main_mp(
        config,
        pretrain_task.num_nodes,
        pretrain_task.relation_cnt,
        downstream_task.dataloader_fine_tune(config),
        test_loader,
        valid_loader=downstream_task.dataloader_valid(config) if config['save_best'] else None,
    )

    del pretrain_task
    del downstream_task
    if test_loader is not None:
        run_test_reasoning(config)


def run_test_reasoning(config):
    from tasks.betae_pretrain import FB15K237
    from tasks.fb15k237_reasoning import FB15K237_reasoning
    pretrain_task = FB15K237(config)
    test_modes = config['reasoning_test_modes']

    from train import ft_test
    for mode in test_modes:
        print('Testing mode', mode)
        downstream_task = FB15K237_reasoning(
            pretrain_task.betae,
            relation_cnt=pretrain_task.relation_cnt,
            train_mode=[],
            test_mode=[mode],
        )
        best_input = config['downstream_task_name']
        if config['from_best']:
            best_result = 0
            valid_dl = downstream_task.dataloader_valid(config)
            for i in range(1, 10000000):
                from pathlib import Path
                task_name = f'{config["downstream_task_name"]}{i}'
                load_path = Path(config['root_dir']) / 'chkpt' / f'{task_name}.pt'
                if not load_path.exists():
                    break
                from train import ft_test
                cur_val = ft_test(
                    valid_dl,
                    num_nodes=pretrain_task.num_nodes,
                    relation_cnt=pretrain_task.relation_cnt,
                    config=config,
                    task_name=task_name,
                    quiet=True,
                )
                if cur_val > best_result:
                    best_input = task_name
                    best_result = cur_val
            del valid_dl
        ft_test(
            downstream_task.dataloader_test(config),
            num_nodes=pretrain_task.num_nodes,
            relation_cnt=pretrain_task.relation_cnt,
            config=config,
            task_name=best_input,
        )


def get_argparser():
    import argparse

    parser = argparse.ArgumentParser(prog='python main.py', description='KGTransformer')
    parser.add_argument('-c', '--config', nargs=1, type=argparse.FileType('r'), help='path to the config file')
    parser.add_argument('tasks', metavar='<task>', nargs='+', type=str, help='tasks from the config to run')

    return parser


def dfs_parsing(config_list, parse_status, task):
    stat = parse_status.get(task)
    if stat == 'Done':
        return
    if stat == 'Parsing':
        assert False, f'Loop detected in config.'
    parse_status[task] = 'Parsing'
    if task not in config_list:
        assert False, f'Task {task} not found'
    config = config_list[task]
    if 'base' in config:
        dfs_parsing(config_list, parse_status, config['base'])
        config_base = config_list[config['base']]
        del config['base']
        for k in config_base:
            if k not in config:
                config[k] = config_base[k]
    put_default_config(config)
    parse_status[task] = 'Done'


def args_to_config(args):
    import json
    config_list = json.load(args.config[0])
    assert isinstance(config_list, dict), "Config should be an dict of tasks."
    parse_status = dict()
    for task in config_list:
        dfs_parsing(config_list, parse_status, task)
    return config_list


def set_seed(seed: int):
    """ Set RNG seeds for python's `random` module, numpy and torch"""
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    print('Running KGTransformer')
    args = get_argparser().parse_args()
    print('Arguments:', args)
    config_list = args_to_config(args)
    for t in args.tasks:
        if t not in config_list:
            assert False, f'Task {t} not found in config'

    import torch.multiprocessing as mp
    mp.set_sharing_strategy('file_system')
    torch.set_printoptions(profile='full')

    for t in args.tasks:
        config = config_list[t]
        print(f'Running task "{t}". Definitive config:')
        import json
        print(json.dumps(config))
        set_seed(config['seed'])
        # Environments
        import os
        os.environ['MASTER_ADDR'] = config['master_addr']
        os.environ['MASTER_PORT'] = str(config['master_port'])

        torch.cuda.empty_cache()
        if config['type'] == 'pretrain':
            run_pretrain(config)
        elif config['type'] == 'reasoning':
            run_reasoning(config)
        elif config['type'] == 'test-reasoning':
            run_test_reasoning(config)
        else:
            assert False, f'Task {t} is not runnable.'


if __name__ == '__main__':
    main()
