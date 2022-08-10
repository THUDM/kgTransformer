from pathlib import Path

import torch
from tqdm import tqdm


def ft_test(test_loader, num_nodes, relation_cnt, config, task_name='', quiet=False):
    device = torch.device(f'cuda:{torch.cuda.current_device()}')

    from model import D_KGTransformer
    model = D_KGTransformer(num_nodes, relation_cnt, config)
    model.to(device)

    load_path = Path(config['root_dir']) / 'chkpt' / f'{task_name}.pt'
    if not quiet:
        print("Testing " + load_path.as_posix())

    checkpoint = torch.load(load_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    from metric import PredictionMetrics
    met = PredictionMetrics()

    with torch.no_grad():
        for batch in tqdm(test_loader):
            graph = batch.to(device)
            x_pred, edge_pred, _ = model.answer_queries(graph)
            met.digest(x_pred, graph.x_ans,
                       weight=graph.x_pred_weight if hasattr(graph, 'x_pred_weight') else None)
            # met.digest(edge_pred, graph.edge_ans)
            # TODO: this is probably too harsh for the prediction since it requires the direction to be correct.
            # Sometimes the direction negligible and should not account for the accuracy.

    from datetime import datetime
    if not quiet:
        print("Current test time:", datetime.now())
        print("MRR:", met.MRR())
        print("Hit@1:", met.hits_at(1))
        print("Hit@3:", met.hits_at(3))

    return met.hits_at(3)


class TrainClient:
    def __init__(self, config, num_nodes, relation_cnt):
        self.num_nodes = num_nodes
        self.relation_cnt = relation_cnt
        import torch.multiprocessing as mp
        ctx = mp.get_context('spawn')
        self.report_loss = ctx.Value('d', 100.0)
        self.report_mrr = ctx.Value('d', 0)
        self.report_hit1 = ctx.Value('d', 0)
        self.report_hit3 = ctx.Value('d', 0)
        self.config = config
        self.first_write = ctx.Value('i', 1)

        if config.get('upstream_task_name') is not None:
            self.load_path = (Path(config['root_dir']) / 'chkpt' / config['upstream_task_name']).with_suffix('.pt')
        else:
            self.load_path = None
        self.save_path = (Path(config['root_dir']) / 'chkpt' / config['downstream_task_name']).with_suffix('.pt')
        self.backup_path = f'{config["root_dir"]}/chkpt/{config["downstream_task_name"]}'
        self.grad_accum = config['grad_accum']

        self.model = None
        self.optimizer = None

    def make_model(self, device):
        from model import D_KGTransformer, D_KGTransformerLoss
        self.model = D_KGTransformer(self.num_nodes, self.relation_cnt, self.config)
        self.model = self.model.to(device)
        return D_KGTransformerLoss(self.model)

    def save_state(self, epoch=None):
        if epoch is None:
            save_path = self.save_path
            if self.first_write.value == 1:
                self.first_write.value = 0
                if save_path.exists():
                    print(f'[INFO] Overwritting "{save_path}"')
        else:
            save_path = self.backup_path + str(epoch) + '.pt'
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)

    def load_state(self, rank, device, is_resume):
        from torch.distributed import barrier
        checkpoint = None
        if rank == 0 or rank is None:
            if self.load_path is not None and self.load_path.exists():
                barrier()
                checkpoint = torch.load(self.load_path, map_location=device)
            else:
                print(f'[WARN] Using random init since "{self.load_path}" does not exist')
                self.save_state()
                barrier()
        else:
            barrier()
            real_load_path = self.load_path if self.load_path is not None and self.load_path.exists() else self.save_path
            checkpoint = torch.load(real_load_path, map_location=device)

        if checkpoint is not None:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            if is_resume and 'optimizer_state_dict' in checkpoint:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f'#{rank}: State dict loaded')

    def ft_train(self, task_queue, is_resume=False, rank=None, world_size=None):
        r"""
        Work as a training pipeline:
        Load from upstream_task_name.pt, save to downstream_task_name.pt
        :param rank: id of the worker
        :param world_size: Total number of workers
        :return:
        """
        config = self.config
        from main import set_seed
        set_seed(config['seed'])

        if rank is not None:
            from torch.distributed import init_process_group
            init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )

        device = torch.device('cuda', rank)
        model = self.make_model(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

        lr = config['lr']
        weights = list(model.parameters())
        grads = [torch.zeros_like(t) for t in weights]
        accum_phase = 0

        if config['optimizer'] == 'AdamW':
            optimizer = torch.optim.AdamW(weights, lr=lr)
        elif config['optimizer'] == 'SGD':
            optimizer = torch.optim.SGD(weights, lr=lr, momentum=0.9)
        else:
            raise f'[ERROR] Unrecognized optimizer {config["optimizer"]}'
        self.optimizer = optimizer

        scheduler = None
        if config['scheduler'] == 'exp':
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, config['exponential_lr_rate'])

        self.load_state(rank, device, is_resume)

        ftest = False

        while True:
            task = task_queue.get()
            if task == 'save_state':
                self.save_state()
            elif isinstance(task, str) and task.startswith('save_state_'):
                epoch = int(task[len('save_state_'):])
                self.save_state(epoch)
            elif task == 'test':
                ftest = True
                model.eval()
                # start to gather test info
                from metric import PredictionMetrics
                met = PredictionMetrics()
                print('process', rank, 'is testing')
            elif task == 'collect_metric':
                # report the test result
                self.report_mrr.value = met.MRR()
                self.report_hit1.value = met.hits_at(1)
                self.report_hit3.value = met.hits_at(3)
                del met
            elif task == 'train':
                # prepare to train
                ftest = False
                model.train()
            elif task == 'step':
                if scheduler is not None:
                    scheduler.step()
            elif task == 'exit':
                from torch.distributed import barrier
                barrier()
                return
            else:
                if ftest:
                    with torch.no_grad():
                        graph = task.to(rank)
                        del task
                        x_pred, edge_pred, _ = self.model.answer_queries(graph)
                        met.digest(x_pred, graph.x_ans,
                                   weight=graph.x_pred_weight if hasattr(graph, 'x_pred_weight') else None)
                        del x_pred
                        del edge_pred
                        del graph
                        del _
                else:
                    optimizer.zero_grad()
                    graph = task.to(rank)
                    del task
                    loss, weight_sum = model(graph)
                    del graph
                    self.report_loss.value = (loss / weight_sum).item()
                    loss.backward()
                    del loss
                    del weight_sum
                    with torch.no_grad():
                        for g, t in zip(grads, weights):
                            g += t.grad
                    accum_phase += 1
                    if accum_phase == self.grad_accum:
                        with torch.no_grad():
                            for t, g in zip(weights, grads):
                                t.grad = g / self.grad_accum
                            grads = [torch.zeros_like(t) for t in weights]
                        optimizer.step()
                        accum_phase = 0
            task_queue.task_done()


def main_mp(config, num_nodes, relation_cnt, train_loader, test_loader=None, valid_loader=None):
    import torch.multiprocessing as mp

    print("Load from: ", config.get('upstream_task_name'))
    print("Save to: ", config.get('downstream_task_name'))

    buff_size = 3
    world_size = torch.cuda.device_count()
    print('Let\'s use', world_size, 'GPUs!')
    ctx = mp.get_context('spawn')
    work_q = [ctx.JoinableQueue(maxsize=buff_size) for i in range(world_size)]
    train_cli = TrainClient(config, num_nodes, relation_cnt)
    proc_pool = [ctx.Process(
        target=TrainClient.ft_train,
        args=(train_cli, work_q[i], config['is_resume'], i, world_size),
    ) for i in range(world_size)]
    for p in proc_pool:
        p.start()
    cur_pt = 0
    global_step = 0

    print('Entering training!')
    for i in range(world_size):
        work_q[i].put('train', block=True)

    batch_que = [None] * world_size
    best_result = 0
    for epoch in range(config['num_epoch']):
        if epoch % config['save_interval'] == 0 and epoch > 0:
            tqdm.write(f'It is the {epoch}-th epoch, saved')
            work_q[cur_pt].put(f'save_state_{epoch}', block=True)
        pbar = tqdm(train_loader)
        for batch_id, batch in enumerate(pbar):
            batch_que[cur_pt] = batch
            cur_pt += 1
            if cur_pt == world_size:
                cur_pt = 0
                pbar.set_postfix(loss=train_cli.report_loss.value, stat='enq')
                for i, v in enumerate(batch_que):
                    work_q[i].put(v, block=True)
                    batch_que[i] = None
                    del v
                pbar.set_postfix(loss=train_cli.report_loss.value, stat='gen')
            global_step += 1

        for que in work_q:
            que.put('step', block=True)

        def do_test(loader):
            work_q[cur_pt].put('test')
            for batch_id, batch in enumerate(loader):
                work_q[cur_pt].put(batch, block=True)
            work_q[cur_pt].put('collect_metric', block=True)
            work_q[cur_pt].join()
            work_q[cur_pt].put('train')

        if valid_loader is None:
            work_q[cur_pt].put('save_state', block=True)
        else:
            do_test(valid_loader)
            cur_val = train_cli.report_hit3.value
            if cur_val > best_result:
                tqdm.write(f'New best hit@3: {cur_val}')
                best_result = cur_val
                work_q[cur_pt].put('save_state', block=True)

        if epoch % config['test_interval'] == 0 and test_loader is not None:
            tqdm.write('Testing: ' + config.get('downstream_task_name'))
            do_test(test_loader)
            from datetime import datetime
            print("Current test time:", datetime.now())
            print("MRR:", train_cli.report_mrr.value)
            print("Hit@1:", train_cli.report_hit1.value)
            print("Hit@3:", train_cli.report_hit3.value)

    for que in work_q:
        que.put('exit')

    for p in proc_pool:
        p.join()
