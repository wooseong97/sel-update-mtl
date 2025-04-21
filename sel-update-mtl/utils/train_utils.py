import os, json
from evaluation.evaluate_utils import PerformanceMeter
from utils.utils import to_cuda
import torch
from tqdm import tqdm
from utils.test_utils import test_phase

from sklearn.cluster import AffinityPropagation

import random
from util_taskonomy.engine_mt import evaluate


def update_tb(tb_writer, tag, loss_dict, iter_no):
    for k, v in loss_dict.items():
        tb_writer.add_scalar(f'{tag}/{k}', v.item(), iter_no)
        
def update_affinity(tb_writer, tag, affinity_map, iter_no, tasks):
    for i, t1 in enumerate(tasks):
        for j, t2 in enumerate(tasks):
            tb_writer.add_scalar(f'{tag}/{t1}_to_{t2}', affinity_map[i,j], iter_no)
            

def train_phase(p, args, train_loader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer, tb_writer_test, iter_count):
    """ Vanilla training with fixed loss weights """
    model.train()

    for i, cpu_batch in enumerate(tqdm(train_loader)):
        # Forward pass
        batch = to_cuda(cpu_batch)
        
        if p['train_db_name']=='Taskonomy': images = batch['rgb'] 
        else: images = batch['image']
        
        output = model(images)
        iter_count += 1
        
        # Measure loss
        loss_dict = criterion(output, batch, tasks=p.TASKS.NAMES)

        if tb_writer is not None:
            update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)
        
        # Backward
        optimizer.zero_grad()
        loss_dict['total'].backward()
        optimizer.step()
        scheduler.step()
        
        # end condition
        if iter_count >= p.max_iter:
            print('Max itereaction achieved.')
            # return True, iter_count
            end_signal = True
        else:
            end_signal = False

        # Evaluate
        if end_signal:
            eval_bool = True
        elif iter_count % p.val_interval == 0:
            eval_bool = True 
        else:
            eval_bool = False

        # Perform evaluation
        if eval_bool and args.local_rank == 0:
            print('Evaluate at iter {}'.format(iter_count))
            
            if p['train_db_name']=='Taskonomy':
                test_stats = evaluate(test_dataloader, model, 0, None, p)
                curr_result = test_stats
            else:
                curr_result = test_phase(p, test_dataloader, model, criterion, iter_count)
            tb_update_perf(p, tb_writer_test, curr_result, iter_count)
            print('Evaluate results at iteration {}: \n'.format(iter_count))
            print(curr_result)
            with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
                json.dump(curr_result, f, indent=4)

            # Checkpoint after evaluation
            print('Checkpoint starts at iter {}....'.format(iter_count))
            torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(), 
                        'epoch': epoch, 'iter_count': iter_count-1}, p['checkpoint'])
            print('Checkpoint finishs.')
            model.train() # set model back to train status

        if end_signal:
            return True, iter_count

    return False, iter_count


def eval_phase(p, args, train_loader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer, tb_writer_test, iter_count):
    """ Vanilla training with fixed loss weights """
    # Perform evaluation
    if args.local_rank == 0:
        print('Evaluate at iter {}'.format(iter_count))
        curr_result = test_phase(p, test_dataloader, model, criterion, iter_count)
        tb_update_perf(p, tb_writer_test, curr_result, iter_count)
        print('Evaluate results at iteration {}: \n'.format(iter_count))
        print(curr_result)
        with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
            json.dump(curr_result, f, indent=4)

        # Checkpoint after evaluation
        print('Checkpoint starts at iter {}....'.format(iter_count))
        torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch, 'iter_count': iter_count-1}, p['checkpoint'])
        print('Checkpoint finishs.')
        model.train() # set model back to train status

    return False, iter_count



def train_phase_affinity(p, args, train_loader, test_dataloader, model, criterion, optimizer, scheduler, epoch, tb_writer, tb_writer_test, iter_count, task_affinity):
    """ Vanilla training with fixed loss weights """
    model.train() 

    for i, cpu_batch in enumerate(tqdm(train_loader)):
        if iter_count <= p['prepos']: train_group = list(task_affinity.group.keys()); random.shuffle(train_group)
        else:   train_group = task_affinity.next_group()
        criterion.group = task_affinity.group
        iter_count += 1
        
        task_affinity.init_pre_loss()
        if len(task_affinity.group) ==1: train_group = train_group*2
        
        for group in train_group:
            # Forward pass
            batch = to_cuda(cpu_batch)
            
            if p['train_db_name']=='Taskonomy': images = batch['rgb'] 
            else: images = batch['image'] 
            
            output = model(images)
            
            # Measure loss
            loss_dict = criterion(output, batch, tasks=p.TASKS.NAMES)
        
            # Backward
            optimizer.zero_grad()
            if 'group_norm' in p.keys() and p['group_norm']: 
                group_loss = loss_dict[group]/len(task_affinity.group[group])
                group_loss.backward()
            else:
                loss_dict[group].backward()

            optimizer.step()
            task_affinity.update(group, loss_dict)
        
        scheduler.step()
            
        if tb_writer is not None:
            update_tb(tb_writer, 'Train_Loss', loss_dict, iter_count)
            update_affinity(tb_writer, 'Affinity', task_affinity.affinity_map, iter_count, p.TASKS.NAMES)
        
        # end condition
        if iter_count >= p.max_iter:
            print('Max itereaction achieved.')
            end_signal = True
        else:
            end_signal = False

        # Evaluate
        if end_signal:
            eval_bool = True
        elif iter_count % p.val_interval == 0:
            eval_bool = True 
        else:
            eval_bool = False

        # Perform evaluation
        if eval_bool and args.local_rank == 0:
            print('Evaluate at iter {}'.format(iter_count))
            
            if p['train_db_name']=='Taskonomy':
                test_stats = evaluate(test_dataloader, model, 0, None, p)
                curr_result = test_stats
            else:
                curr_result = test_phase(p, test_dataloader, model, criterion, iter_count)
            tb_update_perf(p, tb_writer_test, curr_result, iter_count)
            print('Evaluate results at iteration {}: \n'.format(iter_count))
            print(curr_result)
            with open(os.path.join(p['save_dir'], p.version_name + '_' + str(iter_count) + '.txt'), 'w') as f:
                json.dump(curr_result, f, indent=4)

            # Checkpoint after evaluation
            print('Checkpoint starts at iter {}....'.format(iter_count))
            torch.save({'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'model': model.state_dict(), 
                    'epoch': epoch, 'iter_count': iter_count-1}, p['checkpoint'])
            print('Checkpoint finishs.')
            model.train()

        if end_signal:
            return True, iter_count

    return False, iter_count



class form_task_affinity():
    def __init__(self, p):
        self.tasks = p.TASKS.NAMES
        self.group = p.group
        self.affin_decay = p.affin_decay
        if p.preference == 'None': self.preference = None
        else: self.preference=p.preference
        self.affinity_map = torch.zeros(len(self.tasks), len(self.tasks))
        self.pre_loss = {task: -1 for task in p.TASKS.NAMES}
        self.num=0
        self.convergence_iter = 50
    
    def update(self, group, loss_dict):
        for task_s in self.group[group]:
            for task_t in self.tasks:
                if self.pre_loss[task_t]<=0: continue
                if task_t in self.group[group]:
                    
                    if self.pre_loss[task_s]<=0: continue
                    affin_t = 1 - loss_dict[task_t].item()/self.pre_loss[task_t]
                    affin_t /= len(self.group[group])
                    affin_s = 1 - loss_dict[task_s].item()/self.pre_loss[task_s]
                    affin_s /= len(self.group[group])
                    
                    if task_t==task_s:
                        if affin_t < 0: pass
                        else: self.affin_update(affin_t, task_s, task_t)
                    elif affin_t * affin_s >=0:
                        self.affin_update(affin_t, task_s, task_t)
                    else:
                        self.affin_update(-max(affin_t, affin_s), task_s, task_t)
                    
                else:
                    affin = 1 - loss_dict[task_t].item()/self.pre_loss[task_t]
                    affin /= len(self.group[group])
                    self.affin_update(affin, task_s, task_t)
        
        for task in self.tasks: self.pre_loss[task] = loss_dict[task].item()
                
    def affin_update(self, affin, task_s, task_t):
        task_s_i, task_t_i = self.tasks.index(task_s), self.tasks.index(task_t)
        self.affinity_map[task_s_i, task_t_i] = (1-self.affin_decay)*self.affinity_map[task_s_i, task_t_i] + self.affin_decay*affin
        
    def init_pre_loss(self):
        for task in self.tasks: self.pre_loss[task]=-1
        
    def next_group(self):
        convergence_iter = self.convergence_iter
        
        X = self.affinity_map.clone()
        for i in range(len(X)): X[:,i] /= X[i,i]
        X = (X + X.T)/2
        
        for _ in range(10):
            cluster = AffinityPropagation(preference=self.preference, affinity='precomputed', convergence_iter=convergence_iter)
            cls = cluster.fit_predict(X)
            cluster_centers_indices = cluster.cluster_centers_indices_
            labels = cluster.labels_
            n_clusters = len(cluster_centers_indices)
            
            res={}
            for i, center in enumerate(cluster_centers_indices): 
                res['group%d'%(i+1)] = [task for j, task in enumerate(self.tasks) if labels[j]==i]
            
            if len(res) == 0: convergence_iter += 100
            else: break
        
        if len(res) != 0: self.group = res 
        
        train_group = [*self.group.keys()]
        random.shuffle(train_group)
        return train_group


#######################################################################
class PolynomialLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iterations, gamma=0.9, min_lr=0., last_epoch=-1):
        self.max_iterations = max_iterations
        self.gamma = gamma
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        # slight abuse: last_epoch refers to last iteration
        factor = (1 - self.last_epoch /
                  float(self.max_iterations)) ** self.gamma
        return [(base_lr - self.min_lr) * factor + self.min_lr for base_lr in self.base_lrs]

def tb_update_perf(p, tb_writer_test, curr_result, cur_iter):
    if 'semseg' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/semseg_miou', curr_result['semseg']['mIoU'], cur_iter)
    if 'human_parts' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/human_parts_mIoU', curr_result['human_parts']['mIoU'], cur_iter)
    if 'sal' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/sal_maxF', curr_result['sal']['maxF'], cur_iter)
    if 'edge' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/edge_val_loss', curr_result['edge']['loss'], cur_iter)
    if 'normals' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/normals_mean', curr_result['normals']['mean'], cur_iter)
    if 'depth' in p.TASKS.NAMES:
        tb_writer_test.add_scalar('perf/depth_rmse', curr_result['depth']['rmse'], cur_iter)
