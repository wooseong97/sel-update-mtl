from evaluation.evaluate_utils import PerformanceMeter
from tqdm import tqdm
from utils.utils import get_output, mkdir_if_missing
from evaluation.evaluate_utils import save_model_pred_for_one_task
import torch
import os


@torch.no_grad()
def test_phase(p, test_loader, model, criterion, epoch):
    tasks = p.TASKS.NAMES
    performance_meter = PerformanceMeter(p, tasks)
    model.eval()

    if 'tasks_to_save' in p.keys(): tasks_to_save = [key for key, value in p.tasks_to_save.items() if value is True]  
    else: tasks_to_save = []  
    save_dirs = {task: os.path.join(p['save_dir'], task) for task in tasks_to_save}
    for save_dir in save_dirs.values():
        mkdir_if_missing(save_dir)
    
    for i, batch in enumerate(tqdm(test_loader)):
        # Forward pass
        with torch.no_grad():
            images = batch['image'].cuda(non_blocking=True)
            targets = {task: batch[task].cuda(non_blocking=True) for task in tasks}

            if hasattr(model, 'module'): output = model.module(images)
            else: output = model(images)
        
            # Measure loss and performance
            performance_meter.update({t: get_output(output[t], t) for t in tasks}, 
                                    {t: targets[t] for t in tasks})

            for task in tasks_to_save:
                save_model_pred_for_one_task(p, batch, output, save_dirs, task, epoch=epoch)

    eval_results = performance_meter.get_score(verbose = True)

    return eval_results
