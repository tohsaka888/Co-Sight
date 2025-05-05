#!/usr/bin/env python
# coding=utf-8
import os
import json
import shutil

"""
采集中间执行结果和日志的自动化脚本：
执行脚本，work_space_path需要修改为自己的目录，原先的日志是ascii编码的直接操作中文会乱码，
需要新建一个utf-8编码的新文件然后将原先的日志内容放入，查看不乱码即可，
脚本中默认文件名为a.txt,操作的文件夹目录见截图;
"""

def process_workspace(workspace_path):
    # 定义文件路径
    result_file = os.path.join(workspace_path, 'result_level1_20250505071544.json')
    eval_log_file = os.path.join(workspace_path, 'a.txt')
    
    # 读取结果文件
    with open(result_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    # 处理结果文件，找出成功的任务
    successful_tasks = set()
    for detail in results.get('detail', {}):
        if detail.get('score') == 1:
            successful_tasks.add(detail.get('task_id'))
    print(f"successful_tasks:{successful_tasks}, len:{len(successful_tasks)}")
    # 处理日志文件
    task_logs = {}
    current_task_id = None
    current_log = []
    
    with open(eval_log_file, 'r', encoding='utf-8') as f:
        for line in f:
            if 'start task' in line:
                # 保存上一个任务的日志
                if current_task_id is not None:
                    task_logs[current_task_id] = current_log
                
                # 开始新任务
                parts = line.strip().split('start task')
                current_task_id = parts[1].strip()
                current_log = [line]
            else:
                if current_task_id is not None:
                    current_log.append(line)
    
        # 保存最后一个任务的日志
        if current_task_id is not None:
            task_logs[current_task_id] = current_log
    print(f"task_logs:{task_logs.keys()}")
    # 创建任务日志文件并清理失败的任务文件夹
    for task_id, logs in task_logs.items():
        task_folder = os.path.join(workspace_path, task_id)
        
        if task_id in successful_tasks:
            # 创建日志文件
            log_file = os.path.join(task_folder, f'{task_id}.log')
            with open(log_file, 'w', encoding='utf-8') as f:
                f.writelines(logs)
        else:
            # 删除失败的任务文件夹
            if os.path.exists(task_folder):
                shutil.rmtree(task_folder)
                print(f'Deleted failed task folder: {task_folder}')

if __name__ == '__main__':
    work_space_path = r'F:\project\agent\Co-Sight\work_space\level1_20250504_225938\20250504_225938'  # 修改为实际的workspace路径
    process_workspace(work_space_path)