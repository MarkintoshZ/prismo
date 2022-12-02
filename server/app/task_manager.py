import asyncio
from typing import Dict

import traceback

from config import *
from execution_strategy import NerfactoStrategy, InstantNgpStrategy, VanillaNerfStrategy
from models import Task, TaskStatus, ExecutionStrategy


class TaskManager:
    """Keep track of current tasks and process the tasks in the queue"""

    def __init__(self):
        self.tasks: Dict[str, Task] = dict()
        self.queue: asyncio.Queue[Task] = asyncio.Queue(maxsize=10)
        asyncio.get_event_loop()
        self.exec_task = asyncio.create_task(self._execute())

    def add(self, task: Task) -> bool:
        """
        Add new task to the task queue to be executed

        return True on success False on failure (saturated task queue)
        """
        self.tasks[task.id] = task
        try:
            self.queue.put_nowait(task)
            return True
        except asyncio.QueueFull:
            return False

    def get(self, task_id: str):
        """
        Get task by task id

        return None if task not found
        """
        if task_id in self.tasks:
            return self.tasks[task_id]
        return None

    def list(self):
        """Return all tasks"""
        return list(self.tasks.values())

    async def _execute(self):
        """Process tasks added to the task queue"""
        while True:
            task = await self.queue.get()
            if task.execution_strategy == ExecutionStrategy.nerfacto:
                execution_strategy = NerfactoStrategy()
            elif task.execution_strategy == ExecutionStrategy.instant_ngp:
                execution_strategy = InstantNgpStrategy()
            elif task.execution_strategy == ExecutionStrategy.vanilla_nerf:
                execution_strategy = VanillaNerfStrategy()
            else:
                execution_strategy = NerfactoStrategy()

            print(f"running task: {task}")
            try:
                task.status = TaskStatus.PREPROCESSING
                await asyncio.wait_for(
                        execution_strategy.preprocess(task),
                        timeout=execution_strategy.preprocess_timeout())
                task.status = TaskStatus.TRAINING
                await asyncio.wait_for(
                        execution_strategy.train(task),
                        timeout=execution_strategy.train_timeout())
                task.status = TaskStatus.RENDERING
                await asyncio.wait_for(
                        execution_strategy.render(task),
                        timeout=execution_strategy.render_timeout())
                task.status = TaskStatus.DONE
                print("done running task")
            except asyncio.TimeoutError:
                task.error = "Timeout on "
                if task.status == TaskStatus.PREPROCESSING:
                    task.error += "preprocessing"
                elif task.status == TaskStatus.TRAINING:
                    task.error += "training"
                elif task.status == TaskStatus.RENDERING:
                    task.error += "rendering"
                task.status = TaskStatus.FAILED
            except Exception as e:
                print(traceback.print_exc())
                task.error = str(e)
                task.status = TaskStatus.FAILED
