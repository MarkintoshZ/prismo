from abc import ABC, abstractmethod
import asyncio
from typing import List, Dict, Type, Union
from datetime import datetime
from enum import Enum
from os import makedirs, path

from uuid import UUID, uuid4
from fastapi import UploadFile
from pydantic import BaseModel

from config import *


class TaskStatus(Enum):
    QUEUED = 0
    PREPROCESSING = 1
    TRAINING = 2
    RENDERING = 3
    DONE = 4
    FAILED = 5


class Task(BaseModel):
    id: str
    created_at: datetime
    status: TaskStatus
    error: Union[str, None]

    def upload_files(self, files: List[UploadFile]):
        images_dir = self.images_dir()
        if not path.isdir(images_dir):
            makedirs(images_dir, exist_ok=True)
        for i, file in enumerate(files):
            filename = f'image-{i}.jpg'
            with open(path.join(images_dir, filename), 'wb+') as f:
                f.write(file.file.read())

    def task_dir(self):
        return path.join(tasks_dir, str(self.id))

    def dataset_dir(self):
        return path.join(self.task_dir(), 'dataset')

    def images_dir(self):
        return path.join(self.dataset_dir(), 'images')

    def model_dir(self):
        return path.join(self.task_dir(), 'model')

    def log_file(self):
        return path.join(self.task_dir(), 'log.txt')

    def output_file(self):
        return path.join(self.task_dir(), 'output', 'render_output.mp4')

    @staticmethod
    def new():
        return Task(
            id=str(uuid4()),
            status=TaskStatus.QUEUED,
            created_at=datetime.now(),
            error=None)


class TaskManager:
    """Tracks current tasks and manage the task queue"""

    def __init__(self, execution_strategy: Type['ExecutionStrategy']):
        self.tasks: Dict[str, Task] = dict()
        self.queue: asyncio.Queue[Task] = asyncio.Queue(maxsize=10)
        self.execution_strategy = execution_strategy
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

    def get(self, uuid: UUID):
        """
        Get task by task id

        return None if task not found
        """
        task_id = str(uuid)
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
            print("running task")
            try:
                task.status = TaskStatus.PREPROCESSING
                await asyncio.wait_for( 
                        await self.execution_strategy.preprocess(task),
                        timeout=self.execution_strategy.preprocess_timeout)
                task.status = TaskStatus.TRAINING
                await asyncio.wait_for( 
                        await self.execution_strategy.train(task),
                        self.execution_strategy.train_timeout)
                task.status = TaskStatus.RENDERING
                await asyncio.wait_for( 
                        await self.execution_strategy.render(task),
                        self.execution_strategy.render_timeout)
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
            except:
                task.error = "Unknown"
                task.status = TaskStatus.FAILED


class ExecutionStrategy(ABC):
    @property
    @classmethod
    @abstractmethod
    def preprocess_timeout(cls) -> float:
        """Timeout for the preprocessing step in seconds"""
        raise NotImplementedError()

    @property
    @classmethod
    @abstractmethod
    def train_timeout(cls) -> float:
        """Timeout for the training step in seconds"""
        raise NotImplementedError()

    @property
    @classmethod
    @abstractmethod
    def render_timeout(cls) -> float:
        """Timeout for the rendering step in seconds"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def preprocess(task):
        """Run preprocessing on task"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def train(task):
        """Run training on task"""
        raise NotImplementedError()

    @staticmethod
    @abstractmethod
    async def render(task):
        """Run render on task"""
        raise NotImplementedError()


class NerfactoStrategy(ExecutionStrategy):
    @property
    @classmethod
    def preprocess_timeout(cls):
        return 60 * 20  # 20 minutes

    @property
    @classmethod
    def train_timeout(cls):
        return 60 * 20  # 20 minutes

    @property
    @classmethod
    def render_timeout(cls):
        return 60 * 10  # 10 minutes

    @staticmethod
    async def preprocess(task: Task):
        # TODO: redirect stdout and stderr to log file
        proc = await asyncio.create_subprocess_exec(
                "python3.8", 
                path.join(path.curdir, 'scripts', 'colmap2nerf.py'),
                f"--images {task.images_dir()}", 
                f"--out {task.dataset_dir()}", 
                "--colmap_matcher exhaustive", 
                "--run_colmap",
                "--aabb_scale 16")
        status = await proc.wait()
        if status != 0:
            raise RuntimeError()

    @staticmethod
    async def train(task: Task):
        proc = await asyncio.create_subprocess_exec(
                "ns-train", 
                "nerfacto"
                f"--data {task.dataset_dir()}",
                f"--output-dir {task.model_dir()}")
        status = await proc.wait()
        if status != 0:
            raise RuntimeError()

    @staticmethod
    async def render(task: Task):
        proc = await asyncio.create_subprocess_exec("echo", "preprocess")
        status = await proc.wait()
        if status != 0:
            raise RuntimeError()

