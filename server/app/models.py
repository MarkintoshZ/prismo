from typing import List, Union
from datetime import datetime
from enum import Enum
from os import makedirs, path, system

from uuid import uuid4
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


class ExecutionStrategy(str, Enum):
    nerfacto = "nerfacto"
    instant_ngp = "instant-ngp"
    vanilla_nerf = "vanilla-nerf"


class Task(BaseModel):
    id: str
    created_at: datetime
    status: TaskStatus
    error: Union[str, None]
    execution_strategy: ExecutionStrategy

    def upload_images(self, files: List[UploadFile]):
        images_dir = self.images_dir()
        makedirs(images_dir, exist_ok=True)
        for i, file in enumerate(files):
            filename = f'image-{i}.jpg'
            with open(path.join(images_dir, filename), 'wb+') as f:
                f.write(file.file.read())

    def upload_video(self, video: UploadFile):
        images_dir = self.images_dir()
        makedirs(images_dir, exist_ok=True)
        with open(self.input_video_file(), 'wb+') as f:
            f.write(video.file.read())
        system(f"ffmpeg -i {self.input_video_file()} -qscale:v 1 -qmin 1 -vf \"fps=2\" {self.images_dir()}/%04d.jpg")

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

    def transforms_file(self):
        return path.join(self.dataset_dir(), 'transforms.json')

    def output_video_file(self):
        return path.join(self.task_dir(), 'output', 'render_output.mp4')

    def input_video_file(self):
        return path.join(self.dataset_dir(), 'input_video.mp4')

    @staticmethod
    def new(execution_strategy: str):
        return Task(
            id=str(uuid4()),
            status=TaskStatus.QUEUED,
            created_at=datetime.now(),
            error=None,
            execution_strategy=execution_strategy)

