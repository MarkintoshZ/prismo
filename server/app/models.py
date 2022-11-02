from datetime import datetime
from enum import Enum
from os import mkdir, path
from typing import List
from uuid import UUID, uuid4

from fastapi import UploadFile
from pydantic import BaseModel

from config import *


class TaskStatus(Enum):
    QUEUED = 0
    RUNNING = 1
    DONE = 2
    FAILED = 3


class Task(BaseModel):
    id: UUID
    created_at: datetime
    status: TaskStatus
    files: List[str]

    async def upload_files(self, files: List[UploadFile]):
        task_dir = path.join(tasks_dir, str(self.id))
        if not path.isdir(tasks_dir):
            mkdir(tasks_dir)
        mkdir(task_dir)
        for i, file in enumerate(files):
            filename = f'picture-{i}.jpg'
            with open(path.join(task_dir, filename), 'wb+') as f:
                f.write(file.file.read())
            self.files.append(filename)

    @staticmethod
    def new():
        return Task(id=uuid4(), status=TaskStatus.QUEUED, files=[], created_at=datetime.now())
