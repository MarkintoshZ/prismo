from abc import ABC, abstractmethod
import asyncio
from typing import Union
import os
from os import makedirs, path

from models import Task
from config import *


class ExecutionStrategy(ABC):
    @classmethod
    @abstractmethod
    def preprocess_timeout(cls) -> float:
        """Timeout for the preprocessing step in seconds"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def train_timeout(cls) -> float:
        """Timeout for the training step in seconds"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def render_timeout(cls) -> float:
        """Timeout for the rendering step in seconds"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    async def preprocess(task: Task):
        """Run preprocessing on task"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    async def train(task: Task):
        """Run training on task"""
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    async def render(task: Task):
        """Run render on task"""
        raise NotImplementedError()

    @staticmethod
    async def exec_program(task: Task, program: str,
                           *args: str, cwd: Union[str, None]=None):
        """
        Execute program and log the output

        throw RuntimeException on non-zero return code
        """
        log_file = task.log_file()
        makedirs(path.dirname(log_file), exist_ok=True)
        print(f'Executing "{" ".join([program, *args])}"')
        # return
        with open(log_file, "a+", encoding="utf8") as f:
            proc = await asyncio.create_subprocess_exec(
                program, *args, stderr=f, stdout=f, cwd=cwd)
        status = await proc.wait()
        if status != 0:
            err_msg = f'Program "{" ".join([program, *args])}" exit with status code {status}'
            raise RuntimeError(err_msg)


class NerfactoStrategy(ExecutionStrategy):
    @classmethod
    def preprocess_timeout(cls):
        return 60 * 20  # 20 minutes

    @classmethod
    def train_timeout(cls):
        return 60 * 25  # 25 minutes

    @classmethod
    def render_timeout(cls):
        return 60 * 5  # 5 minutes

    @classmethod
    async def preprocess(cls, task: Task):
        args = f"""
            {colmap_script}
            --run_colmap
            --colmap_matcher exhaustive
            --aabb_scale 16
            """
        await cls.exec_program(task, "python3.8", *args.split(),
                               cwd=task.dataset_dir())

    @classmethod
    async def train(cls, task: Task):
        args = f"""
            nerfacto
            --data {task.dataset_dir()}
            --output-dir {task.model_dir()}
            """
        await cls.exec_program(task, "ns-train", *args.split())

    @classmethod
    async def render(cls, task: Task):
        config_file = None
        for dirpath, _dirnames, filenames in os.walk(task.model_dir()):
            for filename in [f for f in filenames if f.endswith(".yml")]:
                config_file = os.path.join(dirpath, filename)
                break
            if config_file:
                break
        if not config_file:
            raise RuntimeError("Cannot find config.yml file for ns-render")
        args = f"""
            --load-config {config_file}
            --traj spiral
            --output-path {task.output_video_file()}
            """
        await cls.exec_program(task, "ns-render", *args.split())


class InstantNgpStrategy(ExecutionStrategy):
    @classmethod
    def preprocess_timeout(cls):
        return 60 * 20  # 20 minutes

    @classmethod
    def train_timeout(cls):
        return 60 * 10  # 10 minutes

    @classmethod
    def render_timeout(cls):
        return 60 * 5  # 5 minutes

    @classmethod
    async def preprocess(cls, task: Task):
        args = f"""
            {colmap_script}
            --run_colmap
            --colmap_matcher exhaustive
            --aabb_scale 16
            """
        await cls.exec_program(task, "python3.8", *args.split(),
                               cwd=task.dataset_dir())

    @classmethod
    async def train(cls, task: Task):
        args = f"""
            instant-ngp
            --data {task.dataset_dir()}
            --output-dir {task.model_dir()}
            """
        await cls.exec_program(task, "ns-train", *args.split())

    @classmethod
    async def render(cls, task: Task):
        config_file = None
        for dirpath, _dirnames, filenames in os.walk(task.model_dir()):
            for filename in [f for f in filenames if f.endswith(".yml")]:
                config_file = os.path.join(dirpath, filename)
                break
            if config_file:
                break
        if not config_file:
            raise RuntimeError("Cannot find config.yml file for ns-render")
        args = f"""
            --load-config {config_file}
            --traj spiral
            --output-path {task.output_video_file()}
            """
        await cls.exec_program(task, "ns-render", *args.split())


class VanillaNerfStrategy(ExecutionStrategy):
    @classmethod
    def preprocess_timeout(cls):
        return 60 * 20  # 20 minutes

    @classmethod
    def train_timeout(cls):
        return 60 * 45  # 45 minutes

    @classmethod
    def render_timeout(cls):
        return 60 * 5  # 5 minutes

    @classmethod
    async def preprocess(cls, task: Task):
        args = f"""
            {colmap_script}
            --run_colmap
            --colmap_matcher exhaustive
            --aabb_scale 16
            """
        await cls.exec_program(task, "python3.8", *args.split(),
                               cwd=task.dataset_dir())

    @classmethod
    async def train(cls, task: Task):
        args = f"""
            vanilla-nerf
            --data {task.dataset_dir()}
            --output-dir {task.model_dir()}
            """
        await cls.exec_program(task, "ns-train", *args.split())

    @classmethod
    async def render(cls, task: Task):
        config_file = None
        for dirpath, _dirnames, filenames in os.walk(task.model_dir()):
            for filename in [f for f in filenames if f.endswith(".yml")]:
                config_file = os.path.join(dirpath, filename)
                break
            if config_file:
                break
        if not config_file:
            raise RuntimeError("Cannot find config.yml file for ns-render")
        args = f"""
            --load-config {config_file}
            --traj spiral
            --output-path {task.output_video_file()}
            """
        await cls.exec_program(task, "ns-render", *args.split())

