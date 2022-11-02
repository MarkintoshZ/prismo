from typing import List

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse

from models import *
from config import *

app = FastAPI()


@app.get("/")
async def index():
    content = """
    <body>
    <form action="/tasks/" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple>
    <input type="submit">
    </form>
    </body>
    """
    return HTMLResponse(content=content)


@app.get("/tasks", response_model=List[Task])
async def get_tasks():
    return []


@app.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str):
    if (task_id is None):
        raise HTTPException(status_code=404, detail="task not found")
    return task_id


@app.post("/tasks/")
async def post_tasks(
    files: List[UploadFile] = File(""),
):
    task = Task.new()
    task.upload_files(files)
    return task

