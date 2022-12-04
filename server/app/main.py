from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

from uvicorn.config import Config
from uvicorn.main import Server

from models import *
from task_manager import TaskManager


app = FastAPI()

manager: TaskManager


@app.on_event("startup")
async def startup_event():
    global manager
    manager = TaskManager()


@app.on_event("shutdown")
async def shutdown_event():
    pass


@app.get("/")
async def index():
    content = """
    <body>
    <form action="/tasks" enctype="multipart/form-data" method="post">
    <label>
        Input Video
        <input name="file" type="file" accept="video/mp4">
    </label>
    <label>
        Processing Strategy
        <input name="strategy" value="nerfacto">
    </label>
    <input type="submit">
    </form>
    </body>
    """
    return HTMLResponse(content=content)


@app.get("/tasks", response_model=List[Task])
async def get_tasks():
    return manager.list()


@app.get("/tasks/{task_id}", response_model=Task)
async def get_task(task_id: str):
    task = manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.get("/tasks/{task_id}/output")
async def get_task_output(task_id: str):
    task = manager.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return FileResponse(task.output_video_file())


@app.post("/tasks")
async def post_tasks(
    file: UploadFile = File(""),
    strategy: ExecutionStrategy = ExecutionStrategy.nerfacto,
):
    if strategy in {"nerfacto", "instant-ngp", "vanilla-nerf"}:
        task = Task.new(strategy)
    else:
        raise HTTPException(status_code=400, detail='"strategy" must be either "nerfacto", "instant-ngp", "vanilla-nerf".')

    if file.content_type in {"video/quicktime", "video/mp4"}:
        task.upload_video(file)
    elif file.content_type in {"img/jpeg"}:
        task.upload_images(file)
    else:
        raise HTTPException(status_code=400, detail="Media type must be either \"video/quicktime\" or \"img/jpeg\"")

    success = manager.add(task)
    if not success:
        raise HTTPException(
                status_code=500, detail="Too many tasks queued. Try again later")
    return JSONResponse(status_code=201, content={'id': task.id})


if __name__ == "__main__":
    config = Config(app=app, loop="asyncio")
    server = Server(config=config)
    server.run()
