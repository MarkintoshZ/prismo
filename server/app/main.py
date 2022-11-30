from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from starlette.responses import JSONResponse

from uvicorn.config import Config
from uvicorn.main import Server

from models import *
from config import *

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
    <form action="/tasks/" enctype="multipart/form-data" method="post">
    <input name="files" type="file" multiple>
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
    task = manager.get(UUID(task_id))
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return task


@app.get("/tasks/{task_id}/output", response_model=Task)
async def get_task_output(task_id: str):
    task = manager.get(UUID(task_id))
    if not task:
        raise HTTPException(status_code=404, detail="task not found")
    return FileResponse(task.output_file())


@app.post("/tasks/")
async def post_tasks(
    images: List[UploadFile] = File(""),
    video: UploadFile = File(""),
    strategy: str = "nerfacto"
):
    if strategy == "nerfacto":
        task = Task.new(NerfactoStrategy)
    elif strategy == "instant_ngp":
        task = Task.new()
        task = Task.new(InstantNGPStrategy)
    elif stategy == "vanilla_nerf":
        task = Task.new(VanillaNerfStrategy)
    else:
        raise HTTPException(status_code=400, detail="\"strategy\" must be either \"nerfacto\", \"instant_ngp\", \"vanilla_nerf\".")
    
    if images:
        task.upload_images(images)
    elif video:
        task.upload_video(video)
    else:
        raise HTTPException(status_code=400, detail="The \"images\" or \"video\" field must contain data.")

    success = manager.add(task)

    if not success:
        raise HTTPException(
                status_code=500, detail="Too many tasks queued. Try again later")
    return JSONResponse(status_code=201, content={'id': task.id})


if __name__ == "__main__":
    config = Config(app=app, loop="asyncio")
    server = Server(config=config)
    server.run()
