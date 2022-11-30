from os import path

# folder structure
# └─ tasks
#    ├── task_id_0
#    │   ├── dataset
#    │   │   ├── images
#    │   │   │   ├── image_0
#    │   │   │   ├── image_1
#    │   │   │   └── ...
#    │   │   ├── colmap.db
#    │   │   └── ...
#    │   ├── model
#    │   │   └── model.ckpt
#    │   ├── output
#    │   │   └── render_output.mp4
#    │   └── log
#    ├── task_id_1
#    └── ...
tasks_dir = path.expanduser("~/tasks")
