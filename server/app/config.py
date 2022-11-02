from os import path

# folder structure
# └─ tasks
#    ├── task_id_0
#    │   ├── input
#    │   │   ├── image_0
#    │   │   ├── image_1
#    │   │   └── ...
#    │   └── output
#    ├── task_id_1
#    └── ...
tasks_dir = path.expanduser("~/tasks")
