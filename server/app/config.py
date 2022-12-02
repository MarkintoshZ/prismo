from os import path, getcwd

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

colmap_tmp_dir = path.expanduser("~/.colmap-tmp")

colmap_script = path.join(getcwd(), path.pardir, 'scripts', 'colmap2nerf.py')