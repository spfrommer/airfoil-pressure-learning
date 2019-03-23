import os.path as op

ROOT_DIR = op.dirname(op.dirname(op.abspath(__file__)))
def root_path(*path):
    return op.join(ROOT_DIR, *path)

RES_DIR = root_path("res")
def res_path(*path):
    return op.join(RES_DIR, *path)

OUT_DIR = root_path("out")
def out_path(*path):
    return op.join(OUT_DIR, *path)

CASE_DIR = root_path("case")
def case_path(*path):
    return op.join(CASE_DIR, *path)

SHELL_DIR = root_path("shell")
def shell_path(*path):
    return op.join(SHELL_DIR, *path)
