"""重新生成 kaggle_all_in_one.py(把项目所有文件压缩内嵌,粘贴到 Kaggle 单元格即可运行)。
用法: python tools/build_all_in_one.py     (在仓库根目录运行;改过任何文件后都要重新运行并提交 kaggle_all_in_one.py)"""
import base64
import re
import subprocess
import sys
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "kaggle_all_in_one.py"
SKIP_DIRS = {".git", "__pycache__", "node_modules", "data", "logs", ".venv"}
SKIP_FILES = {"kaggle_all_in_one.py"}
SKIP_PREFIX = ("data_test",)


def collect():
    files = []
    for p in sorted(ROOT.rglob("*")):
        rel = p.relative_to(ROOT)
        if not p.is_file() or p.name in SKIP_FILES or p.suffix in (".pyc", ".png", ".jpg", ".wav"):
            continue
        if any(part in SKIP_DIRS or part.startswith(SKIP_PREFIX) for part in rel.parts):
            continue
        files.append(rel.as_posix())
    return files


def main():
    src = OUT.read_text("utf-8")
    head, rest = src.split("\nFILES = {\n", 1)
    foot = rest[rest.index("\n}\n") + 3:]
    files = collect()
    head = re.sub(r"内含全部 \d+ 个项目文件", "内含全部 %d 个项目文件" % len(files), head)
    out = [head, "\nFILES = {\n"]
    for rel in files:
        b64 = base64.b64encode(zlib.compress((ROOT / rel).read_bytes(), 9)).decode()
        out.append("    %r: (\n" % rel)
        for i in range(0, len(b64), 110):
            out.append('        "%s"\n' % b64[i:i + 110])
        out.append("    ),\n")
    out.append("}\n" + foot)
    OUT.write_text("".join(out), "utf-8")
    compile(OUT.read_text("utf-8"), str(OUT), "exec")  # 语法自检
    print("已生成 %s:%d 个文件,%d KB" % (OUT.name, len(files), OUT.stat().st_size // 1024))


if __name__ == "__main__":
    main()
