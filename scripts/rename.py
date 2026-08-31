from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    Image = None  # type: ignore[assignment]


@dataclass(frozen=True)
class FilePair:
    image: Path
    annotation: Path


def normalize_extension(value: str) -> str:
    value = value.strip().lower()
    if not value:
        raise argparse.ArgumentTypeError("文件后缀不能为空。")
    return value if value.startswith(".") else f".{value}"


def natural_sort_key(path: Path) -> list[tuple[int, object]]:
    """让 file2 排在 file10 前面，并忽略英文大小写。"""
    parts = re.split(r"(\d+)", path.stem.casefold())
    return [
        (0, int(part)) if part.isdigit() else (1, part)
        for part in parts
        if part
    ]


def choose_directory() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        return None

    root = tk.Tk()
    root.withdraw()
    root.update()
    selected = filedialog.askdirectory(title="选择包含图片和 JSON 的文件夹")
    root.destroy()
    return Path(selected) if selected else None


def index_by_stem(files: list[Path], description: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in files:
        key = path.stem.casefold()
        if key in result:
            raise ValueError(
                f"发现两个{description}的文件名仅大小写不同，无法确定配对：\n"
                f"  {result[key].name}\n  {path.name}"
            )
        result[key] = path
    return result


def find_pairs(
    source_dir: Path, image_ext: str, json_ext: str
) -> tuple[list[FilePair], list[Path], list[Path]]:
    files = [path for path in source_dir.iterdir() if path.is_file()]
    images = [path for path in files if path.suffix.lower() == image_ext]
    annotations = [path for path in files if path.suffix.lower() == json_ext]

    image_map = index_by_stem(images, "图片")
    json_map = index_by_stem(annotations, "JSON")
    common_stems = image_map.keys() & json_map.keys()

    pairs = [
        FilePair(image=image_map[stem], annotation=json_map[stem])
        for stem in common_stems
    ]
    pairs.sort(key=lambda pair: natural_sort_key(pair.image))

    unmatched_images = sorted(
        (image_map[key] for key in image_map.keys() - common_stems),
        key=natural_sort_key,
    )
    unmatched_json = sorted(
        (json_map[key] for key in json_map.keys() - common_stems),
        key=natural_sort_key,
    )
    return pairs, unmatched_images, unmatched_json


def convert_to_jpeg(source: Path, destination: Path, quality: int) -> None:
    if Image is None:
        raise RuntimeError(
            "缺少 Pillow。请先运行：pip install pillow"
        )

    with Image.open(source) as image:
        image.load()
        if image.mode in ("RGBA", "LA") or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            background = Image.new("RGB", rgba.size, (255, 255, 255))
            background.paste(rgba, mask=rgba.getchannel("A"))
            converted = background
        else:
            converted = image.convert("RGB")

        converted.save(
            destination,
            format="JPEG",
            quality=quality,
            subsampling=0,
            optimize=True,
        )


def copy_annotation(
    source: Path,
    destination: Path,
    original_image_name: str,
    new_image_name: str,
) -> None:
    """复制 JSON；若是 LabelMe 格式，则同步其图片文件名。"""
    try:
        with source.open("r", encoding="utf-8-sig") as file:
            data = json.load(file)
    except (UnicodeDecodeError, json.JSONDecodeError):
        shutil.copy2(source, destination)
        return

    if not isinstance(data, dict) or "imagePath" not in data:
        shutil.copy2(source, destination)
        return

    old_image_path = data.get("imagePath")
    if isinstance(old_image_path, str):
        old_name = Path(old_image_path.replace("\\", "/")).name
        if old_name.casefold() == original_image_name.casefold():
            data["imagePath"] = new_image_name
            # 新 JPG 与 JSON 放在一起时无需保留旧 PNG 的嵌入数据。
            if isinstance(data.get("imageData"), str):
                data["imageData"] = None
            with destination.open("w", encoding="utf-8", newline="\n") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
                file.write("\n")
            return

    shutil.copy2(source, destination)


def destination_names(
    count: int, prefix: str, start: int, json_ext: str
) -> list[tuple[str, str]]:
    return [
        (f"{prefix}{start + offset}.jpg", f"{prefix}{start + offset}{json_ext}")
        for offset in range(count)
    ]


def ensure_no_collisions(
    destination_dir: Path,
    names: list[tuple[str, str]],
    source_files: set[Path],
    in_place: bool,
) -> None:
    conflicts: list[Path] = []
    for image_name, json_name in names:
        for name in (image_name, json_name):
            destination = destination_dir / name
            if not destination.exists():
                continue
            if in_place and destination.resolve() in source_files:
                continue
            conflicts.append(destination)

    if conflicts:
        preview = "\n".join(f"  {path.name}" for path in conflicts[:10])
        remaining = len(conflicts) - 10
        suffix = f"\n  ……另有 {remaining} 个" if remaining > 0 else ""
        raise FileExistsError(
            "目标文件已存在。为防止覆盖，操作已取消：\n"
            f"{preview}{suffix}\n"
            "请更换输出目录、起始编号，或先处理这些冲突文件。"
        )


def print_plan(
    pairs: list[FilePair],
    names: list[tuple[str, str]],
    destination_dir: Path,
    unmatched_images: list[Path],
    unmatched_json: list[Path],
) -> None:
    print(f"\n找到 {len(pairs)} 对同名文件。")
    print(f"输出目录：{destination_dir}")
    print("\n重命名预览：")

    preview_count = min(10, len(pairs))
    for pair, (image_name, json_name) in zip(
        pairs[:preview_count], names[:preview_count]
    ):
        print(
            f"  {pair.image.name} + {pair.annotation.name}"
            f"  ->  {image_name} + {json_name}"
        )
    if len(pairs) > preview_count:
        print(f"  ……另有 {len(pairs) - preview_count} 对")

    if unmatched_images:
        print(f"\n跳过 {len(unmatched_images)} 张没有同名 JSON 的图片：")
        for path in unmatched_images[:10]:
            print(f"  {path.name}")
    if unmatched_json:
        print(f"\n跳过 {len(unmatched_json)} 个没有同名图片的 JSON：")
        for path in unmatched_json[:10]:
            print(f"  {path.name}")


def process_pairs(
    pairs: list[FilePair],
    names: list[tuple[str, str]],
    destination_dir: Path,
    quality: int,
    in_place: bool,
) -> None:
    destination_dir.mkdir(parents=True, exist_ok=True)

    # 先在临时目录完成全部转换；任何一个文件失败都不会删除原文件。
    with tempfile.TemporaryDirectory(
        prefix="batch_rename_", dir=destination_dir.parent
    ) as temporary:
        temporary_dir = Path(temporary)

        for number, (pair, (image_name, json_name)) in enumerate(
            zip(pairs, names), start=1
        ):
            convert_to_jpeg(pair.image, temporary_dir / image_name, quality)
            copy_annotation(
                pair.annotation,
                temporary_dir / json_name,
                pair.image.name,
                image_name,
            )
            print(f"\r正在处理：{number}/{len(pairs)}", end="", flush=True)

        print()
        for image_name, json_name in names:
            shutil.move(
                str(temporary_dir / image_name),
                str(destination_dir / image_name),
            )
            shutil.move(
                str(temporary_dir / json_name),
                str(destination_dir / json_name),
            )

    if in_place:
        generated = {
            (destination_dir / name).resolve()
            for pair_names in names
            for name in pair_names
        }
        for pair in pairs:
            for original in (pair.image, pair.annotation):
                if original.resolve() not in generated:
                    original.unlink()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "按文件名自然排序，将同名图片和 JSON 配对后从指定编号开始重命名，"
            "并把图片真正转换为 JPG。"
        )
    )
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        help="源目录；省略时弹出文件夹选择窗口。",
    )
    parser.add_argument(
        "--image-ext",
        type=normalize_extension,
        default=".png",
        help="源图片后缀，默认 .png，例如：--image-ext .webp",
    )
    parser.add_argument(
        "--json-ext",
        type=normalize_extension,
        default=".json",
        help="标注文件后缀，默认 .json",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=2140,
        help="起始编号，默认 2140",
    )
    parser.add_argument(
        "--prefix",
        default="img_",
        help="新文件名前缀，默认 img_",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出目录；默认在源目录下创建 renamed_output。",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPG 质量 1-100，默认 95",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="在源目录中处理，并在全部成功后删除已处理的原始配对文件。",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="跳过执行前确认。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示排序与命名结果，不写入文件。",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()

    if args.start < 0:
        print("错误：起始编号不能小于 0。", file=sys.stderr)
        return 1
    if not 1 <= args.quality <= 100:
        print("错误：JPG 质量必须在 1-100 之间。", file=sys.stderr)
        return 1
    if not args.prefix or any(char in args.prefix for char in '<>:"/\\|?*'):
        print("错误：文件名前缀为空或包含 Windows 非法字符。", file=sys.stderr)
        return 1

    source_dir = args.directory or choose_directory()
    if source_dir is None:
        print("未选择文件夹，操作已取消。")
        return 0
    source_dir = source_dir.expanduser().resolve()
    if not source_dir.is_dir():
        print(f"错误：目录不存在：{source_dir}", file=sys.stderr)
        return 1

    if args.in_place and args.output is not None:
        print("错误：--in-place 和 --output 不能同时使用。", file=sys.stderr)
        return 1

    destination_dir = (
        source_dir
        if args.in_place
        else (
            args.output.expanduser().resolve()
            if args.output
            else source_dir / "renamed_output"
        )
    )

    try:
        pairs, unmatched_images, unmatched_json = find_pairs(
            source_dir, args.image_ext, args.json_ext
        )
        if not pairs:
            print(
                f"没有找到同名的 {args.image_ext} + {args.json_ext} 文件对。"
            )
            return 0

        names = destination_names(
            len(pairs), args.prefix, args.start, args.json_ext
        )
        source_files = {
            path.resolve()
            for pair in pairs
            for path in (pair.image, pair.annotation)
        }
        ensure_no_collisions(
            destination_dir, names, source_files, args.in_place
        )
        print_plan(
            pairs,
            names,
            destination_dir,
            unmatched_images,
            unmatched_json,
        )

        if args.dry_run:
            print("\n预览完成，没有写入或删除任何文件。")
            return 0

        if not args.yes:
            prompt = (
                "\n原始配对文件将在成功后删除。输入 YES 继续："
                if args.in_place
                else "\n输入 YES 开始处理："
            )
            if input(prompt).strip() != "YES":
                print("操作已取消。")
                return 0

        process_pairs(
            pairs, names, destination_dir, args.quality, args.in_place
        )
    except (OSError, ValueError, RuntimeError) as exc:
        print(f"\n错误：{exc}", file=sys.stderr)
        return 1

    print(
        f"\n完成：已生成 {len(pairs)} 张 JPG 和 "
        f"{len(pairs)} 个 JSON。\n位置：{destination_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())