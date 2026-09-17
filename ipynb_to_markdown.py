"""
Chương trình chuyển đổi Jupyter Notebook (.ipynb) sang Markdown (.md)
Hỗ trợ 2 phương pháp:
  1. nbconvert (Thư viện chính thức của Jupyter - tự động trích xuất biểu đồ/hình ảnh)
  2. Pure Python (Không cần cài thư viện ngoài, chỉ dùng json và chuẩn hoá cơ bản)
"""

import os
import sys
import re
import json
import base64
import argparse
from typing import Optional

# Dam bao terminal Windows ho tro UTF-8 khong bi loi UnicodeEncodeError
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def clean_ansi(text: str) -> str:
    """Xóa các ký tự ANSI escape (màu sắc trong terminal / traceback)."""
    ansi_regex = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_regex.sub('', text)


def convert_pure_python(ipynb_path: str, output_path: str, include_outputs: bool = True, extract_images: bool = True) -> str:
    """
    Chuyển đổi notebook sang Markdown chỉ bằng thư viện chuẩn của Python (không cần cài thêm gì).
    """
    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    md_lines = []
    base_name = os.path.splitext(os.path.basename(ipynb_path))[0]
    output_dir = os.path.dirname(output_path) or "."
    img_dir_name = f"{base_name}_files"
    img_dir_path = os.path.join(output_dir, img_dir_name)

    image_counter = 0

    cells = nb.get("cells", [])
    for idx, cell in enumerate(cells):
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))

        if cell_type == "markdown":
            md_lines.append(source.strip())
            md_lines.append("\n\n")

        elif cell_type == "code":
            # Code input
            md_lines.append("```python\n")
            md_lines.append(source.rstrip())
            md_lines.append("\n```\n\n")

            # Code outputs
            if include_outputs and "outputs" in cell:
                for out in cell["outputs"]:
                    out_type = out.get("output_type", "")

                    # 1. Stream output (print, stdout, stderr)
                    if out_type == "stream":
                        text = "".join(out.get("text", []))
                        clean_text = clean_ansi(text).rstrip()
                        if clean_text:
                            md_lines.append("```text\n")
                            md_lines.append(clean_text)
                            md_lines.append("\n```\n\n")

                    # 2. Execute result / Display data (tables, text, plots)
                    elif out_type in ("execute_result", "display_data"):
                        data = out.get("data", {})

                        # Xử lý hình ảnh (PNG / JPEG)
                        img_found = False
                        for mime in ("image/png", "image/jpeg", "image/jpg"):
                            if mime in data:
                                img_data = data[mime]
                                if isinstance(img_data, list):
                                    img_data = "".join(img_data)

                                ext = "png" if "png" in mime else "jpg"
                                if extract_images:
                                    os.makedirs(img_dir_path, exist_ok=True)
                                    img_filename = f"output_{idx}_{image_counter}.{ext}"
                                    img_filepath = os.path.join(img_dir_path, img_filename)
                                    with open(img_filepath, "wb") as img_f:
                                        img_f.write(base64.b64decode(img_data))

                                    rel_img_path = f"./{img_dir_name}/{img_filename}"
                                    md_lines.append(f"![image]({rel_img_path})\n\n")
                                else:
                                    md_lines.append(f"![image](data:{mime};base64,{img_data})\n\n")

                                image_counter += 1
                                img_found = True
                                break

                        # Nếu không có ảnh, hiển thị text/plain nếu có
                        if not img_found and "text/plain" in data:
                            plain_text = "".join(data["text/plain"]).rstrip()
                            if plain_text:
                                md_lines.append("```text\n")
                                md_lines.append(clean_ansi(plain_text))
                                md_lines.append("\n```\n\n")

                    # 3. Error traceback
                    elif out_type == "error":
                        tb = "\n".join(out.get("traceback", []))
                        clean_tb = clean_ansi(tb).rstrip()
                        if clean_tb:
                            md_lines.append("```text\n")
                            md_lines.append(clean_tb)
                            md_lines.append("\n```\n\n")

    content = "".join(md_lines).strip() + "\n"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return output_path


def convert_nbconvert(ipynb_path: str, output_path: str) -> str:
    """
    Chuyển đổi notebook sang Markdown bằng nbconvert.
    Toàn bộ hình ảnh sẽ được lưu gọn gàng vào thư mục <tên_file>_files, không lưu ở thư mục gốc.
    """
    import nbformat
    from nbconvert import MarkdownExporter

    with open(ipynb_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    base_name = os.path.splitext(os.path.basename(ipynb_path))[0]
    img_dir_name = f"{base_name}_files"

    exporter = MarkdownExporter()
    # Cấu hình để nbconvert lưu toàn bộ ảnh/biểu đồ vào thư mục _files và cập nhật đường dẫn trong Markdown
    resources = {
        "output_files_dir": f"./{img_dir_name}"
    }
    body, resources = exporter.from_notebook_node(nb, resources=resources)

    # Đảm bảo đường dẫn ảnh trong markdown dùng dấu gạch chéo xuôi '/' và có tiền tố './'
    body = body.replace(f"{img_dir_name}\\", f"{img_dir_name}/")
    body = body.replace(f"./{img_dir_name}\\", f"./{img_dir_name}/")
    # Đảm bảo luôn có tiền tố './' trước tên thư mục ảnh (vd: ![png](./rnn_files/...))
    body = re.sub(rf'\((?!\./)({re.escape(img_dir_name)}/)', r'(./\1', body)

    # Ghi file markdown
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(body)

    # Trích xuất các tài nguyên (hình ảnh) vào đúng thư mục _files
    output_dir = os.path.dirname(output_path) or "."
    outputs = resources.get("outputs", {})
    for filename, data in outputs.items():
        resource_path = os.path.join(output_dir, filename)
        os.makedirs(os.path.dirname(resource_path), exist_ok=True)
        with open(resource_path, "wb") as f:
            f.write(data)

    return output_path


def convert_ipynb_to_md(
    ipynb_path: str,
    output_path: Optional[str] = None,
    method: str = "auto",
    include_outputs: bool = True,
    extract_images: bool = True,
) -> str:
    """
    Hàm tổng quát để chuyển đổi ipynb sang md:
      - method: 'auto', 'nbconvert', hoặc 'pure'
    """
    if not os.path.exists(ipynb_path):
        raise FileNotFoundError(f"Không tìm thấy file: {ipynb_path}")

    if output_path is None:
        base_name = os.path.splitext(ipynb_path)[0]
        output_path = f"{base_name}.md"

    if method == "auto":
        try:
            import nbconvert  # noqa: F401
            import nbformat   # noqa: F401
            method = "nbconvert"
        except ImportError:
            method = "pure"

    if method == "nbconvert":
        try:
            return convert_nbconvert(ipynb_path, output_path)
        except Exception as e:
            print(f"[Cảnh báo] Lỗi khi dùng nbconvert ({e}), tự động chuyển sang Pure Python...")
            return convert_pure_python(ipynb_path, output_path, include_outputs=include_outputs, extract_images=extract_images)
    else:
        return convert_pure_python(ipynb_path, output_path, include_outputs=include_outputs, extract_images=extract_images)


def main():
    parser = argparse.ArgumentParser(description="Chuyển đổi Jupyter Notebook (.ipynb) sang Markdown (.md)")
    parser.add_argument("input", help="Đường dẫn file .ipynb cần chuyển đổi")
    parser.add_argument("-o", "--output", help="Đường dẫn file .md đầu ra (mặc định cùng tên với file gốc)", default=None)
    parser.add_argument(
        "--method",
        choices=["auto", "nbconvert", "pure"],
        default="auto",
        help="Phương pháp chuyển đổi (mặc định: auto - tự nhận diện nbconvert hoặc pure python)",
    )
    parser.add_argument("--no-output", action="store_true", help="Bỏ qua các kết quả output của code cell")
    parser.add_argument("--no-extract-images", action="store_true", help="Không trích xuất ảnh ra thư mục ngoài (chỉ áp dụng với pure python)")

    args = parser.parse_args()

    out_file = convert_ipynb_to_md(
        ipynb_path=args.input,
        output_path=args.output,
        method=args.method,
        include_outputs=not args.no_output,
        extract_images=not args.no_extract_images,
    )
    print(f"Đã chuyển đổi thành công: {args.input} -> {out_file}")


if __name__ == "__main__":
    main()
