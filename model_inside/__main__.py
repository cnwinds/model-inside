"""用法: python -m model_inside <模型目录或 .safetensors 文件> [--port 8000]"""
import argparse

from . import __version__


def main():
    ap = argparse.ArgumentParser(prog="model_inside", description="把模型权重当图片看，并记录/回放激活")
    ap.add_argument("--version", action="version", version=f"model-inside {__version__}")
    ap.add_argument("model", help="模型目录（含 *.safetensors）或单个 .safetensors 文件")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data-dir", default=".", help="瓦片缓存和录制文件的存放目录")
    ap.add_argument("--dtype", default="float32", help="录制时运行模型用的精度，大模型可用 bfloat16 省内存")
    args = ap.parse_args()

    import uvicorn
    from .server import create_app

    uvicorn.run(create_app(args.model, args.data_dir, args.dtype), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
