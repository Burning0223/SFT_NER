from modelscope import snapshot_download

model_dir = snapshot_download(
    model_id = "qwen/Qwen2.5-7B",
    cache_dir = r"E:\demo3\Qwen",
    revision = "master"
)
print("模型下载完成路径：",model_dir)
