## 如何修改Transformers库的默认下载地址

由于 `huggingface_hub` 使用了符号链接（symlinks）来优化缓存存储，这可能导致磁盘空间的消耗。如果不进行修改，默认情况下文件会被缓存到 `C:\Users\Users\.cache\huggingface` 目录中。

![](../Image/HuggingFace缓存路径修改.png)