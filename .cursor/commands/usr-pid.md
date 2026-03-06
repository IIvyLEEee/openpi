请在终端中执行以下命令，帮我查看当前服务器上是谁在占用 GPU 资源（包含用户名、CPU%、显存占用和具体的进程名）：

```bash
nvidia-smi | grep -v '==' | awk '/[0-9]+/{print $5}' | xargs -r ps -up