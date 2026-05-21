python3 scripts/serve_policy.py \
  --port=8000 \
  policy:checkpoint \
  --policy.config=pi05_droid \
  --policy.dir=/home/acts00/openpi/ckpt/pi05_base

uv run examples/simple_client/main.py --env DROID --host 127.0.0.1 --port 8000

---

(rdt2) acts00@acts00-Z690-AORUS-ELITE:~/RDT2$ git commit -m "on 4070"
[main a07b9f5] on 4070
 8 files changed, 443 insertions(+), 64 deletions(-)
 create mode 100644 constrants_20.txt
 create mode 100644 deploy/umi/common/usb_util.py
 create mode 100644 deploy/umi/real_world/camera/multi_uvc_camera.py
 create mode 100644 deploy/umi/real_world/camera/uvc_camera.py
(rdt2) acts00@acts00-Z690-AORUS-ELITE:~/RDT2$ git push
Missing or invalid credentials.
Error: connect ENOENT /run/user/1000/vscode-git-a440812d71.sock
    at PipeConnectWrap.afterConnect [as oncomplete] (node:net:1494:16) {
  errno: -2,
  code: 'ENOENT',
  syscall: 'connect',
  address: '/run/user/1000/vscode-git-a440812d71.sock'
}
Missing or invalid credentials.
Error: connect ENOENT /run/user/1000/vscode-git-a440812d71.sock
    at PipeConnectWrap.afterConnect [as oncomplete] (node:net:1494:16) {
  errno: -2,
  code: 'ENOENT',
  syscall: 'connect',
  address: '/run/user/1000/vscode-git-a440812d71.sock'
}
^C