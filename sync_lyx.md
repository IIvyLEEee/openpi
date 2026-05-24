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

---

acts00@acts00-Z690-AORUS-ELITE:~/openpi$ uv run --group umi deploy/inference_real.py \
>   --policy-server-host=0.0.0.0 \
>   --policy-server-port=8000 \
>   --robot-config=deploy/configs/umi_ur5e_wsg50.yaml \
>   --prompt="pick the red block and put it in the green box."
Installed 3 packages in 90ms
Traceback (most recent call last):
  File "/home/acts00/openpi/deploy/inference_real.py", line 247, in <module>
    main(tyro.cli(Args))
  File "/home/acts00/openpi/deploy/inference_real.py", line 242, in main
    _run_real(args)
  File "/home/acts00/openpi/deploy/inference_real.py", line 145, in _run_real
    from deploy.umi.common.precise_sleep import precise_wait
ModuleNotFoundError: No module named 'deploy'

[DBG] unified tx will be applied
[DBG] [[ 0.70710678  0.70710678  0.          0.        ]
 [-0.70710678  0.70710678  0.          0.        ]
 [ 0.          0.          1.          0.        ]
 [ 0.          0.          0.          1.        ]]
[DBG] [[ 0.70710678 -0.70710678  0.          0.        ]
 [ 0.70710678  0.70710678  0.          0.        ]
 [ 0.          0.          1.          0.        ]
 [ 0.          0.          0.          1.        ]]

---

acts00@acts00-Z690-AORUS-ELITE:~/openpi$ uv run --group umi deploy/inference_real.py   --policy-server-host=localhost   --policy-server-port=8000   --robot-config=deploy/configs/umi_ur5e_wsg50.yaml   --prompt="pick the red block and put it in the green box."--steps-per-inference=6
INFO:root:Waiting for server at ws://localhost:8000...
INFO:__main__:Server metadata: {}
Successfully reset /dev/bus/usb/002/014
[DBG] Robots created.
[DBG] Grippers created.
[DBG] Env created.
INFO:__main__:Waiting for camera and robot buffers.
INFO:__main__:obs state=[ 0.0857 -0.5175  0.1953  0.0052 -2.8392  1.0471  0.1096] image_shape=(224, 224, 3)
INFO:__main__:inference 189.1 ms, got 30 actions, scheduled 26, first_delta=[-0.0012 -0.0003 -0.0032 -0.0074 -0.0025  0.0075], last_delta=[-0.007  -0.0044 -0.0868 -0.139  -0.0158  0.0247]
INFO:__main__:obs state=[ 0.0838 -0.5169  0.1862 -0.0064 -2.843   1.0561  0.0881] image_shape=(224, 224, 3)
INFO:__main__:inference 169.4 ms, got 30 actions, scheduled 26, first_delta=[ 1.4782e-05 -1.2988e-04 -2.9169e-03 -5.6642e-03 -3.3293e-03  5.6391e-03], last_delta=[-0.0063 -0.0019 -0.0865 -0.1188  0.0156  0.026 ]
INFO:__main__:obs state=[ 0.0785 -0.5144  0.1514 -0.0543 -2.8572  1.0866  0.0842] image_shape=(224, 224, 3)
INFO:__main__:inference 169.4 ms, got 30 actions, scheduled 26, first_delta=[ 8.8566e-04 -5.6434e-04 -2.7597e-03 -2.6990e-03 -8.3923e-05  5.9463e-03], last_delta=[-0.0015  0.0071 -0.08   -0.1163  0.0021  0.0212]
INFO:__main__:obs state=[ 0.0784 -0.5117  0.1339 -0.0727 -2.8618  1.0958  0.0833] image_shape=(224, 224, 3)
INFO:__main__:inference 169.1 ms, got 30 actions, scheduled 26, first_delta=[ 0.0005 -0.0004 -0.0016 -0.0022 -0.0006  0.0017], last_delta=[-0.0015  0.0019 -0.0688 -0.1067  0.0135  0.0167]
INFO:__main__:obs state=[ 0.0763 -0.5076  0.1087 -0.1074 -2.8626  1.1207  0.0825] image_shape=(224, 224, 3)
INFO:__main__:inference 169.8 ms, got 30 actions, scheduled 26, first_delta=[ 0.0015 -0.0008 -0.0006 -0.0003 -0.0065 -0.0027], last_delta=[-0.003   0.0107 -0.0532 -0.0868 -0.0161  0.0224]
INFO:__main__:obs state=[ 0.0765 -0.5036  0.0941 -0.1266 -2.8603  1.1301  0.0826] image_shape=(224, 224, 3)
INFO:__main__:inference 169.4 ms, got 30 actions, scheduled 26, first_delta=[ 5.5117e-04 -1.9597e-03 -6.7912e-05  7.0453e-05 -3.3138e-03  3.6101e-03], last_delta=[-0.007   0.011  -0.0469 -0.0814 -0.0039  0.03  ]
INFO:__main__:obs state=[ 0.0775 -0.4984  0.0796 -0.1443 -2.8532  1.1441  0.0824] image_shape=(224, 224, 3)
INFO:__main__:inference 169.5 ms, got 30 actions, scheduled 26, first_delta=[ 0.0005 -0.0005  0.0002  0.0023 -0.0034  0.0035], last_delta=[-0.0029  0.0161 -0.0421 -0.0775 -0.0109  0.024 ]
INFO:__main__:obs state=[ 0.0762 -0.4948  0.0678 -0.1596 -2.8415  1.1562  0.0819] image_shape=(224, 224, 3)
INFO:__main__:inference 169.9 ms, got 30 actions, scheduled 26, first_delta=[ 0.001  -0.0016  0.0009 -0.0004 -0.0022  0.002 ], last_delta=[-0.003   0.0097 -0.0312 -0.0683 -0.0051  0.0051]
INFO:__main__:obs state=[ 0.077  -0.4887  0.0583 -0.1706 -2.839   1.1714  0.0816] image_shape=(224, 224, 3)
INFO:__main__:inference 169.7 ms, got 30 actions, scheduled 26, first_delta=[ 1.0697e-03 -1.5676e-03  7.8723e-05  3.1092e-03 -1.2991e-03 -7.4887e-04], last_delta=[-0.0028  0.0076 -0.0228 -0.0512  0.0159  0.0031]
INFO:__main__:obs state=[ 0.0771 -0.4844  0.0492 -0.1841 -2.831   1.1786  0.0813] image_shape=(224, 224, 3)
INFO:__main__:inference 170.1 ms, got 30 actions, scheduled 26, first_delta=[ 0.0008 -0.0036  0.0014  0.0007  0.0011 -0.0008], last_delta=[-0.0091 -0.0145 -0.0105 -0.043   0.0222  0.0056]
INFO:__main__:obs state=[ 0.0794 -0.4826  0.043  -0.1909 -2.82    1.1812  0.0806] image_shape=(224, 224, 3)
INFO:__main__:inference 169.7 ms, got 30 actions, scheduled 26, first_delta=[ 0.0021 -0.0022  0.0013  0.0014  0.0007 -0.0007], last_delta=[-0.0106 -0.0086 -0.0131 -0.0457  0.0199 -0.0164]
INFO:__main__:obs state=[ 0.079  -0.484   0.0377 -0.2048 -2.8074  1.1825  0.0798] image_shape=(224, 224, 3)
INFO:__main__:inference 169.5 ms, got 30 actions, scheduled 26, first_delta=[ 0.0009 -0.0013  0.0031  0.0022  0.0037 -0.0015], last_delta=[-0.0086 -0.012  -0.0104 -0.039   0.0278 -0.0195]
INFO:__main__:obs state=[ 0.0813 -0.4802  0.0334 -0.2098 -2.7962  1.1756  0.0793] image_shape=(224, 224, 3)
INFO:__main__:inference 169.3 ms, got 30 actions, scheduled 26, first_delta=[ 0.0018 -0.0017  0.0024  0.0042  0.0042 -0.0007], last_delta=[-0.0033 -0.0009 -0.0091 -0.0193  0.0325 -0.0081]
INFO:__main__:obs state=[ 0.0817 -0.4814  0.0298 -0.2155 -2.7851  1.1754  0.0792] image_shape=(224, 224, 3)
INFO:__main__:inference 169.4 ms, got 30 actions, scheduled 26, first_delta=[ 0.0014 -0.0036  0.0032  0.0024 -0.001  -0.0007], last_delta=[-0.0065 -0.0084 -0.0086 -0.0213  0.0205 -0.0229]
INFO:__main__:obs state=[ 0.084  -0.4775  0.0251 -0.2162 -2.7826  1.17    0.0783] image_shape=(224, 224, 3)
INFO:__main__:inference 169.7 ms, got 30 actions, scheduled 26, first_delta=[-0.0004 -0.0011  0.0039  0.0049  0.0008  0.0005], last_delta=[-0.007  -0.0068 -0.0045 -0.0213  0.0384 -0.0036]
INFO:__main__:obs state=[ 0.0844 -0.4767  0.0229 -0.218  -2.7688  1.1745  0.079 ] image_shape=(224, 224, 3)
INFO:__main__:inference 169.3 ms, got 30 actions, scheduled 26, first_delta=[ 0.0003 -0.0026  0.0032  0.0037  0.0002  0.0005], last_delta=[-0.0051 -0.0088 -0.0049 -0.0152  0.0112 -0.0165]
INFO:__main__:obs state=[ 0.0873 -0.4728  0.0202 -0.2174 -2.7553  1.1665  0.0786] image_shape=(224, 224, 3)
INFO:__main__:inference 169.8 ms, got 30 actions, scheduled 26, first_delta=[ 0.0005 -0.0028  0.0044  0.0032 -0.0022 -0.0003], last_delta=[-0.0033 -0.0084 -0.0002 -0.0195  0.0084 -0.0077]
INFO:__main__:obs state=[ 0.0879 -0.4733  0.0177 -0.2171 -2.7503  1.1712  0.078 ] image_shape=(224, 224, 3)
INFO:__main__:inference 169.7 ms, got 30 actions, scheduled 26, first_delta=[ 0.0022 -0.0048  0.0037  0.0038  0.0005  0.0037], last_delta=[-4.7387e-03 -5.7196e-03  6.0417e-05 -1.6786e-02  2.3288e-02 -1.1747e-02]
INFO:__main__:obs state=[ 0.0907 -0.4729  0.0169 -0.2197 -2.7403  1.1673  0.0784] image_shape=(224, 224, 3)
INFO:__main__:inference 169.6 ms, got 30 actions, scheduled 26, first_delta=[ 0.0004 -0.0017  0.0045  0.0054 -0.0008  0.002 ], last_delta=[-0.0014 -0.0078 -0.0064 -0.0241  0.0215 -0.0133]
INFO:__main__:obs state=[ 0.0916 -0.4699  0.0133 -0.2219 -2.7295  1.1747  0.0785] image_shape=(224, 224, 3)
INFO:__main__:inference 169.5 ms, got 30 actions, scheduled 26, first_delta=[ 0.0002 -0.0041  0.0032  0.0028  0.0005  0.002 ], last_delta=[-0.0073 -0.0047  0.0037 -0.012   0.0185  0.0009]
INFO:__main__:obs state=[ 0.0935 -0.4694  0.0114 -0.2231 -2.7242  1.1687  0.0785] image_shape=(224, 224, 3)
INFO:__main__:inference 169.3 ms, got 30 actions, scheduled 26, first_delta=[-0.0008 -0.0032  0.0041  0.0045 -0.0019  0.0002], last_delta=[-0.002  -0.0037 -0.0035 -0.0221  0.0198 -0.0069]
INFO:__main__:obs state=[ 0.093  -0.4675  0.0108 -0.2185 -2.7072  1.1819  0.0786] image_shape=(224, 224, 3)
INFO:__main__:inference 169.2 ms, got 30 actions, scheduled 26, first_delta=[ 5.3842e-04 -4.0995e-03  3.5175e-03  7.3054e-03  3.3140e-05  1.8229e-03], last_delta=[-0.0018 -0.0007 -0.0026 -0.0219  0.013  -0.009 ]
INFO:__main__:obs state=[ 0.095  -0.4656  0.0082 -0.2195 -2.702   1.173   0.0787] image_shape=(224, 224, 3)
INFO:__main__:inference 170.3 ms, got 30 actions, scheduled 26, first_delta=[ 0.0014 -0.0028  0.0032  0.005   0.0004 -0.0006], last_delta=[-0.0025 -0.0094 -0.0008 -0.0247  0.0254 -0.017 ]
INFO:__main__:obs state=[ 0.0958 -0.4645  0.0057 -0.2205 -2.6902  1.1842  0.0783] image_shape=(224, 224, 3)
INFO:__main__:inference 169.5 ms, got 30 actions, scheduled 26, first_delta=[ 0.0002 -0.0032  0.0032  0.0051  0.0043 -0.0014], last_delta=[-0.0026 -0.0078 -0.0007 -0.0152  0.0334 -0.0056]
INFO:__main__:obs state=[ 0.0985 -0.4639  0.0038 -0.2221 -2.6755  1.178   0.0786] image_shape=(224, 224, 3)
INFO:__main__:inference 169.4 ms, got 30 actions, scheduled 26, first_delta=[-0.0006 -0.003   0.0034  0.0052  0.002   0.0037], last_delta=[-0.0003 -0.005  -0.0014 -0.0125  0.0409  0.0034]
INFO:__main__:obs state=[ 9.9079e-02 -4.6169e-01  2.3589e-03 -2.2078e-01 -2.6568e+00  1.1881e+00
  7.8772e-02] image_shape=(224, 224, 3)
INFO:__main__:inference 169.0 ms, got 30 actions, scheduled 26, first_delta=[-0.0001 -0.0019  0.0035  0.0068  0.0009  0.0034], last_delta=[ 0.0004 -0.001   0.002  -0.0136  0.0343  0.0074]
INFO:__main__:obs state=[ 1.0212e-01 -4.5882e-01  8.0974e-04 -2.1928e-01 -2.6444e+00  1.1855e+00
  7.8543e-02] image_shape=(224, 224, 3)
INFO:__main__:inference 169.2 ms, got 30 actions, scheduled 26, first_delta=[-0.0002 -0.0028  0.0021  0.0058  0.0028  0.0007], last_delta=[ 0.0026  0.0014 -0.0017 -0.0143  0.0159 -0.0013]
INFO:__main__:obs state=[ 1.0396e-01 -4.5509e-01 -1.9579e-03 -2.2003e-01 -2.6310e+00  1.1876e+00
  7.7727e-02] image_shape=(224, 224, 3)
INFO:__main__:inference 169.4 ms, got 30 actions, scheduled 26, first_delta=[ 2.2538e-05 -2.4785e-03  3.0731e-03  5.5178e-03  9.5654e-04  2.5685e-03], last_delta=[-0.002   0.0052 -0.0018 -0.0173  0.0223  0.0122]
INFO:__main__:obs state=[ 1.0687e-01 -4.5299e-01 -2.1690e-03 -2.1719e-01 -2.6202e+00  1.1867e+00
  7.7628e-02] image_shape=(224, 224, 3)
INFO:__main__:inference 169.9 ms, got 30 actions, scheduled 26, first_delta=[-0.0003 -0.0044  0.0041  0.0046  0.0026  0.0036], last_delta=[-0.0005  0.0007 -0.0006 -0.0157  0.0326  0.0181]
INFO:__main__:obs state=[ 0.11   -0.45   -0.0035 -0.2192 -2.6055  1.1881  0.0782] image_shape=(224, 224, 3)
/home/acts00/openpi/deploy/collision_utils.py:32: UserWarning: End-effector pose [ 0.11650658 -0.44351885 -0.0094563  -0.24501295 -2.5898464   1.1878823 ] is too low, adjusting by 0.073 cm to avoid collision with the table.
  warnings.warn(
/home/acts00/openpi/deploy/collision_utils.py:32: UserWarning: End-effector pose [ 0.11495928 -0.44302124 -0.01069462 -0.24791119 -2.590752    1.1886333 ] is too low, adjusting by 0.179 cm to avoid collision with the table.
  warnings.warn(
/home/acts00/openpi/deploy/collision_utils.py:32: UserWarning: End-effector pose [ 0.11558986 -0.4432606  -0.01093196 -0.2469411  -2.589718    1.195143  ] is too low, adjusting by 0.177 cm to avoid collision with the table.
  warnings.warn(
INFO:__main__:inference 170.0 ms, got 30 actions, scheduled 26, first_delta=[ 0.0007 -0.0029  0.0044  0.0053  0.0007  0.0012], last_delta=[ 0.0007  0.0069 -0.0023 -0.026   0.0132  0.0083]
INFO:__main__:obs state=[ 0.1134 -0.4496 -0.004  -0.2203 -2.5936  1.1886  0.0778] image_shape=(224, 224, 3)
/home/acts00/openpi/deploy/collision_utils.py:32: UserWarning: End-effector pose [ 0.12113275 -0.44026116 -0.00712628 -0.21964683 -2.5595875   1.1952003 ] is too low, adjusting by 0.007 cm to avoid collision with the table.
  warnings.warn(
/home/acts00/openpi/deploy/collision_utils.py:32: UserWarning: End-effector pose [ 0.11847714 -0.43715352 -0.00955663 -0.2401853  -2.5801556   1.2024796 ] is too low, adjusting by 0.093 cm to avoid collision with the table.
  warnings.warn(
INFO:__main__:inference 169.6 ms, got 30 actions, scheduled 26, first_delta=[ 0.0003 -0.0026  0.0023  0.0056 -0.0005  0.0014], last_delta=[ 0.0002  0.0088 -0.0008 -0.026   0.0128  0.026 ]
INFO:__main__:obs state=[ 0.1175 -0.4458 -0.0056 -0.2242 -2.5798  1.1903  0.0779] image_shape=(224, 224, 3)
INFO:__main__:inference 169.8 ms, got 30 actions, scheduled 26, first_delta=[-0.0005 -0.0036  0.0038  0.0068 -0.0027 -0.0023], last_delta=[-0.0016  0.0021  0.0006 -0.0223  0.0257  0.0131]
INFO:__main__:obs state=[ 0.1191 -0.444  -0.0052 -0.2206 -2.5647  1.1909  0.0782] image_shape=(224, 224, 3)
/home/acts00/openpi/deploy/collision_utils.py:32: UserWarning: End-effector pose [ 0.12386245 -0.4383601  -0.00804051 -0.241162   -2.5462587   1.2041419 ] is too low, adjusting by 0.036 cm to avoid collision with the table.
  warnings.warn(
INFO:__main__:inference 169.4 ms, got 30 actions, scheduled 26, first_delta=[-0.0012 -0.0036  0.0037  0.0062 -0.0071  0.0009], last_delta=[-0.0019  0.0054  0.0004 -0.0201  0.0254  0.0273]
INFO:__main__:obs state=[ 0.1193 -0.4458 -0.0041 -0.2219 -2.5584  1.188   0.0784] image_shape=(224, 224, 3)
/home/acts00/openpi/deploy/collision_utils.py:32: UserWarning: End-effector pose [ 0.12445789 -0.43838325 -0.00635405 -0.21674573 -2.5245075   1.1974883 ] is too low, adjusting by 0.046 cm to avoid collision with the table.
  warnings.warn(
/home/acts00/openpi/deploy/collision_utils.py:32: UserWarning: End-effector pose [ 0.12446973 -0.43829113 -0.00616957 -0.21667297 -2.5227568   1.2039798 ] is too low, adjusting by 0.025 cm to avoid collision with the table.
  warnings.warn(
/home/acts00/openpi/deploy/collision_utils.py:32: UserWarning: End-effector pose [ 0.12513481 -0.43725395 -0.0065406  -0.21797769 -2.520961    1.2063972 ] is too low, adjusting by 0.061 cm to avoid collision with the table.
  warnings.warn(