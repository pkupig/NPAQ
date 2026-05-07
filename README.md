# NPAQ 接手说明（已实跑）

本仓库是一个几何处理项目，核心目标是从点云/网格预测局部度量场并生成四边形网格（quad mesh）。

## 1. 我已在本机实跑通过（2026-05-07）

### 训练 smoke test（通过）
命令：

```bash
python scripts/train.py --config configs/train_smoke.yaml --logdir logs_smoke
```

本次产物：

- `logs_smoke/20260507_194620/best.pth`
- `logs_smoke/20260507_194620/latest.pth`
- `logs_smoke/20260507_194620/train.log`

### 重建 smoke test（通过）
如果 `smoke/smoke_patch.obj` 不存在，可先生成：

```bash
python -c "import numpy as np; n=8; xs=np.linspace(-1,1,n); ys=np.linspace(-1,1,n); V=[]; F=[]; [V.append((x,y,0.15*np.sin(np.pi*x)*np.cos(np.pi*y))) for y in ys for x in xs]; [F.extend([(j*n+i,j*n+i+1,(j+1)*n+i+1),(j*n+i,(j+1)*n+i+1,(j+1)*n+i)]) for j in range(n-1) for i in range(n-1)]; open('smoke/smoke_patch.obj','w',encoding='utf-8').write(''.join([f'v {a} {b} {c}\\n' for a,b,c in V] + [f'f {a+1} {b+1} {c+1}\\n' for a,b,c in F]))"
```

命令：

```bash
python scripts/reconstruct.py --input smoke/smoke_patch.obj --output smoke/smoke_quad.obj --config configs/reconstruct_smoke_py.yaml --checkpoint logs_smoke/20260507_194620/best.pth
```

本次结果：

- 成功输出 `smoke/smoke_quad.obj`
- 日志显示 `Initial mesh: 67 vertices, 35 quads`

## 2. 快速开始

建议 Python 版本：`3.9+`（当前环境为 3.9.1）。

先安装最小依赖：

```bash
pip install torch numpy scipy pyyaml tqdm tensorboard
```

可选依赖（部分 I/O 或点云流程会用到）：

```bash
pip install trimesh open3d
```

## 3. 项目结构（接手优先看）

- `scripts/train.py`：训练入口
- `scripts/reconstruct.py`：重建入口（指标场预测 + quad 拓扑 + PD 优化）
- `scripts/run_miq.py`：独立 MIQ 路径
- `configs/*.yaml`：训练/重建配置
- `src/`：核心实现（dataset、geometry、optimization、models）

## 4. 这次接手新增内容

- `configs/train_smoke.yaml`：最小训练配置（1 epoch）
- `configs/reconstruct_smoke_py.yaml`：最小重建配置（Python 后端）
- `smoke/smoke_patch.obj`：小型三角网格测试输入

另外修复了一个 Windows 控制台编码坑：

- `src/geometry/parametrization.py` 中错误提示的 `•` 改为 `-`，避免 `gbk` 下异常打印再次崩溃。

## 5. 已知注意事项

1. 当前仓库没有 `requirements.txt` / `environment.yml`，请以本 README 为准先搭环境。  
2. `trimesh` 缺失时会退回简化 OBJ/OFF 解析器，可运行但功能较弱。  
3. `open3d` 在部分 Windows 环境可能出现 DLL 加载失败；mesh 输入路径可绕开该依赖。  
4. `cpp_miq` 目前仅看到 `run_miq.cpp` 与 `build.sh`，若要启用 C++ IGL MIQ 路径，请先确认 `cpp_miq` 下构建文件是否完整（如 `CMakeLists.txt`）。  
