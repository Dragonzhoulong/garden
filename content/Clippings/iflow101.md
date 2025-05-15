## readme编写的问题

``` shell
# 建议添加一行，一些小白可能在这个项目才开始用docker
docker run -it iedaopensource/iflow:latest
```
## 使用docker的问题

我使用mac连接ubuntu远程(没太解决明白这个远程图像显示到mac上)。
然后使用docker来作为运行环境，也面临一个宿主机访问远程生成图像的问题,希望也能在文档中提到该问题的解决方案。
``` zsh
root@a6b92f5c0a99:/opt/iFlow/scripts# Warning: could not connect to display 
Info: Could not load the Qt platform plugin "xcb" in "" even though it was found.
root@a6b92f5c0a99:/opt/iFlow/scripts# ls
```
这个错误是因为在Docker容器中尝试运行图形界面程序（KLayout）时，无法连接到X11显示服务器。要解决这个问题，有几种方法：

1. **使用X11转发**（推荐）：
```bash
# 在宿主机上允许X11转发
xhost +local:docker

# 启动Docker容器时添加X11相关参数
docker run -it \
    -e DISPLAY=$DISPLAY \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    iedaopensource/iflow:latest
```

2. **使用VNC**：
```bash
# 在容器中安装VNC服务器
apt-get update
apt-get install -y x11vnc

# 启动VNC服务器
x11vnc -display :0
```

3. **将文件复制到宿主机查看**：
```bash
# 在宿主机上执行
docker cp [容器ID]:/opt/iFlow/result/uart.layout.klayout_0.26.2.sky130.HS.TYP.V1/uart.gds ./uart.gds

# 然后在宿主机上使用KLayout查看
klayout uart.gds
```

4. **使用卷挂载**：
```bash
# 启动容器时挂载目录
docker run -it \
    -v /host/path:/opt/iFlow/result \
    iedaopensource/iflow:latest
```

5. **安装必要的X11依赖**：
```bash
# 在容器中安装X11相关包
apt-get update
apt-get install -y \
    libxcb1 \
    libxcb-icccm4 \
    libxcb-image0 \
    libxcb-keysyms1 \
    libxcb-randr0 \
    libxcb-render-util0 \
    libxcb-shape0 \
    libxcb-sync1 \
    libxcb-xfixes0 \
    libxcb-xinerama0 \
    libxcb-xkb1
```

最佳实践建议：

1. **开发环境**：
   - 使用X11转发方式，这样可以直接在容器中运行图形界面程序
   - 确保宿主机已安装X11服务器

2. **生产环境**：
   - 将文件复制到宿主机查看
   - 使用VNC远程查看
   - 使用Web界面查看

3. **文件管理**：
   - 使用卷挂载方式，方便文件共享
   - 保持文件组织结构清晰
   - 定期备份重要文件

4. **性能考虑**：
   - X11转发可能有一定性能开销
   - 考虑使用VNC或Web界面查看大文件
   - 注意网络带宽使用

5. **安全考虑**：
   - 限制X11转发范围
   - 使用安全的传输方式
   - 定期清理临时文件

这些方法可以根据具体需求选择使用。对于iFlow项目，推荐使用X11转发方式，或者将文件复制到宿主机查看。


太绕了，我选择直接源码部署。



## 我的平台
```
(base) along@alongforhappy: /home/along/Desktop/iFlow/scripts git:(master) 
➜   neofetch               
            .-/+oossssoo+/-.               along@alongforhappy 
        `:+ssssssssssssssssss+:`           ------------------- 
      -+ssssssssssssssssssyyssss+-         OS: Ubuntu 20.04.6 LTS x86_64 
    .ossssssssssssssssssdMMMNysssso.       Host: 81YW Lenovo Legion Y9000K2020H 
   /ssssssssssshdmmNNmmyNMMMMhssssss/      Kernel: 5.15.0-131-generic 
  +ssssssssshmydMMMMMMMNddddyssssssss+     Uptime: 7 hours, 16 mins 
 /sssssssshNMMMyhhyyyyhmNMMMNhssssssss/    Packages: 3076 (dpkg), 12 (snap) 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   Shell: zsh 5.8 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   Resolution: 1920x1080 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   Terminal: /dev/pts/0 
ossyNMMMNyMMhsssssssssssssshmmmhssssssso   CPU: Intel i7-10875H (16) @ 5.100GHz 
+sssshhhyNMMNyssssssssssssyNMMMysssssss+   GPU: NVIDIA 01:00.0 NVIDIA Corporation Device 1f55 
.ssssssssdMMMNhsssssssssshNMMMdssssssss.   GPU: Intel UHD Graphics 
 /sssssssshNMMMyhhyyyyhdNMMMNhssssssss/    Memory: 3881MiB / 64002MiB 
  +sssssssssdmydMMMMMMMMddddyssssssss+
   /ssssssssssshdmNNNNmyNMMMMhssssss/                              
    .ossssssssssssssssssdMMMNysssso.                               
      -+sssssssssssssssssyyyssss+-
        `:+ssssssssssssssssss+:`
            .-/+oossssoo+/-.

```
## iflow使用


## 使用配置脚本

### ubuntu20.04 报错

``` shell
➜   ./run_flow.py -d aes_cipher_top -s synth

Current  full name : aes_cipher_top.synth.yosys_0.9.sky130.HS.TYP.default
Previous full name : aes_cipher_top...sky130.HS.TYP.default

../../../iFlow/tools/yosys4be891e8/bin/yosys: error while loading shared libraries: libtcl8.5.so: cannot open shared object file: No such file or directory

```


这个错误说明你在执行综合步骤时调用的 Yosys 依赖 libtcl8.5.so，而你的系统（Ubuntu 20.04）没有这个版本的动态库。Ubuntu 20.04 默认提供的是 libtcl8.6，所以你需要手动安装旧版本的库。

⸻

推荐解决方案：手动安装 libtcl8.5

步骤 1：下载 .deb 包

wget http://security.ubuntu.com/ubuntu/pool/universe/t/tcl8.5/libtcl8.5_8.5.19-4_amd64.deb

 步骤 2：安装包

`sudo dpkg -i libtcl8.5_8.5.19-4_amd64.deb`

步骤 3：如果有依赖缺失，修复它们

`sudo apt --fix-broken install`

验证是否成功

ls /usr/lib/x86_64-linux-gnu/libtcl8.5.so

然后重新运行你的 iFlow 命令：

`./run_flow.py -d aes_cipher_top -s synth`


⸻

📦 可选方案（如果系统无法装 libtcl8.5）

如果遇到冲突或不能安装 .deb 包，你也可以：
	•	将 libtcl8.5.so 放到 iFlow/tools/yosys 的 lib 目录下
	•	设置环境变量：

`export LD_LIBRARY_PATH=/path/to/libtcl8.5:$LD_LIBRARY_PATH`


### uart示例运行

``` python
uart            = Flow('uart','asap7','HS','TYP','TYP')
```

1. 首先，这个文件定义了一个名为 `Flow` 的类（从导入语句 `from data_def import *` 可以看出），用于配置不同的设计流程。

2. 每个设计流程包含5个参数：
   - 设计名称（design name）
   - 工艺库（technology library）
   - 速度等级（speed grade）
   - 工艺角（process corner）
   - 额外参数（additional parameter）

3. 以 `ibex` 为例：
```python
ibex = Flow('ibex_core', 'sky130', 'HS', 'TYP', '')
```
这表示：
- 设计名称：`ibex_core`（一个RISC-V处理器核心）
- 工艺库：`sky130`（SkyWater 130nm工艺）
- 速度等级：`HS`（High Speed）
- 工艺角：`TYP`（Typical）
- 额外参数：空

4. 其他设计包括：
- `asic_top`：使用SMIC 110nm工艺
- `aes`：使用SkyWater 130nm工艺的AES加密模块
- `gcd`：使用SkyWater 130nm工艺的GCD模块
- `uart`：使用ASAP7 7nm工艺的UART模块
- `picorv32`：使用SkyWater 130nm工艺的PicoRV32处理器

这个配置文件主要用于设置不同数字电路设计项目的工艺参数，这些参数将用于后续的物理设计流程，如综合、布局布线等。每个设计都根据其特定需求选择了合适的工艺库和参数配置。


我来为您详细解读这个UART设计的流程：

1. **设计概述**
- UART（通用异步收发器）是一个百门级到千门级的数字设计
- 用于异步通信的串行数据总线设计

2. **准备工作**
- 首先需要将UART的RTL代码放在 `iFlow/rtl` 目录下
- 在 `iFlow/scripts/cfg/flow_cfg.py` 中定义UART设计的流程参数
- 从配置文件中可以看到，UART设计最初配置为使用ASAP7工艺库，但也可以使用sky130工艺库

3. **脚本准备**
- 进入 `iFlow/scripts` 目录
- 使用命令 `cp -r gcd uart` 复制GCD设计的脚本作为UART设计的基础
- 需要修改的主要是综合脚本 `synth.yosys_0.9.tcl`，将输入文件改为UART的RTL代码

4. **布局规划**
- 由于UART设计与GCD设计规模相近，可以沿用GCD的floorplan设置
- 可以根据需要调整DIE_AREA（芯片面积）和CORE_AREA（核心区域）

5. **运行流程**
使用以下命令运行完整的后端流程：
```bash
./run_flow.py -d uart -s synth,floorplan,tapcell,pdn,gplace,resize,dplace,cts,filler,groute,droute,layout -f sky130 -t HS -c TYP -v V1 -l V1
```

这个命令包含了以下步骤：
- `synth`: 综合
- `floorplan`: 布局规划
- `tapcell`: 添加tap单元
- `pdn`: 电源网络设计
- `gplace`: 全局布局
- `resize`: 单元大小调整
- `dplace`: 详细布局
- `cts`: 时钟树综合
- `filler`: 填充单元
- `groute`: 全局布线
- `droute`: 详细布线
- `layout`: 生成最终版图

参数说明：
- `-d uart`: 指定设计名称为uart
- `-f sky130`: 使用sky130工艺库
- `-t HS`: 使用高速工艺角
- `-c TYP`: 使用典型工艺角
- `-v V1 -l V1`: 版本和标签设置

这个流程是一个完整的数字后端设计流程，从RTL代码开始，经过综合、布局、布线等步骤，最终生成可以用于制造的版图。



## RTL2GDS体验流程

## 完全按照流程走

会一个一个爆错，提示缺失module

``` shell
Traceback (most recent call last):
  File "/usr/lib/python3.10/runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "/usr/lib/python3.10/runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "/rtl2gds/src/rtl2gds/__main__.py", line 47, in <module>
    main()
  File "/rtl2gds/src/rtl2gds/__main__.py", line 39, in main
    flow.rtl2gds_flow.run(chip_design)
  File "/rtl2gds/src/rtl2gds/flow/rtl2gds_flow.py", line 20, in run
    runner.run_floorplan()
  File "/rtl2gds/src/rtl2gds/flow/step_wrapper.py", line 73, in run_floorplan
    metrics, artifacts = step.floorplan.run(
  File "/rtl2gds/src/rtl2gds/step/floorplan.py", line 69, in run
    raise subprocess.CalledProcessError(ret_code, shell_cmd)
subprocess.CalledProcessError: Command '['iEDA', '-script', '/rtl2gds/tools/iEDA/script/iFP_script/run_iFP.tcl']' returned non-zero exit status 127.

```

单独运行
``` shell
root@7a627ed631f6:/rtl2gds/design_zoo/picorv32a# iEDA -script /rtl2gds/tools/iEDA/script/iFP_script/run_iFP.tcl
iEDA: error while loading shared libraries: libglog.so.0: cannot open shared object file: No such file or directory

```

GPT给的答案不对


```
apt install libgoogle-glog-dev
```


脚本设置的环境变量大有问题。包安装的有问题。待整理待提交若干PR

