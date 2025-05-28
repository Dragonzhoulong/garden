## 首先运行uart设计熟悉iFlow流程

uart设计是一种通用串行数据总线的设计，是一个百门级到千门级的设计，用于异步通信。下面将讲述一下如何将iFlow中的design更换为uart设计。

首先，需要把uart相关的rtl代码放在“iFlow/rtl”目录下，如图34所示，然后要定义uart设计的flow，进入到“iFlow/scripts/cfg”目录下，编辑脚本“flow_cfg.py”，加入uart设计的默认flow参数，如图35所示，这里设置的默认foundry为“asap7”，改为“sky130”也是可以的。

其余的步骤可以不用修改，进入目录“iFlow/scripts”，运行命令：

```
./run_flow.py -d uart -s synth,floorplan,tapcell,pdn,gplace,resize,dplace,cts,filler,groute,droute,layout -f sky130 -t HS -c TYP -v V1 -l V1

```

即可基于sky130工艺跑uart设计的后端流程，结果如图所示。
![uartgds](posts/images/uartgds.png)

iFlow的文档看的一头雾水，我将先参考一下OpenLane的文档跑出来picorv32先。

## OpenLane
### OpenLane架构

	

![[posts/images/openlane.png]]

## iFlow报错

### 先卡在了布局阶段

``` zsh
ABC: Scl_LibertyReadGenlib() skipped three-state cell "sky130_fd_sc_hs__ebufn_2".
ABC: Scl_LibertyReadGenlib() skipped three-state cell "sky130_fd_sc_hs__ebufn_4".
ABC: Scl_LibertyReadGenlib() skipped three-state cell "sky130_fd_sc_hs__ebufn_8".
ABC: Scl_LibertyReadGenlib() skipped sequential cell "sky130_fd_sc_hs__edfxbp_1".
ABC: Scl_LibertyReadGenlib() skipped sequential cell "sky130_fd_sc_hs__edfxtp_1".
ABC: Scl_LibertyReadGenlib() skipped three-state cell "sky130_fd_sc_hs__einvn_1".
ABC: Scl_LibertyReadGenlib() skERROR: Can't open ABC output file `/tmp/yosys-abc-KVSWxH/output.blif'.
```
在确认磁盘`tmp` 空间正常后，继续**尝试手动运行 abc 看报错**,

####  尝试手动运行 abc 看报错

观察到你运行时 /tmp/yosys-abc-*/ 目录下：
	•	有 abc.script、input.blif、stdcells.genlib
	•	但始终没有 output.blif 被生成
	•	说明 ABC 在执行过程中根本没有成功生成输出文件

⸻

✅ 你现在的情况基本确认是：

ABC 执行失败，没有报出详细错误，但未生成 output.blif。

这通常是因为：
	1.	.lib（转换成 .genlib）中大部分逻辑单元不被 ABC 接受或不包含功能定义；
	2.	ABC 运行时遇到非法逻辑结构但不报错；
	3.	环境变量或文件权限/路径问题（现在暂时不考虑这个，因为你能读写该目录）。

⸻

✅ 建议下一步这样处理：

🔧 方法一：手动运行 ABC 并开启 debug 模式

复制当前生成的文件夹进行调试：

``` zsh
cp -r /tmp/yosys-abc-KVSWxH ~/abc_debug
cd ~/abc_debug
~/Desktop/iFlow/tools/yosys4be891e8/bin/yosys-abc -s -f abc.script
```

如果能跑出来会生成 output.blif，否则就会直接在终端报错。

你还可以尝试：

`~/Desktop/iFlow/tools/yosys4be891e8/bin/yosys-abc`



错误信息中有关键点：

``` zsh
+ read_constr -v /home/along/Desktop/iFlow/rtl/picorv32/picorv32.sdc 
Cannot open input file "/home/along/Desktop/iFlow/rtl/picorv32/picorv32.sdc". 
** cmd error: aborting 'source abc.script'
```
说明 ABC 在执行 abc.script 里的 read_constr 命令时，找不到时序约束文件 picorv32.sdc。


这就是导致 ABC 无法完成并报错的根本原因。



解决办法：
	1.	确认文件是否存在

ls -l /home/along/Desktop/iFlow/rtl/picorv32/picorv32.sdc

	•	如果不存在，说明路径错了或者文件没放好。

	2.	如果文件不存在，你有该 SDC 文件吗？
	•	如果没有，你需要从项目或者网络资源获取这个时序约束文件。
	•	如果暂时不需要约束，可以临时修改 abc.script 去掉或注释掉 read_constr 这一行。
	3.	修改 abc.script
找到 read_constr -v /home/along/Desktop/iFlow/rtl/picorv32/picorv32.sdc 这一行，注释掉（前面加 #）或者删除。
	4.	再重新运行 ABC

~/Desktop/iFlow/tools/yosys4be891e8/bin/yosys-abc -s -f /tmp/yosys-abc-KVSWxH/abc.script

或者你复制到本地目录后手动运行调试。


发现原来是我的sdc文件名后缀命名错误。同时单步执行也让我更容易定位问题。



### [ERROR GPL-0301] Utilization exceeds 100%.

需要修改`/iFlow/scripts/picorv32/floorplan.openroad_1.2.0.tcl`

``` tcl
#set DIE_AREA            "0 0 220.2 220.2" 
#set CORE_AREA           "1.08 1.08 219.12 219.12" 
set DIE_AREA            "0 0 2020.2 2020.2" 
set CORE_AREA           "1.08 1.08 2019.12 2019.12" 

```



	1.	Core Area定义错误
	•	你日志里看到CoreAreaLxLy: 960 0和CoreAreaUxUy: 218880 216450，core面积计算很大（4.7e10 dbu²），确认这是否合理。
	•	Core定义如果不合理，会导致单元布局范围混乱。
	2.	实例数量和面积统计异常
	•	总实例数14714，放置实例14082，面积累积巨大。
	•	但是面积统计（PlaceInstsArea: 176776646400）比Core面积还大，造成超利用率。
	3.	DEF或LEF文件是否匹配
	•	LEF中芯片尺寸、宏单元和标准单元尺寸需与DEF和设计匹配。
	•	栅格和site尺寸是否与LEF定义一致。
	4.	Floorplan设置不合理
	•	floorplan脚本中的core区域定义，是否误设成了太小的范围，导致放置实例溢出。
	•	重新确认floorplan配置，确保core区域足够且合理。
	5.	PDN或tapcell阶段对布局没有限制或约束
	•	tapcell插入通常不会改变实例数，但可能对布局边界有要求。


``` bash
[INFO ODB-0225]     Created 1 library cells
[INFO ODB-0226] Finished LEF file:  /home/along/Desktop/iFlow/foundry/sky130/lef/sky130_ef_io__vssio_lvc_pad.lef
[INFO ODB-0222] Reading LEF file: /home/along/Desktop/iFlow/foundry/sky130/lef/sky130_fd_io__top_xres4v2.lef
[INFO ODB-0225]     Created 1 library cells
[INFO ODB-0226] Finished LEF file:  /home/along/Desktop/iFlow/foundry/sky130/lef/sky130_fd_io__top_xres4v2.lef
[INFO ODB-0222] Reading LEF file: /home/along/Desktop/iFlow/foundry/sky130/lef/sky130io_fill.lef
[INFO ODB-0225]     Created 2 library cells
[INFO ODB-0226] Finished LEF file:  /home/along/Desktop/iFlow/foundry/sky130/lef/sky130io_fill.lef
[INFO ODB-0222] Reading LEF file: /home/along/Desktop/iFlow/foundry/sky130/lef/sky130_sram_1rw1r_128x256_8.lef
[INFO ODB-0221] 1000000 lines parsed!
[INFO ODB-0225]     Created 1 library cells
[INFO ODB-0226] Finished LEF file:  /home/along/Desktop/iFlow/foundry/sky130/lef/sky130_sram_1rw1r_128x256_8.lef
[INFO ODB-0222] Reading LEF file: /home/along/Desktop/iFlow/foundry/sky130/lef/sky130_sram_1rw1r_44x64_8.lef
[INFO ODB-0225]     Created 1 library cells
[INFO ODB-0226] Finished LEF file:  /home/along/Desktop/iFlow/foundry/sky130/lef/sky130_sram_1rw1r_44x64_8.lef
[INFO ODB-0222] Reading LEF file: /home/along/Desktop/iFlow/foundry/sky130/lef/sky130_sram_1rw1r_64x256_8.lef
[INFO ODB-0221] 1000000 lines parsed!
[INFO ODB-0225]     Created 1 library cells
[INFO ODB-0226] Finished LEF file:  /home/along/Desktop/iFlow/foundry/sky130/lef/sky130_sram_1rw1r_64x256_8.lef
[INFO ODB-0222] Reading LEF file: /home/along/Desktop/iFlow/foundry/sky130/lef/sky130_sram_1rw1r_80x64_8.lef
[INFO ODB-0225]     Created 1 library cells
[INFO ODB-0226] Finished LEF file:  /home/along/Desktop/iFlow/foundry/sky130/lef/sky130_sram_1rw1r_80x64_8.lef
[INFO ODB-0127] Reading DEF file: /home/along/Desktop/iFlow/result/picorv32.resize.openroad_1.2.0.sky130.HS.TYP.default/picorv32.def
[INFO ODB-0128] Design: picorv32
[INFO ODB-0130]     Created 409 pins.
[INFO ODB-0131]     Created 59149 components and 281634 component-terminals.
[INFO ODB-0132]     Created 2 special nets and 236596 connections.
[INFO ODB-0133]     Created 14264 nets and 45002 connections.
[INFO ODB-0134] Finished DEF file: /home/along/Desktop/iFlow/result/picorv32.resize.openroad_1.2.0.sky130.HS.TYP.default/picorv32.def
Placement Analysis
---------------------------------
total displacement     735617.1 u
average displacement       12.4 u
max displacement          371.7 u
original HPWL          932041.3 u
legalized HPWL        2120837.4 u
delta HPWL                  128 %
```



## report解读
