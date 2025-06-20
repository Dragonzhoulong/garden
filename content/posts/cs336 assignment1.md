
### 首先配置好 `vscode` 的 `settings.json`

因为 uv 项目会在工程目录下新建一个 `.venv` 在这里面放置各种环境依赖，而 vscode 进行文件解析（如果把解释器错误的设置为平时用的路径，那么大概率会出现各种不能成功导入包，出现各种报错, 这种错误我曾经一笑而过，但是也确实在潜意识里让我觉得很难受，直到上 jyy 的课时我知道了 vscode 是 `需要` 也是 `可以` 去设置类似解释器，类似调试器路径，类似等等等操作，不过后来还是不太知道该怎么设置，不过让我知道了可以这么设置，而大模型就完全能够让我正确设置 `settings.json`）

``` json
{
    "python.analysis.typeCheckingMode": "basic",
    "python.defaultInterpreterPath": "${workspaceFolder}/assignment1-basics/.venv/bin/python",      
}
```


### 对于大模型工具等声明

该声明是该 assignment 里的，强调了三点
1. 最好不要用各种大模型自动补全工具
2. 也没有大模型补全工具能够完成该作业
3. 使用大模型补全工具会让使用者不能深层地理解该作业



