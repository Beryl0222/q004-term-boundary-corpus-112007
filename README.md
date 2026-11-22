# 术语语料的词边界与发布隔离

使用虚构设备文献和问答制作的术语抽取语料。文献、短答及合作方编号为合成。

资料为 UTF-8 JSON。各类原始记录按编号分组，字段约定见 [资料格式](data/资料格式.md)。资料浏览命令会原样输出记录，添加 `--list` 可只查看各类数量。

使用 Python 3.11，命令只依赖标准库。请在项目根目录运行。

```sh
python3 inspect_records.py data/records.json --list
python3 -m unittest discover -s tests -v
```

## 工具

`termcorpus` 在原始资料之上做对齐、校对和发布隔离。所有命令只读资料包，
需要落盘时显式给出 `--output` 或 `--save`。

```sh
# 读取检查：位置口径、对齐问题、pin 与发布记录一致性
python3 -m termcorpus.cli data/records.json check

# 两家分词到原稿的对齐：原样文本、展示文本、来源区间与对齐后区间
python3 -m termcorpus.cli data/records.json align --record D1

# 语境判定：否定辖域、定义/肯定使用/否定使用/术语提及/比较提及
python3 -m termcorpus.cli data/records.json terms --record D3

# 两种边界、跨词实体、同位置但词义或边界不同的标注
python3 -m termcorpus.cli data/records.json boundaries --record D1

# 泄漏分组及每组证据；--preview 预览某组裁定会牵动哪些样本
python3 -m termcorpus.cli data/records.json groups --preview G1

# 训练材料到校验 pin 的具体泄漏依据
python3 -m termcorpus.cli data/records.json leaks --training D1 D4

# 清理泄漏后的训练集、校验编号与缺失的领域覆盖
python3 -m termcorpus.cli data/records.json plan --training D1 D4 D7

# 已交付版本通知清单
python3 -m termcorpus.cli data/records.json impact --training D1

# 校对状态与局部裁定
python3 -m termcorpus.cli data/records.json review list
python3 -m termcorpus.cli data/records.json review preview A1:m1
python3 -m termcorpus.cli data/records.json review set_boundary A1:m1 --span 8 13 \
    --state state.json --save state.json
python3 -m termcorpus.cli data/records.json review confirm A1:m2 \
    --state state.json --save state.json

# 只导出已确认标注；默认跳过固定校验集同组材料
python3 -m termcorpus.cli data/records.json export --document D7

# 同一字面在不同原稿中的边界、词义与差异解释
python3 -m termcorpus.cli data/records.json history 相变存储器

# 每段标注的来源、偏移口径与对齐说明
python3 -m termcorpus.cli data/records.json provenance --record D6

# 生成发布记录（含隔离结果与清理说明）
python3 -m termcorpus.cli data/records.json release --training D7 \
    --id R2 --date 2026-11-22 --state state.json
```

## 约定

- 源甲（`marking_a`）的区间按 Unicode 字符计数，源乙（`marking_b`）按 UTF-16
  编码单元计数。读入时统一换成原文字符区间，来源声明的位置仍保留。
- 对齐只消除排版差异：空白/换行、全角字形（`Ａ`→`A`）、组合字符表示
  （`e`+`◌́`→`é`）。内容差异不会被吞掉，相关标注进入待校对。
- 下游读取使用 UTF-8 字节区间，允许嵌套、禁止交叉。导出时逐条校验，
  交叉的标注列在 `dropped` 中而不是静默丢弃。
- 语境判定只给建议：否定回答里的术语用法与正面定义分开；否定后回指
  （"该部件"）不会被记到术语字面本身。
- 分组证据是可核对的内容：同一文献片段、同一技术问答、带型号的转述、
  正文指向答案出处的引用。单纯共现同一术语不构成分组证据。
- 校验 `pin` 的 `public_id` 不重排、不替换；清理泄漏造成的领域覆盖缺失
  在 `plan` 的 `missing_coverage` 中列明依据与可补材料。
- 局部裁定只重开被影响的标注；已确认且证据未变的标注保持确认。
