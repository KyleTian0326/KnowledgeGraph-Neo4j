from pathlib import Path


NODES = [
    {"id": "中石化某炼化企业", "label": "Company", "desc": "石化企业主体"},
    {"id": "乙烯装置", "label": "Unit", "desc": "生产乙烯和丙烯的生产装置"},
    {"id": "石脑油", "label": "Material", "desc": "乙烯装置的主要裂解原料"},
    {"id": "蒸汽裂解", "label": "Process", "desc": "高温裂解生产低碳烯烃的工艺过程"},
    {"id": "裂解炉", "label": "Equipment", "desc": "乙烯装置中的关键设备"},
    {"id": "炉管出口温度", "label": "Parameter", "desc": "裂解炉运行控制参数"},
    {"id": "稀释蒸汽比", "label": "Parameter", "desc": "裂解炉运行控制参数"},
    {"id": "乙烯", "label": "Product", "desc": "乙烯装置主要产品"},
    {"id": "丙烯", "label": "Product", "desc": "乙烯装置联产产品"},
    {"id": "聚乙烯", "label": "Product", "desc": "乙烯下游产品"},
    {"id": "聚丙烯", "label": "Product", "desc": "丙烯下游产品"},
    {"id": "高温泄漏风险", "label": "Risk", "desc": "重点防范的安全风险"},
    {"id": "安全生产标准", "label": "Standard", "desc": "装置运行需要符合的标准"},
]

LINKS = [
    {"source": "中石化某炼化企业", "target": "乙烯装置", "type": "OWNS"},
    {"source": "乙烯装置", "target": "石脑油", "type": "USES_MATERIAL"},
    {"source": "乙烯装置", "target": "蒸汽裂解", "type": "USES_PROCESS"},
    {"source": "乙烯装置", "target": "裂解炉", "type": "HAS_EQUIPMENT"},
    {"source": "乙烯装置", "target": "乙烯", "type": "PRODUCES"},
    {"source": "乙烯装置", "target": "丙烯", "type": "PRODUCES"},
    {"source": "裂解炉", "target": "炉管出口温度", "type": "CONTROLS"},
    {"source": "裂解炉", "target": "稀释蒸汽比", "type": "CONTROLS"},
    {"source": "乙烯", "target": "聚乙烯", "type": "USED_FOR"},
    {"source": "丙烯", "target": "聚丙烯", "type": "USED_FOR"},
    {"source": "乙烯装置", "target": "高温泄漏风险", "type": "HAS_RISK"},
    {"source": "乙烯装置", "target": "安全生产标准", "type": "COMPLIES_WITH"},
]


HTML = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>石化知识图谱 Demo</title>
  <style>
    body { margin: 0; font-family: "Microsoft YaHei", Arial, sans-serif; background: #f6f7f9; color: #1f2933; }
    header { padding: 20px 28px; background: #102a43; color: white; }
    h1 { margin: 0; font-size: 24px; }
    main { display: grid; grid-template-columns: 320px 1fr; min-height: calc(100vh - 72px); }
    aside { padding: 20px; background: white; border-right: 1px solid #d9e2ec; overflow: auto; }
    .stage { position: relative; overflow: hidden; }
    svg { width: 100%; height: 100%; min-height: 720px; display: block; }
    .step { margin-bottom: 18px; padding-bottom: 16px; border-bottom: 1px solid #e4e7eb; }
    .step h2 { margin: 0 0 8px; font-size: 16px; color: #102a43; }
    .step p { margin: 0; line-height: 1.6; font-size: 14px; }
    .node { cursor: pointer; }
    .node circle { stroke: white; stroke-width: 2; }
    .node text { font-size: 13px; paint-order: stroke; stroke: white; stroke-width: 4px; stroke-linejoin: round; fill: #1f2933; }
    .link { stroke: #829ab1; stroke-width: 1.8; marker-end: url(#arrow); }
    .link-label { font-size: 11px; fill: #52606d; paint-order: stroke; stroke: white; stroke-width: 3px; }
    .legend { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 10px; }
    .chip { font-size: 12px; padding: 4px 7px; border-radius: 4px; background: #eef2f7; }
    @media (max-width: 900px) { main { grid-template-columns: 1fr; } aside { border-right: 0; border-bottom: 1px solid #d9e2ec; } }
  </style>
</head>
<body>
  <header><h1>石化知识图谱构建 Demo</h1></header>
  <main>
    <aside>
      <div class="step">
        <h2>1. 原始资料</h2>
        <p>乙烯装置以石脑油为原料，通过裂解炉进行蒸汽裂解，生产乙烯和丙烯。</p>
      </div>
      <div class="step">
        <h2>2. 抽取实体</h2>
        <p>识别企业、装置、原料、设备、工艺、产品、参数、风险和标准。</p>
      </div>
      <div class="step">
        <h2>3. 抽取关系</h2>
        <p>形成 OWNS、PRODUCES、CONTROLS、USED_FOR 等可查询关系。</p>
      </div>
      <div class="step">
        <h2>4. 图谱展示</h2>
        <p>节点代表实体，箭头代表关系。后续接入 Neo4j 后，这些节点和关系会写入图数据库。</p>
      </div>
      <div class="legend" id="legend"></div>
    </aside>
    <section class="stage">
      <svg id="graph" viewBox="0 0 980 720" role="img" aria-label="知识图谱"></svg>
    </section>
  </main>
  <script>
    const nodes = __NODES__;
    const links = __LINKS__;
    const colors = {
      Company: "#2f80ed", Unit: "#00a676", Material: "#f2994a", Process: "#9b51e0",
      Equipment: "#eb5757", Parameter: "#56ccf2", Product: "#27ae60", Risk: "#d64545", Standard: "#607d8b"
    };
    const positions = {
      "中石化某炼化企业": [490, 70], "乙烯装置": [490, 210], "石脑油": [210, 250],
      "蒸汽裂解": [330, 360], "裂解炉": [610, 360], "炉管出口温度": [515, 505],
      "稀释蒸汽比": [705, 505], "乙烯": [365, 505], "丙烯": [855, 320],
      "聚乙烯": [270, 625], "聚丙烯": [855, 475], "高温泄漏风险": [115, 420],
      "安全生产标准": [115, 560]
    };
    const byId = Object.fromEntries(nodes.map(n => [n.id, n]));
    const svg = document.getElementById("graph");

    function el(name, attrs = {}) {
      const node = document.createElementNS("http://www.w3.org/2000/svg", name);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      return node;
    }

    const defs = el("defs");
    defs.innerHTML = '<marker id="arrow" viewBox="0 0 10 10" refX="18" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill="#829ab1"></path></marker>';
    svg.appendChild(defs);

    for (const link of links) {
      const [x1, y1] = positions[link.source];
      const [x2, y2] = positions[link.target];
      svg.appendChild(el("line", { class: "link", x1, y1, x2, y2 }));
      const label = el("text", { class: "link-label", x: (x1 + x2) / 2, y: (y1 + y2) / 2 - 6, "text-anchor": "middle" });
      label.textContent = link.type;
      svg.appendChild(label);
    }

    for (const node of nodes) {
      const [x, y] = positions[node.id];
      const group = el("g", { class: "node" });
      group.appendChild(el("circle", { cx: x, cy: y, r: 28, fill: colors[node.label] || "#888" }));
      const text = el("text", { x, y: y + 46, "text-anchor": "middle" });
      text.textContent = node.id;
      group.appendChild(text);
      group.appendChild(el("title")).textContent = `${node.label}: ${node.desc}`;
      svg.appendChild(group);
    }

    const legend = document.getElementById("legend");
    for (const label of Object.keys(colors)) {
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.style.borderLeft = `10px solid ${colors[label]}`;
      chip.textContent = label;
      legend.appendChild(chip);
    }
  </script>
</body>
</html>
"""


def main() -> None:
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    html = HTML.replace("__NODES__", repr(NODES)).replace("__LINKS__", repr(LINKS))
    output_path = output_dir / "demo_kg.html"
    output_path.write_text(html, encoding="utf-8")
    print(output_path.resolve())


if __name__ == "__main__":
    main()
