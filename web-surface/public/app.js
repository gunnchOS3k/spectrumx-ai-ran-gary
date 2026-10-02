const link = document.getElementById("portal-return");
const themeButton = document.getElementById("theme-toggle");
const root = document.body;
function applyTheme(theme) {
  root.setAttribute("data-theme", theme);
  themeButton.textContent = theme === "dark" ? "Light theme" : "Dark theme";
}
themeButton.addEventListener("click", () => {
  applyTheme(root.getAttribute("data-theme") === "dark" ? "light" : "dark");
});
const saved = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
applyTheme(saved);

const streamlitFrame = document.getElementById("spectrumx-streamlit-app");
const streamlitStatus = document.getElementById("streamlit-status");
if (streamlitFrame && streamlitStatus) {
  streamlitFrame.addEventListener("load", () => {
    streamlitStatus.textContent = "The Streamlit frame loaded. If the app is waking from sleep, wait here or open the full Streamlit app.";
  });
}

function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }
function cell(tag, text) {
  const el = document.createElement(tag);
  el.textContent = text == null ? "" : String(text);
  return el;
}

Promise.all([
  fetch("/project-surface.json").then((res) => res.json()),
  fetch("/explore.json").then((res) => res.json())
]).then(([meta, explore]) => {
  if (meta && typeof meta.returnUrl === "string" && /^https:\/\//.test(meta.returnUrl)) {
    link.href = meta.returnUrl;
  }
  const note = document.getElementById("diagram-note");
  note.textContent = explore.diagram_note || "";
  const svg = document.getElementById("diagram");
  const nodes = Array.isArray(explore.nodes) ? explore.nodes : [];
  nodes.forEach((node, index) => {
    const x = 40 + index * 200;
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "node");
    group.setAttribute("tabindex", "0");
    group.setAttribute("role", "button");
    group.setAttribute("aria-label", node.label || "node");
    const rect = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    rect.setAttribute("x", String(x));
    rect.setAttribute("y", "48");
    rect.setAttribute("width", "160");
    rect.setAttribute("height", "64");
    rect.setAttribute("rx", "12");
    rect.setAttribute("fill", "#20343a");
    const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
    text.setAttribute("x", String(x + 80));
    text.setAttribute("y", "86");
    text.setAttribute("text-anchor", "middle");
    text.setAttribute("fill", "#eef2f4");
    text.setAttribute("font-size", "14");
    text.textContent = node.label || "";
    group.appendChild(rect);
    group.appendChild(text);
    group.addEventListener("click", () => { document.getElementById("row-filter").value = node.label || ""; draw(); });
    svg.appendChild(group);
  });
  const select = document.getElementById("row-select");
  const filter = document.getElementById("row-filter");
  const rows = Array.isArray(explore.rows) ? explore.rows : [];
  function visible() {
    const q = filter.value.trim().toLowerCase();
    return rows.filter((row) => !q || JSON.stringify(row).toLowerCase().includes(q));
  }
  function draw() {
    const list = visible();
    clear(select);
    list.forEach((row, index) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = row.scenario_id || row.source || ("Record " + (index + 1));
      select.appendChild(option);
    });
    const keys = [];
    list.forEach((row) => Object.keys(row).forEach((key) => { if (!keys.includes(key) && key !== "ethical_framing") keys.push(key); }));
    const head = document.getElementById("evidence-head");
    const body = document.getElementById("evidence-body");
    clear(head); clear(body);
    keys.slice(0, 6).forEach((key) => head.appendChild(cell("th", key)));
    list.forEach((row) => {
      const tr = document.createElement("tr");
      keys.slice(0, 6).forEach((key) => tr.appendChild(cell("td", row[key] || "")));
      body.appendChild(tr);
    });
    const current = list[Number(select.value)] || list[0];
    const detail = document.getElementById("detail");
    clear(detail);
    if (!current) {
      detail.appendChild(cell("p", "No repository record matches that filter."));
      return;
    }
    detail.appendChild(cell("h3", current.scenario_id || "Record"));
    detail.appendChild(cell("p", current.ethical_framing || current.source || ""));
    detail.appendChild(cell("p", "Source file: " + (current.source || "repository")));
    const directory = document.getElementById("directory");
    clear(directory);
    (explore.cards || []).forEach((card) => {
      const article = document.createElement("article");
      article.appendChild(cell("h3", card.title || ""));
      article.appendChild(cell("p", card.body || ""));
      directory.appendChild(article);
    });
  }
  select.addEventListener("change", draw);
  filter.addEventListener("input", draw);
  draw();
}).catch(() => {});
