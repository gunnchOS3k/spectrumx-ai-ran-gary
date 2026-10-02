import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.join(path.dirname(fileURLToPath(import.meta.url)), "public");
const html = fs.readFileSync(path.join(root, "index.html"), "utf8");
const css = fs.readFileSync(path.join(root, "styles.css"), "utf8");
const js = fs.readFileSync(path.join(root, "app.js"), "utf8");
const meta = JSON.parse(fs.readFileSync(path.join(root, "project-surface.json"), "utf8"));
assert.match(html, /<main/);
assert.match(html, /id="content"/);
assert.match(html, /aria-label="Return to gunnchOS"/);
assert.match(html, /← gunnchOS/);
assert.match(html, /STAGING_WORKER/);
assert.match(html, /What is this\?/);
assert.match(html, /What is not finished\?/);
assert.match(css, /@media \(max-width: 800px\)/);
assert.match(css, /a:focus-visible/);
for (const [blob, name] of [[html, "html"], [css, "css"], [js, "js"]]) {
  assert.equal(blob.includes("workers.dev"), false, name);
  assert.equal(blob.includes("gunnchos-finds"), false, name);
  assert.equal(blob.includes("FINDS"), false, name);
}
assert.equal(meta.returnUrl.startsWith("https://"), true);
assert.equal(meta.status, "STAGING_WORKER");
assert.ok(meta.classification);
const low = html.toLowerCase();
for (const phrase of ["play now", ">live<", "official oulu"]) {
  assert.equal(low.includes(phrase), false, phrase);
}
const text = html + css + js + JSON.stringify(meta);
assert.equal(/AKIA[0-9A-Z]{16}|BEGIN (RSA |OPENSSH )?PRIVATE KEY|api_key\s*=\s*['"]/.test(text), false);
console.log("surface tests ok", meta.repo);
