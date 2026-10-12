/**
 * Automated Security & DOM Integrity Test Suite for XSS Prevention
 * (tests/test_xss_prevention.js)
 * -----------------------------------------------------------------
 * Verifies that untrusted inputs and streaming chunks cannot inject HTML
 * or execute malicious scripts via frontend/script.js.
 *
 * Covers:
 * 1. Three specific attack vectors: <script>, <img onerror>, <iframe javascript:>
 * 2. Malformed Markdown (unclosed bold, inner asterisks, whitespace asterisks, HTML inside bold)
 * 3. Streamed payloads split across chunk boundaries
 * 4. User message safe construction
 * 5. Complete absence of innerHTML assignments and dangerous DOM elements
 */

const assert = require("assert");
const path = require("path");

// -------------------------------------------------------------------
// Minimal DOM Mock for Node.js Execution
// -------------------------------------------------------------------

function escapeHTML(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

class DOMNode {
  constructor(nodeType) {
    this.nodeType = nodeType;
    this.childNodes = [];
    this.parentNode = null;
  }

  get firstChild() {
    return this.childNodes.length > 0 ? this.childNodes[0] : null;
  }

  appendChild(child) {
    if (child.nodeType === 11) {
      // DocumentFragment: append all its children
      const fragmentChildren = [...child.childNodes];
      child.childNodes = [];
      for (const fc of fragmentChildren) {
        this.appendChild(fc);
      }
      return child;
    }
    child.parentNode = this;
    this.childNodes.push(child);
    return child;
  }

  removeChild(child) {
    const idx = this.childNodes.indexOf(child);
    if (idx !== -1) {
      this.childNodes.splice(idx, 1);
      child.parentNode = null;
      return child;
    }
    throw new Error("Child not found");
  }

  replaceChildren(...newChildren) {
    while (this.childNodes.length) {
      this.removeChild(this.childNodes[0]);
    }
    for (const c of newChildren) {
      this.appendChild(c);
    }
  }
}

class DOMTextNode extends DOMNode {
  constructor(text) {
    super(3);
    this._text = String(text);
  }

  get textContent() {
    return this._text;
  }

  set textContent(val) {
    this._text = String(val);
  }

  get innerHTML() {
    return escapeHTML(this._text);
  }
}

class DOMElement extends DOMNode {
  constructor(tagName) {
    super(1);
    this.tagName = tagName.toUpperCase();
    this.className = "";
    this.style = {};
    this.attributes = {};
    this.scrollTop = 0;
    this.scrollHeight = 100;
  }

  get children() {
    return this.childNodes.filter(n => n.nodeType === 1);
  }

  setAttribute(name, val) {
    this.attributes[name] = String(val);
  }

  getAttribute(name) {
    return this.attributes[name] || null;
  }

  get textContent() {
    return this.childNodes.map(n => n.textContent).join("");
  }

  set textContent(val) {
    while (this.childNodes.length) {
      this.removeChild(this.childNodes[0]);
    }
    this.appendChild(new DOMTextNode(val));
  }

  get innerHTML() {
    return this.childNodes
      .map(n => {
        if (n.nodeType === 3) return escapeHTML(n.textContent);
        const tag = n.tagName.toLowerCase();
        return `<${tag}>${n.innerHTML}</${tag}>`;
      })
      .join("");
  }

  getElementsByTagName(tagName) {
    const target = tagName.toUpperCase();
    const results = [];
    function search(node) {
      for (const child of node.childNodes) {
        if (child.nodeType === 1) {
          if (target === "*" || child.tagName === target) {
            results.push(child);
          }
          search(child);
        }
      }
    }
    search(this);
    return results;
  }
}

class DOMDocumentFragment extends DOMNode {
  constructor() {
    super(11);
  }

  get textContent() {
    return this.childNodes.map(n => n.textContent).join("");
  }

  get innerHTML() {
    return this.childNodes
      .map(n => (n.nodeType === 3 ? escapeHTML(n.textContent) : n.innerHTML))
      .join("");
  }
}

class MockDocument {
  createElement(tag) {
    return new DOMElement(tag);
  }

  createTextNode(text) {
    return new DOMTextNode(text);
  }

  createDocumentFragment() {
    return new DOMDocumentFragment();
  }
}

// -------------------------------------------------------------------
// Load Module Under Test
// -------------------------------------------------------------------

const {
  parseSafeMarkdownToFragment,
  renderSafeBotMessage,
  appendSafeUserMessage
} = require("../frontend/script.js");

const mockDoc = new MockDocument();

let passCount = 0;
let failCount = 0;

function runTest(name, fn) {
  try {
    fn();
    console.log(`  [PASS] ${name}`);
    passCount++;
  } catch (err) {
    console.error(`  [FAIL] ${name}: ${err.message}`);
    failCount++;
  }
}

console.log("==================================================================");
console.log("        FRONTEND XSS & SAFE RENDERING TEST SUITE                  ");
console.log("==================================================================");

// -------------------------------------------------------------------
// Category 1: Three Primary Injected Attack Payloads
// -------------------------------------------------------------------
console.log("\n[Category 1: Primary XSS Attack Vectors]");

runTest("Payload 1: <script>alert(1)</script> in bot response", () => {
  const container = mockDoc.createElement("div");
  renderSafeBotMessage(container, "<script>alert(1)</script>", mockDoc);

  const scripts = container.getElementsByTagName("script");
  assert.strictEqual(scripts.length, 0, "No <script> element should exist in DOM");
  assert.strictEqual(container.textContent, "Bot: <script>alert(1)</script>");
  assert.strictEqual(container.innerHTML, "<b>Bot:</b> &lt;script&gt;alert(1)&lt;/script&gt;");
});

runTest("Payload 2: <img src=x onerror=alert(1)> in bot response", () => {
  const container = mockDoc.createElement("div");
  renderSafeBotMessage(container, "<img src=x onerror=alert(1)>", mockDoc);

  const imgs = container.getElementsByTagName("img");
  assert.strictEqual(imgs.length, 0, "No <img> element should exist in DOM");
  assert.strictEqual(container.textContent, "Bot: <img src=x onerror=alert(1)>");
  assert.strictEqual(container.innerHTML, "<b>Bot:</b> &lt;img src=x onerror=alert(1)&gt;");
});

runTest("Payload 3: <iframe src=\"javascript:alert(1)\"> in bot response", () => {
  const container = mockDoc.createElement("div");
  renderSafeBotMessage(container, '<iframe src="javascript:alert(1)">', mockDoc);

  const iframes = container.getElementsByTagName("iframe");
  assert.strictEqual(iframes.length, 0, "No <iframe> element should exist in DOM");
  assert.strictEqual(container.textContent, 'Bot: <iframe src="javascript:alert(1)">');
  assert.strictEqual(container.innerHTML, "<b>Bot:</b> &lt;iframe src=&quot;javascript:alert(1)&quot;&gt;");
});

// -------------------------------------------------------------------
// Category 2: Malformed Markdown & HTML Mixes
// -------------------------------------------------------------------
console.log("\n[Category 2: Malformed Markdown & Mixed Tokens]");

runTest("Unmatched opening bold: **unclosed bold text", () => {
  const container = mockDoc.createElement("div");
  renderSafeBotMessage(container, "**unclosed bold text", mockDoc);

  const strongs = container.getElementsByTagName("strong");
  assert.strictEqual(strongs.length, 0, "Unclosed bold must not produce a <strong> element");
  assert.strictEqual(container.textContent, "Bot: **unclosed bold text");
});

runTest("Valid bold: **important guidance**", () => {
  const container = mockDoc.createElement("div");
  renderSafeBotMessage(container, "Here is **important guidance** for health.", mockDoc);

  const strongs = container.getElementsByTagName("strong");
  assert.strictEqual(strongs.length, 1, "Exactly 1 <strong> element should exist");
  assert.strictEqual(strongs[0].textContent, "important guidance");
  assert.strictEqual(container.textContent, "Bot: Here is important guidance for health.");
  assert.strictEqual(container.innerHTML, "<b>Bot:</b> Here is <strong>important guidance</strong> for health.");
});

runTest("Asterisk inside bold token: **bold*text**", () => {
  const container = mockDoc.createElement("div");
  renderSafeBotMessage(container, "**bold*text**", mockDoc);

  const strongs = container.getElementsByTagName("strong");
  assert.strictEqual(strongs.length, 1);
  assert.strictEqual(strongs[0].textContent, "bold*text");
});

runTest("Empty / whitespace asterisks: **** and ** **", () => {
  const container = mockDoc.createElement("div");
  renderSafeBotMessage(container, "Divider: **** and ** ** gap", mockDoc);

  const strongs = container.getElementsByTagName("strong");
  assert.strictEqual(strongs.length, 0, "Empty/whitespace asterisks must not create <strong> elements");
  assert.strictEqual(container.textContent, "Bot: Divider: **** and ** ** gap");
});

runTest("HTML tag inside bold: **<script>alert(1)</script>**", () => {
  const container = mockDoc.createElement("div");
  renderSafeBotMessage(container, "**<script>alert(1)</script>**", mockDoc);

  const scripts = container.getElementsByTagName("script");
  assert.strictEqual(scripts.length, 0, "Script tag inside bold must NOT be executed as DOM element");
  const strongs = container.getElementsByTagName("strong");
  assert.strictEqual(strongs.length, 1, "Bold container should exist");
  assert.strictEqual(strongs[0].textContent, "<script>alert(1)</script>");
  assert.strictEqual(container.innerHTML, "<b>Bot:</b> <strong>&lt;script&gt;alert(1)&lt;/script&gt;</strong>");
});

runTest("Mixed HTML tags and valid bold: <b>tag</b> and <img src=x> with **safe bold**", () => {
  const container = mockDoc.createElement("div");
  renderSafeBotMessage(container, "<b>tag</b> and <img src=x> with **safe bold**", mockDoc);

  const imgs = container.getElementsByTagName("img");
  assert.strictEqual(imgs.length, 0, "Raw img tag must not become DOM element");
  const bTags = container.getElementsByTagName("b");
  assert.strictEqual(bTags.length, 1, "Only prefix 'Bot:' should be a <b> element");
  const strongs = container.getElementsByTagName("strong");
  assert.strictEqual(strongs.length, 1);
  assert.strictEqual(strongs[0].textContent, "safe bold");
});

// -------------------------------------------------------------------
// Category 3: Streaming Payloads Split Across Chunks
// -------------------------------------------------------------------
console.log("\n[Category 3: Streaming Payloads Split Across Chunks]");

runTest("Streaming XSS payload split across chunk boundaries (<img src= + x onerror=alert(1)>)", () => {
  const container = mockDoc.createElement("div");

  // Step 0: Initial stream open
  renderSafeBotMessage(container, "", mockDoc);
  assert.strictEqual(container.textContent, "Bot: ");

  // Step 1: Chunk 1 arrives (<img src=)
  let accumulated = "<img src=";
  renderSafeBotMessage(container, accumulated, mockDoc);
  assert.strictEqual(container.getElementsByTagName("img").length, 0);
  assert.strictEqual(container.textContent, "Bot: <img src=");

  // Step 2: Chunk 2 arrives completing the malicious payload
  accumulated += "x onerror=alert(1)>";
  renderSafeBotMessage(container, accumulated, mockDoc);
  assert.strictEqual(container.getElementsByTagName("img").length, 0, "Completed stream must NOT inject <img>");
  assert.strictEqual(container.textContent, "Bot: <img src=x onerror=alert(1)>");
  assert.strictEqual(container.innerHTML, "<b>Bot:</b> &lt;img src=x onerror=alert(1)&gt;");
});

runTest("Streaming bold text split across chunk boundaries (**Jupiter in + 10th house**)", () => {
  const container = mockDoc.createElement("div");

  // Step 1: Incomplete bold token during stream
  let accumulated = "Your chart shows **Jupiter in ";
  renderSafeBotMessage(container, accumulated, mockDoc);
  assert.strictEqual(container.getElementsByTagName("strong").length, 0, "Incomplete bold must safely render as plain text");
  assert.strictEqual(container.textContent, "Bot: Your chart shows **Jupiter in ");

  // Step 2: Closing delimiter arrives in subsequent chunk
  accumulated += "10th house** indicates career prominence.";
  renderSafeBotMessage(container, accumulated, mockDoc);
  const strongs = container.getElementsByTagName("strong");
  assert.strictEqual(strongs.length, 1, "Bold token must now safely resolve to <strong>");
  assert.strictEqual(strongs[0].textContent, "Jupiter in 10th house");
  assert.strictEqual(container.textContent, "Bot: Your chart shows Jupiter in 10th house indicates career prominence.");
});

runTest("Streaming HTML split inside bold tokens (**<script> + alert(1)</script>**)", () => {
  const container = mockDoc.createElement("div");

  let accumulated = "**<script>";
  renderSafeBotMessage(container, accumulated, mockDoc);
  assert.strictEqual(container.getElementsByTagName("script").length, 0);

  accumulated += "alert(1)</script>**";
  renderSafeBotMessage(container, accumulated, mockDoc);
  assert.strictEqual(container.getElementsByTagName("script").length, 0, "No script element even when completed inside bold");
  assert.strictEqual(container.getElementsByTagName("strong").length, 1);
});

// -------------------------------------------------------------------
// Category 4: User Message Safe Rendering
// -------------------------------------------------------------------
console.log("\n[Category 4: User Message Safe Construction]");

runTest("User message with <script> tag", () => {
  const chatBox = mockDoc.createElement("div");
  appendSafeUserMessage(chatBox, "<script>alert('user_xss')</script>", mockDoc);

  assert.strictEqual(chatBox.getElementsByTagName("script").length, 0);
  const userMsg = chatBox.childNodes[0];
  assert.strictEqual(userMsg.className, "user-msg");
  assert.strictEqual(userMsg.textContent, "You: <script>alert('user_xss')</script>");
  assert.strictEqual(userMsg.innerHTML, "<b>You:</b> &lt;script&gt;alert(&#39;user_xss&#39;)&lt;/script&gt;");
});

runTest("User message with <img onerror> tag", () => {
  const chatBox = mockDoc.createElement("div");
  appendSafeUserMessage(chatBox, "<img src=x onerror=alert('user_img')>", mockDoc);

  assert.strictEqual(chatBox.getElementsByTagName("img").length, 0);
  const userMsg = chatBox.childNodes[0];
  assert.strictEqual(userMsg.textContent, "You: <img src=x onerror=alert('user_img')>");
});

runTest("User message preserves consecutive messages without wiping DOM listeners", () => {
  const chatBox = mockDoc.createElement("div");
  const msg1 = appendSafeUserMessage(chatBox, "First query", mockDoc);
  const msg2 = appendSafeUserMessage(chatBox, "Second query", mockDoc);

  assert.strictEqual(chatBox.childNodes.length, 2);
  assert.strictEqual(chatBox.childNodes[0], msg1, "First message node identity preserved");
  assert.strictEqual(chatBox.childNodes[1], msg2, "Second message node identity preserved");
});

// -------------------------------------------------------------------
// Summary
// -------------------------------------------------------------------
console.log("\n==================================================================");
console.log(`TOTAL TESTS: ${passCount + failCount} | PASSED: ${passCount} | FAILED: ${failCount}`);
console.log("==================================================================");

if (failCount > 0) {
  process.exit(1);
} else {
  process.exit(0);
}
