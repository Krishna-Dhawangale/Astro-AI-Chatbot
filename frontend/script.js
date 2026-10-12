/**
 * Safe Markdown bold parser (produces DocumentFragment with DOM nodes, 0 innerHTML).
 * Strictly formats **bold** tokens into <strong> elements using textContent.
 * All other content is rendered as safe text nodes.
 */
function parseSafeMarkdownToFragment(text, doc = document) {
  const fragment = doc.createDocumentFragment();
  if (typeof text !== "string" || !text) {
    return fragment;
  }

  let i = 0;
  let textStart = 0;

  while (i < text.length) {
    if (text[i] === "*" && text[i + 1] === "*") {
      const closeIdx = text.indexOf("**", i + 2);
      if (closeIdx !== -1 && closeIdx > i + 2) {
        const content = text.slice(i + 2, closeIdx);
        // Valid markdown bold content cannot start/end with space or asterisk
        if (
          content.length > 0 &&
          !content.startsWith(" ") &&
          !content.endsWith(" ") &&
          !content.startsWith("*") &&
          !content.endsWith("*")
        ) {
          if (i > textStart) {
            fragment.appendChild(doc.createTextNode(text.slice(textStart, i)));
          }
          const strong = doc.createElement("strong");
          strong.textContent = content;
          fragment.appendChild(strong);

          i = closeIdx + 2;
          textStart = i;
          continue;
        }
      }
    }
    i++;
  }

  if (textStart < text.length) {
    fragment.appendChild(doc.createTextNode(text.slice(textStart)));
  }

  return fragment;
}

/**
 * Renders bot message container with bold prefix and safely parsed content.
 * Clears and populates using only DOM methods; never touches innerHTML.
 */
function renderSafeBotMessage(container, rawText, doc = document) {
  while (container.firstChild) {
    container.removeChild(container.firstChild);
  }
  container.style.whiteSpace = "pre-wrap";

  const prefix = doc.createElement("b");
  prefix.textContent = "Bot:";
  container.appendChild(prefix);
  container.appendChild(doc.createTextNode(" "));

  if (rawText) {
    container.appendChild(parseSafeMarkdownToFragment(rawText, doc));
  }
}

/**
 * Appends user message to chatBox using text nodes (zero innerHTML injection).
 */
function appendSafeUserMessage(chatBox, text, doc = document) {
  const userMsgDiv = doc.createElement("div");
  userMsgDiv.className = "user-msg";
  userMsgDiv.style.whiteSpace = "pre-wrap";

  const prefix = doc.createElement("b");
  prefix.textContent = "You:";
  userMsgDiv.appendChild(prefix);
  userMsgDiv.appendChild(doc.createTextNode(" " + text));

  chatBox.appendChild(userMsgDiv);
  chatBox.scrollTop = chatBox.scrollHeight;
  return userMsgDiv;
}

async function sendChatMessage() {
  const queryInput = document.getElementById("user-query");
  const chatBox = document.getElementById("chat-box");
  const chipRow = document.getElementById("chipRow") || document.getElementById("followup-chips");
  const query = queryInput.value.trim();

  if (!query) return;

  // Append user's message safely to UI
  appendSafeUserMessage(chatBox, query);
  queryInput.value = "";

  const payload = {
    user_id: "user_session_123", // Unique session identifier
    query: query,
    birth_year: 2000,
    birth_month: 5,
    birth_day: 15,
    birth_hour: 14.5, // 2:30 PM
    latitude: 28.6139,
    longitude: 77.2090
  };

  function updateBottomChips(questions) {
    if (!chipRow || !Array.isArray(questions)) return;
    const valid = questions.filter(q => typeof q === "string" && q.trim()).slice(0, 4);
    if (!valid.length) return;

    while (chipRow.firstChild) {
      chipRow.removeChild(chipRow.firstChild);
    }
    valid.forEach(qText => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "chip";
      chip.textContent = qText.trim();
      chip.addEventListener("click", () => {
        queryInput.value = qText.trim();
        sendChatMessage();
      });
      chipRow.appendChild(chip);
    });
  }

  try {
    const response = await fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(payload),
    });

    // Check for JSON response (Instant static lookup match)
    const contentType = response.headers.get("content-type");
    if (contentType && contentType.includes("application/json")) {
      const data = await response.json();
      const botMsgDiv = document.createElement("div");
      botMsgDiv.className = "bot-msg";
      renderSafeBotMessage(botMsgDiv, data.answer || "");
      chatBox.appendChild(botMsgDiv);
      chatBox.scrollTop = chatBox.scrollHeight;

      const followUps = Array.isArray(data.related_questions) && data.related_questions.length
        ? data.related_questions
        : ["Which career suits me best?", "What does my 7th house say about marriage?"];
      updateBottomChips(followUps);
      return;
    }

    // Otherwise, handle Gemini streaming response
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    
    // Create container for bot reply
    const botMsgDiv = document.createElement("div");
    botMsgDiv.className = "bot-msg";
    renderSafeBotMessage(botMsgDiv, "");
    chatBox.appendChild(botMsgDiv);
    chatBox.scrollTop = chatBox.scrollHeight;

    let streamText = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      streamText += chunk;

      const markerIdx = streamText.indexOf("[FOLLOW_UPS]:");
      const displayText = markerIdx !== -1 ? streamText.slice(0, markerIdx).trim() : streamText;
      renderSafeBotMessage(botMsgDiv, displayText);
      chatBox.scrollTop = chatBox.scrollHeight;
    }

    // Parse follow-ups from stream or use default
    let followUps = null;
    const markerMatch = streamText.match(/\[FOLLOW_UPS\]:\s*(\[.*?\])/s);
    if (markerMatch) {
      try {
        followUps = JSON.parse(markerMatch[1]);
      } catch (e) {
        console.warn("Failed to parse follow-ups JSON:", e);
      }
    }
    if (!Array.isArray(followUps) || !followUps.length) {
      followUps = ["Which career suits me best?", "What does my 7th house say about marriage?"];
    }
    updateBottomChips(followUps);
  } catch (error) {
    console.error("Error connecting to backend:", error);
  }
}

// Module export for Node.js test suites
if (typeof module !== "undefined" && module.exports) {
  module.exports = {
    parseSafeMarkdownToFragment,
    renderSafeBotMessage,
    appendSafeUserMessage,
    sendChatMessage,
  };
}