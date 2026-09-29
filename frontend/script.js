async function sendChatMessage() {
  const queryInput = document.getElementById("user-query");
  const chatBox = document.getElementById("chat-box");
  const chipRow = document.getElementById("chipRow") || document.getElementById("followup-chips");
  const query = queryInput.value.trim();

  if (!query) return;

  // Append user's message to UI
  chatBox.innerHTML += `<div class="user-msg"><b>You:</b> ${query}</div>`;
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

    chipRow.innerHTML = "";
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
      botMsgDiv.innerHTML = `<b>Bot:</b> ${data.answer}`;
      chatBox.appendChild(botMsgDiv);

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
    botMsgDiv.innerHTML = "<b>Bot:</b> ";
    chatBox.appendChild(botMsgDiv);

    let streamText = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      streamText += chunk;

      const markerIdx = streamText.indexOf("[FOLLOW_UPS]:");
      const displayText = markerIdx !== -1 ? streamText.slice(0, markerIdx).trim() : streamText;
      botMsgDiv.innerHTML = `<b>Bot:</b> ${displayText}`;
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