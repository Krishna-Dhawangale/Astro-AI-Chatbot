async function sendChatMessage() {
  const queryInput = document.getElementById("user-query");
  const chatBox = document.getElementById("chat-box");
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

  try {
    const response = await fetch("/chat", { // Relative path if mounted via FastAPI
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
      chatBox.innerHTML += `<div class="bot-msg"><b>Bot:</b> ${data.response}</div>`;
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

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      botMsgDiv.innerHTML += chunk;
      chatBox.scrollTop = chatBox.scrollHeight;
    }
  } catch (error) {
    console.error("Error connecting to backend:", error);
  }
}