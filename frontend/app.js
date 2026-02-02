const micButton = document.getElementById("micButton");
const micText = document.getElementById("micText");
const connectionStatus = document.getElementById("connectionStatus");
const listenStatus = document.getElementById("listenStatus");
const burgerName = document.getElementById("burgerName");
const heardText = document.getElementById("heardText");
const ingredientList = document.getElementById("ingredientList");
const debugPanel = document.getElementById("debugPanel");
const debugText = document.getElementById("debugText");

let burgersByKey = new Map();
let ws = null;
let isListening = false;
let usesNewProtocol = false;
const params = new URLSearchParams(window.location.search);
const wsPort = params.get("ws") || "2700";
const wsHost = window.location.hostname || "localhost";
const wsUrl = `ws://${wsHost}:${wsPort}`;
const handsfree = params.get("handsfree") !== "0";
const debugMode = params.get("debug") === "1";
const demoMode = params.get("demo") === "1";
const demoKey = params.get("demoKey");
const demoHeard = params.get("demoHeard");
const demoState = params.get("demoState");
const autoListen = handsfree;
let reconnectTimer = null;
let pendingStart = false;
let startTimer = null;

const ICON_BASE = "assets/ingredients";
const ICONS = [
  { icon: "bread.jpg", match: ["pan", "brioche"] },
  { icon: "beef.jpg", match: ["ternera", "vaca", "hamburguesa", "rabo de toro"] },
  { icon: "chicken.jpg", match: ["pollo", "panko", "rebozado"] },
  { icon: "vegan.jpg", match: ["vegana"] },
  { icon: "cheese.jpg", match: ["cheddar", "queso", "raclette", "gorgonzola", "curado", "4 quesos"] },
  { icon: "onion.jpg", match: ["cebolla"] },
  { icon: "lettuce.jpg", match: ["lechuga", "rucula", "col"] },
  { icon: "tomato.jpg", match: ["tomate"] },
  { icon: "pickle.jpg", match: ["pepinillo"] },
  { icon: "bacon.jpg", match: ["bacon", "guanciale", "jamon", "sobrasada", "pulled pork"] },
  { icon: "egg.jpg", match: ["huevo"] },
  { icon: "sauce.jpg", match: ["salsa", "mayonesa", "barbacoa", "teriyaki"] },
  { icon: "mushroom.jpg", match: ["champi", "seta"] },
  { icon: "pepper.jpg", match: ["jalapeno", "picante"] },
  { icon: "avocado.jpg", match: ["guacamole"] },
  { icon: "carrot.jpg", match: ["zanahoria"] },
  { icon: "sweet.jpg", match: ["mermelada", "miel", "foie"] },
];
const DEFAULT_ICON = "generic.jpg";

async function loadBurgers() {
  const response = await fetch("data/burgers.json", { cache: "no-store" });
  const data = await response.json();
  burgersByKey = new Map(data.burgers.map((item) => [item.key, item]));
}

function normalizeIngredient(text) {
  return text
    .toLowerCase()
    .normalize("NFD")
    .replace(/[\u0300-\u036f]/g, "")
    .replace(/[^a-z0-9 ]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function iconForIngredient(name) {
  const normalized = normalizeIngredient(name);
  for (const entry of ICONS) {
    if (entry.match.some((key) => normalized.includes(key))) {
      return `${ICON_BASE}/${entry.icon}`;
    }
  }
  return `${ICON_BASE}/${DEFAULT_ICON}`;
}

function setConnection(status) {
  connectionStatus.textContent = status ? "Conectado" : "Desconectado";
  connectionStatus.classList.toggle("connected", status);
}

function setListening(state) {
  isListening = state;
  micButton.classList.toggle("listening", state);
  micText.textContent = state ? "Escuchando" : "Escuchar";
  listenStatus.textContent = state ? "Escuchando" : "Listo";
}

function setState(state, level) {
  const labels = {
    listening: "Escuchando",
    detecting: "Detectando",
    cooldown: "Cooldown",
  };
  listenStatus.textContent = labels[state] || "Listo";
  if (debugMode && level && debugText) {
    debugText.textContent = `RMS ${level.rms?.toFixed(3)} | Peak ${level.peak?.toFixed(
      3
    )} | Noise ${level.noise_floor?.toFixed(3)}`;
  }
}

function updateMicVisibility() {
  if (handsfree || usesNewProtocol) {
    micButton.style.display = "none";
  } else {
    micButton.style.display = "";
  }
}

function scheduleReconnect() {
  if (reconnectTimer) return;
  listenStatus.textContent = "Reconectando...";
  reconnectTimer = setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, 1500);
}

function requestStart(delay = 0) {
  if (!autoListen) return;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    pendingStart = true;
    connect();
    return;
  }
  if (isListening) return;
  if (startTimer) {
    clearTimeout(startTimer);
  }
  startTimer = setTimeout(() => {
    if (ws && ws.readyState === WebSocket.OPEN && !isListening) {
      ws.send(JSON.stringify({ action: "start" }));
    }
  }, delay);
}

function showBurger(item, heard, confidence) {
  if (!item) {
    burgerName.textContent = "No encontrado";
    heardText.textContent = heard ? `Oido: ${heard}` : "";
    ingredientList.innerHTML = "";
    return;
  }

  burgerName.textContent = item.display;
  const confText = confidence ? ` (${Math.round(confidence * 100)}%)` : "";
  heardText.textContent = heard ? `Oido: ${heard}${confText}` : "";

  ingredientList.innerHTML = "";

  const ingredients = item.ingredients || [];
  if (!ingredients.length) {
    ingredientList.innerHTML = "<li>Ingredientes no definidos</li>";
    return;
  }

  ingredients.forEach((ingredient, index) => {
    const li = document.createElement("li");
    li.className = "ingredient-item";

    const iconWrap = document.createElement("span");
    iconWrap.className = "ingredient-icon";
    const icon = document.createElement("img");
    icon.src = iconForIngredient(ingredient);
    icon.alt = "";
    icon.loading = "lazy";
    iconWrap.appendChild(icon);
    li.appendChild(iconWrap);

    const label = document.createElement("span");
    label.textContent = ingredient;
    li.appendChild(label);

    ingredientList.appendChild(li);
  });
}

function handleMessage(event) {
  let payload;
  try {
    payload = JSON.parse(event.data);
  } catch {
    return;
  }

  if (payload.type === "state") {
    usesNewProtocol = true;
    setConnection(true);
    setState(payload.state, payload.level);
    updateMicVisibility();
    if (payload.debug && debugMode && debugText) {
      debugText.textContent = payload.debug;
    }
    return;
  }

  if (payload.type === "detect") {
    usesNewProtocol = true;
    const item = burgersByKey.get(payload.key);
    showBurger(item, payload.text, payload.confidence);
    updateMicVisibility();
    if (payload.debug && debugMode && debugText) {
      debugText.textContent = JSON.stringify(payload.debug);
    }
    return;
  }

  if (payload.type === "status") {
    if (payload.status === "connected") {
      setConnection(true);
    }
    if (payload.status === "listening") {
      setListening(true);
    }
    if (payload.status === "idle") {
      setListening(false);
      requestStart(200);
    }
    if (payload.status === "audio_warning") {
      listenStatus.textContent = "Aviso de audio";
    }
  }

  if (payload.type === "partial") {
    // Skip partial text to avoid showing non-catalog words.
  }

  if (payload.type === "result") {
    const item = burgersByKey.get(payload.key);
    showBurger(item, payload.heard, payload.confidence);
    setListening(false);
    requestStart(400);
  }

  if (payload.type === "error") {
    listenStatus.textContent = "Error audio";
  }
}

function connect() {
  if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
    return;
  }
  listenStatus.textContent = "Conectando...";
  ws = new WebSocket(wsUrl);
  ws.addEventListener("open", () => {
    setConnection(true);
    listenStatus.textContent = "Listo";
    if (pendingStart || autoListen) {
      pendingStart = false;
      requestStart(0);
    }
  });
  ws.addEventListener("close", () => {
    setConnection(false);
    setListening(false);
    scheduleReconnect();
  });
  ws.addEventListener("error", () => {
    listenStatus.textContent = "Error WS";
  });
  ws.addEventListener("message", handleMessage);
}

micButton.addEventListener("click", () => {
  if (usesNewProtocol) {
    return;
  }
  if (autoListen) {
    return;
  }
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    pendingStart = true;
    connect();
    return;
  }

  if (isListening) {
    ws.send(JSON.stringify({ action: "stop" }));
    setListening(false);
  } else {
    ws.send(JSON.stringify({ action: "start" }));
  }
});

(async () => {
  try {
    await loadBurgers();
  } catch (err) {
    console.error(err);
  }
  updateMicVisibility();
  if (debugPanel && debugMode) {
    debugPanel.style.display = "block";
  }
  if (demoMode) {
    setConnection(true);
    listenStatus.textContent = "Demo";
    const chosen = demoKey ? burgersByKey.get(demoKey) : burgersByKey.values().next().value;
    showBurger(chosen, demoHeard || "doble cheeseburger", 0.92);
    if (demoState === "listening") {
      setListening(true);
    } else {
      setListening(false);
    }
    return;
  }
  connect();
})();
