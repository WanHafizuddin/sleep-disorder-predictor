// Seeded from the DOM once wireSegmented/wireRange/wireNumber run below, so that
// values the browser restores across a soft reload (form controls, but not these
// plain-text labels) are reflected here instead of silently reverting to defaults.
const state = {};

const RESULT_CONTENT = {
  None: {
    title: "No Sleep Disorder Detected",
    tagLabel: "Healthy",
    tagClass: "tag-outline",
    message: "Your sleep health looks good. Keep up your current routine — consistent sleep and activity habits are working in your favor.",
    suggestions: [],
    icon: '<path d="M20 6 9 17l-5-5"></path>',
  },
  Insomnia: {
    title: "Insomnia Risk",
    tagLabel: "Needs attention",
    tagClass: "tag-accent",
    message: "Your answers point to insomnia — trouble falling or staying asleep. A few small changes can make a real difference.",
    suggestions: [
      "Wind down earlier — ease off screens and stress before bed.",
      "Keep your sleep schedule steady, even on weekends.",
      "Skip caffeine in the afternoon and evening.",
    ],
    icon: '<path d="M12 9v4M12 17h.01M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"></path>',
  },
  "Sleep Apnea": {
    title: "Sleep Apnea Risk",
    tagLabel: "Needs attention",
    tagClass: "tag-accent",
    message: "Your answers suggest sleep apnea, where breathing repeatedly stops during sleep. It's worth looking into further.",
    suggestions: [
      "Talk to a doctor about a proper sleep study.",
      "Work toward a healthy BMI — it eases the load on your airway.",
      "Try sleeping on your side instead of your back.",
    ],
    icon: '<circle cx="12" cy="12" r="10"></circle><path d="M12 8v4M12 16h.01"></path>',
  },
};

const formScreen = document.getElementById("form-screen");
const loadingScreen = document.getElementById("loading-screen");
const resultScreen = document.getElementById("result-screen");

function showScreen(name) {
  formScreen.classList.toggle("hidden", name !== "form");
  loadingScreen.classList.toggle("hidden", name !== "loading");
  resultScreen.classList.toggle("hidden", name !== "result");
}

function wireSegmented(containerId, stateKey) {
  const container = document.getElementById(containerId);
  const active = container.querySelector(".seg-opt.active");
  if (active) {
    state[stateKey] = active.dataset.value;
  }
  container.querySelectorAll(".seg-opt").forEach((btn) => {
    btn.addEventListener("click", () => {
      state[stateKey] = btn.dataset.value;
      container.querySelectorAll(".seg-opt").forEach((b) => b.classList.toggle("active", b === btn));
    });
  });
}

function wireRange(inputId, labelId, stateKey, parse, format) {
  const input = document.getElementById(inputId);
  const label = document.getElementById(labelId);
  const initial = parse(input.value);
  state[stateKey] = initial;
  label.textContent = format(initial);
  input.addEventListener("input", () => {
    const value = parse(input.value);
    state[stateKey] = value;
    label.textContent = format(value);
  });
}

function wireNumber(inputId, stateKey) {
  const input = document.getElementById(inputId);
  state[stateKey] = parseInt(input.value, 10) || 0;
  input.addEventListener("input", () => {
    state[stateKey] = parseInt(input.value, 10) || 0;
  });
}

wireSegmented("gender-seg", "gender");
wireSegmented("bmi-seg", "bmi");
wireRange("age", "age-value", "age", (v) => parseInt(v, 10), (v) => String(v));
wireRange("sleep-duration", "sleep-duration-value", "sleepDuration", (v) => parseFloat(v), (v) => v.toFixed(1));
wireRange("quality-of-sleep", "quality-of-sleep-value", "qualityOfSleep", (v) => parseInt(v, 10), (v) => String(v));
wireRange("physical-activity", "physical-activity-value", "physicalActivity", (v) => parseInt(v, 10), (v) => String(v));
wireRange("stress-level", "stress-level-value", "stressLevel", (v) => parseInt(v, 10), (v) => String(v));
wireNumber("heart-rate", "heartRate");
wireNumber("daily-steps", "dailySteps");
wireNumber("bp-systolic", "bpSystolic");
wireNumber("bp-diastolic", "bpDiastolic");

function renderResult(label) {
  const content = RESULT_CONTENT[label] || RESULT_CONTENT.None;
  document.getElementById("result-icon").innerHTML = content.icon;
  document.getElementById("result-title").textContent = content.title;
  const tag = document.getElementById("result-tag");
  tag.textContent = content.tagLabel;
  tag.className = "tag " + content.tagClass;
  document.getElementById("result-message").textContent = content.message;

  const suggestionsBlock = document.getElementById("suggestions-block");
  const suggestionsList = document.getElementById("suggestions-list");
  suggestionsList.innerHTML = "";
  if (content.suggestions.length > 0) {
    content.suggestions.forEach((s) => {
      const li = document.createElement("li");
      li.textContent = s;
      suggestionsList.appendChild(li);
    });
    suggestionsBlock.classList.remove("hidden");
  } else {
    suggestionsBlock.classList.add("hidden");
  }
}

function renderError(message) {
  document.getElementById("result-icon").innerHTML = "";
  document.getElementById("result-title").textContent = "Something went wrong";
  const tag = document.getElementById("result-tag");
  tag.textContent = "";
  tag.className = "tag";
  document.getElementById("result-message").textContent = message;
  document.getElementById("suggestions-block").classList.add("hidden");
}

async function predict() {
  showScreen("loading");
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        gender: state.gender,
        age: state.age,
        sleep_duration: state.sleepDuration,
        quality_of_sleep: state.qualityOfSleep,
        physical_activity: state.physicalActivity,
        stress_level: state.stressLevel,
        bmi_category: state.bmi,
        heart_rate: state.heartRate,
        daily_steps: state.dailySteps,
        bp_systolic: state.bpSystolic,
        bp_diastolic: state.bpDiastolic,
      }),
    });
    if (response.status === 422) {
      renderError("Some of your values are out of range. Please check your entries and try again.");
    } else if (!response.ok) {
      throw new Error("Server returned " + response.status);
    } else {
      const data = await response.json();
      renderResult(data.result);
    }
  } catch (err) {
    renderError("We couldn't reach the prediction service. Please try again.");
  }
  showScreen("result");
}

document.getElementById("predict-btn").addEventListener("click", predict);
document.getElementById("reset-btn").addEventListener("click", () => showScreen("form"));
