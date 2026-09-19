// ============================================================
// PredictOps Frontend
// ============================================================

const API_BASE_URL = "http://127.0.0.1:8000";

const predictionForm = document.getElementById("predictionForm");
const resultContent = document.getElementById("resultContent");
const refreshButton = document.getElementById("refreshMachines");
const machineSelect = document.getElementById("machineId");


// ============================================================
// LOAD DASHBOARD
// ============================================================

async function loadMachines() {
    try {
        const response = await fetch(`${API_BASE_URL}/machines`);

        if (!response.ok) {
            throw new Error("Unable to load machine data.");
        }

        const data = await response.json();

        updateSummary(data);
        updateMachineSelect(data.machines);
        updateMachineTable(data.machines);

        updateSystemStatus(true);

        const endpointEl = document.getElementById("endpointStatus");
        if (endpointEl) {
            endpointEl.textContent = "InService";
            endpointEl.style.color = "#4ade80";
        }

        await updateDriftStatus();

    } catch (error) {
        console.error("Dashboard error:", error);
        updateSystemStatus(false);

        const endpointEl = document.getElementById("endpointStatus");
        if (endpointEl) {
            endpointEl.textContent = "OFFLINE";
            endpointEl.style.color = "#f87171";
        }
    }
}


// ============================================================
// SYSTEM STATUS
// ============================================================

function updateSystemStatus(online) {

    const statusElements = document.querySelectorAll(
        ".header-status, .aws-status"
    );

    statusElements.forEach(element => {

        if (online) {
            element.innerHTML = `
                <span class="status-dot"></span>
                ${element.classList.contains("aws-status")
                    ? "AWS Connected"
                    : "System Operational"}
            `;
        } else {
            element.innerHTML = `
                <span class="status-dot offline"></span>
                ${element.classList.contains("aws-status")
                    ? "API Offline"
                    : "System Offline"}
            `;
        }

    });
}


// ============================================================
// DRIFT STATUS
// ============================================================

async function updateDriftStatus() {
    const driftEl = document.getElementById("driftStatus");
    if (!driftEl) return;

    try {
        const response = await fetch(`${API_BASE_URL}/metrics`);
        const text = await response.text();

        const match = text.match(/predictops_data_drift_percentage\s+([\d.]+)/);

        if (match) {
            const pct = parseFloat(match[1]);
            driftEl.textContent = `${pct.toFixed(1)}% drift`;
            driftEl.style.color =
                pct < 10 ? "#4ade80" : pct < 25 ? "#fb923c" : "#f87171";
        } else {
            driftEl.textContent = "No data";
        }
    } catch (error) {
        driftEl.textContent = "Unavailable";
    }
}


// ============================================================
// SUMMARY CARDS
// ============================================================

function updateSummary(data) {

    const cards = document.querySelectorAll(".card-value");

    if (cards.length >= 3) {

        cards[0].textContent = data.count ?? 0;

        cards[1].textContent = data.healthy ?? 0;

        cards[2].textContent =
            data.maintenance_required ?? 0;
    }

}


// ============================================================
// MACHINE DROPDOWN
// ============================================================

function updateMachineSelect(machines) {

    if (!machineSelect) return;

    machineSelect.innerHTML = `
        <option value="">Select Machine</option>
    `;

    machines.forEach(machine => {

        const option = document.createElement("option");

        option.value = machine.machine_id;

        option.textContent =
            `MACHINE-${String(machine.machine_id).padStart(3, "0")}`;

        machineSelect.appendChild(option);

    });

}


// ============================================================
// MACHINE TABLE
// ============================================================

function updateMachineTable(machines) {

    const table = document.querySelector(".machine-table");

    if (!table) return;

    table.innerHTML = `
        <div class="table-row table-heading">
            <span>Machine</span>
            <span>Temperature</span>
            <span>Vibration</span>
            <span>Health</span>
            <span>RUL</span>
        </div>
    `;

    machines.forEach(machine => {

        let healthClass = "healthy";

        if (machine.status === "Critical") {
            healthClass = "critical";
        }
        else if (
            machine.status === "Maintenance Required"
        ) {
            healthClass = "warning";
        }

        const row = document.createElement("div");

        row.className = "table-row";

        row.innerHTML = `
            <span>
                MACHINE-${String(machine.machine_id).padStart(3, "0")}
            </span>

            <span>
                ${Number(machine.temperature).toFixed(1)}°C
            </span>

            <span>
                ${Number(machine.vibration).toFixed(2)} mm/s
            </span>

            <span>
                <span class="health ${healthClass}">
                    ${machine.status}
                </span>
            </span>

            <span>
                ${Number(machine.rul).toFixed(1)} cycles
            </span>
        `;

        table.appendChild(row);

    });

}


// ============================================================
// PREDICTION
// ============================================================

if (predictionForm) {

    predictionForm.addEventListener(
        "submit",
        async function (event) {

            event.preventDefault();

            const button =
                document.querySelector(".predict-button");

            if (!machineSelect || !machineSelect.value) {

                alert("Please select a machine.");

                return;
            }

            const unitId = Number(machineSelect.value);

            if (!unitId) {

                alert("Invalid machine selected.");

                return;
            }


            // --------------------------------------------------
            // LOADING STATE
            // --------------------------------------------------

            button.disabled = true;

            button.innerHTML =
                "⏳ Running ML Prediction...";


            resultContent.innerHTML = `

                <div class="result-placeholder">

                    <div class="placeholder-icon">
                        ⚙️
                    </div>

                    <h3>
                        Running Prediction
                    </h3>

                    <p>
                        Fetching machine data and
                        running the Random Forest model...
                    </p>

                </div>

            `;


            try {

                // --------------------------------------------------
                // CALL FASTAPI
                // --------------------------------------------------

                const response = await fetch(
                    `${API_BASE_URL}/predict-machine/${unitId}`,
                    {
                        method: "POST",
                        headers: {
                            "Accept": "application/json"
                        }
                    }
                );


                const data = await response.json();


                if (!response.ok) {

                    throw new Error(
                        data.detail ||
                        "Prediction failed."
                    );

                }


                console.log(
                    "Prediction response:",
                    data
                );


                // --------------------------------------------------
                // RESULT
                // --------------------------------------------------

                const rul =
                    Number(
                        data.predicted_rul_cycles
                    );


                let healthClass = "healthy";


                if (
                    data.status === "Critical"
                ) {

                    healthClass = "critical";

                }
                else if (
                    data.status === "Maintenance Required"
                ) {

                    healthClass = "warning";

                }


                resultContent.innerHTML = `

                    <div class="result-success">

                        <div class="rul-number">
                            ${rul.toFixed(1)}
                        </div>

                        <div class="rul-label">
                            Estimated Remaining Useful Life
                            (cycles)
                        </div>


                        <div class="
                            result-health
                            ${healthClass}
                        ">
                            ${data.status}
                        </div>


                        <div class="prediction-details">

                            <div>
                                Machine:
                                <strong>
                                    MACHINE-${String(
                                        data.machine_id
                                    ).padStart(3, "0")}
                                </strong>
                            </div>


                            <div>
                                Latest Cycle:
                                <strong>
                                    ${data.cycle}
                                </strong>
                            </div>


                            <div>
                                Features Used:
                                <strong>
                                    ${data.features_used}
                                </strong>
                            </div>


                            <div>
                                Model:
                                <strong>
                                    Random Forest
                                </strong>
                            </div>

                        </div>

                    </div>

                `;


            } catch (error) {

                console.error(
                    "Prediction error:",
                    error
                );


                resultContent.innerHTML = `

                    <div class="result-placeholder">

                        <div class="placeholder-icon">
                            ⚠️
                        </div>

                        <h3>
                            Prediction Failed
                        </h3>

                        <p style="color:#f87171;">
                            ${error.message}
                        </p>

                        <p>
                            Make sure FastAPI is running
                            on port 8000.
                        </p>

                    </div>

                `;

            }


            // --------------------------------------------------
            // RESET BUTTON
            // --------------------------------------------------

            button.disabled = false;

            button.innerHTML =
                "<span>⚡</span> Predict Remaining Useful Life";

        }
    );

}


// ============================================================
// REFRESH MACHINES
// ============================================================

if (refreshButton) {

    refreshButton.addEventListener(
        "click",
        async function () {

            refreshButton.disabled = true;

            refreshButton.innerHTML =
                "⟳ Refreshing...";


            await loadMachines();


            refreshButton.innerHTML =
                "✓ Updated";


            setTimeout(() => {

                refreshButton.innerHTML =
                    "↻ Refresh";

                refreshButton.disabled =
                    false;

            }, 1200);

        }
    );

}


// ============================================================
// INITIAL LOAD
// ============================================================

loadMachines();