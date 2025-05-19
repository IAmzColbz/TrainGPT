// web_dashboard/static/js/main.js
document.addEventListener('DOMContentLoaded', function() {
    const socket = io.connect(location.protocol + '//' + document.domain + ':' + location.port);

    const trainingStatusEl = document.getElementById('training-status');
    const currentPhaseEl = document.getElementById('current-phase');
    const currentEpochEl = document.getElementById('current-epoch'); // Renamed from pretrain-epoch
    const totalEpochsEl = document.getElementById('total-epochs');   // Renamed
    const globalTrainBatchEl = document.getElementById('global-train-batch'); // Renamed
    const logOutputEl = document.getElementById('log-output');
    const startTrainBtn = document.getElementById('start-train-btn');
    const stopTrainBtn = document.getElementById('stop-train-btn');
    const modelRunNameInput = document.getElementById('model-run-name-input');

    const overallProgressBar = document.getElementById('overall-progress-bar');
    const progressBarText = document.getElementById('progress-bar-text');
    const progressPhaseLabel = document.getElementById('progress-phase-label');
    const progressCurrentStep = document.getElementById('progress-current-step');
    const progressTotalSteps = document.getElementById('progress-total-steps');
    const progressEta = document.getElementById('progress-eta');

    const chartOptions = (title, yAxisLabel = 'Value') => ({
        responsive: true, maintainAspectRatio: false,
        scales: {
            x: { type: 'linear', position: 'bottom', title: { display: true, text: 'Step/Epoch' } },
            y: { title: { display: true, text: yAxisLabel } }
        },
        plugins: { legend: { display: false }, title: { display: true, text: title } },
        animation: false // Disable animation for performance with frequent updates
    });
    const lineChartConfig = (label, data, borderColor, yAxisLabel = 'Value') => ({
        type: 'line',
        data: { datasets: [{ label: label, data: data, borderColor: borderColor, fill: false, tension: 0.1 }] },
        options: chartOptions(label, yAxisLabel)
    });

    let charts = {
        trainBatchLoss: new Chart(document.getElementById('trainBatchLossChart').getContext('2d'), lineChartConfig('Train Batch Loss', [], 'rgb(255, 99, 132)', 'Loss')),
        trainEpochLoss: new Chart(document.getElementById('trainEpochLossChart').getContext('2d'), lineChartConfig('Train Epoch Loss', [], 'rgb(255, 159, 64)', 'Loss')),
        validationLoss: new Chart(document.getElementById('validationLossChart').getContext('2d'), lineChartConfig('Validation Loss', [], 'rgb(54, 162, 235)', 'Loss')),
        validationBleu: new Chart(document.getElementById('validationBleuChart').getContext('2d'), lineChartConfig('Validation BLEU Score', [], 'rgb(75, 192, 192)', 'BLEU'))
    };

    function formatChartData(historyArray) {
        if (!Array.isArray(historyArray)) return [];
        return historyArray.map(item => ({ x: item[0], y: item[1] }));
    }

    socket.on('connect', function() {
        console.log('Socket.IO connected!');
        socket.emit('request_initial_data');
    });

    socket.on('initial_data', function(data) {
        console.log('Received initial_data:', data);
        if (data.status) {
            updateStatusDisplay(data.status);
            if (data.status.live_metrics) {
                 updateMetricsDisplay(data.status.live_metrics);
            }
        }
        if (data.config) populateConfigTable(data.config);
    });

    socket.on('status_update', function(data) {
        // console.log('Received status_update:', data);
        updateStatusDisplay(data);
        if (data.live_metrics) updateMetricsDisplay(data.live_metrics);
    });

    socket.on('log_update', function(data) { // This might be removed if logs are part of status_update
        const logEntry = document.createElement('div');
        logEntry.textContent = data.message; // Assuming 'message' key from backend
        logOutputEl.insertBefore(logEntry, logOutputEl.firstChild);
        if (logOutputEl.children.length > 200) {
            logOutputEl.removeChild(logOutputEl.lastChild);
        }
    });
    
    socket.on('training_started', function(data) {
        showToast(data.message, 'success');
        startTrainBtn.disabled = true;
        stopTrainBtn.disabled = false;
    });

    socket.on('training_stopped', function(data) {
        showToast(data.message, 'info');
        startTrainBtn.disabled = false;
        stopTrainBtn.disabled = true;
    });
    
    socket.on('training_error', function(data) {
        showToast(data.message, 'error');
        startTrainBtn.disabled = false;
        stopTrainBtn.disabled = true;
    });

    socket.on('config_updated', function(response) {
        const statusEl = document.getElementById('config-save-status');
        if (response.success) {
            statusEl.textContent = response.message || 'Config saved successfully!';
            statusEl.style.color = 'green';
            populateConfigTable(response.new_config);
        } else {
            statusEl.textContent = response.message || 'Error saving config.';
            statusEl.style.color = 'red';
        }
        setTimeout(() => { statusEl.textContent = ''; }, 5000);
    });

    function updateStatusDisplay(statusData) {
        trainingStatusEl.textContent = statusData.is_training ? 'Training' : 'Idle';
        currentPhaseEl.textContent = statusData.current_phase || 'N/A';
        currentEpochEl.textContent = statusData.current_epoch || '0'; // Updated key
        totalEpochsEl.textContent = statusData.total_epochs || '0';   // Updated key
        globalTrainBatchEl.textContent = statusData.live_metrics?.current_global_train_batch_step || '0'; // Updated key

        startTrainBtn.disabled = statusData.is_training;
        stopTrainBtn.disabled = !statusData.is_training;
        
        let currentStep = 0;
        let totalSteps = 1;
        let progressPercent = 0;
        progressPhaseLabel.textContent = statusData.current_phase || 'Idle';

        if (statusData.is_training && statusData.current_phase !== "Idle") {
            currentStep = statusData.current_epoch || 0;
            totalSteps = statusData.total_epochs || 1;
            if (statusData.current_phase.startsWith("Training Epoch") || statusData.current_phase.startsWith("Validating")) {
                 // Finer grain progress based on global batch step within the epoch phase
                const currentGlobalBatch = statusData.live_metrics?.current_global_train_batch_step || 0;
                const approxBatchesPerEpoch = statusData.live_metrics?.approx_batches_per_epoch || 200; // Estimate if not available
                const epochProgressBatch = currentGlobalBatch % approxBatchesPerEpoch;
                
                // Overall progress by epoch
                progressPercent = totalSteps > 0 ? ((currentStep -1 + (epochProgressBatch / approxBatchesPerEpoch)) / totalSteps) * 100 : 0;
                progressCurrentStep.textContent = `${currentStep} (Batch ${currentGlobalBatch})`;

            } else { // Default to epoch progress
                progressPercent = totalSteps > 0 ? (currentStep / totalSteps) * 100 : 0;
                progressCurrentStep.textContent = currentStep;
            }
             progressTotalSteps.textContent = totalSteps;
        } else {
             progressCurrentStep.textContent = 0;
             progressTotalSteps.textContent = 0;
        }
        
        overallProgressBar.style.width = Math.min(100, Math.max(0, progressPercent)) + '%';
        progressBarText.textContent = Math.round(Math.min(100, Math.max(0, progressPercent))) + '%';
        progressEta.textContent = "ETA: N/A";
    }

    function updateMetricsDisplay(metrics) {
        if (!metrics) return;
        charts.trainBatchLoss.data.datasets[0].data = formatChartData(metrics.train_batch_loss_history);
        charts.trainEpochLoss.data.datasets[0].data = formatChartData(metrics.train_epoch_loss_history);
        charts.validationLoss.data.datasets[0].data = formatChartData(metrics.validation_loss_history);
        charts.validationBleu.data.datasets[0].data = formatChartData(metrics.validation_bleu_history);

        Object.values(charts).forEach(chart => chart.update()); // Default 'quiet' update

        if (metrics.log_messages && Array.isArray(metrics.log_messages)) {
            logOutputEl.innerHTML = '';
            metrics.log_messages.forEach(msg => {
                const logEntry = document.createElement('div');
                logEntry.textContent = msg;
                logOutputEl.appendChild(logEntry);
            });
            logOutputEl.scrollTop = logOutputEl.scrollHeight;
        }
    }

    const configContainer = document.getElementById('config-params-container');
    const saveConfigBtn = document.getElementById('save-config-btn');
    const paramDescriptionsSupervised = {
        "NN_LEARNING_RATE": "Initial learning rate for the optimizer.",
        "SUPERVISED_EPOCHS": "Total number of epochs for supervised training.",
        "SAVE_EVERY_N_EPOCHS": "Frequency (in epochs) for saving model checkpoints.",
        "VALIDATE_EVERY_N_BATCHES": "Frequency (in global batches) to run validation.",
        "D_MODEL": "Model embedding/hidden state dimensionality. (Restart required)",
        "N_HEADS": "Number of attention heads. (Restart required)",
        "NUM_ENCODER_LAYERS": "Number of encoder layers. (Restart required)",
        "NUM_DECODER_LAYERS": "Number of decoder layers. (Restart required)",
        "D_FF": "Feed-forward layer dimensionality. (Restart required)",
        "TRANSFORMER_DROPOUT": "Dropout rate in Transformer. (Restart required)",
        "SUPERVISED_BATCH_SIZE": "Batch size for training and validation data.",
        "LR_SCHEDULER_PATIENCE": "Epochs for ReduceLROnPlateau scheduler to wait for improvement.",
        "LR_SCHEDULER_FACTOR": "Factor by which LR is reduced by scheduler."
    };

    function populateConfigTable(configParams) {
        if (!configParams || Object.keys(configParams).length === 0) {
            configContainer.innerHTML = "<p>No configuration parameters loaded.</p>"; return;
        }
        configContainer.innerHTML = '';
        Object.entries(configParams).forEach(([key, value]) => {
            const div = document.createElement('div');
            const label = document.createElement('label');
            label.htmlFor = `config-${key}`;
            label.textContent = key + ":";
            
            const input = document.createElement('input');
            const type = typeof value;
            input.type = (type === 'boolean') ? 'checkbox' : (type === 'number') ? 'number' : 'text';
            if (input.type === 'number') input.step = (Number.isInteger(value) && value < 100 && value > -100) ? "1" : "any";
            input.id = `config-${key}`;
            input.name = key;
            if (type === 'boolean') input.checked = value;
            else input.value = value;
            
            div.appendChild(label);
            div.appendChild(input);

            if (paramDescriptionsSupervised[key]) {
                const descSpan = document.createElement('span');
                descSpan.className = 'param-description';
                descSpan.textContent = `(${paramDescriptionsSupervised[key]})`;
                div.appendChild(descSpan);
            }
            configContainer.appendChild(div);
        });
    }

    saveConfigBtn.addEventListener('click', function() {
        const updatedConfig = {};
        configContainer.querySelectorAll('input').forEach(input => {
            updatedConfig[input.name] = (input.type === 'checkbox') ? input.checked : input.value;
        });
        socket.emit('update_config', updatedConfig);
        document.getElementById('config-save-status').textContent = 'Saving...';
    });

    startTrainBtn.addEventListener('click', function() {
        const modelRunName = modelRunNameInput.value.trim();
        if (!modelRunName) {
            showToast("Model Run Name cannot be empty!", "error");
            modelRunNameInput.focus(); return;
        }
        console.log("Start button clicked, model run name:", modelRunName);
        socket.emit('start_training', { model_run_name: modelRunName });
    });

    stopTrainBtn.addEventListener('click', function() {
        socket.emit('stop_training');
    });

    const tabs = document.querySelectorAll('.tab-button');
    const tabContents = document.querySelectorAll('.tab-content');
    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            tabs.forEach(item => item.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));
            this.classList.add('active');
            document.getElementById(this.dataset.tab).classList.add('active');
        });
    });

    function showToast(message, type = 'info') { alert(`${type.toUpperCase()}: ${message}`); }
});