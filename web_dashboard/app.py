# web_dashboard/app.py
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import os
import sys
import time # For the background thread example

# Add project root to sys.path to import training_manager etc.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

try:
    from training_manager import shared_training_state, start_training_manager_thread, stop_training_manager_thread
    # tokenizer_wrapper might be imported by training_manager or other core modules
    # Ensure config is accessible as well
    import config 
except ImportError as e:
    print(f"ERROR: Could not import core modules from project root: {e}")
    print(f"Ensure '{project_root}' is the correct project root and is in sys.path.")
    print(f"Current sys.path: {sys.path}")
    sys.exit(1)


app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
socketio = SocketIO(app, async_mode='threading')

status_thread = None
status_thread_started = False

def emit_status_updates():
    """Periodically emits status updates to all connected clients."""
    while True:
        # Structure based on the new TrainingState for supervised learning
        status_data = {
            'is_training': shared_training_state.is_training,
            'current_phase': shared_training_state.current_phase,
            'current_epoch': shared_training_state.current_epoch, # Changed from current_epoch_pretrain
            'total_epochs': shared_training_state.total_epochs,   # Changed from total_epochs_pretrain
            # 'current_iteration_self_play' & 'total_iterations_self_play' are removed
            'live_metrics': shared_training_state.get_metrics() # get_metrics() will return new structure
        }
        # Add approx_batches_per_epoch for better progress bar estimation if available
        # This would require TrainingState or training_logic to estimate/store it.
        # For now, JS will use a default if not present.
        # if hasattr(shared_training_state, 'approx_batches_per_epoch'):
        #    status_data['live_metrics']['approx_batches_per_epoch'] = shared_training_state.approx_batches_per_epoch

        socketio.emit('status_update', status_data)
        socketio.sleep(1) # Emit updates every 1 second

@socketio.on('connect')
def handle_connect():
    global status_thread, status_thread_started
    print('Client connected:', request.sid)
    if not status_thread_started:
        status_thread = socketio.start_background_task(target=emit_status_updates)
        status_thread_started = True
        print("Background status update thread started.")
    # It's good practice for client to request, or send initial data here directly
    # Forcing client to request via emit('request_initial_data') ensures it's ready.
    emit('request_initial_data') # Prompt client to ask for initial data


@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected:', request.sid)

@socketio.on('request_initial_data')
def handle_request_initial_data():
    print("Received request_initial_data. Sending current state.")
    # Align with the new TrainingState structure
    initial_data = {
        'status': {
            'is_training': shared_training_state.is_training,
            'current_phase': shared_training_state.current_phase,
            'current_epoch': shared_training_state.current_epoch,
            'total_epochs': shared_training_state.total_epochs,
            'live_metrics': shared_training_state.get_metrics()
        },
        'config': shared_training_state.get_config() # get_config() will return new structure
    }
    emit('initial_data', initial_data)

@socketio.on('start_training')
def handle_start_training(data):
    model_run_name = data.get('model_run_name')
    if not model_run_name:
        emit('training_error', {'message': 'Error: Model Run Name is required.'})
        return

    print(f"Received start_training event for run: {model_run_name}")
    shared_training_state.add_log_message(f"UI command: Attempting to start training for run '{model_run_name}'...")
    
    success = start_training_manager_thread(model_run_name) # This function is now simpler
    if success:
        emit('training_started', {'message': f'Training started for run "{model_run_name}".'})
        shared_training_state.add_log_message(f"Training successfully initiated for '{model_run_name}'.")
    else:
        emit('training_error', {'message': f'Failed to start training for "{model_run_name}". Check server logs or UI logs.'})

@socketio.on('stop_training')
def handle_stop_training():
    print("Received stop_training event")
    shared_training_state.add_log_message("UI command: Attempting to stop training...")
    success = stop_training_manager_thread()
    if success:
        emit('training_stopped', {'message': 'Stop signal sent. Training will halt.'})
        shared_training_state.add_log_message("Stop signal processed by manager.")
    else:
        emit('training_error', {'message': 'Could not stop training (possibly not running).'})
        shared_training_state.add_log_message("Stop signal failed (manager indicated not running or already stopping).")

@socketio.on('update_config')
def handle_update_config(data):
    print(f"Received update_config event with data: {data}")
    shared_training_state.add_log_message(f"UI command: Attempting to update configuration...")
    all_success = True
    messages = []

    for key, value in data.items():
        # TrainingState.update_config_param now handles casting and logging internally
        success = shared_training_state.update_config_param(key, value)
        if not success:
            all_success = False
            messages.append(f"Failed to update '{key}'.")
        # else: # No need to append success message here, update_config_param logs it
            # messages.append(f"'{key}' updated.")

    if all_success:
        response_message = "Configuration updated successfully."
        # This message is already logged by update_config_param on success for each key
    else:
        response_message = "Some configuration updates failed. Check logs for details."
        # This specific message is useful to summarize for the UI if any failed

    emit('config_updated', {
        'success': all_success,
        'message': response_message,
        'new_config': shared_training_state.get_config() # Send back the full, current config
    })

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    print("Starting Flask-SocketIO server for Supervised Training Dashboard...")
    print(f"Project root: {project_root}")
    # print(f"Python sys.path includes: {sys.path}") # Useful for debugging imports
    socketio.run(app, debug=True, host='0.0.0.0', port=5001, allow_unsafe_werkzeug=True)