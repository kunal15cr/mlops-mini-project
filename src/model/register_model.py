import json
import mlflow
import logging
from mlflow.tracking import MlflowClient

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
# 👉 Change this to your MLflow server URI
# Example if running locally: "http://localhost:5000"
# Example if hosted on a VM: "http://<your-server-ip>:5000"
mlflow.set_tracking_uri("http://localhost:5000")

# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------
logger = logging.getLogger('model_registration')
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

file_handler = logging.FileHandler('model_registration_errors.log')
file_handler.setLevel(logging.ERROR)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# ---------------------------------------------------------------------
# FUNCTIONS
# ---------------------------------------------------------------------
def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logger.debug('✅ Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logger.error('❌ File not found: %s', file_path)
        raise
    except Exception as e:
        logger.error('❌ Unexpected error while loading model info: %s', e)
        raise

def register_model(model_name: str, model_info: dict):
    """Register the model to the MLflow Model Registry."""
    try:
        model_uri = f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        logger.debug(f"Registering model from URI: {model_uri}")

        # Register model
        model_version = mlflow.register_model(model_uri, model_name)
        logger.info(f"✅ Model '{model_name}' registered as version {model_version.version}")

        # Transition model to 'Staging'
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version.version,
            stage="Staging"
        )

        logger.info(f"✅ Model '{model_name}' transitioned to 'Staging'")
        return model_version.version

    except Exception as e:
        logger.error('❌ Error during model registration: %s', e)
        raise

# ---------------------------------------------------------------------
# MAIN SCRIPT
# ---------------------------------------------------------------------
def main():
    try:
        model_info_path = 'reports/experiment_info.json'
        model_info = load_model_info(model_info_path)

        model_name = "my_model"
        version = register_model(model_name, model_info)
        print(f"✅ Model '{model_name}' successfully registered (version {version}) and moved to Staging.")

    except Exception as e:
        logger.error('❌ Model registration process failed: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
