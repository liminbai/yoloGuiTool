import fiftyone as fo

# Load your FiftyOne dataset
dataset = fo.load_dataset("ppe_dataset")

# Launch the app
session = fo.launch_app(dataset, port=5151)