# Number-Recognizer
A minimal handwritten digit recognizer trained on MNIST with TensorFlow/Keras. The repo shows the full pipeline from training to evaluating to predicting on your own PNGs.


Key ideas

	•	Keep the model simple (MLP) to highlight the end-to-end steps.
	•	Use consistent normalization between training and inference.
	•	Make inference easy: point at a folder of images and get predicted digits.

How it works

	1.	Load MNIST and normalize images to [0,1].
	2.	Train a small dense network (ReLU hidden layers, Softmax output).
	3.	Save the trained model to handwritten.keras.
	4.	Evaluate on the test set for a sanity check.
	5.	Infer on external images: read → grayscale → resize to 28×28 → normalize → predict.

Run it

(Optional) Train
python train.py  # or uncomment the training block in main script

Evaluate & predict on your images (named like x1.png, x2.png, ...)
python main.py

Next steps

	•	Swap in a CNN for better accuracy on real handwriting.
	•	Add data augmentation, early stopping, and a tiny web demo (Flask/Streamlit).
	•	Package a CLI (predict.py --path /folder/of/pngs) and a Dockerfile for reproducibility.
