> ## Archived, and it does not run
>
> This repo is archived and read only. Do not clone it expecting working code.
>
> `NumberRecognizer.py` calls `tf.keras.models.load_model('handwritten.keras')` on line 37, but
> `handwritten.keras` was never committed here, so the script raises on a clean clone. The training
> block that would produce that file is commented out, lines 14 to 30. The inference loop also reads
> from a hardcoded `~/Downloads/nums4` folder that does not exist for anyone else.
>
> The instructions below are also wrong. There is no `train.py` and no `main.py` in this repo. The
> only script is `NumberRecognizer.py`.
>
> Everything under this line is the original README, kept as written.
>
> For work that actually runs, see [ESG Investment Screener](https://github.com/utosrad/ESG-Investment-Screener)
> (live at https://esg-investment-screener.streamlit.app/) or the rest of
> [github.com/utosrad](https://github.com/utosrad).

---

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
