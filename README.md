# Number-Recognizer

A model that scores 98.77% on MNIST reads 26.81% of distorted digits correctly. Preprocessing is the difference.

## The result

MNIST images are not raw drawings. Each one was size-normalised to fit a 20x20 box, then centred by
centre of mass in a 28x28 field, bright ink on a dark background. A classifier trained on that data
expects that exact framing at inference time. Feed it a drawing resized straight to 28x28 and it
guesses.

One model, trained once. Two preprocessing paths. Same images through both.

| set | `naive` | `mnist_style` | delta |
| --- | ---: | ---: | ---: |
| clean MNIST test | 98.77% | 98.10% | -0.67 |
| synthetic distorted | 26.81% | 95.69% | +68.88 |
| distorted, upright | 36.70% | 95.87% | +59.17 |
| distorted, inverted | 17.24% | 95.51% | +78.28 |

10,000 images per set, distortion seed 0. Produced by `python -m numrec evaluate` on the weights in
`models/mnist_cnn.pt`, on an Apple M5 CPU. `naive` on the clean set is a no-op resize, which is why
that cell matches the training script's reported test accuracy exactly.

**The distorted set is synthetic.** It is MNIST *test* digits with capture distortions applied in
code: random off-centering, random scale between 0.55x and 1.5x, a random canvas of 36 to 64 pixels
so the digit sits inside a margin, stroke thickening via a 3x3 max filter on half the images,
contrast compressed to a random ink and background level, and polarity inverted on half the images.
No photographs. No real scanned handwriting. See the limitations section.

Two things worth noting. `mnist_style` costs 0.67 points on already-normalised MNIST, because
re-cropping and re-centring an image that is already cropped and centred throws away a little
information. And `naive` on the upright half of the distorted set still only reaches 36.70%, so
polarity is not the whole story. Geometry alone accounts for most of the gap.

## Quickstart

```bash
pip install -r requirements.txt
python -m numrec train                    # ~50s on CPU, writes models/mnist_cnn.pt
python -m numrec evaluate                 # the table above, ~5s
python -m numrec predict examples         # mnist_style by default
python -m numrec predict examples --preprocess naive   # watch it fail
```

The three sample images in `examples/` are MNIST test digits with the same synthetic distortions
applied. They are not photographs. On those three:

```
preprocessing: mnist_style          preprocessing: naive
sample_0_true_7.png  7  (100.0%)    sample_0_true_7.png  8  (21.9%)
sample_1_true_2.png  2  (100.0%)    sample_1_true_2.png  1  (43.8%)
sample_2_true_1.png  1   (99.9%)    sample_2_true_1.png  1  (28.1%)
```

Three out of three, versus one out of three at a third of the confidence.

`predict` takes any folder. There are no hardcoded paths in this repo.

```bash
python -m numrec predict ~/my-drawings
```

Training is seeded. Running `train` on a fresh clone rewrote `models/mnist_cnn.pt` byte for byte
identically to the committed file, and `evaluate` reproduced the table above to the last decimal.
Verified on Python 3.14 in a clean virtualenv.

## How the preprocessing works

`naive` is the path most MNIST demos use, and the one the earlier version of this repo used.

```
image -> grayscale -> resize whole frame to 28x28 -> divide by 255
```

That is it. The digit keeps whatever position, scale and polarity it had in the frame.

`mnist_style` reproduces what the MNIST authors did to the original NIST scans.

Walking a made-up 56x56 capture through it, dark ink on light paper, digit off to one side:

```
1. input frame                 2. contrast stretched to [0,1].
   +--------------------+         The border ring is brighter than
   |                    |         the midpoint of the range, so the
   |   ####             |         image is inverted to bright ink.
   |      #             |
   |     #              |      3. threshold at 0.2, take the tight
   |    #               |         ink bounding box: 22 rows x 14 cols
   |    #               |         +------+
   |                    |         | #### |
   +--------------------+         |    # |
                                  |   #  |
                                  |  #   |
                                  |  #   |
                                  +------+

4. scale the LONG side to 20px      5. paste into a 28x28 field of zeros
   aspect ratio preserved              +--------------------------+
   22x14 -> 20x13                      |                          |
   +-----+                             |        +-----+           |
   |     |                             |        |20x13|           |
   | 20  |                             |        +-----+           |
   | x13 |                             |                          |
   +-----+                             +--------------------------+

6. translate so the intensity-weighted centre of mass lands on
   (13.5, 13.5), the centre of a 28-wide field. The shift is rounded
   to whole pixels, so the residual offset is at most half a pixel
   on each axis.
```

Polarity is decided by the border ring. The mean of the outermost pixels is compared against the
midpoint of the image's own dynamic range, not against a fixed 0.5. That is what makes it survive a
washed-out capture where every pixel sits between 0.3 and 0.6.

Both paths live in `numrec/preprocess.py` and both are selectable everywhere.

## Commands

| command | what it does |
| --- | --- |
| `train` | trains the CNN on MNIST, saves weights, prints clean test accuracy |
| `evaluate` | runs both preprocessing paths over clean and synthetic distorted sets |
| `predict FOLDER` | classifies every PNG or JPG in a folder, `--preprocess naive` to compare |
| `make-examples FOLDER` | regenerates the sample images |

The model is a 216,170 parameter CNN: two conv blocks, max pooling, a 128-unit head with dropout.
Three epochs of Adam at 1e-3, batch size 128, no augmentation. Training took 51.5 seconds on an
Apple M5 CPU and reached 98.77% on the MNIST test split.

No augmentation is deliberate. Augmenting the training set is the other way to close this gap, and
it would hide the measurement this repo exists to make.

## Tests

```bash
pip install pytest==9.1.1
pytest
```

75 tests, all passing, under a second. They assert derived facts rather than recorded outputs: that
centring an off-centre synthetic blob lands its centre of mass within half a pixel of the field
centre, that a 40x20 shape scales to 20x10 so the aspect ratio survives, that the bounding box on a
known rectangle is exactly that rectangle, that polarity detection fires on both polarities and on a
low-contrast capture, that a blank image returns an empty field instead of raising, that the two
paths disagree on an off-centre input while agreeing that the output is 28x28 in [0, 1], and that the
distorted set is drawn from the MNIST test split and never the training split.

The tests do not download MNIST.

## Limitations

The distorted set is synthetic. It is built by transforming MNIST test images in code, so it inherits
MNIST's stroke style, its writer population and its digitisation. Real captures bring things this
generator does not model: perspective, uneven lighting, JPEG artefacts, ruled paper, ballpoint versus
marker, digits touching a box edge. The 95.69% figure should be read as "the preprocessing recovers
almost everything these particular distortions destroy", not as an accuracy estimate on photographs.

Honest testing on real handwriting would need a few hundred scans or photos from several writers,
labelled by hand, held out entirely, and ideally captured on more than one device. That set does not
exist here and nothing in this repo claims otherwise.

Other known limits. Preprocessing assumes one digit per image, so it will crop to the union of
several digits rather than segmenting them. The threshold at 0.2 of the dynamic range is fixed and
will struggle with very faint pencil. The centring shift is rounded to whole pixels rather than
resampled. The polarity heuristic reads the border ring, so a frame with a dark border and a light
interior will fool it.

## What this replaces

The earlier version of this repo was a TensorFlow script whose training block was commented out and
whose model file, `handwritten.keras`, was never committed. It raised on any clean clone. The
inference loop read a hardcoded `~/Downloads/nums4` folder and resized whatever it found straight to
28x28, which is exactly the `naive` path measured above. That silent preprocessing gap is the failure
this repo now measures and closes.

## License

MIT. See [LICENSE](LICENSE).
