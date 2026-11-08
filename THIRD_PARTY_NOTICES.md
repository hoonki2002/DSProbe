# Third-party notices

## MVProbe

The probing architectures and selected training conventions are adapted from
[MVProbe](https://github.com/AI-hew-math/MVProbe), distributed under the MIT
License. Its copyright and permission notice are retained in [LICENSE](LICENSE).
This package reorganizes the required components into a standalone experiment
runner and adds displacement references and the accompanying comparisons.
[LICENSE](LICENSE) therefore carries two copyright lines: the upstream notice,
retained as the MIT License requires, and one for the code added here.

## ProbeX and Model Jungle

The ProbeX baseline and Model Jungle task follow
[Learning on Model Weights using Tree Experts](https://github.com/eliahuhorwitz/ProbeX).
The baseline architecture is adapted from the MIT-licensed MVProbe implementation.
No ProbeX downloader or model-bank files are redistributed here. Dataset and
pretrained-model access remain subject to the providers' licenses and terms.

## OpenAI CLIP

[OpenAI CLIP](https://github.com/openai/CLIP) is an optional external dependency
for preparing frozen SD-LoRA text features. It is distributed under the MIT
License and is not vendored in this directory. CLIP weights are not included.

## StatNN statistics and Model-J adaptation

The per-tensor summary-statistic definition follows Unterthiner et al. (2020),
*Predicting Neural Network Accuracy from Weights*, and the Google Research
[`dnn_predict_accuracy` notebook](https://github.com/google-research/google-research/blob/60a55a0fb30eb0722fa49aa7bb3412312d57c314/dnn_predict_accuracy/colab/dnn_predict_accuracy.ipynb)
at commit `60a55a0fb30eb0722fa49aa7bb3412312d57c314`. The upstream project is
licensed under Apache License 2.0; its notebook and checkout are not bundled.

This package independently implements the seven statistics (five quantiles,
population standard deviation, and mean) for every named parameter tensor and
adapts them to Model-J's 100 binary class-membership outputs using XGBoost.
It is not the original LightGBM accuracy-regression pipeline. The audited
parameter-name/shape manifest is packaged locally so runtime extraction does
not require Google Research code or `transformers`.

XGBoost is an optional, separately installed dependency, pinned to version
3.2.0 by the `statnn` extra, and retains its own license. The extra also installs
scikit-learn 1.7.2 for the classifier API; it retains its BSD license. No source
from either dependency or trained classifier is bundled.

## Other dependencies

PyTorch, NumPy, safetensors, Matplotlib, and the optional torchvision dependency
are installed separately and retain their respective licenses. The bundled
layer-order tables and ImageNet class-name mapping are taken from the
MIT-licensed MVProbe codebase.
