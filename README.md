# Biometrics

Anti-spoof algorithm based on fine-tuning MobileNet on CelebaSpoof dataset.

The data vas cleaned by [deduplicator](https://github.com/idealo/imagededup)

Test accuracy = 0.88

# Inference

Use `Inference.ipynb`

or

1. Place images for testing in `data/my_test/`

2. Ensure `src/config.py` contains correct paths

3. Run from project root: 

    ```python -m src.predict```

# Training

Write the correct paths for data (path_local) in `config.py`

Load the Data and place it to `data/`:

https://github.com/ZhangYuanhan-AI/CelebA-Spoof

For creating `.json` with unique test and train labels, run: ```python -m src.deduplicator```

For training run from project root: 

```python -m src.train```

for convert model in ONNX:

```python -m src.convert_to_onnx```


# links

[Presentation](https://docs.google.com/presentation/d/1OsWTKHuWqyzJY5QuILpMl7KXF-wb59mlq48AoqmYqF4)

[Video demonstration](https://drive.google.com/file/d/1-5gc4cWe3OZ6t8y_U1Wyo8beI3_7mYhQ)

[ClearML experiment](https://app.clear.ml/projects/b224527669464b0294fd6bf796590d63/experiments/6b89fc0d4fc94598874358f887b29c8c/output/execution)

