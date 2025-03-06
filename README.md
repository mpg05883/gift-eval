## Installation
1. Clone the repository and change the working directory to `GIFT-EVAL`.
2. Create a conda environment and activate it:
```
conda create -n gift python=3.11
```

3. Install required pamyenvckages:

If you just want to explore the dataset, you can install the required dependencies as follows:
```
pip install -e .
```

If you want to run baselines, you can install the required dependencies as follows:
```
pip install -e .[baseline]
```
Note: The specific instructions for installing the [Moirai](notebooks/moirai.ipynb) and [Chronos](notebooks/chronos.ipynb) models are available in their relevant notebooks.

4. Get the train/test dataset from [huggingface](https://huggingface.co/datasets/Salesforce/GiftEval).

```
huggingface-cli download Salesforce/GiftEval --repo-type=dataset --local-dir ../data
```

5. Set up the environment variables and add the path to the data:
```
echo "GIFT_EVAL=../data" >> .env
```
