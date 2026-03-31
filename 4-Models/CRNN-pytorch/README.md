# Convolutional Recurrent Neural Network (CRNN) Model
  We constructed a CRNN model for this task. The model processes raw data using CNNs, and then feed its output to RNNs, forming a Convolutional Recurrent Neural Network (CRNN). In such case, convolutional layers extract local features, and recurrent layers combine it to extract temporal features. We take the original data which contains 2500 time steps and 12 leads (<batch_size>, 1, 2500, 12) as the input of the model. 
  
  ![alt text](./CRNN_model_architecture.png)

Use the plot(history) method to visulize the training and validationg accuracy, loss and validation AUC, depending on the metric used.

Now this model achieves AUC ROC of 0.86 on the validation set on the amyloid pace removed data in the data file. If higher performance of this model is desired, tune the model through the hyperparameter tuning scheme provided in this repository. 

Another advantage of this model is that this model is relatively shallow and very efficient as well. Currently, the number of parameters of the CRNN model we implemented is around 500,000, which is much lower than the 2D CNN model. This model is not resource-intensive and time-consuming.

## Dependencies

Install the Python packages in [`requirements-crnn.txt`](./requirements-crnn.txt). A clean environment is recommended:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r IntroECG/4-Models/CRNN-pytorch/requirements-crnn.txt
python -m ipykernel install --user --name introecg-crnn
```

The notebook metadata was originally created with Python 3.7.5, but a modern Python 3.10 or 3.11 environment generally works better with current PyTorch builds.

## End-To-End Pipeline

The CRNN notebook is the last stage of a larger pipeline. The intended repo flow is:

1. Extract 12-lead waveform arrays from source ECGs.
2. Build a metadata table with labels and filter out paced and poor-quality ECGs.
3. Assemble waveform arrays into a dataset and run baseline-wander removal, truncation, and normalization.
4. Train and evaluate the CRNN.
5. Plot the final accuracy, loss, and ROC AUC curves.

### 1. Extract waveforms

If your source data comes from GE Muse XML exports, use:

- [`muse_xml_to_array.py`](../../1-Waveform%20Extraction/muse_xml_to_array.py)

It writes one waveform per ECG as a `.npy` file with shape `(2500, 12, 1)` in lead order:

`I, II, III, aVR, aVL, aVF, V1, V2, V3, V4, V5, V6`

Example:

```bash
python IntroECG/1-Waveform\ Extraction/muse_xml_to_array.py /path/to/xml_directory
```

If your source data is PDF-only, use:

- [`ecg_pdf_to_dataframe.py`](../../1-Waveform%20Extraction/ecg_pdf_to_dataframe.py)

That script extracts vector traces into a dataframe first. Additional conversion work is typically needed to match the CRNN input shape.

### 2. Build metadata and labels

The preprocessing notebook expects a metadata CSV with at least:

- `PatientID`
- `AcquisitionDateTime_DT`
- `UniqueECGID`
- `Ventricular_Pacing_Present`
- `Poor_Data_Quality_Present`
- `ANY_AMYLOID`

The filename convention used to join metadata to waveform arrays is:

`{PatientID}_{AcquisitionDate}_{UniqueECGID}.npy`

The notebook in [`Waveform_Array_Generation_Truncation_Normalization.ipynb`](../../3-Preprocessing/Waveform_Array_Generation_Truncation_Normalization.ipynb) contains the example SQL and metadata processing steps used by the authors.

### 3. Preprocess waveforms

Run [`Waveform_Array_Generation_Truncation_Normalization.ipynb`](../../3-Preprocessing/Waveform_Array_Generation_Truncation_Normalization.ipynb) after updating its path variables. That notebook does the following:

- converts XML exports to `.npy` arrays if needed
- matches available waveform files to metadata rows
- keeps the newest ECG per patient
- removes paced and poor-quality ECGs
- removes baseline wander
- truncates outliers per lead
- normalizes each lead
- saves the final model-ready arrays with shape `(N, 1, 2500, 12)`

Important: several path variables in that notebook are placeholders and must be changed before running:

- `XML_input_directory_path`
- `query_metadata_result`
- `fpath_eval_df_newest_ecg_waveform_data_prebuilt`

### 4. Prepare CRNN train and eval files

The CRNN notebook currently expects prebuilt train and eval artifacts:

- train metadata CSV
- eval metadata CSV
- train waveform array `.npy`
- eval waveform array `.npy`

In the original notebook these were hard-coded under `/data/ECGnet/data/`. If your files live somewhere else, update those paths in [`Y_CRNN.ipynb`](./Y_CRNN.ipynb) before running.

The final waveform tensors loaded by the model must be shaped:

`(N, 1, 2500, 12)`

Labels are taken from the metadata column:

`ANY_AMYLOID`

### 5. Train and inspect results

Open and run [`Y_CRNN.ipynb`](./Y_CRNN.ipynb). The notebook:

- loads cached dataloaders from `loaders.data` if available
- otherwise builds `train_loader` and `test_loader`
- instantiates `CRNN(hidR=256, layerR=1, hidC=256)`
- trains with `BCEWithLogitsLoss` and `Adam`
- records train loss, validation loss, train accuracy, validation accuracy, and ROC AUC
- plots the learning curves with `plot(history)`

The main result outputs are:

- best validation ROC AUC
- best validation accuracy
- per-epoch history dataframe
- training/validation accuracy, loss, and ROC plots

## Practical Notes

- GPU is optional but strongly recommended for training.
- `torchvision` is imported in the notebook but is not central to the CRNN model itself.
- The notebook caches dataloaders to `loaders.data`, which speeds up reruns after the first preprocessing pass.
- This repo does not ship the original amyloid train/eval datasets, so you must provide your own waveform arrays and metadata labels.
- The EchoNext waveform splits are large. In this repo, the train split is about 16 GB on disk, so loading it with `np.array(...)` can crash a notebook kernel. For EchoNext, prefer the streaming helper in [`echonext_dataset.py`](./echonext_dataset.py), which uses memory mapping and loads samples batch-by-batch.
