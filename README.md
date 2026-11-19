# Implicit neural representation of lake dry bathymetry and water level from SWOT data

## Project Description

The project aims to estimate lake variables (volume change,water surface elevation, area, bathymetry) from SWOT point cloud data (PIXC)

![combinaison_predicted_z](https://gitlab.cnes.fr/campus/stages/2024-tpa-nicolas/-/raw/develop/example_config/combinaison_predicted_z_COP30.png)

# Step 1 : Dataset creation

SWOT data is freely available online on 2 platforms:
- NASA EarthData https://www.earthdata.nasa.gov/
- hydroweb.next https://hydroweb.next.theia-land.fr/

Configuration files in yaml are needed :

Code-speaking, the **dataset_CLI.py** called the the class **MultiDateLakeDataset** (defined in the file **datasetSWOT.py**). This class has methods to create a PyTorch dataset for the model. It contains the methods to
- first, convert multiple SWOT pixel cloud .nc files(raw SWOT data) to one unique .parquet (intermediate data)
- second, convert the .parquet to pytorch datasets .pt (normalized data ready for the model).

In the unique .parquet, all the SWOT points are present. For the pytorch dataset, it is possible to filter some of the points :
- based on their wse (ex : we remove the points that have wse < min_wse, because we know there are artefacts)
- based on the SWOT classification (ie : we want just the points classified as "water").

# Step 2 : Model training


Training the model is done by calling the training CLI like this : **train_CLI.py --config config_training.yaml**.
A configuration file in yaml is needed : you will find examples in the tutorial folder.
For training, the path to the dataset will be precised in the training config file.

In the tutorial folder, there is a template "sbatch file", that activates the virtualenv environnement and executes the CLI for you on TREX. You have to modify the name of your yaml config file within this sbatch file.

Code-speaking, the **training_CLI.py** calls a class **Trainer** (defined in the file **trainer.py**).
- The class StandardTrainer implements the training of "standard" (non-bayesian) model.
- The class BayesianTrainer, is a child class of StandardTrainer, and implements the training of a bayesian model.
- Many methods are in common, except for **instantiate_model** and **train**.

When you lauch the CLI, a folder is automatically created. Every N epoch (N = plot_every_epoch), the model is saved in this folder in .pth and a plot of the losses is saved as well.


### Training a Standard (non bayesian) model
The standard model has to estimate (at best) the height of each SWOT data point (x, y, t) :
- The training data is the SWOT data : points in space (x,y) at a certain time (t).
- The ground truth is the associated WSE (= Water Surface Elevation) measured by SWOT.
- The Loss is the Mean Squared Error = the squared difference between the height computed by the model and the true WSE.

After each epoch, the model gives a label to each point : either ground of water. A ground ratio is computed :
- if the ground ratio is 1, all points have been classified as ground.
- if the ground ratio is 0, all points have been classified as water.
The value of the ground ratio is an indication of the state of the training.

During training, plots are generated with the MSE Loss and the Ground Ratio.

### Training a Bayesian model
Everything explained for the Standard model is still valid. In addition, the model must also estimate the variance of the data. So the model will first output 2 probability distributions:
- 1 probability distribution for water points, with mu_water and sigma_water, that is estimated gaussian.
- 1 probability distribution for ground points, with mu_ground and sigma_ground. The user can change the type of the distribution.

The final output is also a probability distribution, which is the maximum of the 2 probability distributions water/ground.

Since this is a difficult task, there are 3 Losses that appears sequentially :
- first, the Loss is a Negative-Log-Likelihood
- then a custom "uncertainty loss"
- finally, a KL divergence loss
The user chooses at which epoch the Losses are introduced.
For more information, refer to the paper.

Example : training of a bayesian_GROUND-garf-w-16-d-4-sigma-0-1_WATER-mlp-w-16-d-4 model
- Epoch500 : we can see the evolution of the "ground ratio" (in blue) at the begining. The losses 2 (red) and 3 (green) have not been introduced yet

- End of training : Epoch9999. The model stabilized around a ground ratio ~0.5. The 2nd loss (red) was introduced at epoch 5500 and the 3rd loss (green) was introduced at epoch 7500.

# Inference

Making predictions is done by calling the inference CLI like this : **inference_CLI.py --config config_inference.yaml**.
A configuration file in yaml is needed : in the config file, the user can specify the kind of plots needed to assess the performance of the model :
- cross-section plots: for a fixed x, the model must predict h for each point along the y axis. The plots shows hg (the model proposition for the ground height), hw (the model proposition for the water height) and h (the final model output). The real SWOT data points are also showed for comparison, as well as the reference DEM.

- water probability on Sentinel-2 images: we plot the delimation water/non-water on Sentinel-2 optic images with a probability of 0.9 or 0.99 or 0.999.

- error histograms: after the introduction of the uncertainty loss, some points are considered "water" by the model with 99% confidence. These points are a "silver standard", an "intermediate ground truth". We compute the histogram of these points "true" wse (from SWOT) and we assess the error between the histogram and the distribution predicted by the model.

- DSM-like plots: we create a 2D grid (x,y) of the space, and the model must predict h in each point of the mesh. We can also plot the binary mask for a certain wse.

Code-speaking, the **inference_CLI.py** calls 3 classes (defined in the file **inference_classes.py**):
- The class SimulateInputs simulates inputs for a 2D-cut or a 3D-grid.
- The class Processor loads the model, perform prediction on the simulated input and format the output of the model.
- The class Plotter and the child classes PlotterErrorHistograms, PlotterCrossSection, PlotterWaterProbaOnS2, PlotterDSM do the actual plots defined above.

## Installing / Getting started

### Conda installation on HPC
```bash
mkdir implicit_bathy_project
cd implicit_bathy_project
git clone git@gitlab.cnes.fr:campus/stages/2024-tpa-nicolas.git implicit_bathy
export TMPDIR=/work/scratch/data/name/tmp_dir
ml conda/22.11.1
conda create --name implicit_bathy
conda activate env_implicit_bathy
conda install python==3.12
cd implicit_bathy
pip install -e . #Getting sources of the project
```

### Pixi installation

Get pixi and [install](https://pixi.prefix.dev/latest/installation/) it if needed:
```bash
wget -qO- https://pixi.sh/install.sh | sh
```
Then install this project:

```bash
git clone git@gitlab.cnes.fr:campus/stages/2024-tpa-nicolas.git implicit_bathy
cd implicit_bathy
pixi install
```

### Pixi installation on CNES HPC

> **⚠️ Warning**
>
> This environnement requires at least 8gb and write numerous files.
> Be careful in which folder you clone the source before installing the pixi env.


1. Replace standard channels by CNES artifactory in the pyproject.toml
```diff

 [tool.pixi.workspace]
-channels = ["conda-forge"]
+channels = ["https://artifactory.cnes.fr/artifactory/api/conda/conda-forge-remote", "conda-forge"]
 platforms = ["linux-64"]

+[tool.pixi.pypi-options]
+# Priority betwee extra and main index is counter-intuitive
+# see https://pixi.prefix.dev/latest/reference/pixi_manifest/#alternative-registries
+extra-index-urls = ["https://artifactory.cnes.fr/artifactory/api/pypi/pypi/simple"]
+index-url = "https://pypi.org/simple"
```
2. Ensure pip have the certificats

```conf
[global]
    cert = /etc/pki/tls/certs/ca-bundle.crt
```

3. Export 6 environnements variables
```bash
export UV_NATIVE_TLS=true
export SSL_CERT_FILE=/etc/pki/tls/certs/ca-bundle.crt
export CURL_CA_BUNDLE=/etc/pki/tls/cert.pem
export RATTLER_CACHE_DIR=/work/scratch/data/$USER/tmp
export TMPDIR=/work/scratch/data/$USER/tmp
export UV_CACHE_DIR=/work/scratch/data/$USER/tmp

#For the tests inside the code
export MODE_TEST=True
```

4. Login to artifactory
```sh
pixi auth login --username $USER --password $IDENTITY_TOKEN_ARTIFACTORY https://artifactory.cnes.fr
```
The identity token must be used as the password, not the public identity token.

5. Activate the project environment
```
pixi install
pixi shell
```

## Check the installation

```bash
ImplicitBathyOrchestrator -h
```

##  CURL error: SSL certificate problem

If you encounter the following error:
```
SSL certificate problem: self-signed certificate in certificate chain
```

You need to :

```
Conda :
cp /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem /chemin/vers/mon/environnement/lib/python3.X/site-packages/certifi/cacert.pem

Pixi :
cp /etc/pki/ca-trust/extracted/pem/tls-ca-bundle.pem .pixi/envs/default/lib/python3.12/site-packages/certifi/cacert.pem
```

## Requirement before downloading the data

### Download DSM 

In order to provide comparisons with other DSM, we propose automatic downloading (Corpernicus30, SRTM, ...)

Need to have an account : https://portal.opentopography.org/login
And in your env, export the API_KEY

```bash
export OPENTOPOGRAPHY_API_KEY=MY_API_KEY
```

### Download SWOT .nc files with EarthAccess tool -> podaac

You need to create an account : https://urs.earthdata.nasa.gov/

```bash
export EARTHDATA_TOKEN=YOUR_TOKEN
export EARTHDATA_USERNAME=USERNAME
export EARTHDATA_PASSWORD=PASSWORD

## Launch orchestrator

To launch the orchestrator, you need : a config file containing the parameters, a resource file and an optional CSV file.
Each configuration in the CSV file must have an ID. Configuration examples can be found in example_config/.

*For launching tasks :*

Simple Orchestrator :

```bash

ImplicitBathyOrchestrator -config config_file.yaml -config_ressource config_resources.yaml -scheduler_type Slurm -nb_parallel_tasks 10 (-starting_step 1 -ending_step 5)

```

Multiple Orchestrator :

```bash

ImplicitBathyOrchestrator -config config_file.yaml -config_ressource config_resources.yaml -csv_file parameters.csv -scheduler_type Slurm -nb_parallel_tasks 10 (-starting_step 1 -ending_step 6)

```

*In a local environment :*

Simple Orchestrator :

```bash

ImplicitBathy -config config_file.yaml (-starting_step 1 -ending_step 5)

```

Multiple Orchestrator

```bash

ImplicitBathy -config config_file.yaml -csv_file parameters.csv (-starting_step 1 -ending_step 6)

```


## Output Structure

Structure simple orchestrator :

```bash
output/
├── cfg/
│   ├── config_data_gpkg.yaml
│   ├── config_data_pt.yaml
│   ├── config_inference.yaml
│   └── config_train.yaml
├── data/
│   ├── dataset_classes_4_1.pt
│   └── dataset.gpkg
├── dsm/
│   └── COP30.tif
├── inference/
│   ├── absolute_surface_time_serie.png
│   ├── ...
│   ├── error_histograms_time_0.0.png
│   ├── ...
│   ├── plot_cross_section...png
│   ├── ...
│   ├── plot_water_proba_on_s2...png
│   ├── ...
│   ├── relative_volume...png
│   └── ...
├── inputs_nc/
│   └── SWOT_L2_HR_PIXC_016_264_069R_20240607T143155_20240607T143207_PIC0_01.nc ...
├── logs/
│   ├── DatasetGpkgStep/
│   ├── ...
│   ├── TrainStep/
│   ├── implicit_bathy_err.log
│   └── implicit_bathy_out.log
└── train/
    ├── bayesian_GROUND-garf-w-16-d-4-sigma-0-1_WATER-mlp-w-16-d-4_epoch9999.pth
    └── bayesian_GROUND-garf-w-16-d-4-sigma-0-1_WATER-mlp-w-16-d-4_epoch9999.png
```

Structure multiple orchestrator :

```bash
output/
├── cfg/
│   ├── config_data_gpkg.yaml
│   ├── config_data_pt.yaml
│   ├── config_inference.yaml
│   └── config_train.yaml
├── data/
│   └── dataset.gpkg
├── dsm/
│   └── COP30.tif
├── inputs_nc/
│   └── SWOT_L2_HR_PIXC_016_264_069R_20240607T143155_20240607T143207_PIC0_01.nc ...
├── logs/
│   ├── DatasetGpkgStep/dataset_gpkg.err
│   ├── DatasetGpkgStep/dataset_gpkg.out
│   ├── ...
│   ├── DatasetPtStep/dataset_pt_1.err
│   ├── DatasetPtStep/dataset_pt_1.out
│   ├── ...
│   ├── TrainStep/...
│   ├── implicit_bathy.slurm
│   ├── implicit_bathy_err.log
│   ├── implicit_bathy_out.log...
│   └── tasks_status_ImplicitBathy_1.svg
└── out_path_1/
│   ├── data/dataset_classes_4_1.pt
│   ├── inference/
│   ├── logs/DatasetPtStep/dataset_pt_1.err ...
│   ├── logs/DatasetPtStep/dataset_pt_1.out ...
│   └── train/
└── out_path_2/
│   ├── data/dataset_classes_4_1.pt
│   ├── inference/
│   ├── logs/DatasetPtStep/dataset_pt_1.err ...
│   ├── logs/DatasetPtStep/dataset_pt_1.out ...
│   └── train/
└── synthesis/
    ├── absolute_surface_time_serie.png
    ├── absolute_wse_time_serie.png
    ├──  ...
    ├── global_report.xlsx
    └── ...

```

## Launch test :

You can set the following environment variable : 

-MODE_TEST to true if you want to run the tests inside the code.

```bash
pytest -v -s --path_yaml tests/input/config_global.yaml --tmp_path tests/
```