<h1 align="center">
  ω-0: A Latent Predictive World Action Model<br>
  for Concurrent Humanoid Loco-Manipulation
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2608.06375"><img src="https://img.shields.io/badge/arXiv-2608.06375-b31b1b.svg" alt="arXiv"></a>
  <a href="https://gentlefress.github.io/OMEGA-0_page/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/keycharon/omega-HOME"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-%CF%89--HOME-yellow.svg" alt="Hugging Face Dataset: ω-HOME"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-green.svg" alt="MIT License"></a>
  <a href="https://x.com/Jianfei_AI/status/2086677871960236291?s=20"><img src="https://img.shields.io/badge/Twitter%20%2F%20X-000000.svg" alt="Twitter / X"></a>
  <a href="https://xhslink.cn/o/11fRT5XexqR"><img src="https://img.shields.io/badge/%E5%B0%8F%E7%BA%A2%E4%B9%A6-FF2442.svg" alt="小红书 / Xiaohongshu"></a>
  <a href="https://www.linkedin.com/posts/jianfei-yang-55560386_humanoidrobotics-robotics-embodiedai-activity-7492444308520718336-YSDh?utm_source=share&amp;utm_medium=member_desktop&amp;rcm=ACoAAFw0zd8BUbAGtl28gqUhAJW6kfODKtA5wFA"><img src="https://img.shields.io/badge/LinkedIn-0A66C2.svg" alt="LinkedIn"></a>
</p>

<p align="center">
  Zhe Li<sup>1,3*†</sup> · Zhenzhe Zhang<sup>2,3*</sup> · Yangyang Wei<sup>3*</sup> · Wenjie Zhang<sup>4*</sup> · Xichen Yuan<sup>1*</sup><br>
  Peiyuan Zhi<sup>3</sup> · Gen Li<sup>1</sup> · Xinying Guo<sup>1</sup> · Fengjie Gao<sup>1</sup> · Jianfei Yang<sup>1♣</sup> · Shanghang Zhang<sup>2♣</sup>
</p>

<p align="center">
  <sup>1</sup> MARS Lab, Nanyang Technological University (NTU)<br>
  <sup>2</sup> Peking University · <sup>3</sup> Beijing Academy of Artificial Intelligence (BAAI)<br>
  <sup>4</sup> The Hong Kong University of Science and Technology (Guangzhou)<br>
  <strong>*</strong> Equal contribution · <strong>†</strong> Project lead · <strong>♣</strong> Corresponding authors
</p>

<p align="center">
  <img src="https://gentlefress.github.io/OMEGA-0_page/assets/logo/mars_lab_logo.png" alt="MARS Lab" height="80" align="middle">
  &nbsp;&nbsp;
  <img src="https://gentlefress.github.io/OMEGA-0_page/assets/logo/hmi_logo.png" alt="HMI Lab" height="155" align="middle">
</p>

<p align="center">
  <a href="https://gentlefress.github.io/OMEGA-0_page/assets/videos/teaser.mp4">
    <img src="assets/teaser.svg" alt="OMEGA-0 overview: the ω-HOME dataset, latent world–action learning, and humanoid household demonstrations" width="100%">
  </a>
</p>
<p align="center">
  <a href="https://gentlefress.github.io/OMEGA-0_page/assets/videos/teaser.mp4">Watch the teaser video</a>
</p>

<a id="introduction"></a>

## 🌟 Introduction

**ω-0 (OMEGA-0)** is a latent predictive world action model for concurrent humanoid
locomotion and manipulation. It maps language instructions, visual observations,
and robot proprioceptive state to whole-body action latents, enabling coordinated
movement and object interaction through the SONIC controller.

This repository provides model training, inference, teleoperation, and episode
recording for the Unitree G1. The sections below describe environment setup,
robot operation, and the two training stages.

OMEGA-0 is released under the [MIT License](LICENSE). Third-party code and files
with their own license headers remain subject to their original licenses.

## 🎬 Demos

[▶ Watch the full introduction](https://gentlefress.github.io/OMEGA-0_page/assets/videos/intro.mp4)

<table>
  <tr>
    <td align="center" width="50%">
      <strong>Pick Garbage</strong><br>
      <a href="https://gentlefress.github.io/OMEGA-0_page/assets/videos/demo_lib/pick_garbage/exo.mp4"><img src="https://gentlefress.github.io/OMEGA-0_page/assets/videos/demo_lib/pick_garbage/exo.jpg" alt="Pick Garbage demo — click to watch" width="100%"></a><br>
    </td>
    <td align="center" width="50%">
      <strong>Retrieve From Fridge</strong><br>
      <a href="https://gentlefress.github.io/OMEGA-0_page/assets/videos/demo_lib/retrieve_from_upper_fridge/exo.mp4"><img src="https://gentlefress.github.io/OMEGA-0_page/assets/videos/demo_lib/retrieve_from_upper_fridge/exo.jpg" alt="Retrieve From Upper Fridge demo — click to watch" width="100%"></a><br>
    </td>
  </tr>
  <tr>
    <td align="center" width="50%">
      <strong>Pick Clothes From Washing Machine</strong><br>
      <a href="https://gentlefress.github.io/OMEGA-0_page/assets/videos/demo_lib/pick_clothes_from_washing_machine/exo.mp4"><img src="https://gentlefress.github.io/OMEGA-0_page/assets/videos/demo_lib/pick_clothes_from_washing_machine/exo.jpg" alt="Pick Clothes From Washing Machine demo — click to watch" width="100%"></a><br>
    </td>
    <td align="center" width="50%">
      <strong>Clean Bed</strong><br>
      <a href="https://gentlefress.github.io/OMEGA-0_page/assets/videos/demo_lib/clean_bed/exo.mp4"><img src="https://gentlefress.github.io/OMEGA-0_page/assets/videos/demo_lib/clean_bed/exo.jpg" alt="Clean Bed demo — click to watch" width="100%"></a><br>
    </td>
  </tr>
</table>

## ✅ TODO

- [x] Provide training and inference code.
- [x] Provide teleoperation, recording, and robot deployment instructions.
- [x] Release the [ω-HOME dataset](https://huggingface.co/datasets/keycharon/omega-HOME) on Hugging Face.
- [ ] Publish pretrained checkpoints.

<a id="installation"></a>

## 📦 Installation

### Python Environment

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from the repository root:

```bash
uv sync --locked --extra train --extra deploy --python 3.10 --inexact
source .venv/bin/activate
uv pip install flash-attn --no-build-isolation
```

The `train` extra installs training and inference dependencies; `deploy` adds
the Python robot runtime. Building FlashAttention requires a compatible CUDA
development toolkit.

Verify the installation and CUDA availability:

```bash
python -c "import omega, torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

Activate this environment in each workstation terminal used below. Run Python
commands from the repository root unless another directory is specified.

<a id="hardware-setup"></a>

## 🛠️ Hardware Setup

The supplied configurations target a Unitree G1 with Inspire hands and a
ZED Mini egocentric camera. Teleoperation uses Pico body tracking through
XRoboToolkit. The SONIC controller can run on the workstation or the G1's Orin.
Configure the network addresses according to where each service runs.

### SONIC Controller

OMEGA-0 uses [SONIC](https://github.com/NVlabs/GR00T-WholeBodyControl) for low-level
whole-body control. The bundled [gear_sonic_deploy](thirdparty/gear_sonic_deploy/) backend
includes support for Inspire dexterous hands.

Follow the [SONIC installation guide](https://nvlabs.github.io/GR00T-WholeBodyControl/getting_started/installation_deploy.html)
for controller dependencies, build instructions, and model assets. Use the
[launch commands below](#start-robot-services) to connect SONIC to OMEGA-0.
The original upstream [README](thirdparty/gear_sonic_deploy/README.md) and
[license](thirdparty/gear_sonic_deploy/LICENSE) are preserved alongside the backend.

### Inspire Hands

On the robot, install the [Inspire Hand SDK](https://github.com/NaCl-1374/inspire_hand_ws)
and its dependencies in the environment used for the hand driver:

```bash
git clone --recurse-submodules https://github.com/NaCl-1374/inspire_hand_ws.git
cd inspire_hand_ws
python -m pip install -r requirements.txt
python -m pip install -e ./unitree_sdk2_python -e ./inspire_hand_sdk
```

### ZED Mini Camera

Install the [ZED SDK](https://www.stereolabs.com/en-sg/developers/release) version
compatible with the robot's JetPack installation. Build the bundled Orin Video
Sender on the robot with CUDA and the ZED SDK installed:

```bash
sudo apt-get install -y build-essential pkg-config libzmq3-dev libopencv-dev \
  libssl-dev libglib2.0-dev libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev libavcodec-dev libavformat-dev \
  libavutil-dev libswscale-dev libavdevice-dev
cd /path/to/OMEGA-0/thirdparty/XRoboToolkit-Orin-Video-Sender
make
```

The build produces `OrinVideoSender_jpeg`. The Python camera module receives its
JPEG stream and sends `OPEN_CAMERA` through the TCP control service.

### Pico Tracking and Optional Exocentric Camera

Configure Pico body tracking and install the XRoboToolkit Python SDK
(`xrobotoolkit_sdk`) in the workstation environment. The collection configuration
also enables an exocentric ZED camera, which requires the ZED SDK and its Python
bindings (`pyzed`) on the workstation.

For an ego-only setup, remove the `exocentric_camera` module from
[collect.yaml](real/configs/collect.yaml), along with the recorder's `exo_image`
and `exo_depth` inputs and their entries under `recorder.options.layout.videos`
and `recorder.options.layout.images`.

### Network Connections

Set these variables in each workstation terminal running a collection or
deployment client. Replace the example addresses with your setup:

```bash
# Camera service on the G1
export CAMERA_HOST='<G1-ip-addr>'

# Commands published by Python; telemetry published by SONIC
export COMMAND_ENDPOINT='tcp://*:5556'
export TELEMETRY_ENDPOINT='tcp://<addr-of-machine-running-sonic>:5557'
```

### Start Robot Services

Start each service in a separate terminal and keep it running throughout
collection or deployment. Use the hand-driver environment on the robot and
the project environment on the workstation.

**Hand driver:**

```bash
cd /path/to/inspire_hand_ws/inspire_hand_sdk/example
python Headless_driver_double.py
```

**Camera sender:**

```bash
cd /path/to/OMEGA-0/thirdparty/XRoboToolkit-Orin-Video-Sender
./OrinVideoSender_jpeg \
  --listen 192.168.123.164:13579 \
  --zmq-raw 'tcp://*:5555'
```

Set `--listen` to the robot's address. In this sender, `--zmq-raw` publishes
JPEG frames; `--zmq` publishes H.264. The Python configs expect JPEG on port
5555 and camera control on port 13579. If the camera is not detected after boot,
reconnect it before restarting the sender.

**SONIC:**

After installing the controller dependencies and placing its model assets at
the paths expected by `deploy.sh`, launch the container:

```bash
cd /path/to/OMEGA-0/thirdparty/gear_sonic_deploy
export TensorRT_ROOT=/path/to/TensorRT
./docker/run-ros2-dev.sh --with-opengl
```

Inside the container, select the network interface connected to the G1 and the
workstation address:

```bash
./deploy.sh eth0 \
  --input-type zmq_manager \
  --output-type zmq \
  --zmq-host 192.168.123.100
```

Replace `eth0` and `192.168.123.100` with your interface and workstation address.
Use `--zmq-host 127.0.0.1` when SONIC and the Python client run on the same host.

<a id="data-collection"></a>

## 🎥 Data Collection

Start the robot services and Pico tracking, then activate the workstation
environment. Set the variables in [Network Connections](#network-connections)
and provide the skeleton and robot description from your SONIC checkout:

```bash
export SKELETON_PATH=/path/to/sonic/data/human/human_joints_info.pkl
export ROBOT_URDF_PATH=/path/to/sonic/data/robots/g1/g1_29dof_with_hand.urdf
export INSTRUCTION='Example instruction.'

python -m omega_real.run --config real/configs/collect.yaml --check
python -m omega_real.run --config real/configs/collect.yaml --arm
```

Run `--check` to validate the configuration and module bindings without opening
hardware or sockets. Launch with `--arm` to allow command transmission, then
start the controller using the controls below.

| Control | Action |
| --- | --- |
| A + B + X + Y | Start and calibrate from OFF; stop when active |
| A + X | Switch between planner and pose teleoperation |
| B + Y | Toggle frozen upper-body mode from pose teleoperation |
| Hold left menu | Pause pose teleoperation; release to resume |
| Left grip + A, in pose mode | Start recording or finish the current episode |
| Left grip + B, in pose mode | Abort and discard the current episode |
| `q` | Request a software emergency stop |
| Ctrl+C | Stop the Python runtime; send a stop command by default |

Recordings are saved under `recorder.options.root` (`artifacts/robot` by default).
Each completed episode contains `state_action.hdf5`, `ego.mp4`, and
`session_meta.json`. Exocentric RGB (`exo.mp4`) and depth (`exo_depth/`) are
optional; depth images are stored separately from HDF5. Prepare these recordings
in the dataset layout described in [Training](#training) before fine-tuning.

<a id="deployment"></a>

## 🤖 Deployment

Start the robot services, activate the workstation environment, and set the
variables in [Network Connections](#network-connections). Stop the collection
client before starting deployment: both clients bind the command publisher to
TCP 5556. Run the inference server and robot client in separate terminals.

### Inference Server

Set the checkpoint and external model paths in
[serve.yaml](src/configs/serve.yaml). Its `checkpoint` and `external_models`
options load an original-format checkpoint. For an export produced by this
repository's training code, replace the `model` block with:

```yaml
model:
  artifact: /path/to/artifacts/finetune/policy-00080000
  device: cuda
```

External model assets referenced in the export must remain available at their
configured paths. Launch the server with the project environment active:

```bash
python -m omega.inference.run --config src/configs/serve.yaml --check
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=4 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m omega.inference.run --config src/configs/serve.yaml
```

The default server listens on `127.0.0.1:8014`. If the Python client runs on
another computer, set `host` to the server's reachable interface (or `0.0.0.0`)
and use that computer's address in `INFERENCE_ENDPOINT`.

### Robot Client

In the client terminal, set the variables from
[Network Connections](#network-connections), then specify the inference endpoint
and task instruction:

```bash
export INFERENCE_ENDPOINT=http://127.0.0.1:8014
export INSTRUCTION='Example instruction.'

python -m omega_real.run --config real/configs/deploy.yaml --check
python -m omega_real.run --config real/configs/deploy.yaml --arm
```

Set `model.options.view` to `ego` or `exo` to match the input camera. The default
configuration uses a 640 × 360 crop from the left side of the stereo ego image.
Adjust `camera.options.crop` for other camera layouts.

Press **Enter** to start the controller in planner mode. Wait for the ready
message after the configured 2.5-second settling period, then press **Enter**
again to enable model commands.

| Control | Action |
| --- | --- |
| `p` | Pause or resume model output |
| `q` | Request a software emergency stop |
| Ctrl+C | Exit the Python runtime |

Each module reports its current state. Use `--status-interval` to adjust the
reporting interval.

<a id="training"></a>

## 🧠 Training

The supplied training recipes cover action-token pretraining and world action
model (WAM) fine-tuning. Both stages use a shared Accelerate training loop with checkpoint
saving and resume support.

| Stage | Objective | Configuration |
| --- | --- | --- |
| Action-Token Pretraining | Autoregressive prediction of FAST whole-body motion tokens with Qwen3-VL | [vlm_pretrain.yaml](src/configs/vlm_pretrain.yaml) |
| WAM Fine-Tuning | Action prediction and future visual representation learning | [finetune.yaml](src/configs/finetune.yaml) |

Configuration values of the form `${VLM_PATH}` are resolved from environment
variables at launch. Paths in the examples are placeholders and must be replaced
with the corresponding local assets.

### VLM Pretraining

This stage trains Qwen3-VL to associate language and visual observations with
whole-body motion tokens. The default configuration uses 75-dimensional SMPL
actions and a FAST vocabulary of 2,048 codes. Each sample predicts one action
three frames after the sampled observation.

**Data preparation.** Organize the HDF5 annotations and corresponding videos with
matching filenames:

```text
pretrain_data/
├── annotation_smpl/
│   └── episode_0000.hdf5
└── video_smpl/
    └── episode_0000.mp4
```

Each annotation must contain the following datasets:

| Dataset | Content |
| --- | --- |
| `motion` | Frame-aligned motion array with 85 columns; the reader selects the first 75 columns for this recipe |
| `instruction` | Task instruction |
| `view` | `first` for egocentric observations or `third` for exocentric observations |

The normalization JSON must provide 75-element `q01` and `q99` arrays under the
`wam_smpl` key. Video frames and motion annotations must be temporally aligned.

**Training.** Specify the pretrained backbone, FAST tokenizer, dataset, and
normalization statistics, then launch:

```bash
export VLM_PATH=/path/to/Qwen3-VL
export FAST_TOKENIZER_PATH=/path/to/tokenizer_2048_smpl_75
export DATA_ROOT=/path/to/pretrain_data
export ACTION_STATS_PATH=/path/to/action_stats.json

python -m omega.training.run --config src/configs/vlm_pretrain.yaml
```

**Outputs.** Checkpoints are saved under `artifacts/vlm_pretrain/`. Each contains a
`backbone/` directory with the trained Qwen model and processor. This directory
can be assigned to `VLM_PATH` for subsequent WAM fine-tuning.

### WAM Fine-Tuning

The default fine-tuning configuration optimizes the predictor while keeping the
vision-language backbone and frame encoder frozen. It uses 30-step action
chunks with 66-dimensional targets and initializes the predictor from an
existing checkpoint through `predictor_init`.

**Model assets.** Set the following environment variables to local paths:

| Variable | Required asset |
| --- | --- |
| `VLM_PATH` | Qwen model and processor directory, such as a pretraining `backbone/` export |
| `T5_PATH` | Local T5 model and tokenizer directory |
| `VJEPA_PATH` | V-JEPA frame-encoder checkpoint matching the configured architecture |
| `WAN_PATH` | Path to `Wan2.2-TI2V-5B/Wan2.2_VAE.pth` (48-channel Wan2.2 VAE) |
| `PREDICTOR_INIT_PATH` | Predictor initialization checkpoint matching the configured architecture |
| `DATA_ROOT` | Prepared fine-tuning dataset directory |

**Data preparation.** Organize the training annotations and ego videos as follows:

```text
finetune_data/
├── annotation/
│   └── episode_0000.hdf5
└── first/
    └── episode_0000.mp4
```

Each annotation must follow the format consumed by
[FinetuneDataset](src/data/wholebody/finetune.py): frame-aligned `latent` and
`state` arrays, an `instruction` string, and a `view` string (`first` or `third`).
The reader removes the final six state channels (linear acceleration and angular
velocity).

Use `annotation_dir` and `ego_video_dir` in the reader options for other directory
layouts, including `annotation_pure_zup_zeroyaw/` and `video_smpl_first/`.

The default recipe uses egocentric video without state conditioning. Action
normalization is set to `none`. If normalized targets are required, configure
both `NormalizeFields` in the data transforms and `artifact.normalization` with
the same statistics so inference can denormalize the predictions correctly.

**Training.** After configuring the asset and dataset paths, launch:

```bash
python -m omega.training.run --config src/configs/finetune.yaml
```

**Outputs.** Training checkpoints are saved under `artifacts/finetune/`. At the
end of training, model weights and metadata are exported to `policy-XXXXXXXX/`,
where `XXXXXXXX` is the number of completed optimizer steps. Use this export
with the inference server as described in the deployment section.

### Distributed Training

Use `torchrun` for distributed training. Configure `train.batch_size`,
`train.gradient_accumulation_steps`, and `train.workers` for the available
hardware. The effective global batch size is:

```text
global batch size = batch size per process × number of processes × accumulation steps
```

For single-node training with four GPUs:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m omega.training.run --config src/configs/finetune.yaml
```

### Resuming Training

Resume from a checkpoint directory using the original training configuration:

```bash
python -m omega.training.run --config src/configs/vlm_pretrain.yaml \
  --resume artifacts/vlm_pretrain/checkpoint-00005000
```

Resuming requires the same process count, dataset batching, and gradient
accumulation settings. Exact mid-epoch reproduction of data transformations
requires `train.workers: 0` and `train.exact_resume: true`. The provided
configurations use multiple data-loading workers and set `exact_resume: false`.

<a id="acknowledgements"></a>

## 🙏 Acknowledgements

We would like to acknowledge the following projects from which parts of the code in this repo are derived from:

- [Ψ₀](https://github.com/physical-superintelligence-lab/Psi0)
- [GR00T-WholeBodyControl](https://github.com/NVlabs/GR00T-WholeBodyControl)

<a id="citation"></a>

## 📚 Citation

If you use OMEGA-0 in your research, please cite
[OMEGA-0](https://arxiv.org/abs/2608.06375):

```bibtex
@article{li2026omega0,
  title   = {{$\omega$-0: A Latent Predictive World Action Model for Concurrent Humanoid Loco-Manipulation}},
  author  = {Zhe Li and Zhenzhe Zhang and Yangyang Wei and Wenjie Zhang and
             Xichen Yuan and Peiyuan Zhi and Gen Li and Xinying Guo and
             Fengjie Gao and Jianfei Yang and Shanghang Zhang},
  journal = {arXiv preprint arXiv:2608.06375},
  year    = {2026},
  doi     = {10.48550/arXiv.2608.06375},
  url     = {https://arxiv.org/abs/2608.06375}
}
```
