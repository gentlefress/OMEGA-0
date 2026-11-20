# ω-0 fine-tuning on HA-Lab 38-task data

ω-0 (HF `keycharon/Omega-zero`)를 HA-Lab 38-task 데이터로 fine-tune하는 코드입니다. OMEGA-0 원본
(commit `2ca35a3`)은 수정하지 않았고, 여기 `halab/` 파일들이 원본 학습 루프(`omega.training.engine`)를
그대로 불러 씁니다.

| 파일 | 역할 |
|---|---|
| `omega_halab_data.py` | LeRobot 데이터 → ω-0 학습 형식(hdf5 + stats) 변환, 학습용 reader |
| `omega_halab_train.py` | 학습 실행: config, weight surgery, resume, `load_policy` |
| `omega0_home_stats.py` | 몸통 회전(rot6d) 계산 함수 |
| `vla_task_registry.py`, `episode_split.py` | HA-Lab `data_loader/ha/`의 수정 없는 복사본 (task 탐색, val split 규칙) |
| `jobs/omega_halab38_c50_rtc_50k.sbatch` | 학습 job (A40 4장) |
| `jobs/omega_halab38_c50_rtc_50k_l40s.sbatch` | 같은 job을 L40S 4장으로 |

## 학습 설정

공개 ckpt의 `run_config.json`(원래 fine-tune 설정)을 따르고, action 공간과 chunk만 바꿨습니다.

- action: `DIM=76` = SONIC token 64 + 손가락 12 (Inspire openness, 1 = open) / `DIM=78` = + 카메라 2. GMR 팔 없음.
- state: 45D = 관절 29 + 손 5+5 + 몸통 rot6d (에피소드 첫 yaw 제거). ω-0 `sonic_state`와 같은 형식.
- chunk 50, 학습 시 RTC (prefix 0~7행), predictor만 학습 (Qwen3-VL / V-JEPA ViT-g / T5 고정).
- AdamW lr 1e-4, betas (0.95, 0.999), wd 1e-6, warmup 1000, cosine, grad clip 1, bf16.
- global batch 256 = GPU당 8 × 4장 × accumulation 8, 50k step.
- 정규화: action std, state bounds_q99. stats는 train split에서 계산 (`stats.json`).
- val split: GR00T 38-task run과 같은 규칙 (전체 episode의 5%, seed 7).

weight surgery (ckpt → 새 모양, 나머지는 전부 그대로 strict load):

- action head `input_proj` / `final_layer.linear` 66 → 76/78: token 64는 복사, 손가락은 각 손 trigger weight의
  부호 반대 (openness = 1 − trigger, 입력 쪽은 /6), 카메라는 0.
- `motion_queries` 25 → 50행, `ego_queries` 1792 → 3328 (Wan latent 7 → 13 프레임): 선형 보간.
- `temporal_pos_emb`: 쓰이지 않아 새로 초기화.

원본 학습 루프에 없어서 추가한 것: DDP `find_unused_parameters` (원본은 DeepSpeed ZeRO-3), 최신 checkpoint
자동 resume, 10k마다 `keep-<step>` 저장, 로그 시각. ViT-g에는 현재 프레임 1장만 넣습니다 (원본은 51장을 넣고
1장만 씀, loss 동일). 영상은 PyAV로 읽습니다.

## 1. 환경

```bash
cd OMEGA-0
uv sync --locked --extra train --python 3.10 --inexact      # OMEGA-0 README의 학습 환경
uv pip install wandb pandas pyarrow                          # halab/ 코드가 추가로 쓰는 패키지
```

## 2. 파일 준비

`OMEGA_GB`(기본 base 폴더) 아래에 아래처럼 둡니다. 각 경로는 환경변수로 따로 지정할 수도 있습니다.

```
$OMEGA_GB/
├── datasets/ha-lab-dataset-38tasks/halab/   HA-Lab 38-task LeRobot 데이터 (OMEGA_HALAB_DATA)
├── omega/                                   (OMEGA_ASSETS)
│   ├── Omega-zero/                          HF keycharon/Omega-zero
│   ├── external/t5-base/                    HF google-t5/t5-base
│   ├── external/Wan2.2_VAE.pth              HF Wan-AI/Wan2.2-TI2V-5B
│   └── halab38/                             첫 실행 때 자동 생성 (OMEGA_DATA)
├── huggingface/                             HF_HOME
└── outputs/                                 학습 출력 (OMEGA_OUT_ROOT)
```

```bash
A=$OMEGA_GB/omega
hf download keycharon/Omega-zero --local-dir $A/Omega-zero --exclude "random_states_*" zero_to_fp32.py scheduler.bin
hf download google-t5/t5-base --local-dir $A/external/t5-base --include "*.json" spiece.model model.safetensors
hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir $A/external
```

HA-Lab 데이터는 기존 run과 같은 revision(HF `sehoonha/ha-lab-dataset`, rev `702ef02b`, 2026-10-25, 38 task)을
써야 합니다. 데이터가 다르면 split, stats, epoch당 batch 수가 달라져서 기존 checkpoint에서 이어 학습할 수 없습니다.

## 3. 실행

`jobs/*.sbatch`는 Skynet 기준입니다. 다른 클러스터에서는 `#SBATCH` 줄 (partition, qos, GPU 종류,
`--exclude`, `--output`/`--error` 경로)을 맞게 고쳐 주세요. OMEGA-0 루트에서 제출합니다.

```bash
cd OMEGA-0 && OMEGA_GB=<base> DIM=76 sbatch halab/jobs/omega_halab38_c50_rtc_50k.sbatch
```

- 첫 실행은 학습 전에 데이터 변환을 한 번 합니다 (`$OMEGA_GB/omega/halab38/`, CPU).
- 7일 job 하나가 끝나도 50k에 못 갔으면 마지막 checkpoint에서 다음 job을 자동으로 제출합니다.
- slurm 없이 돌릴 때는 sbatch 안의 `torch.distributed.run ... omega_halab_train.py` 명령을 그대로 쓰면 됩니다
  (`PYTHONPATH`에 `halab/` 추가).

## 4. 이어서 학습

`checkpoint-XXXXXXXX` 폴더 (`progress.json`이 있는 것)를 `$OMEGA_GB/outputs/<run 이름>/` 안에 넣고 같은 명령으로
제출하면 그 step부터 이어집니다. 기본 run 이름은 `halab38_omega0_76d_c50_rtc8_g256_50k`입니다 (`EXP`로 변경).
checkpoint가 없으면 0 step부터 시작합니다.

이어서 학습하려면 아래가 기존 run과 같아야 합니다 (다르면 engine이 resume을 거부합니다).

- GPU 4장 × `--batch-size 8` × `--accumulation 8`
- 같은 데이터 revision (epoch당 batch 수)
- `MAX_STEPS=50000` (cosine lr이 50k 기준)

## 5. 출력

- `checkpoint-*`: 100 step마다, 최근 2개만 유지 (model + optimizer, 약 13GB).
- `keep-<step>/`: 10k마다 model weight (약 9GB). `omega_halab_train.load_policy(<keep dir>)`로
  `(model, metadata)`를 불러 `omega.inference.model.WAMInference(model, metadata)`에 넣습니다 (아직 실제로 돌려보지는
  않았습니다).
- `metrics-XXXXXXXX.json`: 1000 step마다 validation loss (wandb에는 train loss만 올라감).
- wandb project `halab-omega0`, run 이름 = run 폴더 이름.

추론/배포 때는 학습과 같게: ego 이미지는 LeRobot `observation.images.left` 전체 (crop 없음), state는
`observation.state` 기준 45D, action은 정규화를 풀면 [token 64 | 손가락 openness 12 (| 카메라 2)].

## 현재 상태 (2026-11-18)

- A40 run (`halab38_omega0_76d_c50_rtc8_g256_50k`): step 27,400. 38.4초/step, GPU당 VRAM 약 38.8GB.
  `keep-10000`, `keep-20000` 있음.
- L40S run (`..._l40s`, 같은 설정과 seed): step 6,200에서 중단. 23.3초/step.
- 78D (`DIM=78`)는 코드만 준비돼 있고 돌려본 적 없습니다.
