# ART-Embodied

[OpenPipe ART](https://github.com/OpenPipe/ART)를 기반으로 Physical AI를 위한
trajectory-aware reinforcement learning을 제공하며,
[LeRobot](https://github.com/huggingface/lerobot) 워크플로를 중심으로 설계된
실험적 프레임워크입니다.

[English](README.md) · [日本語](README.ja.md) ·
[简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) ·
[Embodied RL 가이드](docs/experimental/embodied-rl.mdx) ·
[예제](examples/embodied/README.md) ·
[릴리스 검증](docs/experimental/embodied-release-validation.mdx) ·
[ART 공식 문서](https://art.openpipe.ai)

<p align="center">
  <img src="docs/assets/art-embodied-trajectory-rl-dashboard.gif" alt="PI0.5 성공률 곡선과 네 개의 LIBERO Long 로봇 rollout을 보여 주는 W&B 대시보드" width="920">
</p>
<p align="center">
  <strong>PI0.5 / LIBERO Long을 trajectory-level Flow-SDE GRPO로 학습합니다.</strong><br>
  성공률 곡선과 grouped rollout을 하나의 live experiment 화면에서 추적합니다.
</p>

> [!WARNING]
> **ART-Embodied는 리서치 프리뷰 단계입니다.**
>
> - **OpenVLA-OFT / GRPO:** LIBERO에서 롤아웃, 분산 LoRA 학습, 체크포인트 복원, 평가를 검증했습니다.
> - **OpenVLA-OFT / GSPO:** 분산 실행과 체크포인트를 검증했습니다. 목적함수 검증과 학습 결과 비교는 진행 중입니다.
> - **PI0, PI0.5, SmolVLA / Flow-SDE GRPO:** 고정 개발 세트에서 성공률이 향상되었습니다. 여러 시드와 sealed test 검증은 남아 있습니다.
> - **GR00T N1.7 / Flow-SDE GRPO:** RoboCasa 단일 태스크에서 학습하고 192개 에피소드로 sealed test를 진행했습니다.
> - **PI0-FAST / GRPO:** LIBERO Long 단일 태스크에서 학습하고 개발 평가와 sealed test를 각각 100개 에피소드로 진행했습니다. 결과의 불확실성과 검증 범위는 아래를 참고하세요.

## 무엇을 추가하는가

LeRobot은 정책, 전처리·후처리, 데이터셋, 로봇 환경을 제공합니다.
ART-Embodied는 궤적 그룹 수집, GRPO/GSPO 학습, 체크포인트 관리,
평가와 W&B·Weave 기록 기능을 추가합니다.

```text
LeRobot policy + processors + environment
                   │
                   ▼
       grouped trajectories and rewards
                   │
                   ▼
      trajectory/action-token GRPO or GSPO
                   │
                   ▼
       versioned LoRA checkpoints
                   │
                   ▼
 fixed evaluation + W&B Models + Weave
```

학습 성과는 정책의 원래 실행 환경에서 측정한 태스크 성공률로 평가합니다.

## ART lifecycle과 LeRobot의 소유 범위

LeRobot은 정책, 전처리·후처리, 환경, 액션 샘플링을 담당합니다.
ART-Embodied는 ART의 모델 관리 API를 사용해 궤적 그룹, 학습, 업데이트,
체크포인트, 평가, 로그를 관리합니다.

| 기능 | API |
| --- | --- |
| 모델 관리 | `art.TrainableModel`을 상속하는 `EmbodiedTrainableModel` |
| 학습 | `backend.train(model, trajectory_groups, learning_rate=...)` |
| 궤적 그룹 수집 | `trajectory_group(...)`, `gather_trajectory_groups(...)` |
| 결과와 업데이트 | `TrainResult`, `get_step()` |
| 로깅 | 모델별 W&B 지표·영상 및 Weave 트레이스 |
| 정책 실행 | LeRobot 전처리·후처리 및 액션 샘플러 |

전용 백엔드가 이미지, 로봇 상태, 액션 청크, 샘플러별 우도를 처리합니다.
OpenVLA 백엔드는 ART의 `LocalBackend`, AOM, Serverless Training과
독립적으로 실행됩니다.

학습 루프 전체를 실행하려면 LeRobot 스타일 고수준 API를, 궤적 수집과 업데이트를
직접 제어하려면 ART 스타일 저수준 API를 사용하세요. 두 API는 동일한 모델 등록
로직과 학습 백엔드를 사용합니다.

## 검증된 결과

동일한 초기 상태에서 SFT와 학습 후 정책의 성공 횟수를 비교했습니다.

| Policy / suite | Objective | Update | SFT | ART-Embodied | Paired lift |
| --- | --- | ---: | ---: | ---: | ---: |
| [OpenVLA-OFT / LIBERO Object](https://wandb.ai/wandb-japan/art-embodied-openvla) | Action-token GRPO | 200 | 34/100 | **100/100** | **+66 point** |
| [OpenVLA-OFT / LIBERO Spatial](https://wandb.ai/wandb-japan/art-embodied-openvla-spatial) | Action-token GRPO | 100 | 48/100 | **88/100** | **+40 point** |
| [PI0 / LIBERO Spatial](https://wandb.ai/wandb-japan/art-embodied-pi0-positive-control-reference/runs/941byojx) | Flow-SDE GRPO | 100 | 63/100 | **99/100** | **+36 point** |
| [PI0.5 / LIBERO Long](https://wandb.ai/wandb-japan/art-embodied-pi05-positive-control-reference/runs/t0a9mnd3) | Flow-SDE GRPO | 250 (best dev) | 48/100 | **84/100** | **+36 point** |
| [SmolVLA / LIBERO Long](https://wandb.ai/wandb-japan/art-embodied-smolvla-positive-control-v3/runs/s4xwc2jm) | Flow-SDE GRPO | 180 (best dev) | 42/100 | **69/100** | **+27 point** |
| GR00T N1.7 / RoboCasa Cuttingboard-to-Pan ([개발 run](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task/runs/1970sjop), [sealed test](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/kjpzsj11)) | Flow-SDE GRPO | 100 (sealed) | 111/192 | **139/192** | **+14.6 point** |
| [PI0-FAST / LIBERO Long (단일 태스크)](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/a1hx8rsf) | Action-token GRPO | 100 (개발 평가) | 70/100 | **89/100** | **+19 %p** |
| [PI0-FAST / LIBERO Long (단일 태스크)](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/ww9b5upu) | Action-token GRPO | 100 (sealed) | [73/100](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/r56aa76y) | **83/100** | **+10 %p** |

**평가 조건.** 처음 다섯 행은 학습에 사용하지 않은 초기 상태 100개를 고정한
개발 평가입니다. 태스크와 지시문은 학습 때와 같습니다. 이 세트는 개발 중 반복해서
사용했으며, 여러 시드와 sealed test 검증은 남아 있습니다. PI0.5와 SmolVLA는
개발 평가가 가장 높았던 체크포인트의 결과이며, 최종 점수는 아래에 기재했습니다.

GR00T는 체크포인트를 선택한 뒤 개발에 사용하지 않은 환경 시드 192개로 sealed test를
진행했습니다. RoboCasa의 한 태스크를 평가했으며, 멀티태스크와 새로운 태스크에서의
평가는 아직 진행하지 않았습니다.

PI0-FAST는 LIBERO Long의 "put both moka pots on the stove" 태스크를 대상으로,
학습 시드 하나와 격리된 MuJoCo 3.3 환경을 사용했습니다. 표의 두 행 모두 sealed test
전에 선택한 최종 업데이트 100 체크포인트의 결과입니다. Sealed 향상 폭 +10%p의
대응표본 95% 신뢰구간은 [0, 20]%p(p=0.099)로, 5% 유의수준에서 통계적 유의성은
확인되지 않았습니다. 멀티태스크와 강건성 평가는 남아 있습니다.
[레시피와 상세 결과](docs/experimental/pi0-fast-long-result.md)를 참고하세요.

<details>
<summary>학습 조건과 대응표본 평가 상세</summary>

- **OpenVLA-OFT Object:** rank 32/alpha 32 LoRA로 200회 업데이트했습니다. 평가용 초기 상태 100개는 학습용 500개와 독립적으로 생성했습니다. 최종 점수 100/100은 공개 RLinf GRPO 체크포인트와 같았습니다.
- **OpenVLA-OFT Spatial:** 같은 백엔드를 사용하되 SFT 체크포인트, 태스크 모음, 평가 세트, W&B 프로젝트를 분리했습니다. 업데이트 30에서 82/100, 업데이트 100에서 88/100을 기록했습니다. 비교 기준은 SFT입니다.
- **PI0 / PI0.5:** K4/noise 0.5 Flow-SDE 샘플러와 업데이트당 1,024개 궤적을 사용했습니다. PI0는 업데이트 130까지 97--99/100을 유지했습니다. PI0.5는 rank 32/alpha 32 LoRA로 300회 업데이트했으며, 업데이트 250의 최고 점수 84/100에 비해 최종 점수는 81/100이었습니다.
- **SmolVLA:** 200회 업데이트했으며, 업데이트 100 이후 액션 전문가의 어댑터 적용 범위와 rank를 확장했습니다. 업데이트 180에서 69/100으로 최고점을 기록한 뒤 최종 업데이트 200에서는 61/100으로 하락했습니다.
- **GR00T N1.7:** RoboCasa GR1 `PnPCounterToCab`의 Cuttingboard-to-Pan 태스크입니다. NVIDIA 레시피로 60k 스텝 학습한 SFT 체크포인트에서 rank 64/alpha 64 LoRA로 100회 연속 업데이트했습니다. 고정 개발 에피소드 64개로 체크포인트를 선택하고 독립 감사를 거쳐 sealed 평가를 한 번 진행했습니다. [SFT 111/192](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/j11avofl) 대비 [GRPO 139/192](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/kjpzsj11)였습니다. 학습에는 Flow-SDE, 평가에는 공식 ODE 샘플러를 사용하며 공식 프로세서, 로봇 구성, 정규화를 유지합니다.

위 W&B 링크에서 지표, 영상, 모델 아티팩트, 궤적 트레이스를 확인할 수 있습니다.
재현 레시피와 초기 상태 명세는 `examples/embodied/`에 있습니다.

| 비교 | 개선 / 악화된 에피소드 수 | 대응표본 95% 신뢰구간 (%p) | 정확 McNemar 검정 p |
| --- | ---: | ---: | ---: |
| OpenVLA-OFT Object | 66 / 0 | | `2.71e-20` |
| OpenVLA-OFT Spatial | 43 / 3 | | `4.62e-10` |
| PI0 | 36 / 0 | `[+27,+46]` | `2.91e-11` |
| PI0.5 | 38 / 2 | `[+26,+46]` | `1.49e-9` |
| SmolVLA | 33 / 6 | `[+16,+38]` | `1.43e-5` |
| GR00T N1.7 | 순 개선 28건 | `[+5.2,+24.0]` | `0.00335` |

</details>

## 현재 지원 현황

| 기능 | 상태 |
| --- | --- |
| LeRobot / Gymnasium 롤아웃 어댑터 | 구현 완료 |
| 고정 초기 상태의 대응표본 평가 | 구현 완료 |
| 단일 GPU·로컬 다중 GPU LoRA 학습 | 구현 완료 |
| 여러 actor가 공유하는 정책의 배치 추론 | 구현 완료 |
| optimizer·RNG 상태를 포함한 체크포인트와 재개 | 구현 완료 |
| W&B 지표·영상·Table·모델 아티팩트 | 구현 완료 |
| Weave 궤적 트레이스와 영상 링크 | 구현 완료 |

### 1 GPU도 지원되는 resource profile

GPU 한 개로 실행하려면 두 장치 목록에 같은 GPU를 지정하고 `per_update`를 사용하세요.
평가, 궤적 수집, 학습을 순서대로 실행하며 학습 전에 롤아웃 모델을 해제합니다.

```yaml
runtime:
  rollout_devices: [cuda:0]
  training_devices: [cuda:0]
  distributed_training: false
  rollout_execution:
    lifecycle: per_update
```

`per_update`는 단계마다 워커를 종료해 GPU와 호스트 메모리 사용량을 줄입니다.
`cpu_offload`는 워커를 호스트 RAM에 유지해 시작 시간을 줄입니다.
[PI0.5 단일 GPU 예제](examples/embodied/pi05_libero_object_flow_sde_grpo_single_gpu.yaml)를 참고하세요.

## 설치

> **LIBERO 초기 상태 검사:** 의존성 검사를 통과해도 의도한 초기 배치가 보장되지는 않습니다.
> MuJoCo 업데이트로 reset 중 물체 배치가 달라질 수 있습니다(Spatial task 5에서 보고됨).
> 레시피의 시뮬레이터 버전과 reset 설정을 함께 재현하고, 모델이 필요 없는 [검사 및 재현 가이드](docs/experimental/libero-reset-health.md)를 참고하세요.
> 검사는 현재 명시적으로 실행해야 하며, 기존 레시피를 변경하거나 초기 상태를 자동 보정하지 않습니다.

### 검증된 호환성

ART-Embodied는 OpenPipe ART와 함께 `art_embodied` 패키지를 설치합니다.
`0.1.0rc2`는 실행 환경별로 ART 0.5.18과 0.5.20을 지원합니다.

| 실행 환경 | OpenPipe ART | 용도 |
| --- | --- | --- |
| Python 3.12 이상 기본 환경 | `0.5.20` | PI0 / PI0.5, PI0-FAST, SmolVLA 구성 |
| Python 3.11 | `0.5.18` | 기존 OpenVLA-OFT 구성(LeRobot `>=0.4.4,<0.5`) |
| GR00T N1.7 전용 Python 3.12 환경 | `0.5.18` | NVIDIA가 의존성을 고정한 네이티브 환경 |

아래 설치 명령으로 ART 버전을 지정합니다. NVIDIA는 SciPy 1.15.3을,
ART 0.5.20은 SciPy 1.17을 요구하므로 GR00T 설치 스크립트는 ART 0.5.18을
유지합니다. 기존 ART 0.5.18 환경도 계속 지원합니다. ART 0.5.20의 패키지,
수치 연산, 업데이트 및 재개, W&B 검증 범위는
[호환성 검증 보고서](docs/experimental/upstream-art-compatibility.md)를 참고하세요.
표준 환경에는 `constraints/security.txt`를 적용하며 NVIDIA 전용 환경의 버전은 별도로 관리합니다.
설치 방법별 차이와 주의 사항은 [의존성 구성과 보안](docs/experimental/dependency-security.md)을 참고하세요.

ART는 `import art`, 애드온은 `import art_embodied as embodied`로 불러옵니다.

ART 0.5.18과 ART-Embodied는 Python 3.11을 지원합니다. 검증된 OpenVLA-OFT
구성에서는 ART와 정책을 하나의 격리 환경에서 실행할 수 있습니다.

격리된 환경을 사용하십시오. Robotics policy stack은 ART의 LLM backend와
다른 Torch, Transformers, simulator version을 고정하는 경우가 많습니다.

깨끗한 환경에서 add-on 설치 순서를 재현하려면 다음을 실행합니다.

```bash
git clone https://github.com/wandb/ART-Embodied.git
cd ART-Embodied
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/security.txt 'openpipe-art==0.5.18'
python -m pip install -c constraints/security.txt '.[libero]'
art-embodied doctor --profile libero
```

새 Python 3.11 환경에서 이 설치 절차를 검증했습니다. 설치된 패키지로
OpenVLA-OFT를 로드하고 LoRA GRPO 업데이트 1회, 어댑터·학습 상태 저장,
체크포인트 복원까지 확인했습니다.

Checkout에서 uv 기반 개발을 하려면 다음을 사용합니다.

uv 0.12.0 이상을 사용하세요. `uv sync --locked`는 LiteLLM 1.101.0과
Diffusers 0.38.0을 사용하도록 의존성 조건을 재정의합니다. GR00T N1.7 설치 스크립트는
Diffusers에만 이 설정을 적용하고 Safetensors 0.8.0을 사용합니다. 상위 패키지를 수정하지 않고
기존 버전 조건을 대체합니다. 일반 pip 설치에는 이 설정이 적용되지 않습니다.

```bash
git clone https://github.com/wandb/ART-Embodied.git
cd ART-Embodied
uv sync --python 3.11 --extra lerobot
```

검증된 LIBERO integration:

```bash
uv sync --python 3.11 --extra libero
```

PI0/PI0.5 Flow-SDE에는 LeRobot 0.6과 Transformers 5를 사용하는
별도의 Python 3.12 환경을 준비하세요:

```bash
python3.12 -m venv .venv-pi
source .venv-pi/bin/activate
python -m pip install -c constraints/security.txt 'openpipe-art==0.5.20' '.[pi-libero]'
art-embodied doctor --profile pi
```

용도별로 환경을 나누어 설치하세요.

- `lerobot`: 모델별·시뮬레이터 추가 의존성을 제외한 LeRobot.
- `libero`: 검증된 OpenVLA-OFT/LIBERO 구성. Torch 2.6.0, Transformers 4.40.1, PEFT 0.11.1, NumPy 1.26.4를 사용합니다.
- `pi` / `pi-libero`: LeRobot 0.6과 PI 샘플러.

이 구성을 같은 환경에 섞어 설치하지 마세요. `libero`를 `lerobot[libero,peft]`로
대체하면 LeRobot 0.4.4에서 Transformers와 PEFT 버전이 바뀌어 체크포인트가
정상적으로 로드되더라도 액션 로짓이 달라질 수 있습니다.

GPU를 할당하기 전에 설치된 control-plane stack을 확인하십시오.

```bash
uv run art-embodied doctor

# Generic LeRobot worker profile에서 사용합니다.
uv run art-embodied doctor --require-lerobot

# OpenVLA-OFT/LIBERO profile에서 GPU를 할당하기 전에 사용합니다.
uv run art-embodied doctor --profile libero
```

Process-isolated policy 환경에서는 `--worker`를 사용하십시오. 이는 worker
process에 ART를 요구하지 않고 선택된 package profile을 검증합니다.
`--require-lerobot`은 generic LeRobot profile에만 추가하십시오. CI나 launch
script에서 같은 report를 programmatically 사용하려면 `--json`을 추가합니다.

OpenVLA-OFT v0.1에는 전용 `libero` 환경이나 동일한 구성의 컨테이너가 필요합니다.
시작 시 추론 결과에 영향을 주는 의존성 버전 차이를 검사하고 중단합니다.

Native policy dependency가 여전히 process isolation을 요구한다면 두 환경 모두에
`art-embodied` wheel을 설치하고 policy 환경을 명시적으로 선택합니다.

```yaml
runtime:
  # Rollout actor, batched inference server, training worker가 사용합니다.
  worker_python_executable: /opt/venv/openvla/bin/python
```

`null`이면 현재 Python을 사용합니다. 별도 워커 환경에는 `art-embodied`와
정책·시뮬레이터 의존성이 필요합니다. 워커는 셸을 거치지 않고 실행되며,
ART를 초기화하지 않고 정책 의존성을 로드합니다.

GR00T N1.7과 RoboCasa는 policy와 simulator를 각각 고정된 Python 3.12 환경으로
분리합니다.

GR00T N1.7 설치에는 Git LFS, micromamba, CMake, C++ 빌드 도구가 필요합니다.

```bash
./scripts/install-gr00t-n1d7-runtime.sh
./scripts/install-robocasa-gr1-runtime.sh
./scripts/download-robocasa-gr1-dataset.sh
```

설치 스크립트는 NVIDIA Isaac-GR00T와 CUDA/Torch를 RoboCasa·robosuite·MuJoCo와
분리해 버전을 고정합니다. 데이터셋도 지정한 리비전으로 내려받고 학습 스크립트용
검증 보고서를 생성합니다. 개발·학습 재개·sealed 평가 절차는
[GR00T 예제](examples/embodied/README.md#gr00t-n17--robocasa-flow-sde)를 참고하세요.

### 이식 가능한 Slurm 실행

Slurm에서는 저장소 래퍼가 체크아웃 경로를 찾고 `.env`를 읽습니다.
클러스터별 경로와 계정 설정은 재사용할 작업 파일 밖에서 관리하세요:

```bash
sbatch --gres=gpu:h100:8 --cpus-per-task=96 --mem=690G \
  scripts/slurm/run-in-repo.sh \
  uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml
```

Secret이 checkout 밖에 있을 때만 submit 시점에 `ART_EMBODIED_ENV_FILE`을
설정합니다.

```bash
ART_EMBODIED_ENV_FILE="${HOME}/.config/art-embodied/secrets.env" \
  sbatch --export=ALL scripts/slurm/run-in-repo.sh COMMAND [ARG ...]
```

Slurm은 기본적으로 submit 환경 변수를 export합니다. 따라서 다른 user, home
directory, cluster, region에서 clone하더라도 job file을 수정하지 않고 wrapper를
사용할 수 있습니다. Experiment 조건은 YAML에 유지하고, 환경 변수 override는
secret과 machine-local 위치에만 사용합니다.

## OpenVLA-OFT control 실행

Experiment 조건은 YAML에 저장합니다. 환경 변수는 `WANDB_API_KEY` 같은
secret에만 사용합니다.

```bash
uv run art-embodied validate \
  examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml

uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml \
  --preflight

uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml
```

`validate`와 `--preflight`는 모델 로드 전에 궤적 형태, optimizer 행, 장치,
평가 시나리오, 시뮬레이터 에셋을 확인합니다. 새로운 연동은
[`lerobot_action_token_grpo.template.yaml`](examples/embodied/lerobot_action_token_grpo.template.yaml)에서 시작하세요.

학습 전에 학습용 초기 상태와 분리된 고정 평가 명세를 만들고 SFT를 한 번 평가하세요.
정기 평가에도 같은 명세와 SFT 결과를 사용합니다. 레시피나 체크포인트를 선택하는 동안은
`evaluation.data_role: development`를 지정하세요.

방법과 체크포인트 선택 규칙을 정한 뒤, 새 명세를 `data_role: sealed_test`,
`checkpoint_selection: last`로 평가합니다. Sealed 결과에서 `best`를 선택하는
설정은 거부됩니다. 명세 준비 절차는 예제 가이드를 참고하세요.

고정 평가마다 원시 결과와 재현 정보를 저장합니다. 재현 정보에는 결과·명세·소스 해시,
Git 상태, 의존성 및 실행 환경·컨테이너 버전, 평가한 모델·어댑터 또는 체크포인트 식별자가
포함됩니다. W&B가 활성화된 경우 `log_evaluation_artifacts: true`로 두 파일을
버전별 평가 아티팩트로 업로드합니다. 같은 파일은 로컬에도 저장됩니다.

Optimizer를 만들지 않고 고정 SFT baseline을 평가하려면:

```bash
uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_object_sft_baseline_eval.yaml \
  --evaluate-only --evaluation-step 0
```

Optimizer를 resume하지 않고 저장된 ART-Embodied policy snapshot을 평가하려면:

```bash
uv run python examples/embodied/libero/train_openvla_oft.py \
  --config experiment-eval.yaml --evaluate-only --evaluation-step 5 \
  --policy-checkpoint outputs/my-run/checkpoints/step-000005/policy
```

평가 YAML의 정책 종류, 기본 모델 리비전, 전처리·후처리, 환경, 고정 시나리오는
체크포인트와 일치해야 합니다.

Candidate preparation, paired evaluation, generated state manifest,
conformance tool은 [`examples/embodied/README.md`](examples/embodied/README.md)을
참조하십시오.

## 기존 LeRobot workflow에 연결

애플리케이션의 LeRobot 정책, 전처리·후처리, 환경을 `run_lerobot_experiment`에
전달하세요. 모델 등록, 궤적 그룹 수집, 학습, 로그, 체크포인트, 종료 처리를 관리합니다.
액션 토큰 정책의 어댑터는 샘플링한 토큰과 롤아웃 시점의 로그 확률을 기록합니다.

```python
import asyncio

import art_embodied as embodied
from my_robot_app import (
    evaluation_scenarios,
    make_environment,
    policy,
    postprocessor,
    preprocessor,
    record_action_tokens,
    train_scenarios,
)


async def main() -> None:
    config = embodied.EmbodiedExperimentConfig.from_yaml("experiment.yaml")
    adapter = embodied.LeRobotPolicyAdapter(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        device=config.policy.device,
        record_action_fn=record_action_tokens,
    )
    result = await embodied.run_lerobot_experiment(
        config=config,
        policy=policy,
        train_scenarios=train_scenarios,
        evaluation_scenarios=evaluation_scenarios,
        environment_factory=make_environment,
        policy_adapter=adapter,
    )
    print(result.config_fingerprint)


asyncio.run(main())
```

궤적 그룹 수집과 업데이트를 직접 제어하려면 저수준 API를 사용하세요:

```python
import art_embodied as embodied

config = embodied.EmbodiedExperimentConfig.from_yaml("experiment.yaml")
model = embodied.EmbodiedTrainableModel(policy=policy, config=config)
native_backend = embodied.make_embodied_backend(config, policy=policy)
backend = embodied.EmbodiedBackend(native_backend, config=config)
await model.register(backend)

groups = await embodied.gather_trajectory_groups(
    [
        embodied.trajectory_group(
            (rollout(model.policy, scenario) for _ in range(config.algorithm.group_size)),
            metadata={"scenario_id": scenario.id},
        )
        for scenario in train_scenarios
    ]
)
result = await backend.train(
    model,
    groups,
    learning_rate=config.training.optimizer.learning_rate,
)
await model.log(groups, split="train", metrics=result.metrics, step=result.step)
await model.close()
```

`EmbodiedTrainableModel`은 `art.TrainableModel`을 상속합니다. 각 정책 백엔드는
해당 샘플러에 맞는 목적함수로 학습합니다.

## Experiment 조건을 바꾸지 않고 scale하기

Slurm은 선택 사항입니다. 장치, actor 수, 추론 복제본 수, 워커 수명, 마이크로배치를
YAML에서 조정해 워크스테이션과 클러스터에서 같은 실험을 실행할 수 있습니다.

1. 롤아웃 GPU당 actor 하나와 모델 하나로 업데이트 1회를 확인합니다.
2. actor를 늘려 시뮬레이션과 추론을 병렬로 진행합니다.
3. `batched_server`로 여러 actor가 정책을 공유하게 합니다.
4. GPU 메모리와 측정한 처리량에 따라 추론 복제본을 추가합니다.
5. 로컬 분산 학습으로 optimizer 시간을 줄입니다.
6. 롤아웃과 학습이 GPU를 공유하면 `cpu_offload`를 사용하고 모든 복제본을 수용할 호스트 RAM을 확보합니다.

80 GB H100에서 OpenVLA-OFT를 GPU당 추론 복제본 3개, actor 6개, 배치 크기 2,
학습 마이크로배치 12로 측정했습니다. 1,024개 궤적을 `1.536 trajectories/s`로
수집해 초기 구현의 두 배 이상 처리량을 기록했습니다. 정책과 하드웨어에 맞게 조정하고
GPU 메모리에 10–20% 여유를 두세요.

## W&B와 Weave

W&B는 SFT 평가를 Step 0에 기록하고 이후 정기 평가를 같은
`validation/success_rate` 곡선에 추가합니다. 학습 업데이트마다 이력을 한 행씩
기록하고 영상, 평가 Table, 버전별 모델·학습 상태 아티팩트를 저장합니다.
Weave는 업데이트, 그룹, 궤적 순서로 트레이스를 구성하고 영상을 연결합니다.

| 섹션 | 내용 |
| --- | --- |
| `train/*` | `train/success_rate`, `train/reward_mean` |
| `validation/*` | 집계된 평가 지표 |
| `signal/*` | 보상, advantage, 유효 그룹 |
| `optimization/*` | 손실, KL, 우도비, 기울기 |
| `performance/*` | 시간, 처리량, 메모리 |
| `train_details/*` | 에피소드 수와 길이 |
| `media/simulation/*` | 학습·평가 영상 |

태스크별·에피소드별 결과는 Table과 평가 아티팩트에 저장합니다.
W&B 지원에는 `observability` extra를 설치하세요. Weave 클라이언트는
ART 0.5.18에 포함됩니다. 두 연동 모두 YAML에서 활성화한 경우에만 데이터를 전송합니다.

```yaml
observability:
  delivery_failure_policy: fail_run
  wandb:
    enabled: true
    project: art-embodied
    mode: online
    log_model_artifacts: true
    log_evaluation_table: true
  weave:
    enabled: true
    project: art-embodied
    trace_trajectories: true
    max_groups_per_update: 4
    max_trajectories_per_group: 4
  videos_per_update: 2
  videos_per_evaluation: 8
```

두 연동을 비활성화해도 체크포인트, 영상, JSON 평가 결과는 로컬에 저장됩니다.

체크포인트에는 정책, optimizer, 난수 생성기(RNG) 상태와 각 해시를 저장합니다.
완료 마커와 원자적 게시를 통해 저장을 마친 체크포인트부터 사용할 수 있습니다.
재개 시 파일 무결성과 학습 설정의 호환성을 확인합니다.

예제의 `delivery_failure_policy: fail_run` 설정은 W&B·Weave 전송 오류를
크기가 제한된 `telemetry_failures.jsonl`에 기록하고 실행을 중단합니다.
전송 실패 시에도 로컬 기록으로 계속하려면 `best_effort`를 사용하세요.
두 모드 모두 완료된 업데이트와 체크포인트를 보존합니다.

Step 0 평가가 있는 실험을 재개할 때는 결과 파일을
`evaluation.baseline_outcomes_path`에 지정하고 `evaluate_before_training`을
비활성화하세요. 시작 전 검사에서 이 참조를 확인해 대응표본 비교, 신뢰구간,
McNemar 검정을 이어갈 수 있도록 합니다.

## 경계

- GRPO 업데이트에는 완전한 궤적 그룹이 필요합니다.
- 학습 진단 지표와 정책의 기본 평가 결과는 따로 기록합니다.
- 영상과 트레이스 수는 YAML로 설정합니다.
- 환경별 동작은 시뮬레이터 어댑터가 처리합니다.

## 문서

- [Embodied RL 개념과 설정](docs/experimental/embodied-rl.mdx)
- [Runtime 및 scaling architecture](docs/experimental/embodied-runtime-architecture.mdx)
- [Release validation contract](docs/experimental/embodied-release-validation.mdx)
- [OpenVLA-OFT 및 LIBERO 예제](examples/embodied/README.md)

## 감사의 말

구현, 레시피, 체크포인트를 공개한 [RLinf](https://github.com/RLinf/RLinf)
개발자들에게 감사드립니다. OpenVLA-OFT 롤아웃, 액션 토큰 마스킹,
advantage·손실 집계, LIBERO 평가를 검증할 때 참고했습니다.

`rlinf_v01`은 이 참조 조건에 해당하는 검증 프로파일 이름입니다.
ART-Embodied는 자체 실행 환경(`model_loader: native`)을 사용하며 RLinf에
런타임 의존성이 없습니다.

## ART와의 관계

ART-Embodied는 OpenPipe ART의 애드온이며 로봇 관련 의존성은 선택적으로 설치합니다.
재현 가능한 벤치마크, 정책·시뮬레이터 어댑터, W&B·Weave 연동에 대한 기여를
환영합니다. [CONTRIBUTING.md](CONTRIBUTING.md)를 참고하세요.
