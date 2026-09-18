# ART-Embodied

[OpenPipe ART](https://github.com/OpenPipe/ART)をPhysical AI向けに拡張し、
[LeRobot](https://github.com/huggingface/lerobot)のワークフローで
trajectory-aware RLを行うための実験的フレームワークです。

[English](README.md) · [한국어](README.ko.md) ·
[简体中文](README.zh-CN.md) · [繁體中文](README.zh-TW.md) ·
[Embodied RLガイド](docs/experimental/embodied-rl.mdx) ·
[実行例](examples/embodied/README.md) ·
[リリース検証](docs/experimental/embodied-release-validation.mdx) ·
[ART公式ドキュメント](https://art.openpipe.ai)

<p align="center">
  <img src="docs/assets/art-embodied-trajectory-rl-dashboard.gif" alt="PI0.5の成功率カーブと4本のLIBERO Longロボットrolloutを表示するW&Bダッシュボード" width="920">
</p>
<p align="center">
  <strong>PI0.5 / LIBERO Longをtrajectory-level Flow-SDE GRPOで学習。</strong><br>
  成功率カーブとgrouped rolloutを、1つのlive実験画面で追跡できます。
</p>

> [!WARNING]
> **ART-Embodiedはリサーチプレビュー版です。**
>
> - **OpenVLA-OFT / GRPO:** LIBEROでロールアウト、分散LoRA学習、チェックポイント復元、評価を検証済み。
> - **OpenVLA-OFT / GSPO:** 分散実行とチェックポイントを検証済み。目的関数の検証と学習結果の比較は進行中です。
> - **PI0・PI0.5・SmolVLA / Flow-SDE GRPO:** 固定開発セットで成功率が向上。複数シードとsealed testでの検証は未完了です。
> - **GR00T N1.7 / Flow-SDE GRPO:** RoboCasaの単一タスクで学習し、192エピソードのsealed testを実施済み。
> - **PI0-FAST / GRPO:** LIBERO Longの単一タスクで学習し、開発評価・sealed testを各100エピソードで実施済み。結果の解釈と検証範囲は下記を参照してください。

## 何を追加するのか

LeRobotはポリシー、前処理・後処理、データセット、ロボット環境を提供します。
ART-Embodiedは軌道のグループ収集、GRPO/GSPO学習、チェックポイント管理、
評価とW&B・Weaveへの記録を追加します。

```text
LeRobot policy + processor + environment
                   │
                   ▼
       grouped trajectories and rewards
                   │
                   ▼
      trajectory/action-token GRPOまたはGSPO
                   │
                   ▼
       versioned LoRA checkpoints
                   │
                   ▼
 fixed evaluation + W&B Models + Weave
```

学習の成果は、ポリシー本来の実行環境でのタスク成功率で評価します。

## ART lifecycleとLeRobotの所有範囲

ポリシー、前処理・後処理、環境、アクションのサンプリングはLeRobotが担当します。
ART-EmbodiedはARTのモデル管理APIを使い、軌道のグループ化、学習、更新ステップ、
チェックポイント、評価、ログを管理します。

| 機能 | API |
| --- | --- |
| モデル管理 | `art.TrainableModel`を継承した`EmbodiedTrainableModel` |
| 学習 | `backend.train(model, trajectory_groups, learning_rate=...)` |
| 軌道のグループ収集 | `trajectory_group(...)`と`gather_trajectory_groups(...)` |
| 結果と更新ステップ | `TrainResult`と`get_step()` |
| ログ | モデルごとのW&B指標・動画とWeaveトレース |
| ポリシー実行 | LeRobotの前処理・後処理とサンプラー |

画像、ロボット状態、アクションチャンク、サンプラー固有の尤度は専用バックエンドで
扱います。OpenVLAバックエンドはARTの`LocalBackend`、AOM、Serverless Trainingから
独立して動作します。

学習ループをまとめて実行するLeRobot形式の高水準APIと、軌道収集や更新を個別に
制御するART形式の低水準APIを用意しています。どちらも同じモデル登録処理と
学習バックエンドを使います。

## 検証済みの結果

同じ初期状態で、SFTと学習後の成功数を比較しています。

| Policy / suite | Objective | Update | SFT | ART-Embodied | Paired lift |
| --- | --- | ---: | ---: | ---: | ---: |
| [OpenVLA-OFT / LIBERO Object](https://wandb.ai/wandb-japan/art-embodied-openvla) | Action-token GRPO | 200 | 34/100 | **100/100** | **+66 point** |
| [OpenVLA-OFT / LIBERO Spatial](https://wandb.ai/wandb-japan/art-embodied-openvla-spatial) | Action-token GRPO | 100 | 48/100 | **88/100** | **+40 point** |
| [PI0 / LIBERO Spatial](https://wandb.ai/wandb-japan/art-embodied-pi0-positive-control-reference/runs/941byojx) | Flow-SDE GRPO | 100 | 63/100 | **99/100** | **+36 point** |
| [PI0.5 / LIBERO Long](https://wandb.ai/wandb-japan/art-embodied-pi05-positive-control-reference/runs/t0a9mnd3) | Flow-SDE GRPO | 250（best dev） | 48/100 | **84/100** | **+36 point** |
| [SmolVLA / LIBERO Long](https://wandb.ai/wandb-japan/art-embodied-smolvla-positive-control-v3/runs/s4xwc2jm) | Flow-SDE GRPO | 180（best dev） | 42/100 | **69/100** | **+27 point** |
| GR00T N1.7 / RoboCasa Cuttingboard-to-Pan（[開発run](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task/runs/1970sjop)、[sealed test](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/kjpzsj11)） | Flow-SDE GRPO | 100（sealed） | 111/192 | **139/192** | **+14.6 point** |
| [PI0-FAST / LIBERO Long (単一タスク)](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/a1hx8rsf) | Action-token GRPO | 100 (開発評価) | 70/100 | **89/100** | **+19 ポイント** |
| [PI0-FAST / LIBERO Long (単一タスク)](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/ww9b5upu) | Action-token GRPO | 100 (sealed) | [73/100](https://wandb.ai/wandb-japan/art-embodied-pi0-fast-long/runs/r56aa76y) | **83/100** | **+10 ポイント** |

**評価条件。** 最初の5行は、学習に使わなかった初期状態100件を固定した開発評価です。
タスクと指示文は学習時と共通です。このセットは開発中に繰り返し使用しており、
複数シードとsealed testでの検証は未完了です。PI0.5とSmolVLAは開発評価が最も
高かったチェックポイントの値で、最終スコアは下記に記載しています。

GR00Tはチェックポイント選定後、開発に使っていない192シードでsealed testを
行いました。対象はRoboCasaの1タスクで、マルチタスクや未知タスクでの評価は未実施です。

PI0-FASTはLIBERO Longの「2つのモカポットをコンロに置く」が対象です。
学習シードは1つ、MuJoCo 3.3の隔離環境を使用しています。表の2行はともに、
sealed test前に選んだ最終更新100のチェックポイントを評価した結果です。
sealedの改善幅+10ポイントの対応付き95%信頼区間は[0, 20]ポイント（p=0.099）で、
5%水準での統計的有意差は確認できていません。マルチタスクや頑健性の評価は今後の課題です。
[レシピと結果の詳細](docs/experimental/pi0-fast-long-result.md)を参照してください。

<details>
<summary>学習条件と対応付き評価の詳細</summary>

- **OpenVLA-OFT Object:** rank 32/alpha 32のLoRAで200更新。評価用の100初期状態は、学習用の500状態とは独立に生成しています。最終スコア100/100は公開RLinf GRPOチェックポイントと同じでした。
- **OpenVLA-OFT Spatial:** 同じバックエンドを使用し、SFTチェックポイント、タスク群、評価セット、W&Bプロジェクトを分けて検証。更新30で82/100、更新100で88/100でした。比較対象はSFTです。
- **PI0 / PI0.5:** K4/noise 0.5のFlow-SDEサンプラーを使用し、1更新あたり1,024軌道を収集。PI0は更新130まで97--99/100を維持しました。PI0.5はrank 32/alpha 32のLoRAで300更新し、更新250の最高値84/100に対し最終値は81/100でした。
- **SmolVLA:** 200更新。更新100後にアクションエキスパートのLoRA適用範囲とrankを拡大しました。更新180で最高値69/100となり、最終更新200では61/100に低下しました。
- **GR00T N1.7:** RoboCasa GR1の`PnPCounterToCab`、Cuttingboard-to-Panが対象です。NVIDIAのレシピで60kステップ学習したSFTから、rank 64/alpha 64のLoRAで100更新を連続実行。固定した開発用64エピソードでチェックポイントを選び、独立監査後に1回のsealed評価を実施しました。[SFT 111/192](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/j11avofl)に対し、[GRPOは139/192](https://wandb.ai/wandb-japan/art-embodied-gr00t-n1d7-robocasa-single-task-sealed/runs/kjpzsj11)でした。学習時はFlow-SDE、評価時は公式のODEサンプラーを使い、前処理・後処理、身体構成、正規化も公式設定を維持しています。

上のW&Bリンクから指標、動画、モデルアーティファクト、軌道トレースを確認できます。
再現レシピと初期状態のマニフェストは`examples/embodied/`にあります。

| 比較 | 改善 / 悪化したエピソード数 | 対応付き95%信頼区間（ポイント） | 正確McNemar検定 p |
| --- | ---: | ---: | ---: |
| OpenVLA-OFT Object | 66 / 0 | | `2.71e-20` |
| OpenVLA-OFT Spatial | 43 / 3 | | `4.62e-10` |
| PI0 | 36 / 0 | `[+27,+46]` | `2.91e-11` |
| PI0.5 | 38 / 2 | `[+26,+46]` | `1.49e-9` |
| SmolVLA | 33 / 6 | `[+16,+38]` | `1.43e-5` |
| GR00T N1.7 | 差し引き28件改善 | `[+5.2,+24.0]` | `0.00335` |

</details>

## 現在の対応状況

| 機能 | 状況 |
| --- | --- |
| LeRobot / Gymnasiumのロールアウトアダプター | 実装済み |
| 固定初期状態での対応付き評価 | 実装済み |
| 単一GPU・ローカル複数GPUでのLoRA学習 | 実装済み |
| 複数actorで共有するポリシーのバッチ推論 | 実装済み |
| optimizer・RNG状態を含むチェックポイントと再開 | 実装済み |
| W&B指標・動画・Table・モデルアーティファクト | 実装済み |
| Weaveの軌道トレースと動画リンク | 実装済み |

### 1 GPU構成を正式にサポート

1 GPUで動かす場合は、両方のデバイス一覧に同じGPUを指定し、`per_update`を使います。
評価、軌道収集、学習を順に実行し、学習前にロールアウト用モデルを解放します。

```yaml
runtime:
  rollout_devices: [cuda:0]
  training_devices: [cuda:0]
  distributed_training: false
  rollout_execution:
    lifecycle: per_update
```

`per_update`は処理段階ごとにワーカーを終了し、GPUとホストのメモリ使用量を抑えます。
`cpu_offload`はワーカーをホストRAMに保持して、起動時間を短縮します。
[PI0.5の1 GPU実行例](examples/embodied/pi05_libero_object_flow_sde_grpo_single_gpu.yaml)を参照してください。

## インストール

> **LIBEROの初期状態検査:** 依存関係の検査が通っても、想定した初期配置になる保証はありません。
> MuJoCoの更新でreset中の物体配置が変わる事例があります（Spatial task 5）。
> レシピのシミュレータ版とreset設定をセットで再現し、モデル不要の[検査・再現ガイド](docs/experimental/libero-reset-health.md)を参照してください。
> 検査は現在明示的な実行が必要です。既存レシピの変更や初期状態の自動補正は行いません。

### 検証済み互換性

ART-EmbodiedはOpenPipe ARTと併せて`art_embodied`パッケージをインストールします。
`0.1.0rc2`は、実行環境ごとにART 0.5.18と0.5.20に対応します。

| 実行環境 | OpenPipe ART | 用途 |
| --- | --- | --- |
| Python 3.12以上の標準環境 | `0.5.20` | PI0 / PI0.5、PI0-FAST、SmolVLA構成 |
| Python 3.11 | `0.5.18` | 既存のOpenVLA-OFT構成（LeRobot `>=0.4.4,<0.5`） |
| GR00T N1.7専用のPython 3.12環境 | `0.5.18` | NVIDIAが依存関係を固定した実行環境 |

以下のインストール手順でARTの版を指定します。GR00Tのインストーラーは、
NVIDIAがSciPy 1.15.3を要求し、ART 0.5.20がSciPy 1.17を要求するため、
ART 0.5.18を維持します。既存のART 0.5.18環境も引き続き使用できます。
ART 0.5.20ではパッケージ、数値計算、更新・再開、W&Bの検証を実施しています。
対象範囲は[互換性検証レポート](docs/experimental/upstream-art-compatibility.md)を参照してください。
標準環境には`constraints/security.txt`を適用し、NVIDIA専用環境の固定版は別に管理します。
導入方法による違いと残る注意点は[依存構成とセキュリティ](docs/experimental/dependency-security.md)を参照してください。

ARTは`import art`、アドオンは`import art_embodied as embodied`で読み込みます。

ART 0.5.18とART-EmbodiedはPython 3.11に対応しています。検証済みの
OpenVLA-OFT構成では、ARTとポリシーを同じ隔離環境で実行できます。

専用環境を使用してください。Robot policyはARTのLLM backendと異なる
Torch、Transformers、simulatorのversionを要求することがあります。

まっさらな環境でアドオンの導入順序を再現する場合:

```bash
git clone https://github.com/wandb/ART-Embodied.git
cd ART-Embodied
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints/security.txt 'openpipe-art==0.5.18'
python -m pip install -c constraints/security.txt '.[libero]'
art-embodied doctor --profile libero
```

この手順は新規のPython 3.11環境で検証済みです。インストールしたパッケージから
OpenVLA-OFTをロードし、LoRA GRPOを1更新、アダプターと学習状態の保存、
チェックポイント復元まで確認しています。

checkout内でuvを使って開発する場合:

uv 0.12.0以降を使用してください。`uv sync --locked`はLiteLLM 1.101.0と
Diffusers 0.38.0への互換性上書きを適用します。GR00T N1.7用インストーラーは
Diffusersのみを上書きし、Safetensors 0.8.0を使います。上流パッケージは改変せず、
古い依存バージョン指定を置き換えます。通常のpipではこれらの上書きは適用されません。

```bash
git clone https://github.com/wandb/ART-Embodied.git
cd ART-Embodied
uv sync --python 3.11 --extra lerobot
```

LIBERO integrationを使用する場合:

```bash
uv sync --python 3.11 --extra libero
```

PI0/PI0.5 Flow-SDEには、LeRobot 0.6とTransformers 5を使う
独立したPython 3.12環境を用意してください。

```bash
python3.12 -m venv .venv-pi
source .venv-pi/bin/activate
python -m pip install -c constraints/security.txt 'openpipe-art==0.5.20' '.[pi-libero]'
art-embodied doctor --profile pi
```

用途ごとに環境を分けてインストールしてください。

- `lerobot`: モデル固有・シミュレータ用の追加依存を含まないLeRobot。
- `libero`: 検証済みOpenVLA-OFT/LIBERO構成。Torch 2.6.0、Transformers 4.40.1、PEFT 0.11.1、NumPy 1.26.4を使用。
- `pi` / `pi-libero`: LeRobot 0.6とPI用サンプラー。

これらの構成は同じ環境に混在させないでください。`libero`を`lerobot[libero,peft]`で
置き換えると、LeRobot 0.4.4ではTransformersとPEFTのバージョンが変わり、
チェックポイントを読み込めてもアクションのlogitsが変化することがあります。

GPUを確保する前にcontrol-plane環境の互換性を確認できます。

```bash
uv run art-embodied doctor

# generic LeRobot worker profileの場合
uv run art-embodied doctor --require-lerobot

# OpenVLA-OFT/LIBERO profileでGPUを確保する前に実行
uv run art-embodied doctor --profile libero
```

policy環境をprocess分離する場合は`--worker`を指定します。このmodeではARTを
要求せず、選択したpackage profileを検証します。`--require-lerobot`はgeneric
LeRobot profileにだけ追加します。CIやlaunch scriptから同じ結果を扱う場合は
`--json`を追加してください。

OpenVLA-OFT v0.1には専用の`libero`環境か、同じ構成のコンテナが必要です。
推論結果に影響する依存バージョンの違いは、起動時に検出して停止します。

native policy依存をprocess分離する必要がある場合は、両環境に
`art-embodied` wheelをinstallし、policy側のPythonをYAMLで明示します。

```yaml
runtime:
  # rollout actor、batched inference server、training workerで共通です。
  worker_python_executable: /opt/venv/openvla/bin/python
```

`null`では現在のPythonを使います。別のワーカー環境には`art-embodied`と
ポリシー・シミュレータの依存パッケージが必要です。ワーカーはシェルを介さず起動し、
ARTの初期化をせずにポリシー用の依存を読み込みます。

GR00T N1.7とRoboCasaでは、policyとsimulatorを別々の固定Python 3.12環境に
分離します。

GR00T N1.7のインストールにはGit LFS、micromamba、CMake、C++ビルドツールが必要です。

```bash
./scripts/install-gr00t-n1d7-runtime.sh
./scripts/install-robocasa-gr1-runtime.sh
./scripts/download-robocasa-gr1-dataset.sh
```

インストーラーはNVIDIA Isaac-GR00TとCUDA/Torchを、RoboCasa・robosuite・MuJoCoとは
別々にバージョン固定します。データセットもリビジョンを指定して取得し、学習スクリプトが
参照する検証レポートを生成します。開発・継続学習・sealed評価の手順は
[GR00Tの実行例](examples/embodied/README.md#gr00t-n17--robocasa-flow-sde)を参照してください。

### 移植可能なSlurm実行

Slurmでは、リポジトリのラッパーがチェックアウト先を解決して`.env`を読み込みます。
クラスタ固有のパスやアカウント設定は、再利用するジョブファイルの外で管理してください。

```bash
sbatch --gres=gpu:h100:8 --cpus-per-task=96 --mem=690G \
  scripts/slurm/run-in-repo.sh \
  uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml
```

secretをcheckout外へ置く場合だけ、submit時に`ART_EMBODIED_ENV_FILE`を
指定します。

```bash
ART_EMBODIED_ENV_FILE="${HOME}/.config/art-embodied/secrets.env" \
  sbatch --export=ALL scripts/slurm/run-in-repo.sh COMMAND [ARG ...]
```

Slurmは標準でsubmit元の環境変数をexportします。このため別user、別home、
別cluster、別regionへcloneしてもjob fileの編集は不要です。実験条件はYAMLに
残し、環境変数overrideはsecretとmachine-localな場所だけに限定します。

## OpenVLA-OFT controlを実行する

実験条件はYAMLをsingle source of truthとします。環境変数は
`WANDB_API_KEY`などのsecretに限定します。

```bash
uv run art-embodied validate \
  examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml

uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml \
  --preflight

uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_grpo_rlinf_positive_control.yaml
```

`validate`と`--preflight`はモデルのロード前に、軌道の形状、optimizerの行、
デバイス、評価シナリオ、シミュレータ資産を確認します。新しい連携の実装には
[`lerobot_action_token_grpo.template.yaml`](examples/embodied/lerobot_action_token_grpo.template.yaml)
を使用してください。

学習前に、学習用と分離した初期状態の評価マニフェストを作り、SFTを一度評価します。
定期評価でも同じマニフェストとSFTの結果を使ってください。レシピやチェックポイントを
選ぶ間は`evaluation.data_role: development`を指定します。

手法とチェックポイント選択規則を決めた後、新しいマニフェストを
`data_role: sealed_test`、`checkpoint_selection: last`で評価します。
sealed結果から`best`を選ぶ設定は拒否されます。作成手順は実行例ガイドを参照してください。

固定評価ごとに、生の評価結果と再現情報を保存します。再現情報には結果・マニフェスト・
ソースのハッシュ、Gitの状態、依存パッケージと実行環境・コンテナのバージョン、
評価したモデル・アダプターまたはチェックポイントの識別情報が含まれます。
W&B有効時は`log_evaluation_artifacts: true`で両ファイルを評価アーティファクトとして
アップロードします。同じファイルはローカルにも保存されます。

optimizerを構築せず固定SFT baselineを評価できます。

```bash
uv run python examples/embodied/libero/train_openvla_oft.py \
  --config examples/embodied/openvla_oft_libero_object_sft_baseline_eval.yaml \
  --evaluate-only --evaluation-step 0
```

optimizerを再開せず、保存済みART-Embodied policy snapshotを評価できます。

```bash
uv run python examples/embodied/libero/train_openvla_oft.py \
  --config experiment-eval.yaml --evaluate-only --evaluation-step 5 \
  --policy-checkpoint outputs/my-run/checkpoints/step-000005/policy
```

評価YAMLのポリシー種、ベースモデルのリビジョン、前処理・後処理、環境、
固定シナリオはチェックポイントと一致させてください。

candidate生成、paired evaluation、初期状態manifest、conformance toolは
[`examples/embodied/README.md`](examples/embodied/README.md)にまとめています。

## 既存のLeRobotワークフローへ接続する

アプリケーションで使っているLeRobotのポリシー、前処理・後処理、環境を
`run_lerobot_experiment`に渡します。モデル登録、軌道のグループ収集、学習、ログ、
チェックポイント、終了処理をまとめて実行できます。アクショントークン型のポリシーでは、
アダプターがサンプリングしたトークンとロールアウト時の対数確率を記録します。

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

軌道グループの収集や更新を個別に制御する場合は、低水準APIを使います。

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

`EmbodiedTrainableModel`は`art.TrainableModel`を継承します。各ポリシーの
バックエンドは、そのサンプラーに対応した目的関数で学習します。

## 実験条件を変えずにスケールする

Slurmは任意です。デバイス、actor数、推論レプリカ数、ワーカーの生存期間、
マイクロバッチ数をYAMLで調整し、ワークステーションでもクラスタでも同じ実験を実行できます。

1. ロールアウト用GPUごとに1 actor・1モデルで1更新を確認します。
2. actorを増やし、シミュレーションと推論を並行させます。
3. `batched_server`で複数actorからポリシーを共有します。
4. GPUメモリの余裕と実測スループットに応じて推論レプリカを増やします。
5. ローカル分散学習でoptimizerの処理時間を短縮します。
6. ロールアウトと学習でGPUを共有する場合は`cpu_offload`を使い、全レプリカ分のホストRAMを確保します。

80 GB H100では、OpenVLA-OFTでGPUあたり推論レプリカ3・actor 6、バッチサイズ2、
学習マイクロバッチ12の構成を測定しました。1,024軌道を`1.536 trajectories/s`で収集し、
初期実装の2倍以上のスループットでした。モデルとハードウェアに合わせて調整し、
GPUメモリには10–20%の余裕を残してください。

## W&BとWeave

W&BではSFTの評価をStep 0に記録し、その後の定期評価を同じ
`validation/success_rate`の曲線に追加します。学習更新ごとに履歴を1行記録し、
動画、評価Table、バージョン付きモデル・学習状態アーティファクトを保存します。
Weaveでは更新・グループ・軌道の順にトレースをまとめ、動画をリンクします。

| セクション | 内容 |
| --- | --- |
| `train/*` | `train/success_rate`、`train/reward_mean` |
| `validation/*` | 評価の集約指標 |
| `signal/*` | 報酬、advantage、有効グループ |
| `optimization/*` | 損失、KL、尤度比、勾配 |
| `performance/*` | 処理時間、スループット、メモリ |
| `train_details/*` | エピソード数と長さ |
| `media/simulation/*` | 学習・評価の動画 |

タスク別・エピソード別の結果はTableと評価アーティファクトに保存します。
W&B連携には`observability` extraをインストールしてください。Weaveクライアントは
ART 0.5.18に含まれます。どちらもYAMLで有効にした場合にデータを送信します。

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

両方の連携を無効にしても、チェックポイント、動画、JSON形式の評価結果はローカルに保存されます。

チェックポイントにはポリシー、optimizer、乱数生成器（RNG）の状態と各ハッシュを
保存します。完了マーカーを書き、アトミックに公開してから再開可能になります。
再開時はファイルの整合性と学習設定の互換性を確認します。

実行例の`delivery_failure_policy: fail_run`では、W&B・Weaveの送信エラーを
容量制限付きの`telemetry_failures.jsonl`に記録して停止します。送信が失敗しても
ローカル記録で続行するには`best_effort`を使います。どちらも完了済みの更新と
チェックポイントは保持します。

Step 0の評価がある実験を再開する場合は、その結果ファイルを
`evaluation.baseline_outcomes_path`に指定し、`evaluate_before_training`を無効にします。
対応付き比較、信頼区間、McNemar検定を継続できるよう、起動前にこの参照を確認します。

## 境界

- GRPOの更新には完全な軌道グループが必要です。
- 学習中の診断指標と、ポリシー本来の評価系での結果は分けて記録します。
- 動画とトレースの件数はYAMLで指定できます。
- シミュレータ固有の処理は環境アダプターが担当します。

## ドキュメント

- [Embodied RLの概念と設定](docs/experimental/embodied-rl.mdx)
- [Runtimeとscaling architecture](docs/experimental/embodied-runtime-architecture.mdx)
- [リリース検証契約](docs/experimental/embodied-release-validation.mdx)
- [OpenVLA-OFT / LIBERO実行例](examples/embodied/README.md)

## 謝辞

実装、レシピ、チェックポイントを公開している[RLinf](https://github.com/RLinf/RLinf)の
開発者に感謝します。OpenVLA-OFTのロールアウト、アクショントークンのマスク、
advantageと損失の集約、LIBERO評価を検証する際に参照しました。

`rlinf_v01`は、この参照条件に対応する検証プロファイル名です。ART-Embodiedは
独自の実行系（`model_loader: native`）を使い、RLinfへの実行時依存はありません。

## ARTとの関係

ART-EmbodiedはOpenPipe ARTのアドオンで、ロボティクス関連の依存は追加インストールできます。
再現可能なベンチマーク、ポリシー・シミュレータのアダプター、W&B・Weave連携への
貢献を歓迎します。[CONTRIBUTING.md](CONTRIBUTING.md)を参照してください。
