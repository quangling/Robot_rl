# Kiến thức project `hw3_imitation_learning`

## 1. Tóm tắt nhanh

Project này xây dựng một pipeline **Imitation Learning (IL)** cho cánh tay robot SO-101 trong mô phỏng MuJoCo. Robot học cách bắt chước thao tác của con người để gắp một khối lập phương, tránh vật cản và thả khối vào hộp.

Ba bài tập tương ứng với ba ý tưởng tăng dần về độ khó:

1. **Behavioral Cloning với MSE:** học trực tiếp ánh xạ từ trạng thái hiện tại sang một chuỗi hành động tương lai.
2. **DAgger:** cho policy tự chạy ở phân phối khó hơn, con người can thiệp tại các trạng thái policy xử lý kém, rồi gộp dữ liệu mới để huấn luyện lại.
3. **Goal-conditioned Imitation Learning:** một policy xử lý nhiều nhiệm vụ; đầu vào cho biết cần gắp khối màu nào và vị trí hộp đích.

Đây là một **bộ khung bài tập**, không phải project đã hoàn thiện. Người học phải cài đặt model, loss, training loop, optimizer, scheduler và chọn hyperparameter.

---

## 2. Bài toán robot đang giải quyết

### 2.1 Môi trường

- Robot: SO-101, gồm 5 khớp tay và 1 khớp hàm kẹp (`Jaw`).
- Simulator: MuJoCo.
- Điều khiển: 10 Hz trong các script thu thập/evaluate mặc định.
- Quan sát: trạng thái vector từ simulator, không dùng ảnh làm đầu vào policy trong code hiện tại.
- Ảnh từ ba camera (`left_wrist`, `angle`, `top`) chỉ phục vụ hiển thị cho người điều khiển.

### 2.2 Mục tiêu

Với trạng thái hiện tại \(s_t\), policy dự đoán một đoạn gồm \(H\) hành động:

\[
\pi_\theta(s_t) = [\hat a_t, \hat a_{t+1}, \ldots, \hat a_{t+H-1}]
\]

Trong đó:

- \(H\) là `chunk_size`, mặc định trong CLI là 16.
- Mỗi hành động robot arm là một **delta** so với target hiện tại.
- Hành động gripper là lệnh điều khiển tuyệt đối đã được ghi khi teleoperate.

Trong lúc evaluate, toàn bộ chunk được đưa vào hàng đợi và thực thi lần lượt. Policy chỉ suy luận lại khi hàng đợi rỗng. Vì vậy đây là thực thi **open-loop theo từng chunk**, không phải dự đoán lại ở mọi timestep.

---

## 3. Toàn bộ pipeline

```mermaid
flowchart LR
    A[Cấu hình phím] --> B[Teleoperation]
    B --> C[Raw Zarr<br/>state theo timestep]
    C --> D[Compute actions<br/>a_t = s_{t+1} - s_t]
    D --> E[Processed Zarr<br/>state/action/episode boundaries]
    E --> F[Chunk dataset<br/>s_t -> a_t:t+H]
    F --> G[Train policy<br/>MSE]
    G --> H[Checkpoint<br/>weights + normalizer + metadata]
    H --> I[Rollout trong MuJoCo]
    I --> J{Policy lỗi ở OOD?}
    J -- Có --> K[Con người takeover<br/>thu dữ liệu DAgger]
    K --> D
    J -- Không --> L[Đánh giá và nộp bài]
```

Các bước thực tế:

1. Cấu hình bàn phím bằng `scripts/configure_keys.py`.
2. Người dùng teleoperate robot và ghi demonstration.
3. Dữ liệu thô được lưu theo từng timestep trong Zarr.
4. `scripts/compute_actions.py` chuyển chuỗi state thành transition `(state, action)`.
5. `SO100ChunkDataset` tạo các cặp `(state_t, action_chunk)`.
6. Policy được train bằng supervised learning.
7. Checkpoint lưu đủ thông tin để dựng lại model và tiền xử lý khi inference.
8. `scripts/eval.py` rollout policy, áp dụng action và tính success rate.
9. Với DAgger, con người takeover ở tình huống policy thất bại rồi train lại trên dữ liệu đã gộp.

---

## 4. Dữ liệu và biểu diễn trạng thái

### 4.1 Vì sao dùng Zarr?

Zarr phù hợp với dữ liệu robot vì lưu được nhiều array lớn theo chunk, nén được, truy cập từng phần và dễ mở rộng dần trong lúc record. Cấu trúc chính là:

```text
dataset.zarr/
├── data/
│   ├── state_joints
│   ├── state_ee hoặc state_ee_full/state_ee_xyz
│   ├── state_cube
│   ├── state_gripper
│   ├── state_obstacle
│   ├── action_...
│   └── action_gripper
└── meta/
    └── episode_ends
```

`episode_ends` chứa các chỉ số kết thúc tích lũy. Ví dụ `[100, 230]` nghĩa là episode đầu là `[0, 100)`, episode thứ hai là `[100, 230)`.

### 4.2 Các state key

| Key | Số chiều | Ý nghĩa |
|---|---:|---|
| `state_ee_xyz` | 3 | Vị trí Cartesian `(x, y, z)` của end-effector |
| `state_ee_full` | 7 | Vị trí 3D và quaternion `wxyz` |
| `state_joints` | 6 | Góc 5 khớp tay và `Jaw`; khi evaluate phần Jaw bị loại khỏi state khớp |
| `state_gripper` | 1 | Góc mở hiện tại của hàm kẹp |
| `state_cube` | 7 | Vị trí và quaternion của cube mục tiêu |
| `state_obstacle` | 3 | Vị trí vật cản; bằng vector 0 nếu không có vật cản |
| `goal_pos` | 3 | Tâm của bin trong world frame |

Riêng bài multicube:

| Key | Số chiều | Ý nghĩa |
|---|---:|---|
| `original_pos_cube_red` | 7 | Pose khối đỏ |
| `original_pos_cube_green` | 7 | Pose khối xanh lá |
| `original_pos_cube_blue` | 7 | Pose khối xanh dương |
| `state_goal` | 3 | One-hot mục tiêu theo thứ tự `[red, green, blue]` |

Tên có tiền tố `original_` được tạo trong bước processing từ các raw key `pos_cube_*`; nó không có nghĩa đây chỉ là vị trí đầu episode. Array vẫn được ghi theo từng timestep.

### 4.3 Ghép và cắt state

`load_zarr()` cho phép ghép nhiều key thành một vector phẳng. Cú pháp slice hỗ trợ:

```text
key
key[:N]
key[M:]
key[M:N]
```

Ví dụ:

```bash
--state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle
```

Tạo state có kích thước `3 + 1 + 3 + 3 = 10`.

Việc chỉ lấy `state_cube[:3]` bỏ quaternion của cube là hợp lý nếu orientation không cần thiết để hoàn thành nhiệm vụ. Đây là cách giảm số chiều và làm bài toán dễ học hơn.

---

## 5. Không gian hành động

`scripts/compute_actions.py` hỗ trợ ba lựa chọn:

| CLI `--action-space` | Action key | Chiều arm | Cách tính |
|---|---|---:|---|
| `ee` | `action_ee_xyz` | 3 | Delta vị trí end-effector |
| `ee_full` | `action_ee_full` | 6 | Delta vị trí 3D + delta Euler 3D |
| `joints` | `action_joints` | 5 | Delta góc của 5 khớp, không gồm Jaw |

Với position hoặc joint state thông thường:

\[
a_t = s_{t+1} - s_t
\]

Timestep cuối của mỗi episode bị bỏ vì không có \(s_{t+1}\). Việc tính delta luôn diễn ra **bên trong từng episode**, nên không tạo transition sai giữa cuối episode này và đầu episode sau.

Với pose đầy đủ, không lấy hiệu trực tiếp giữa hai quaternion. Code tính relative rotation:

\[
q_{rel} = q_{t+1} q_t^{-1}
\]

sau đó đổi `q_rel` thành Euler `(roll, pitch, yaw)`.

### Điểm đặc biệt của gripper

`action_gripper` **không phải delta**. Nó là actuator command tuyệt đối được record khi teleoperate. Lý do là chỉ nhìn thay đổi góc Jaw có thể không biểu diễn được lực kẹp đang tác dụng lên cube. Khi inference:

- arm action: cộng delta vào target hiện tại;
- gripper action: gửi trực tiếp làm target của Jaw.

Không được xử lý nhầm hai loại này giống nhau.

### Chọn action space thế nào?

- `ee`: đơn giản nhất nếu nhiệm vụ không cần đổi orientation; thường cần ít dữ liệu hơn.
- `ee_full`: linh hoạt hơn nhưng khó học hơn và Euler có thể tạo biểu diễn không liên tục.
- `joints`: tránh IK/mocap khi chạy, nhưng quan hệ từ trạng thái task-space tới chuyển động khớp phức tạp hơn.

Project tự tắt mocap weld nếu checkpoint dùng `action_joints`; với EE action, mocap target điều khiển end-effector.

---

## 6. Action chunking

Thay vì học một hành động duy nhất, dataset tạo mẫu:

\[
(s_t, [a_t, a_{t+1}, \ldots, a_{t+H-1}])
\]

Shape của một batch:

```text
state:        (B, state_dim)
action_chunk: (B, H, action_dim)
```

`build_valid_indices()` chỉ giữ timestep mà toàn bộ chunk nằm trong cùng episode. Với episode dài \(L\), số mẫu hợp lệ là:

\[
\max(0, L - H + 1)
\]

Ưu điểm của action chunking:

- model học một đoạn chuyển động có cấu trúc thay vì từng lệnh rời rạc;
- inference ít lần hơn;
- hành vi thường mượt hơn.

Đổi lại, trong implementation này robot chạy hết chunk trước khi quan sát và lập kế hoạch lại. Chunk quá dài làm feedback chậm và lỗi tích lũy; chunk quá ngắn làm hành động dễ rung và phải inference thường xuyên.

---

## 7. Chuẩn hóa dữ liệu

`Normalizer` tính mean và standard deviation riêng cho từng feature:

\[
\tilde s = \frac{s - \mu_s}{\max(\sigma_s, 10^{-6})}
\]

\[
\tilde a = \frac{a - \mu_a}{\max(\sigma_a, 10^{-6})}
\]

Model học trên state và action đã chuẩn hóa. Khi inference, predicted action được đưa về đơn vị thật:

\[
a = \tilde a \odot \sigma_a + \mu_a
\]

Chuẩn hóa quan trọng vì tọa độ mét, góc khớp, quaternion và gripper command có scale khác nhau. Nếu không chuẩn hóa, MSE có thể bị feature có biên độ lớn chi phối.

Mean/std phải được tính từ training data và lưu trong checkpoint. Không được tính lại bằng dữ liệu evaluate vì sẽ gây data leakage và làm preprocessing không nhất quán.

---

## 8. Behavioral Cloning và MSE policy

### 8.1 Behavioral Cloning

Behavioral Cloning coi imitation learning như supervised learning:

\[
\mathcal D = \{(s_t, a_t^{expert})\}
\]

và tối ưu policy sao cho hành động dự đoán gần hành động expert.

Trong project, `ObstaclePolicy` được gợi ý là một MLP:

```text
state vector
    ↓
MLP encoder / hidden layers
    ↓
H × action_dim giá trị
    ↓ reshape
(H, action_dim)
```

Loss tự nhiên là:

\[
\mathcal L_{MSE} = \frac{1}{B H D_a}
\sum_{b=1}^{B}\sum_{h=1}^{H}\sum_{j=1}^{D_a}
(\hat a_{b,h,j} - a_{b,h,j})^2
\]

`compute_loss()` nên gọi forward trên state và so sánh đúng shape `(B, H, action_dim)`. `sample_actions()` dùng cùng forward path nhưng là API dành cho inference.

### 8.2 Vì sao Behavioral Cloning có thể thất bại?

Training data chỉ chứa state expert từng đi qua. Khi rollout, một sai số nhỏ đưa robot sang state khác. Policy chưa từng thấy state đó nên có thể sai thêm; sai số tích lũy và đưa robot ngày càng xa phân phối training. Hiện tượng này gọi là **covariate shift** hoặc compounding error.

Validation loss thấp chưa đảm bảo success rate cao vì:

- MSE đo độ giống action offline, không đo kết quả closed-loop;
- nhiều action expert hợp lệ có thể tồn tại ở cùng một state;
- lỗi nhỏ ở thời điểm gắp/thả có ảnh hưởng lớn hơn lỗi ở đoạn di chuyển dễ;
- các chunk train chồng lấn mạnh với nhau.

Success rate qua rollout mới là metric chính.

---

## 9. DAgger trong project

DAgger giải quyết covariate shift bằng cách thu thập dữ liệu ở các state mà policy thực sự ghé thăm.

Vòng lặp:

1. Train policy từ demonstration ban đầu.
2. Evaluate với phân phối vật cản adversarial.
3. Quan sát failure mode.
4. Khi sắp xảy ra lỗi, nhấn phím `record` để con người takeover.
5. Chỉ các timestep takeover được lưu.
6. Chạy lại `compute_actions.py`; script tìm cả thư mục `teleop/` và `dagger/`, rồi gộp chúng.
7. Train lại policy trên dataset tổng hợp.
8. Lặp đến khi coverage và success rate đủ tốt.

Phân phối obstacle adversarial gồm:

- 20% ở vùng trung tâm;
- 40% lệch phải khoảng `+0.08 m`;
- 40% lệch trái khoảng `-0.08 m`;
- có thêm Gaussian noise nhỏ.

Về mặt học thuật, implementation này gần với **selective intervention / human-in-the-loop DAgger**: expert chỉ cung cấp nhãn trong các đoạn takeover. DAgger cổ điển thường mô tả việc expert gán nhãn cho toàn bộ state mà learner ghé thăm. Cách ở project tiết kiệm công gán nhãn hơn nhưng coverage phụ thuộc vào việc con người phát hiện và can thiệp đủ sớm.

Các thao tác DAgger:

- `record`: chuyển qua lại giữa policy và human control;
- `reset`: bỏ dữ liệu hiện tại và replay đúng scenario bằng cách phục hồi RNG state;
- `Enter`: bỏ episode hiện tại và sang episode kế;
- `Escape`: bỏ phần đang thu và thoát.

Sau khi human đưa cube vào bin, code tiếp tục record khoảng 1.7 giây để tránh đoạn cuối ngắn hơn một action chunk.

---

## 10. Goal-conditioned imitation learning

Trong bài multicube, cùng một hình trạng vật lý có thể yêu cầu hành động khác nhau tùy cube mục tiêu. Vì vậy policy phải phụ thuộc vào goal:

\[
\pi_\theta(a \mid s, g)
\]

Trong đó `g = state_goal` là one-hot:

```text
red   = [1, 0, 0]
green = [0, 1, 0]
blue  = [0, 0, 1]
```

State nên chứa ít nhất:

- pose của end-effector và trạng thái gripper;
- pose/vị trí của cả ba cube;
- `state_goal` để biết cube cần chọn;
- `goal_pos` vì vị trí bin thay đổi giữa các episode.

Nếu thiếu `state_goal`, cùng một state đầu vào có ba action expert khả dĩ. Với MSE, model dễ học “trung bình” giữa các hướng đi và không tới cube nào. Nếu thiếu `goal_pos`, policy không biết nơi cần thả cube khi bin bị shuffle.

Một hướng thiết kế tốt là biến đầu vào về các quan hệ tương đối, ví dụ:

\[
p_{target} - p_{ee}, \qquad p_{bin} - p_{ee}, \qquad p_{bin} - p_{target}
\]

So với chỉ dùng tọa độ tuyệt đối, relative features thường làm cấu trúc bài toán rõ hơn và hỗ trợ generalization giữa các layout. Tuy nhiên mọi feature dùng khi train cũng phải được tái tạo y hệt trong `eval_utils.py` hoặc trong forward của model từ các state key đã lưu.

---

## 11. Luồng inference và điều khiển

Khi load checkpoint:

1. Đọc `state_dim`, `action_dim`, `chunk_size`, `policy_type`.
2. Dựng lại policy với cùng kiến trúc.
3. Load `model_state_dict`.
4. Khôi phục normalizer và danh sách `state_keys`, `action_keys`.
5. Mỗi lần queue rỗng, ghép observation theo đúng thứ tự `state_keys`.
6. Normalize state, gọi `model.sample_actions()` và denormalize action chunk.
7. Tách từng action theo `action_keys` rồi áp dụng vào simulator.

Ví dụ checkpoint dùng:

```text
state_keys  = [state_ee_xyz, state_gripper, state_cube[:3], state_obstacle]
action_keys = [action_ee_xyz, action_gripper]
```

thì output mỗi timestep có 4 chiều:

```text
[dx, dy, dz, gripper_target]
```

`apply_action()` cộng ba delta đầu vào mocap position hiện tại và gửi phần cuối trực tiếp cho actuator Jaw.

---

## 12. Điều kiện đánh giá

Một episode thành công khi cube mục tiêu:

- nằm đủ gần tâm bin theo cả `x` và `y`;
- có `0 < z < 0.04 m`;
- với multicube, ngưỡng XY tối đa là `0.04 m`.

Single-cube dùng ngưỡng XY mặc định `0.05 m`. Episode thất bại sớm nếu cube rơi khỏi workspace hoặc xuống dưới bàn.

Mỗi rollout chính thức có tối đa 800 step. Project còn yêu cầu cube phải được **thả** vào bin, không còn bị giữ trong gripper ở ngưỡng thành công; đây là yêu cầu chấm bổ sung được mô tả trong README.

Với multicube, script còn báo nếu một cube không phải mục tiêu nằm trong bin và thống kê success rate riêng theo từng màu.

---

## 13. Vai trò của từng file

| File | Trách nhiệm |
|---|---|
| `README.md` | Đề bài, yêu cầu từng exercise, cách nộp và thang điểm |
| `hw3/model.py` | Interface `BasePolicy`; nơi phải cài `ObstaclePolicy`, `MultiTaskPolicy`, `build_policy` |
| `hw3/dataset.py` | Đọc/ghép Zarr, parse key slice, normalize, tạo action chunks không vượt episode |
| `hw3/sim_env.py` | Wrapper MuJoCo, reset/randomization, observation, control, render, môi trường single/multicube |
| `hw3/eval_utils.py` | Dựng state từ observation, load checkpoint, inference, tách và áp dụng action, kiểm tra success |
| `hw3/teleop_utils.py` | Keymap, điều khiển tay, quaternion, giao diện camera, writer Zarr |
| `scripts/configure_keys.py` | Tạo `hw3/keymap.json` theo bàn phím người dùng |
| `scripts/record_teleop_demos.py` | Thu demonstration single-cube hoặc multicube |
| `scripts/compute_actions.py` | Gộp raw Zarr, tính delta action, căn chỉnh array và ghi processed Zarr |
| `scripts/train.py` | Training/validation loop, checkpointing; hiện còn TODO |
| `scripts/eval.py` | Evaluate trực quan hoặc headless |
| `scripts/dagger_eval.py` | Policy rollout có human takeover và ghi dữ liệu DAgger |
| `student_eval/run_eval.py` | Gọi binary harness, chạy đánh giá chính thức và tạo file `.hwresult` đã ký |
| `so101_gym/assets/*.xml` | Scene, robot, actuator, camera, keyframe và geometry MuJoCo |

---

## 14. Quy trình chạy đề xuất

Chạy các lệnh từ thư mục `hw3_imitation_learning`.

### 14.1 Cài đặt

```bash
uv venv --python 3.12
source .venv/bin/activate
uv pip install -e .
```

### 14.2 Exercise 1

```bash
python scripts/configure_keys.py
python scripts/record_teleop_demos.py
python scripts/compute_actions.py --action-space ee
```

Sau khi hoàn thiện `model.py` và `train.py`, một cấu hình state/action đơn giản có thể là:

```bash
python scripts/train.py \
  --zarr datasets/processed/single_cube/processed_ee_xyz.zarr \
  --policy obstacle \
  --chunk-size 16 \
  --state-keys state_ee_xyz state_gripper "state_cube[:3]" state_obstacle \
  --action-keys action_ee_xyz action_gripper
```

Evaluate:

```bash
python scripts/eval.py \
  --checkpoint checkpoints/single_cube/<checkpoint>.pt \
  --num-episodes 100 \
  --headless \
  --seed 42
```

### 14.3 Exercise 2

Quan sát phân phối khó:

```bash
python scripts/eval.py \
  --checkpoint checkpoints/single_cube/<checkpoint>.pt \
  --adversarial-obstacle
```

Thu corrective demonstration:

```bash
python scripts/dagger_eval.py \
  --checkpoint checkpoints/single_cube/<checkpoint>.pt \
  --num-episodes 10
```

Sau đó chạy lại processing. Vì mặc định script tìm đệ quy trong `datasets/raw/single_cube`, nó sẽ lấy cả `teleop` và `dagger`:

```bash
python scripts/compute_actions.py --action-space ee
```

Cuối cùng train lại từ processed dataset mới và evaluate với `--adversarial-obstacle`.

### 14.4 Exercise 3

```bash
python scripts/record_teleop_demos.py --multicube
python scripts/compute_actions.py \
  --action-space ee \
  --datasets-dir datasets/raw/multi_cube
```

Ví dụ input goal-conditioned:

```bash
python scripts/train.py \
  --zarr datasets/processed/multi_cube/processed_ee_xyz.zarr \
  --policy multitask \
  --chunk-size 16 \
  --state-keys state_ee_xyz state_gripper \
    "original_pos_cube_red[:3]" \
    "original_pos_cube_green[:3]" \
    "original_pos_cube_blue[:3]" \
    state_goal goal_pos \
  --action-keys action_ee_xyz action_gripper
```

Evaluate:

```bash
python scripts/eval.py \
  --checkpoint checkpoints/multi_cube/<checkpoint>.pt \
  --multicube \
  --goal-cube all \
  --num-episodes 100 \
  --headless \
  --seed 42
```

Các lệnh train trên chỉ chạy sau khi những TODO và mismatch mô tả ở phần tiếp theo đã được sửa.

---

## 15. Hợp đồng của checkpoint

Checkpoint không chỉ chứa weight. `eval_utils.load_checkpoint()` cần các trường:

```text
model_state_dict
normalizer.state_mean
normalizer.state_std
normalizer.action_mean
normalizer.action_std
chunk_size
policy_type
state_keys
action_keys
state_dim
action_dim
```

Nếu kiến trúc có `d_model`, `depth` hoặc hyperparameter khác ảnh hưởng shape của weight, chúng cũng phải được lưu và dùng lại khi `build_policy()` dựng model.

Thứ tự `state_keys` và `action_keys` là một phần của model contract. Đổi thứ tự nhưng giữ nguyên số chiều vẫn không gây lỗi shape, song ý nghĩa feature bị tráo và policy sẽ hoạt động sai.

Với bài nộp, tên class `ObstaclePolicy` và `MultiTaskPolicy` không được đổi. Default constructor/kiến trúc phải tương thích với checkpoint để autograder tái tạo model.

---

## 16. Những phần chưa hoàn thiện trong snapshot hiện tại

### `hw3/model.py`

- `ObstaclePolicy.forward`, `compute_loss`, `sample_actions` chưa được cài.
- `MultiTaskPolicy` chưa được cài.
- `build_policy()` chưa nhận/truyền `chunk_size` và các tham số kiến trúc.

### `scripts/train.py`

- `EPOCHS`, `BATCH_SIZE`, `LR` vẫn là `...`.
- Training step và validation step còn TODO.
- Optimizer và scheduler chưa được tạo.
- Parser chưa khai báo `--extra-zarr`, nhưng code lại truy cập `args.extra_zarr`; chạy hiện tại sẽ gặp `AttributeError`.
- Chưa truyền `chunk_size` hay hyperparameter kiến trúc vào `build_policy()`.
- Chưa lưu `d_model`/`depth`, trong khi loader có logic đọc hai trường này.

### Mismatch giữa train/model/eval

`eval_utils.load_checkpoint()` đang gọi:

```text
build_policy(..., chunk_size=..., d_model=..., depth=...)
```

nhưng chữ ký `build_policy()` trong scaffold hiện chỉ khai báo `state_dim` và `action_dim`. Vì vậy cần thống nhất constructor, builder, phần lưu checkpoint và loader trước khi train/evaluate.

Đây là các phần được giao cho sinh viên, không phải lỗi môi trường cài đặt.

---

## 17. Các quyết định thiết kế quan trọng

### 17.1 Chọn state vừa đủ

Single-cube obstacle cần policy biết ít nhất:

- robot/end-effector đang ở đâu;
- gripper đang mở hay đóng;
- cube ở đâu;
- obstacle ở đâu.

Nếu bin cố định, có thể không cần `goal_pos`. Nếu muốn model tổng quát hơn hoặc dùng multicube, nên thêm nó.

### 17.2 Model capacity

MLP quá nhỏ dễ underfit các pha khác nhau: tiếp cận, hạ tay, đóng gripper, nâng, tránh obstacle, tới bin và nhả. Model quá lớn so với lượng demonstration dễ ghi nhớ dữ liệu. README gợi ý dưới một triệu tham số đã đủ cho Exercise 1 và 2.

### 17.3 Dữ liệu quan trọng hơn chỉ số loss

Demonstration nên:

- có quỹ đạo nhất quán;
- bao phủ biến thiên vị trí cube/obstacle;
- không đứng yên quá lâu khi đang record;
- không kết thúc episode quá sớm, để các chunk cuối không bị mất;
- có đủ ví dụ quanh các trạng thái khó như tiếp xúc, kẹp và thả.

### 17.4 Validation split

Scaffold dùng `random_split` trên các sliding window. Hai window lân cận chia sẻ gần như toàn bộ action, nên train và validation có thể chứa các mẫu rất giống nhau từ cùng episode. Validation loss vì thế có thể lạc quan.

Để đánh giá tổng quát hóa nghiêm túc hơn, nên split theo **episode** trước rồi mới tạo chunks. Dù vậy, metric cuối vẫn phải là rollout success rate trên seed/scenario chưa dùng khi train.

---

## 18. Lỗi thường gặp và cách suy luận

### Policy đứng yên

Nguyên nhân thường gặp:

- demonstration có quá nhiều timestep không thao tác;
- action delta rất nhỏ và model học gần mean;
- quên denormalize action;
- action key trong checkpoint không khớp dataset.

### Robot đi đúng hướng nhưng không gắp được

- thiếu `state_gripper` hoặc `action_gripper`;
- dữ liệu đóng gripper không nhất quán;
- coi gripper command là delta;
- chunk quá dài làm thời điểm đóng kẹp thiếu feedback.

### Tốt offline nhưng kém khi rollout

- covariate shift;
- validation leakage do overlapping windows;
- không đủ dữ liệu ở failure states;
- state thiếu obstacle/goal;
- horizon quá dài nên open-loop error lớn.

### Exercise 1 tốt nhưng Exercise 2 kém

Đây là kết quả được dự đoán trước: vị trí obstacle adversarial nằm ngoài phân phối demonstration ban đầu. Cần DAgger hoặc thu dữ liệu đa dạng hơn, không chỉ tăng số epoch.

### Multicube lấy sai màu

- thiếu `state_goal`;
- mất cân bằng số demonstration theo màu;
- thứ tự one-hot/cube feature không nhất quán;
- model chưa học interaction giữa goal và pose của từng cube.

### Load checkpoint lỗi shape hoặc unexpected keyword

- constructor và `build_policy()` không khớp lúc train/eval;
- không lưu hyperparameter kiến trúc;
- default của class đã đổi sau khi tạo checkpoint.

---

## 19. Checklist trước khi đánh giá chính thức

- [ ] Hoàn thiện toàn bộ TODO trong `model.py` và `train.py`.
- [ ] Thống nhất chữ ký model giữa train, checkpoint loader và autograder.
- [ ] Kiểm tra state/action dimension bằng output của script train.
- [ ] Kiểm tra đúng thứ tự `state_keys` và `action_keys` trong checkpoint.
- [ ] Bảo đảm normalizer được lưu và load đúng.
- [ ] Evaluate trực quan để xem failure mode, không chỉ nhìn MSE.
- [ ] Evaluate headless với 100 episode và seed 42 trước khi nộp.
- [ ] Với Exercise 2, evaluate đúng chế độ adversarial.
- [ ] Với Exercise 3, kiểm tra success rate riêng cho cả ba màu.
- [ ] Không đổi tên hai policy class mà autograder import.
- [ ] Đặt checkpoint nộp thành `ex1.pt`, `ex2.pt`, `ex3.pt` tương ứng.
- [ ] Không chỉnh sửa file `.hwresult` do evaluation harness tạo.

---

## 20. Kết luận

Kiến thức cốt lõi của project không chỉ là viết một MLP. Đây là một pipeline robot learning hoàn chỉnh ở quy mô nhỏ:

- định nghĩa observation/action phù hợp;
- thu thập và tổ chức demonstration theo episode;
- chuyển state trajectory thành action label;
- chuẩn hóa và tạo action chunks;
- huấn luyện Behavioral Cloning bằng MSE;
- đánh giá closed-loop thay vì chỉ dựa vào validation loss;
- sửa covariate shift bằng human intervention/DAgger;
- mở rộng thành policy có điều kiện theo goal;
- đóng gói mọi preprocessing và kiến trúc vào checkpoint có thể tái tạo.

Thông điệp quan trọng nhất là: **chất lượng policy phụ thuộc đồng thời vào biểu diễn state/action, coverage của dữ liệu, thiết kế horizon, model capacity và tính nhất quán giữa training với inference**. Chỉ tối ưu loss mà bỏ qua một trong các mắt xích đó thường không tạo được robot policy chạy tốt trong rollout.
