# MuJoCo trong project `ethz-course-2026`

Tài liệu này trình bày kiến thức nền tảng, kiến trúc tích hợp, cách cài đặt, sử dụng, kiểm thử và triển khai MuJoCo trong toàn bộ project. Nội dung bám theo mã nguồn hiện tại của ba phần:

- `hw2_robot_control_mdps`: inverse kinematics, trajectory, PID và PPO với Stable-Baselines3;
- `hw3_imitation_learning`: teleoperation, thu thập demonstration, DAgger và imitation learning;
- `hw4_reinforcement_learning`: môi trường Gymnasium cho PPO và SAC tự cài đặt.

> Tên đúng của thư viện là **MuJoCo** (Multi-Joint dynamics with Contact), không phải “Mujuco”. Package Python được import bằng `import mujoco`.

---

## 1. Bắt đầu nhanh

### 1.1 Môi trường đã có trong repository cục bộ

Repository hiện có hai thư mục virtual environment chưa được Git quản lý:

| Thư mục | Python | Trạng thái kiểm tra |
|---|---:|---|
| `mujoco/` | 3.10.12 | Không có bản cài MuJoCo hoàn chỉnh; không nên dùng |
| `mujoco312/` | 3.12.13 | Có `mujoco==3.5.0`, chạy được smoke test HW2/HW4; chưa có `dm_control` cho HW3 |

Để dùng môi trường hiện tại trên Linux:

```bash
cd ethz-course-2026
source mujoco312/bin/activate
python -c "import mujoco; print(mujoco.__version__)"
```

Kết quả tại thời điểm viết tài liệu:

```text
3.5.0
```

Virtual environment là sản phẩm cục bộ, không nên commit. Khi tạo mới nên đặt tên `.venv` thay vì `mujoco` để tránh nhầm thư mục môi trường với package `mujoco`.

### 1.2 Các lệnh chạy quan trọng

Mỗi lệnh nên được chạy từ đúng thư mục homework tương ứng.

```bash
# HW2: mở model và thanh điều khiển MuJoCo
cd hw2_robot_control_mdps
python scripts/interactive.py

# HW2: IK, spline và PID
python scripts/inverse_kinematics.py
python scripts/quintic_splines.py
python scripts/pid_control.py

# HW2: train/evaluate PPO của Stable-Baselines3
python scripts/train.py --num_envs 16 --max_iterations 500
python scripts/evaluate_rand_targets.py --load_run 1 --checkpoint 500

# HW3: cấu hình phím và thu demonstration
cd ../hw3_imitation_learning
python scripts/configure_keys.py
python scripts/record_teleop_demos.py

# HW3: evaluate không mở cửa sổ
python scripts/eval.py --checkpoint PATH_TO_MODEL.pt --headless

# HW4: train/evaluate PPO và SAC
cd ../hw4_reinforcement_learning
python scripts/train_ppo.py
python scripts/eval_ppo.py
python scripts/eval_ppo.py --play
python scripts/train_sac.py
python scripts/eval_sac.py --play
```

---

## 2. MuJoCo là gì và project dùng nó để làm gì?

MuJoCo là physics engine dành cho hệ nhiều vật thể, robot, tiếp xúc và điều khiển. Trong project này, MuJoCo đảm nhiệm bốn lớp công việc:

1. **Mô tả hệ vật lý:** hình học, mesh, khối lượng, quán tính, joint, actuator, contact, camera và sensor được khai báo bằng MJCF/XML.
2. **Tính động học:** forward kinematics, vị trí/orientation của end-effector và Jacobian cho IK.
3. **Mô phỏng động lực học:** tích phân trạng thái theo thời gian khi nhận lệnh actuator.
4. **Hiển thị:** viewer tương tác, camera offscreen và ảnh dùng trong teleoperation.

Luồng dữ liệu chung:

```text
MJCF/XML + STL mesh
        |
        v
  mujoco.MjModel   <-- cấu trúc và tham số gần như tĩnh
        |
        +----> mujoco.MjData <-- trạng thái thay đổi theo thời gian
                              qpos, qvel, ctrl, xpos, site_xpos, ...
                                      |
                       policy/PID ----+
                                      |
                                      v
                              mujoco.mj_step()
                                      |
                                      v
                         observation/reward/render
```

MuJoCo không tự cung cấp thuật toán RL. Gymnasium, Stable-Baselines3 và code PyTorch trong project tạo policy/training loop; MuJoCo chỉ cung cấp môi trường vật lý và kết quả chuyển trạng thái.

---

## 3. Bản đồ tích hợp MuJoCo trong repository

| Khu vực | File chính | Cách dùng MuJoCo |
|---|---|---|
| HW2 model | `hw2_robot_control_mdps/so101_gym/assets/*.xml` | Robot SO-100, position/torque actuator, target mocap |
| HW2 điều khiển | `exercises/ex1.py`, `exercises/ex2.py` | Jacobian, IK DLS, spline, PID |
| HW2 Gym env | `env/so100_tracking_env.py` | Gymnasium env, 500 Hz physics/10 Hz policy |
| HW2 chạy | `scripts/*.py` | Viewer, callback control, PPO train/eval |
| HW3 model | `hw3_imitation_learning/so101_gym/assets/*.xml` | Cube, bin, obstacle, camera, mocap weld, keyframe |
| HW3 sim wrapper | `hw3/sim_env.py` | Reset, action target, substep, state query, offscreen render |
| HW3 thu dữ liệu | `scripts/record_teleop_demos.py` | Keyboard teleop, camera render, ghi Zarr |
| HW3 đánh giá | `scripts/eval.py`, `scripts/dagger_eval.py` | Rollout policy, success metric, headless/GUI |
| HW4 model | `hw4_reinforcement_learning/assets/mujoco/*.xml` | Bản model tracking tương tự HW2 |
| HW4 Gym env | `envs/so100_rl_env.py` | Observation 19 chiều, action 6 chiều, reward, reset |
| HW4 train | `scripts/train_ppo.py`, `scripts/train_sac.py` | Thu transition, update policy, checkpoint, TensorBoard |

### 3.1 Các model đã được kiểm tra compile

Kết quả đọc trực tiếp model bằng MuJoCo 3.5.0:

| Model | `nq` | `nv` | `nu` | `nmocap` | Timestep |
|---|---:|---:|---:|---:|---:|
| HW2/HW4 `so100_pos_ctrl.xml` | 6 | 6 | 6 | 1 | 0.002 s |
| HW3 single-cube | 13 | 12 | 6 | 1 | 0.002 s |
| HW3 multicube | 27 | 24 | 6 | 1 | 0.002 s |

Trong đó:

- `nq`: số tọa độ cấu hình trong `data.qpos`;
- `nv`: số bậc tự do/vận tốc trong `data.qvel`;
- `nu`: số actuator/lệnh trong `data.ctrl`;
- `nmocap`: số mocap body.

`nq` không luôn bằng `nv`. Một free joint của cube dùng 7 số trong `qpos` — vị trí 3 chiều và quaternion `wxyz` 4 chiều — nhưng chỉ có 6 vận tốc trong `qvel` — tịnh tiến 3 chiều và vận tốc góc 3 chiều. Vì vậy single-cube có `nq = 6 + 7 = 13` nhưng `nv = 6 + 6 = 12`.

---

## 4. Mô hình MJCF/XML của project

### 4.1 Cấu trúc include

File scene cấp cao chỉ khai báo timestep, mặt sàn, ánh sáng, target và include robot:

```xml
<mujoco model="so_arm100 pos_ctrl">
  <option timestep="0.002"/>
  <include file="trs_so_arm100/so_arm100.xml"/>
  ...
</mujoco>
```

Đường dẫn trong `<include>` và đường dẫn mesh được giải quyết tương đối với file XML. Vì vậy nên truyền **đường dẫn tuyệt đối** vào `MjModel.from_xml_path`, như code hiện tại đã làm ở phần lớn entry point.

### 4.2 Sáu khớp SO-100

| Index | Tên joint | Trục | Khoảng radian |
|---:|---|---|---|
| 0 | `Rotation` | Y | `[-1.92, 1.92]` |
| 1 | `Pitch` | X | `[-3.32, 0.174]` |
| 2 | `Elbow` | X | `[-0.174, 3.14]` |
| 3 | `Wrist_Pitch` | X | `[-1.66, 1.66]` |
| 4 | `Wrist_Roll` | Y | `[-2.79, 2.79]` |
| 5 | `Jaw` | Z | `[-0.174, 1.75]` |

Thứ tự này được dùng xuyên suốt trong action, `default_qpos`, joint target và dataset. Thay đổi thứ tự XML mà không cập nhật code sẽ làm policy điều khiển sai joint.

### 4.3 Position actuator và motor actuator

Project có hai model HW2:

#### Position control

```xml
<position name="Rotation" joint="Rotation" inheritrange="1" />
```

Giá trị `data.ctrl[i]` là vị trí joint mục tiêu. XML cấu hình servo với `kp="50"`, `dampratio="1"` và giới hạn force `[-3.5, 3.5]`. Đây là model dùng cho môi trường RL HW2/HW4.

#### Motor/torque-style control

```xml
<motor name="Rotation" joint="Rotation" gear="50" ctrlrange="-100 100"/>
```

Đây là model dùng bởi `scripts/pid_control.py`. PID ghi tín hiệu vào `data.ctrl`; actuator nhân lệnh với `gear=50` để tạo generalized force. Vì có gear, không nên diễn giải số trong `ctrl` là mô-men khớp theo tỉ lệ 1:1.

Quy tắc quan trọng: trước khi viết `data.ctrl`, phải biết actuator đang là `position`, `velocity`, `motor` hay loại khác. Cùng một vector số có ý nghĩa vật lý khác nhau tùy MJCF.

### 4.4 Site, body và mocap target

- `Base` là body gốc của robot và được dùng làm hệ tọa độ tham chiếu.
- `ee_site` đánh dấu pose của end-effector.
- `target` hoặc `mocap_target` là body có `mocap="true"`.
- `data.site("ee_site").xpos` là vị trí EE trong world frame.
- `data.body("Base").xpos` là vị trí base trong world frame.
- `data.mocap_pos[0]` và `data.mocap_quat[0]` là pose của mocap body đầu tiên.

Trong HW2/HW4, mocap body chủ yếu là target trực quan và target cho reward. Trong HW3, `mocap_target` được weld với `ee_site`; di chuyển mocap target tạo teleoperation Cartesian của EE.

### 4.5 Keyframe trong HW3

Scene HW3 khai báo các keyframe như `home`, `rest`, `student_start`. Reset đúng cách:

```python
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "student_start")
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)
```

Keyframe lưu đồng bộ nhiều phần trạng thái hơn việc chỉ gán `qpos`, nên phù hợp để tạo initial state tái lập cho robot và cube.

---

## 5. Hai đối tượng cốt lõi: `MjModel` và `MjData`

### 5.1 Khởi tạo tối thiểu

```python
from pathlib import Path
import mujoco

xml_path = Path("so101_gym/assets/so100_pos_ctrl.xml").resolve()
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

mujoco.mj_forward(model, data)
print(model.nq, model.nv, model.nu)
print(data.qpos)
```

### 5.2 `model`: cấu trúc và tham số

Các trường được dùng nhiều:

| Trường/API | Ý nghĩa |
|---|---|
| `model.opt.timestep` | Bước tích phân vật lý |
| `model.nq`, `model.nv`, `model.nu` | Kích thước state/control |
| `model.jnt_range` | Giới hạn các joint |
| `model.actuator_ctrlrange` | Giới hạn actuator control |
| `model.jnt_qposadr` | Địa chỉ bắt đầu của joint trong `qpos` |
| `model.body_pos` | Vị trí body tham số trong model |
| `mujoco.mj_name2id(...)` | Đổi tên object thành ID |

`model` thường được xem là tĩnh. HW3 có sửa `model.body_pos` để randomize obstacle/bin; sau đó phải gọi `mj_forward` để cập nhật đại lượng suy ra.

### 5.3 `data`: trạng thái runtime

| Trường/API | Ý nghĩa |
|---|---|
| `data.qpos` | Vị trí/cấu hình generalized coordinates |
| `data.qvel` | Vận tốc generalized coordinates |
| `data.ctrl` | Lệnh actuator |
| `data.site_xpos`, `data.site_xmat` | Pose site trong world frame |
| `data.xpos`, `data.xmat` | Pose body đã tính |
| `data.mocap_pos`, `data.mocap_quat` | Pose mocap body |
| `data.time` | Thời gian mô phỏng |

Khi sửa các mảng do MuJoCo sở hữu, dùng gán in-place:

```python
data.qpos[:] = new_qpos
data.ctrl[:] = target
```

Không nên thay object mảng bằng `data.qpos = new_qpos`. Binding MuJoCo cung cấp view vào vùng nhớ native; gán slice giữ nguyên liên kết với engine.

### 5.4 `mj_forward`, `mj_kinematics` và `mj_step`

| Hàm | Dùng khi nào |
|---|---|
| `mj_forward(model, data)` | Vừa sửa `qpos`/model và cần cập nhật toàn bộ đại lượng suy ra mà không tăng thời gian |
| `mj_kinematics(model, data)` | Chỉ cần cập nhật động học trong vòng IK |
| `mj_step(model, data)` | Thực hiện một bước vật lý và tăng `data.time` |
| `mj_resetData(model, data)` | Reset runtime state về mặc định model |
| `mj_resetDataKeyframe(...)` | Reset về keyframe cụ thể |

Gán `qpos` rồi gọi `mj_forward` là “teleport” trạng thái, không phải chuyển động vật lý. Gán `ctrl` rồi lặp `mj_step` mới mô phỏng actuator, quán tính, damping và contact.

---

## 6. Tần số mô phỏng và control decimation

Timestep XML là:

```text
dt_sim = 0.002 s
f_sim  = 1 / 0.002 = 500 Hz
```

Policy không cần chạy 500 lần mỗi giây. HW2/HW4 dùng:

```python
self.ctrl_decimation = 50
self.ctrl_timestep = self.model.opt.timestep * self.ctrl_decimation
```

Do đó:

```text
dt_control = 0.002 * 50 = 0.1 s
f_control  = 10 Hz
```

Một bước Gymnasium tương ứng:

```python
data.ctrl[:] = processed_action
for _ in range(50):
    mujoco.mj_step(model, data)
return observation, reward, terminated, truncated, info
```

HW3 tổng quát hóa cùng ý tưởng bằng:

```python
dt_ctrl = 1.0 / control_hz
substeps = round(dt_ctrl / sim_dt)
```

Khi thay `timestep`, `control_hz` hoặc `ctrl_decimation`, cần kiểm tra lại:

- episode length tính theo giây;
- PID derivative/integral;
- độ ổn định của actuator/contact;
- phân phối dữ liệu demonstration;
- checkpoint cũ, vì policy đã học theo dynamics và action frequency cũ.

---

## 7. Động học và IK trong HW2

### 7.1 Forward kinematics

Forward kinematics ánh xạ cấu hình joint `q` sang pose EE:

```text
x_ee = f(q)
```

Trong code:

```python
mujoco.mj_kinematics(model, data)
current_pos = data.site("ee_site").xpos
```

### 7.2 Jacobian

Quan hệ vi phân:

```text
velocity_ee = J(q) qdot
```

MuJoCo tính Jacobian của site:

```python
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
site_id = model.site("ee_site").id
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
```

- `jacp`: Jacobian tịnh tiến;
- `jacr`: Jacobian quay.

### 7.3 Damped Least Squares

`exercises/ex1.py` giải IK vị trí bằng:

```text
qdot = J^T (J J^T + damping I)^(-1) weighted_error
```

Code dùng `np.linalg.solve` thay vì tính inverse trực tiếp để ổn định số tốt hơn. Phần orientation error được đặt 0, vì bài tập chỉ track vị trí. Sau mỗi vòng lặp, `qdot` được clip và tích phân vào `qpos`.

Các giới hạn của solver hiện tại:

- chưa track orientation;
- chưa ép joint limit sau mỗi iteration;
- chưa tránh collision/self-collision;
- chưa tối ưu null-space hoặc posture;
- có thể không hội tụ khi target ngoài workspace hoặc gần singularity;
- `dt` trong IK là step của thuật toán số, không phải timestep vật lý.

### 7.4 Quintic trajectory và PID

Spline bậc năm trong `exercises/ex2.py`:

```text
f(s) = 6s^5 - 15s^4 + 10s^3
p(s) = p_start + (p_end - p_start) f(s)
```

Sau khi IK đổi waypoint Cartesian thành `target_qpos`, PID tạo lệnh motor:

```text
e_k = q_target - q_current
u_k = Kp e_k + Ki sum(e) dt + Kd (e_k - e_(k-1)) / dt
```

`scripts/pid_control.py` đăng ký callback bằng `mujoco.set_mjcb_control`. Callback được engine gọi trong pipeline step, nên phải nhẹ, xác định và không thực hiện I/O chậm.

Sau khi dùng callback toàn cục, cần dọn:

```python
mujoco.set_mjcb_control(None)
```

Nếu process dài hoặc có nhiều environment, callback toàn cục dễ tạo coupling. Với code mới, ghi `data.ctrl` trực tiếp ngay trước `mj_step` thường dễ quản lý hơn.

---

## 8. Môi trường Gymnasium trong HW2/HW4

### 8.1 Vòng đời chuẩn

```python
obs, info = env.reset(seed=0)
done = False

while not done:
    action = policy(obs)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

env.close()
```

- `terminated`: kết thúc do trạng thái terminal của bài toán;
- `truncated`: kết thúc do giới hạn thời gian;
- môi trường hiện tại luôn để `terminated=False` và kết thúc bằng `truncated`.

### 8.2 Action space

Policy sinh 6 số trong `[-1, 1]`. `process_action` ánh xạ tuyến tính tới joint range:

```text
q_target = q_low + (action + 1) / 2 * (q_high - q_low)
```

HW4 clip action trước khi scale. Đây là phòng vệ cần thiết khi policy hoặc caller trả giá trị vượt biên.

### 8.3 Observation space HW4

Observation có 19 phần tử:

| Thành phần | Số chiều |
|---|---:|
| `qpos` | 6 |
| EE position trong base frame | 3 |
| EE quaternion trong base frame | 4 |
| target position trong base frame | 3 |
| position error trong base frame | 3 |
| **Tổng** | **19** |

Chuyển từ world frame sang base frame:

```text
p_base = R_world_base^T (p_world - base_position_world)
```

Orientation tương đối:

```text
q_base_ee = conjugate(q_world_base) * q_world_ee
```

MuJoCo dùng quaternion theo thứ tự `wxyz`. Không trộn với thư viện dùng `xyzw`.

### 8.4 Reward HW4

Reward hiện tại kết hợp:

- exponential dense reward theo tracking error;
- bonus tại các ngưỡng 0.10, 0.05, 0.02 và 0.005 m;
- penalty nhỏ theo bình phương vận tốc joint lớn nhất.

Reward, observation, action scaling và control frequency tạo thành “hợp đồng” của checkpoint. Thay bất kỳ phần nào có thể khiến checkpoint cũ không còn tương thích về hành vi dù tensor shape vẫn giống.

### 8.5 Reproducibility

`env.reset(seed=...)` gọi `super().reset`, nhưng helper hiện dùng `np.random.uniform` toàn cục thay vì `self.np_random`. Vì vậy chỉ truyền seed cho Gymnasium chưa bảo đảm reset noise hoàn toàn tái lập. Khi cần benchmark chặt, nên:

1. truyền RNG của env vào helper; hoặc
2. seed đồng bộ Python, NumPy và PyTorch trước khi tạo env.

---

## 9. Teleoperation và imitation learning trong HW3

### 9.1 Mocap weld

Robot HW3 có equality constraint:

```xml
<equality>
  <weld site1="mocap_target_site" site2="ee_site"/>
</equality>
```

Khi người dùng bấm phím, code thay đổi `data.mocap_pos`/`data.mocap_quat`. Constraint solver làm EE đi theo target. Đây là cách thuận tiện để thu demonstration Cartesian mà không cần tự viết IK teleoperation.

Trong evaluation dùng joint actuator trực tiếp, `BaseSO100SimEnv._disable_mocap_weld()` tắt weld constraint để mocap không tranh quyền điều khiển với policy.

### 9.2 Tra cứu ID an toàn

HW3 không giả định joint/actuator nằm liên tiếp theo một index cố định. Nó tra ID theo tên:

```python
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
qpos_index = model.jnt_qposadr[joint_id]

actuator_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
)
```

Luôn kiểm tra ID khác `-1`. Đây là pattern nên dùng khi mở rộng XML.

### 9.3 State và action

`hw3/sim_env.py` cung cấp:

- joint angles;
- EE position/pose;
- gripper angle;
- cube state;
- obstacle position;
- goal/bin position;
- multicube states và one-hot goal.

Policy có thể dự đoán joint target hoặc delta end-effector tùy cấu hình dataset. Khi train, state/action keys và slicing phải giống lúc evaluate checkpoint.

### 9.4 Render camera

HW3 dùng offscreen renderer:

```python
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="angle")
rgb = renderer.render()
```

MuJoCo trả RGB. OpenCV mặc định dùng BGR, nên code chuyển:

```python
bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
```

Không chuyển đúng thứ tự kênh sẽ làm màu camera sai, đặc biệt nguy hiểm nếu image được dùng làm input policy.

### 9.5 Quy trình dữ liệu

```text
configure keys
     |
     v
teleoperate MuJoCo scene
     |
     v
raw Zarr demonstrations
     |
     v
compute_actions.py
     |
     v
processed state/action Zarr
     |
     v
train.py -> checkpoint .pt
     |
     v
eval.py / dagger_eval.py
```

Physics/control rate, action definition, state keys và camera convention phải nhất quán từ recording đến training và evaluation.

---

## 10. Viewer, rendering và headless

### 10.1 Viewer tương tác

```python
import mujoco.viewer

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
```

`launch_passive` không tự step physics; chương trình chịu trách nhiệm gọi `mj_step` và `viewer.sync`.

`mujoco.viewer.launch(model, data)` mở viewer blocking và phù hợp để inspect model/actuator bằng UI.

### 10.2 Không dùng `sleep` khi train

Các script playback dùng:

```python
time.sleep(model.opt.timestep)
```

để tốc độ hiển thị gần thời gian thật. Training/headless không nên sleep vì sẽ làm chậm thu thập dữ liệu hàng trăm lần.

### 10.3 Headless CPU/server

Trên Linux server không có màn hình, chọn backend trước khi import `mujoco`:

```bash
export MUJOCO_GL=egl
python scripts/eval.py --checkpoint model.pt --headless
```

Nếu máy không có EGL/GPU phù hợp, có thể thử software rendering:

```bash
export MUJOCO_GL=osmesa
python your_script.py
```

Lưu ý:

- `render_mode=None` và không tạo `mujoco.Renderer` là lựa chọn nhanh nhất cho RL state-based;
- `--headless` của HW3 bỏ cửa sổ OpenCV, nhưng `sim_env.py` vẫn khởi tạo `mujoco.Renderer`; vì vậy hệ thống vẫn cần một OpenGL backend hợp lệ;
- biến `MUJOCO_GL` phải được đặt trước lần import MuJoCo đầu tiên trong process;
- EGL thường cần driver GPU tương ứng; OSMesa chạy CPU và chậm hơn.

### 10.4 macOS

Theo hướng dẫn HW4, khi mở GUI trên macOS có thể cần:

```bash
.venv/bin/mjpython scripts/eval_ppo.py --play
```

---

## 11. Cài đặt sạch và tái lập

### 11.1 Khuyến nghị: một virtual environment cho mỗi homework

Ba homework có dependency khác nhau và HW2 pin version chặt. Tách môi trường giúp tránh xung đột và giúp tái lập kết quả.

#### HW2

```bash
cd ethz-course-2026/hw2_robot_control_mdps
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

`requirements.txt` hiện pin:

```text
gymnasium==1.2.3
mujoco==3.5.0
numpy==2.4.2
stable_baselines3==2.7.1
tensorboard==2.20.0
```

#### HW3

```bash
cd ethz-course-2026/hw3_imitation_learning
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

HW3 khai báo `dm-control`, OpenCV, Zarr, PyTorch và các dependency khác trong `pyproject.toml`. `dm-control` sử dụng MuJoCo ở tầng dependency, còn code project vẫn import API Python `mujoco` trực tiếp.

#### HW4

```bash
cd ethz-course-2026/hw4_reinforcement_learning
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 11.2 Linux packages cho GUI/render

Trên Ubuntu/Debian, viewer hoặc EGL có thể cần:

```bash
sudo apt install libegl1-mesa-dev libgl1-mesa-dri libglvnd-dev
```

Đây là system dependency, không được cài bởi `pip`.

### 11.3 Không phụ thuộc working directory ngoài ý muốn

Entry point nên dựng path từ `__file__`:

```python
ROOT_DIR = Path(__file__).resolve().parents[1]
xml_path = ROOT_DIR / "assets" / "mujoco" / "so100_pos_ctrl.xml"
```

Pattern này đang được HW4 dùng tốt. Nó cho phép chạy script từ nhiều current working directory khác nhau và phù hợp với job scheduler/container.

---

## 12. Smoke test và kiểm tra model

### 12.1 Kiểm tra import/version

```bash
python -c "import sys, mujoco; print(sys.version); print(mujoco.__version__)"
```

### 12.2 Compile XML không mở viewer

Chạy từ `hw2_robot_control_mdps`:

```bash
python -c "from pathlib import Path; import mujoco; p=Path('so101_gym/assets/so100_pos_ctrl.xml').resolve(); m=mujoco.MjModel.from_xml_path(str(p)); print(m.nq, m.nv, m.nu, m.opt.timestep)"
```

Kết quả mong đợi:

```text
6 6 6 0.002
```

### 12.3 Smoke test một bước HW4

Chạy từ `hw4_reinforcement_learning`:

```bash
python -c "from pathlib import Path; import numpy as np; from envs.so100_rl_env import SO100RLEnv; env=SO100RLEnv(Path('assets/mujoco/so100_pos_ctrl.xml').resolve()); obs,_=env.reset(seed=0); obs2,reward,term,trunc,info=env.step(np.zeros(env.action_dim,dtype=np.float32)); print(obs.shape,reward,info); env.close()"
```

Shape observation mong đợi là `(19,)`.

### 12.4 Sanity check state

```python
assert np.isfinite(data.qpos).all()
assert np.isfinite(data.qvel).all()
assert np.isfinite(data.ctrl).all()
assert model.nu == data.ctrl.shape[0]
```

Trong train loop dài, nên log thêm:

- reward và episode length;
- EE tracking error;
- maximum `abs(qvel)`;
- số lần action bị clip;
- NaN/Inf count;
- simulation steps/second.

### 12.5 Kiểm tra Gymnasium

Có thể dùng checker trong quá trình phát triển:

```python
from gymnasium.utils.env_checker import check_env

check_env(env, skip_render_check=True)
```

Checker có thể phát hiện sai dtype/shape và sai hợp đồng `reset`/`step` trước khi lỗi xuất hiện trong training.

---

## 13. Training, checkpoint và TensorBoard

### 13.1 HW2 Stable-Baselines3 PPO

Mặc định `scripts/train.py` tạo 16 process environment:

```bash
python scripts/train.py \
  --num_envs 16 \
  --max_iterations 500 \
  --save_checkpt_freq 50 \
  --device cpu
```

Debug với một env có render:

```bash
python scripts/train.py --num_envs 1 --max_iterations 5
```

Không nên dùng render khi chạy training đầy đủ. Với multiprocessing, mỗi worker có `MjModel` và `MjData` riêng; không chia sẻ một `MjData` giữa process/thread.

### 13.2 HW4 PPO/SAC

HW4 lưu artifact tại:

```text
logs/ppo/YY_MM_DD_HH_MM_SS_model/iter_N.pt
logs/sac/YY_MM_DD_HH_MM_SS_model/iter_N.pt
```

Các eval script tự tìm checkpoint iteration lớn nhất trong run mới nhất nếu không truyền path.

### 13.3 TensorBoard

```bash
tensorboard --logdir logs --port 6006
```

Sau đó mở `http://localhost:6006`.

Checkpoint nên đi kèm metadata:

- Git commit;
- Python/MuJoCo/Torch version;
- XML/model checksum hoặc version;
- observation/action schema;
- reward config;
- timestep/control frequency;
- random seed và training config.

Nếu thiếu metadata, checkpoint có thể load thành công nhưng chạy sai vì environment contract đã thay đổi.

---

## 14. Triển khai trên máy khác hoặc server

Trong project nghiên cứu này, “triển khai” thường có ba nghĩa:

1. tái tạo simulator trên máy khác;
2. chạy training batch/headless;
3. chạy inference/evaluation từ checkpoint.

### 14.1 Những gì phải đóng gói

- code Python;
- toàn bộ XML được include;
- toàn bộ `.stl` mesh;
- dependency lock/requirements;
- checkpoint;
- config tương ứng checkpoint;
- nếu HW3: keymap/dataset schema và preprocessing definition;
- driver/OpenGL backend phù hợp nếu có render.

Không chỉ copy file XML cấp cao. Nếu thiếu `trs_so_arm100/*.xml` hoặc mesh, model sẽ không compile.

### 14.2 Checklist deploy headless

```bash
# 1. Checkout đúng commit
git checkout COMMIT_ID

# 2. Tạo môi trường sạch
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 3. Chọn backend nếu cần renderer
export MUJOCO_GL=egl

# 4. Smoke test import + XML + một env step
python -c "import mujoco; print(mujoco.__version__)"

# 5. Chạy train/eval không GUI
python scripts/train_ppo.py
```

### 14.3 Container

Nếu đóng gói Docker/OCI:

- dùng base image có Python 3.12;
- copy requirements trước để tận dụng layer cache;
- copy cả assets/XML/STL;
- đặt `WORKDIR` là thư mục homework;
- dùng `MUJOCO_GL=egl` cho GPU render hoặc `osmesa` cho software render;
- mount `logs/`, dataset và checkpoint thành volume;
- không bake dataset/checkpoint lớn vào image nếu chúng thay đổi thường xuyên;
- với NVIDIA/EGL, host phải có driver tương thích và container runtime phải expose GPU.

State-based HW2/HW4 training không gọi renderer nên có thể chạy headless mà không cần GPU đồ họa. GPU CUDA chỉ tăng tốc neural network; nó không làm physics MuJoCo CPU hiện tại tự động chạy trên GPU.

### 14.4 Job batch/HPC

Một job train nên:

- ghi log/checkpoint ra thư mục bền vững;
- bắt signal và lưu checkpoint nếu scheduler sắp dừng job;
- tránh viewer/OpenCV window;
- đặt số thread BLAS/PyTorch hợp lý để không oversubscribe khi dùng nhiều env process;
- log hostname, CPU/GPU, seed và version;
- chạy smoke test ngắn trước job dài.

### 14.5 Inference

Inference cần bảo đảm:

```text
same XML/model
+ same joint/actuator order
+ same observation transform
+ same action scaling
+ same control frequency
+ same checkpoint architecture
```

Nếu triển khai policy sang robot thật, không được nối thẳng action simulation vào phần cứng. Cần thêm calibration, joint/velocity/torque limits, emergency stop, watchdog, collision handling và kiểm thử sim-to-real. Nội dung đó nằm ngoài phạm vi simulator hiện tại.

---

## 15. Lỗi thường gặp và cách xử lý

### `ModuleNotFoundError: No module named 'mujoco'`

Nguyên nhân thường gặp:

- chưa activate đúng venv;
- cài dependency vào Python khác;
- đang dùng venv `mujoco/` cũ chưa có package.

Kiểm tra:

```bash
which python
python -m pip show mujoco
python -c "import mujoco; print(mujoco.__file__)"
```

### Import `mujoco` nhưng không có `__version__`

Có thể Python đang import nhầm namespace/thư mục tên `mujoco` thay vì package thật. In `mujoco.__file__` và `sys.path` để xác nhận. Dùng `.venv` cho virtual environment mới giúp tránh tên gây nhầm.

### XML compile error hoặc không tìm thấy mesh/include

- kiểm tra đã copy đủ asset;
- dùng `Path(...).resolve()`;
- không đổi cấu trúc thư mục tương đối bên trong `assets`;
- đọc chính xác path được nêu trong exception.

### `data.ctrl` có đúng shape nhưng robot không đi như mong đợi

- kiểm tra actuator type;
- kiểm tra thứ tự actuator bằng tên/ID;
- kiểm tra `ctrlrange`, `forcerange`, `gear`;
- xác nhận đã gọi `mj_step`;
- xác nhận mocap weld không tranh với actuator trong HW3.

### Đọc `site_xpos` ngay sau khi sửa `qpos` nhưng giá trị cũ

Gọi:

```python
mujoco.mj_forward(model, data)
```

### Viewer không cập nhật

Với passive viewer phải gọi:

```python
viewer.sync()
```

### Viewer/EGL không mở trên Linux

- kiểm tra `DISPLAY` nếu dùng window;
- dùng `MUJOCO_GL=egl` cho headless;
- cài Mesa/EGL packages và driver;
- thử `MUJOCO_GL=osmesa` nếu không có GPU/EGL.

### NaN hoặc simulation mất ổn định

- giảm control gain/action magnitude;
- clip action/ctrl;
- kiểm tra timestep;
- kiểm tra pose reset có xuyên vật thể;
- kiểm tra quaternion đã chuẩn hóa và theo `wxyz`;
- log `qpos`, `qvel`, `ctrl` ngay trước bước đầu xuất hiện NaN.

### Policy load được nhưng kết quả rất kém

- sai state/action keys hoặc thứ tự;
- XML/model khác lúc train;
- sai control decimation;
- reward/normalization khác;
- quaternion convention khác;
- checkpoint architecture/default constructor không trùng lúc train.

### Multiprocessing bị treo hoặc lỗi OpenGL

- training worker nên dùng `render_mode=None`;
- không tạo viewer trong subprocess;
- mỗi process tự tạo model/data;
- dùng start method theo platform như code HW2 (`forkserver` trên Linux, `spawn` trên Windows).

---

## 16. Quy tắc khi mở rộng project

### Thêm joint/actuator

1. Sửa MJCF và actuator.
2. Compile XML bằng smoke test.
3. Tra ID theo tên, không hard-code index nếu có thể.
4. Cập nhật action space, `default_qpos`, observation và checkpoint architecture.
5. Kiểm tra `nq`, `nv`, `nu` mới.
6. Train lại policy.

### Thêm object có free joint

Nhớ rằng mỗi free joint thêm:

```text
qpos: 7 = xyz + quaternion(wxyz)
qvel: 6 = linear velocity + angular velocity
```

Không dùng một slice `qpos` để suy luận trực tiếp slice `qvel`.

### Thêm camera

1. Khai báo `<camera name="...">` trong XML.
2. Kiểm tra bằng `mj_name2id`.
3. Render đúng tên camera.
4. Cố định width/height/color order trong dataset metadata.
5. Nếu dùng image policy, không thay intrinsics/extrinsics sau khi train nếu chưa domain-randomize.

### Thêm reward hoặc observation

- giữ dtype `float32` ở boundary Gym/PyTorch;
- cập nhật `observation_space` chính xác;
- version hóa schema;
- viết unit test shape/range;
- coi checkpoint cũ là không tương thích nếu semantic thay đổi.

---

## 17. API cheat sheet

```python
# Load
model = mujoco.MjModel.from_xml_path(str(xml_path))
data = mujoco.MjData(model)

# Reset/update
mujoco.mj_resetData(model, data)
mujoco.mj_resetDataKeyframe(model, data, key_id)
mujoco.mj_forward(model, data)

# Physics
data.ctrl[:] = action
mujoco.mj_step(model, data)

# Named access
ee_pos = data.site("ee_site").xpos.copy()
base_pos = data.body("Base").xpos.copy()

# ID lookup
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_site")
joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "Elbow")

# Jacobian
jacp = np.zeros((3, model.nv))
jacr = np.zeros((3, model.nv))
mujoco.mj_jacSite(model, data, jacp, jacr, site_id)

# Mocap
data.mocap_pos[0] = target_xyz
data.mocap_quat[0] = target_quat_wxyz

# Passive viewer
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()

# Offscreen render
renderer = mujoco.Renderer(model, height=480, width=640)
renderer.update_scene(data, camera="angle")
rgb = renderer.render()
```

---

## 18. Checklist trước khi chạy experiment dài

- [ ] Đúng Python/venv và `mujoco.__version__`.
- [ ] XML compile được, đủ include và mesh.
- [ ] `nq`, `nv`, `nu`, joint order đúng kỳ vọng.
- [ ] Action được clip và scale đúng actuator type.
- [ ] Observation đúng shape, dtype và coordinate frame.
- [ ] Quaternion dùng thứ tự `wxyz` và đã normalize.
- [ ] `sim_dt`, control decimation và episode length đúng.
- [ ] Reset chạy nhiều lần không sinh NaN hoặc contact bất thường.
- [ ] Headless backend hoạt động nếu chạy server.
- [ ] Không mở viewer/sleep trong training.
- [ ] Seed và config được lưu.
- [ ] Logs/checkpoints ghi vào đường dẫn bền vững.
- [ ] Eval ngắn chạy được trước khi train dài.
- [ ] Checkpoint và environment contract cùng version.

---

## 19. Kết luận

MuJoCo trong project này được dùng theo ba mức trừu tượng tăng dần:

1. **API vật lý trực tiếp:** XML → `MjModel`/`MjData` → `mj_forward`/`mj_step`.
2. **Robot control:** Jacobian, IK, spline, PID, mocap và actuator.
3. **Learning environment:** Gymnasium, observation/action/reward, teleoperation, imitation learning, PPO và SAC.

Để làm việc ổn định với repository, cần luôn phân biệt:

- model tĩnh (`MjModel`) và state runtime (`MjData`);
- teleport bằng `qpos + mj_forward` và physics bằng `ctrl + mj_step`;
- position target và motor input;
- simulation frequency và policy/control frequency;
- world frame và robot base frame;
- GUI viewer và headless/offscreen rendering;
- shape tương thích và semantic tương thích của checkpoint.

Nắm chắc các ranh giới này sẽ giúp việc sửa model, thiết kế controller, train policy và triển khai experiment trên máy khác ít lỗi hơn đáng kể.
