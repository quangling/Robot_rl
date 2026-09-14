# Kiến thức dự án `hw2_robot_control_mdps`

## 1. Dự án này dạy điều gì?

Đây là bài tập điều khiển cánh tay robot SO-100 gồm 6 khớp, mô phỏng bằng MuJoCo. Mục tiêu cuối cùng là đưa đầu công tác (end-effector, viết tắt là EE) tới một điểm đích trong không gian 3D.

Dự án được tổ chức thành ba bước có độ trừu tượng tăng dần:

1. **Exercise 1 — Inverse Kinematics (IK):** đổi một điểm đích trong không gian làm việc Cartesian thành cấu hình góc khớp.
2. **Exercise 2 — Lập quỹ đạo và PID:** tạo chuyển động trơn giữa các điểm rồi dùng phản hồi PID để robot thật sự đi theo cấu hình mong muốn trong mô phỏng vật lý.
3. **Exercise 3 — MDP và Reinforcement Learning:** mô hình hóa bài toán thành một MDP và huấn luyện policy PPO tự sinh lệnh điều khiển từ quan sát.

Ba bài tập tương ứng với ba câu hỏi quan trọng trong robot học:

- **Muốn EE ở đâu?** — mô tả đích trong workspace.
- **Các khớp phải ở đâu và đi tới đó thế nào?** — IK, spline và PID.
- **Robot có thể tự học ánh xạ từ trạng thái sang hành động không?** — MDP và PPO.

> Trạng thái hiện tại của mã nguồn: các hàm trong `exercises/ex1.py`, `exercises/ex2.py` và `exercises/ex3.py` vẫn chứa `TODO`/`NotImplementedError`. Vì vậy các script chính chỉ chạy hoàn chỉnh sau khi những phần này được cài đặt.

## 2. Cấu trúc project

| Thành phần | Vai trò |
|---|---|
| `exercises/ex1.py` | Sinh đường số 8 và giải IK bằng Damped Least Squares |
| `exercises/ex2.py` | Sinh quintic spline và tính tín hiệu PID |
| `exercises/ex3.py` | Reset ngẫu nhiên, đổi miền action, reward và observation của bài toán RL |
| `env/so100_tracking_env.py` | Môi trường Gymnasium kết nối các hàm Exercise 3 với MuJoCo |
| `scripts/inverse_kinematics.py` | Kiểm tra IK bằng cách đặt trực tiếp `qpos` |
| `scripts/quintic_splines.py` | Minh họa các waypoint trung gian, vẫn đặt trực tiếp `qpos` |
| `scripts/pid_control.py` | Theo waypoint bằng PID và actuator mô-men |
| `scripts/train.py` | Huấn luyện PPO, mặc định dùng 16 môi trường song song |
| `scripts/evaluate_rand_targets.py` | Đánh giá policy trên 10 đích ngẫu nhiên |
| `scripts/evaluate_trajectory.py` | Thử policy trên quỹ đạo hình số 8 |
| `scripts/utils.py` | Hàm quaternion, marker và callback cho huấn luyện |
| `so101_gym/assets/*.xml` | Mô hình robot, giới hạn khớp và cấu hình actuator MuJoCo |

Hai mô hình MuJoCo phục vụ hai kiểu điều khiển khác nhau:

- `so100_pos_ctrl.xml`: actuator vị trí. Giá trị `data.ctrl` là **vị trí khớp đích**; MuJoCo dùng servo vị trí nội bộ (`kp=50`, `dampratio=1`) để sinh lực.
- `so100_torque_ctrl.xml`: actuator motor. `data.ctrl` là **đầu vào actuator** dùng cho PID của Exercise 2. File XML đặt `gear=50`, vì vậy nói chính xác hơn đây là lệnh motor được quy đổi thành lực/mô-men qua hệ số truyền động, không đơn thuần là mô-men khớp bằng đúng con số trong `ctrl`.

Timestep vật lý của cả hai mô hình là `0.002 s`, tương đương **500 Hz**.

## 3. Các khái niệm robot nền tảng

### 3.1 Không gian khớp và không gian làm việc

- **Joint space:** vector cấu hình khớp

  $$q=[q_1,q_2,\ldots,q_6]^T.$$

- **Workspace/Cartesian space:** vị trí EE

  $$x=[x,y,z]^T.$$

**Forward kinematics** là ánh xạ $x=f(q)$: biết góc khớp để tính vị trí EE. **Inverse kinematics** giải bài toán ngược: biết $x^*$ và tìm $q^*$ sao cho $f(q^*)\approx x^*$.

Một đích Cartesian có thể có nhiều nghiệm IK, không có nghiệm nếu nằm ngoài tầm với, hoặc nằm gần cấu hình kỳ dị khiến việc giải số trở nên khó khăn.

### 3.2 Hệ tọa độ world và base

MuJoCo cung cấp vị trí và ma trận quay trong world frame. Policy nên nhận đại lượng trong base frame để kết quả không phụ thuộc vào việc cả robot được đặt ở đâu hoặc quay theo hướng nào trong thế giới.

Gọi $R_{WB}$ là ma trận đưa vector từ base frame sang world frame. Khi đó:

$$R_{BW}=R_{WB}^T$$

$$p_E^B=R_{BW}(p_E^W-p_B^W)$$

$$p_T^B=R_{BW}(p_T^W-p_B^W).$$

Với quaternion theo thứ tự MuJoCo `wxyz`, orientation tương đối của EE là:

$$q_{BE}=q_{WB}^{-1}\otimes q_{WE},$$

trong đó quaternion của base cần được liên hợp để lấy nghịch đảo (sau khi chuẩn hóa). Kết quả cuối cũng nên được chuẩn hóa để vẫn là một phép quay hợp lệ.

## 4. Exercise 1 — Đường Lemniscate và Inverse Kinematics

### 4.1 Lemniscate of Bernoulli

Dự án tạo đường hình số 8 trong mặt phẳng $Y$-$Z$:

$$y(t)=\frac{a\cos t}{1+\sin^2t},\qquad
z(t)=\frac{a\cos t\sin t}{1+\sin^2t},\quad t\in[0,2\pi).$$

Điểm 3D được tạo bằng:

$$k(t)=[x_{offset},\ y(t),\ z(t)+z_{offset}].$$

`endpoint=False` nên được dùng khi lấy mẫu $t$ để tránh lặp lại cùng một điểm ở $0$ và $2\pi$. Các giá trị mặc định là 16 điểm, `width=0.25`, `x_offset=0.3`, `z_offset=0.25`.

Tham số $a$ điều khiển kích thước đường. Khi tăng $a$, một số điểm có thể ra ngoài workspace khả đạt hoặc tới gần singularity/joint limit, làm IK không hội tụ hoặc cho nghiệm kém ổn định.

### 4.2 Jacobian

Ở lân cận cấu hình hiện tại, chuyển động EE và chuyển động khớp có quan hệ xấp xỉ tuyến tính:

$$\dot{x}=J(q)\dot{q}.$$

Trong code, `mj_jacSite` trả về:

- `jacp`: Jacobian vị trí, kích thước $3\times n_v$;
- `jacr`: Jacobian góc, kích thước $3\times n_v$.

Hai phần được ghép thành $J\in\mathbb{R}^{6\times n_v}$. Bài tập chỉ bám vị trí nên phần sai số orientation trong vector sai số 6 chiều được đặt bằng 0.

### 4.3 Damped Least Squares (DLS)

Pseudo-inverse thông thường có thể bùng nổ gần singularity. DLS làm bài toán ổn định hơn:

$$\dot q=J^T(JJ^T+\lambda I)^{-1}e_w,$$

với

$$e_w=[K_{pos}(x^*-x),0,0,0]^T.$$

Một số tài liệu viết $\lambda^2I$ thay cho $\lambda I$; project dùng quy ước `damping * I`. Không nên tính ma trận nghịch đảo trực tiếp. Cách ổn định hơn là giải hệ tuyến tính

$$(JJ^T+\lambda I)y=e_w$$

rồi tính $\dot q=J^Ty$.

Vòng lặp IK thực hiện:

1. Chạy forward kinematics để cập nhật vị trí EE.
2. Tính $e=x^*-x$ và dừng nếu $\|e\|_2<10^{-3}$ m.
3. Tính Jacobian và bước DLS.
4. Chặn mỗi thành phần $\dot q$ trong `[-2, 2]` để giảm overshoot.
5. Cập nhật $q\leftarrow q+\dot q\,dt$.
6. Lưu nghiệm tìm được, sau đó khôi phục trạng thái ban đầu của simulator trước khi trả nghiệm.

`dt` ở đây là **bước của thuật toán tối ưu IK**, không phải timestep vật lý MuJoCo. Quá lớn có thể gây dao động/vượt quá nghiệm; quá nhỏ làm hội tụ chậm và dễ hết `max_iters`.

### 4.4 Điểm mạnh và giới hạn của IK trong project

Ưu điểm của numerical IK là dễ viết, dùng được cho nhiều cấu trúc robot và không cần tự suy ra công thức đóng. Nhược điểm là cần nghiệm khởi tạo, tốn nhiều vòng lặp, kết quả phụ thuộc tham số và không bảo đảm tìm được mọi nghiệm.

Solver hiện tại còn đơn giản vì:

- không bám orientation;
- không áp joint limits sau mỗi bước;
- không tránh va chạm hoặc self-collision;
- không tối ưu tư thế phụ/null space;
- không tự điều chỉnh damping theo singularity;
- không bảo đảm nghiệm tối ưu toàn cục;
- có thể dùng cả bậc tự do của gripper dù mục tiêu chỉ là vị trí EE.

Các solver hiện đại thường bổ sung ràng buộc bất đẳng thức, collision constraints, joint-limit avoidance, task priorities và tối ưu hóa nhiều mục tiêu.

### 4.5 Lưu ý về script kiểm thử

`inverse_kinematics.py` gán thẳng nghiệm vào `data.qpos`. Robot vì thế bị “teleport” giữa các cấu hình; đây chưa phải điều khiển động lực học. Script chỉ chứng minh ánh xạ workspace → joint space hoạt động.

## 5. Exercise 2 — Quintic spline và PID

### 5.1 Vì sao cần waypoint trung gian?

Nếu nhảy trực tiếp từ một keypoint sang keypoint kế tiếp, lệnh mong muốn thay đổi đột ngột. Điều này có thể sinh vận tốc, gia tốc và mô-men lớn. Ta dùng time scaling bậc năm:

$$f(s)=10s^3-15s^4+6s^5,\qquad s\in[0,1].$$

Waypoint được nội suy:

$$p(s)=p_{start}+(p_{end}-p_{start})f(s).$$

Đa thức này thỏa:

$$f(0)=0,\ f(1)=1,$$

$$f'(0)=f'(1)=0,$$

$$f''(0)=f''(1)=0.$$

Do đó vị trí, vận tốc và gia tốc nối vào/ra mỗi đoạn một cách êm hơn nội suy tuyến tính. Sau khi sinh waypoint Cartesian, project gọi IK để đổi từng waypoint thành `target_qpos`.

### 5.2 Bộ điều khiển PID

Sai số khớp tại thời điểm $k$:

$$e_k=q_k^*-q_k.$$

Ba thành phần:

$$P_k=e_k,$$

$$I_k\approx \Delta t\sum_i e_i,$$

$$D_k\approx\frac{e_k-e_{k-1}}{\Delta t}.$$

Tín hiệu điều khiển:

$$u_k=K_PP_k+K_II_k+K_DD_k.$$

Nếu lịch sử chỉ có một mẫu thì $D=0$. Trong script thực tế, `tracking_error_history` được giữ tối đa 10 mẫu, nên thành phần tích phân là một **moving-window integral**, không phải tích phân từ đầu episode.

Gain mặc định:

```text
Kp = 150.0
Ki = 0.0
Kd = 0.01
```

Ý nghĩa trực giác:

- $K_P$ tạo lực tỉ lệ với sai số hiện tại. Tăng quá cao dễ overshoot, rung hoặc mất ổn định.
- $K_D$ phản ứng với tốc độ đổi của sai số, tạo damping và giảm overshoot/rung. Quá lớn có thể khuếch đại nhiễu.
- $K_I$ tích lũy sai số để loại steady-state error do tải không đổi, ma sát hoặc bias mô hình. Quá lớn gây integral windup.

### 5.3 Pipeline điều khiển cổ điển

```text
keypoint Cartesian
       ↓ quintic time scaling
waypoint Cartesian
       ↓ inverse kinematics
target_qpos
       ↓ so sánh với qpos hiện tại
joint error
       ↓ PID
actuator command
       ↓ MuJoCo dynamics
qpos mới và vị trí EE mới
```

`quintic_splines.py` vẫn teleport robot để chỉ kiểm tra hình dạng quỹ đạo. `pid_control.py` mới chạy vòng kín qua động lực học. Mỗi đoạn mặc định có 5 waypoint. Vì cả đầu và cuối đoạn đều được lấy mẫu, biên giữa hai đoạn liên tiếp có thể xuất hiện hai lần.

## 6. Exercise 3 — Mô hình hóa thành MDP

Một Markov Decision Process thường được viết là

$$\mathcal M=(\mathcal S,\mathcal A,P,R,\gamma,\rho_0).$$

Trong project:

| Thành phần | Cách hiện thực |
|---|---|
| State $\mathcal S$ | Trạng thái vật lý MuJoCo của robot và vị trí target |
| Observation $o_t$ | Vector 16 chiều gồm joint position, pose EE và target trong base frame |
| Action $\mathcal A$ | Vector liên tục 6 chiều trong `[-1,1]` |
| Transition $P$ | Động lực học MuJoCo qua 50 bước vật lý cho mỗi action |
| Reward $R$ | Hàm dense + sparse dựa trên khoảng cách EE–target |
| Discount $\gamma$ | `0.99` trong PPO |
| Initial distribution $\rho_0$ | Home pose có nhiễu và target 3D lấy mẫu ngẫu nhiên |
| Horizon | 10 giây = 100 control steps khi train |

### 6.1 Observation 16 chiều

Thứ tự feature mà `get_obs` phải trả về là:

```text
[qpos(6), ee_pos_base(3), ee_quat_base(4), target_pos_base(3)]
```

Tổng kích thước là $6+3+4+3=16$. Joint position được giữ nguyên vì nó là tọa độ nội tại của robot; các vị trí và orientation tuyệt đối cần đổi sang base frame.

Observation space được khai báo là `Box(-inf, inf, shape=(16,), dtype=float64)`. Action space là `Box(-1, 1, shape=(6,), dtype=float32)`.

Một điểm tinh tế: trạng thái vật lý đầy đủ có cả vận tốc khớp `qvel`, nhưng observation hiện tại không chứa nó. Vì động lực học phụ thuộc vận tốc, observation 16 chiều không hoàn toàn Markov; nhìn từ phía policy, đây gần với một bài toán **partially observable/approximate MDP**. Bổ sung `qvel` là một hướng cải tiến hợp lý.

### 6.2 Phân phối reset

Robot bắt đầu gần home pose:

```text
[0.0, -1.57, 1.0, 1.0, 0.0, 0.02239]
```

rồi cộng nhiễu đều cho từng khớp trong `[-0.5, 0.5]`.

Target được lấy mẫu quanh base với offset:

```text
x ∈ [ 0.2, 0.4]
y ∈ [-0.2, 0.2]
z ∈ [ 0.1, 0.4]
```

Theo skeleton hiện tại, cách đơn giản là lấy `base_pos + sampled_offset`. Cách này đúng khi trục base song song với world; nếu base có thể quay tùy ý, một offset thật sự trong base frame phải được quay bởi $R_{WB}$ trước khi cộng vào vị trí world.

Random reset là một dạng domain randomization nhỏ: policy không thể ghi nhớ duy nhất một trạng thái đầu và phải học hành vi có khả năng khái quát hơn.

### 6.3 Chuẩn hóa action

Policy sinh $a_i\in[-1,1]$. Mỗi phần tử được ánh xạ tuyến tính sang giới hạn khớp $[l_i,u_i]$:

$$q_i^*=l_i+\frac{a_i+1}{2}(u_i-l_i)$$

hay tương đương

$$q_i^*=\frac{l_i+u_i}{2}+a_i\frac{u_i-l_i}{2}.$$

Nên clip action về `[-1,1]` trước khi đổi miền. Các mốc có ý nghĩa:

- $a_i=-1\Rightarrow q_i^*=l_i$;
- $a_i=0\Rightarrow q_i^*=(l_i+u_i)/2$;
- $a_i=1\Rightarrow q_i^*=u_i$.

Trong Exercise 3, XML dùng **position actuators**, vì vậy policy không điều khiển torque trực tiếp. Nó chọn target joint positions; servo của MuJoCo thực hiện tầng điều khiển thấp hơn.

### 6.4 Reward shaping

Sai số theo dõi là khoảng cách Euclid:

$$d=\|p_{EE}-p_{target}\|_2.$$

Reward được yêu cầu:

$$r_{dense}=e^{-2d},$$

$$r_{sparse}=\begin{cases}
1,&d<0.005\\
0,&\text{ngược lại},
\end{cases}$$

$$r=r_{dense}+r_{sparse}.$$

Phần dense cung cấp gradient học ở mọi khoảng cách. Phần sparse thưởng thêm khi đạt vùng chính xác 5 mm. Reward nằm trong khoảng gần $(0,2]$: tiến gần đích làm reward dense tiến tới 1, và vào ngưỡng thành công nhận thêm 1.

Lưu ý reward này không phạt vận tốc, gia tốc, năng lượng, thay đổi action hoặc va chạm. Vì thế policy có thể đạt reward tốt nhưng chuyển động giật hoặc dao động.

### 6.5 Hai thang thời gian

Mỗi lần `env.step(action)`:

1. đổi action chuẩn hóa thành vị trí khớp đích;
2. ghi vào `data.ctrl[:]`;
3. chạy 50 bước MuJoCo, mỗi bước `0.002 s`;
4. sau `0.1 s`, đo lỗi, tính reward và observation mới.

Do đó:

| Vòng lặp | Tần số | Chu kỳ |
|---|---:|---:|
| Physics MuJoCo | 500 Hz | 0.002 s |
| Policy/control | 10 Hz | 0.1 s |

Control decimation làm policy rẻ hơn và action ít nhiễu tần số cao hơn, trong khi mô phỏng vật lý vẫn đủ mịn. Đổi `ctrl_decimation` cũng làm thay đổi bản chất transition và horizon theo số step, vì vậy policy thường phải được train lại.

### 6.6 Gymnasium termination

Environment luôn trả `terminated=False`; episode kết thúc do giới hạn thời gian với `truncated=True` sau 100 control steps. Phân biệt này quan trọng:

- `terminated`: đạt trạng thái kết thúc thuộc bản chất nhiệm vụ;
- `truncated`: bị cắt bởi giới hạn bên ngoài như time limit.

Project không kết thúc sớm khi chạm target; agent tiếp tục nhận reward nếu giữ EE gần target.

## 7. PPO trong project

PPO học một policy $\pi_\theta(a\mid o)$ và một value function. Ý tưởng chính là cập nhật policy để tăng expected return nhưng chặn thay đổi quá mạnh giữa policy cũ và mới bằng clipped surrogate objective. Điều này thường ổn định hơn policy gradient không ràng buộc.

`scripts/train.py` dùng:

```text
algorithm: PPO
policy: MlpPolicy
gamma: 0.99
entropy coefficient: 0.001
value-function coefficient: 1.0
device: CPU mặc định
parallel environments: 16 mặc định
```

Vai trò tham số:

- `gamma=0.99`: coi trọng reward tương lai; ở 10 Hz, reward sau 1 giây được nhân xấp xỉ $0.99^{10}\approx0.904$.
- `ent_coef=0.001`: khuyến khích exploration nhẹ, tránh policy trở nên quyết định quá sớm.
- `vf_coef=1.0`: trọng số loss của value function.
- `MlpPolicy`: mạng fully connected phù hợp observation/action dạng vector.

Script còn có callback tự điều chỉnh learning rate theo approximate KL:

- KL cao hơn vùng quanh `target_kl=0.05` → giảm learning rate bằng hệ số `0.7`;
- KL thấp hơn vùng mục tiêu → tăng learning rate bằng hệ số `1.1`;
- learning rate luôn nằm trong `[1e-5, 1e-3]`.

Mặc định Stable-Baselines3 PPO thu `n_steps=2048` bước trên mỗi environment cho một rollout. Với 16 environment, một rollout chứa 32,768 transition; script tính tổng bước bằng `max_iterations × n_steps × n_envs`. Các environment chạy bằng subprocess để tận dụng nhiều lõi CPU.

## 8. Train, theo dõi và đánh giá

### 8.1 Cài đặt

Python 3.12 là phiên bản được hướng dẫn và đã kiểm thử. Từ thư mục gốc `ethz-course-2026`:

```bash
python3 -m venv mujoco
source mujoco/bin/activate
pip install -r hw2_robot_control_mdps/requirements.txt
pip install -e hw2_robot_control_mdps
cd hw2_robot_control_mdps
```

Kiểm tra viewer:

```bash
python scripts/interactive.py
```

### 8.2 Thứ tự kiểm thử hợp lý

```bash
python scripts/inverse_kinematics.py
python scripts/quintic_splines.py
python scripts/pid_control.py
python scripts/train.py --max_iterations 500 --save_checkpt_freq 50
```

Theo dõi train ở terminal khác:

```bash
tensorboard --logdir=logs --port=6006
```

Sau đó mở `http://localhost:6006`.

### 8.3 Đánh giá random targets

```bash
python scripts/evaluate_rand_targets.py --load_run=1 --checkpoint=500
```

Script dùng policy deterministic, thử 10 episode, mỗi episode 2 giây, rồi in final EE tracking error và trung bình. Tiêu chí full score là:

```text
Average final EE tracking error < 0.05 m
```

Đánh giá 2 giây ngắn hơn episode train 10 giây, nên policy không chỉ cần hội tụ mà còn phải tới đích nhanh.

### 8.4 Đánh giá đường số 8

```bash
python scripts/evaluate_trajectory.py --load_run=1 --checkpoint=500
```

Target đổi sang keypoint tiếp theo khi lỗi nhỏ hơn `0.05 m`. Policy vẫn chỉ phát action ở 10 Hz, còn điều kiện chuyển target được kiểm tra trong callback vật lý.

Đây là một distribution shift đáng chú ý: policy được train với **một target đứng yên trong mỗi episode**, nhưng lúc đánh giá trajectory thì target liên tục chuyển sang điểm kế tiếp. Vì vậy chuyển động thường kém mượt hoặc chậm hơn pipeline spline + IK + PID chuyên biệt.

## 9. So sánh ba cách điều khiển

| Thuộc tính | IK teleport | Spline + IK + PID | PPO policy |
|---|---|---|---|
| Có động lực học | Không | Có | Có |
| Cần mô hình/kinematics | Có Jacobian | Có IK và chỉnh gain | Không cần IK khi triển khai |
| Tín hiệu vào cuối | Gán `qpos` | Motor command từ PID | Joint-position target |
| Khả năng giải thích | Cao | Cao | Thấp hơn |
| Công sức tuning | Damping/step IK | Spline và PID gains | Reward, observation, PPO hyperparameters |
| Khả năng khái quát | Theo solver | Theo thiết kế controller | Phụ thuộc dữ liệu train |
| Độ mượt | Không có chuyển động thật | Tốt nếu tune đúng | Không được bảo đảm bởi reward hiện tại |

Điểm quan trọng là đây không hoàn toàn là so sánh ngang hàng: PID ở Exercise 2 điều khiển motor qua torque-model XML, còn PPO ở Exercise 3 ra position targets cho servo MuJoCo. PPO đang học một tầng điều khiển cao hơn.

## 10. Câu trả lời ngắn cho phần lý thuyết

### Exercise 1

1. **Tăng độ rộng Lemniscate:** các điểm có thể vượt khỏi workspace hoặc gần singularity/joint limit, khiến numerical IK không hội tụ hay cho nghiệm không ổn định.
2. **Thay đổi `dt` của IK:** `dt` lớn dễ overshoot/dao động, còn `dt` nhỏ hội tụ chậm và có thể hết số vòng lặp.
3. **Numerical so với analytical IK:** numerical IK tổng quát và dễ áp dụng nhưng chậm, phụ thuộc khởi tạo và không bảo đảm nghiệm, trong khi analytical IK nhanh/chính xác nhưng khó suy ra và phụ thuộc cấu trúc robot.
4. **Giới hạn solver hiện tại:** nó không xử lý orientation, joint limits, collision, task priorities hay tối ưu toàn cục như các solver tiên tiến.

### Exercise 2

1. **Liên tục tăng $K_P$:** robot dễ overshoot, rung và cuối cùng mất ổn định.
2. **Vai trò $K_D$:** derivative term tạo damping theo tốc độ thay đổi sai số nên giảm overshoot và dao động do $K_P$ cao.
3. **Khi cần $K_I\ne0$:** cần integral khi có steady-state error do tải cố định, ma sát hoặc bias/mismatch của mô hình mà P-D không loại bỏ hết.

## 11. Hướng cải tiến cho bonus RL

Các cải tiến nên gắn với một vấn đề đo được, thay vì thay nhiều thứ cùng lúc:

1. **Thêm `qvel` vào observation:** làm quan sát gần Markov hơn và giúp policy nhận biết robot đang tiến về hay rời khỏi target.
2. **Thêm vector lỗi $p_T^B-p_E^B`:** dù có thể suy ra từ hai vị trí, feature trực tiếp này giúp mạng học dễ hơn.
3. **Reward theo tiến bộ:** thêm $d_{t-1}-d_t$ để thưởng việc giảm khoảng cách, đặc biệt hữu ích khi cần tới đích nhanh.
4. **Phạt thay đổi action:** dùng $-\lambda_a\|a_t-a_{t-1}\|^2$ để giảm giật.
5. **Phạt vận tốc/gia tốc hoặc năng lượng:** tạo chuyển động mượt và thực tế hơn, nhưng trọng số quá lớn có thể khiến robot ngại di chuyển.
6. **Action residual/delta:** policy ra độ thay đổi nhỏ quanh `qpos` hiện tại thay vì nhảy tới bất kỳ điểm nào trong toàn joint range.
7. **Train với target di chuyển hoặc chuỗi target:** giảm distribution shift khi đánh giá Lemniscate.
8. **Curriculum:** ban đầu lấy target gần, sau đó tăng dần vùng lấy mẫu.
9. **Success termination có chủ đích:** có thể kết thúc sau khi giữ target trong vài bước; cần thiết kế bonus để không khuyến khích chỉ chạm đích tức thời.
10. **Chuẩn hóa observation/reward:** giúp tối ưu ổn định khi các feature có thang đo khác nhau.

Khi làm thí nghiệm bonus, nên giữ seed và checkpoint schedule giống nhau, train nhiều seed nếu có thời gian, rồi so sánh ít nhất các đại lượng: final error, thời gian tới target, độ biến thiên action và độ mượt quỹ đạo.

## 12. Các lỗi triển khai dễ gặp

- Gán `data.qpos = new_array` hoặc `data.ctrl = action` thay vì sửa in-place bằng `[:]`; MuJoCo cần các buffer hiện có được cập nhật.
- Quên `mj_forward`/`mj_kinematics` sau khi sửa trực tiếp `qpos`, khiến `site.xpos` còn là giá trị cũ.
- Dùng trực tiếp matrix inverse trong DLS thay vì `np.linalg.solve`.
- Nhầm `dt` của IK với timestep vật lý.
- Quên đặt phần orientation error bằng 0 khi Jacobian đã ghép thành 6 hàng.
- Không clip action trước khi ánh xạ sang joint limits.
- Biến đổi vị trí nhưng quên trừ vị trí base trước khi quay sang base frame.
- Nhân quaternion sai thứ tự; phép nhân quaternion không giao hoán.
- Không chuẩn hóa quaternion sau phép nhân.
- Tính derivative PID khi lịch sử mới có một mẫu.
- Để integral tích lũy vô hạn và gặp windup.
- Đánh giá bằng stochastic action dù mục tiêu là đo policy tốt nhất; script đã dùng `deterministic=True`.
- Dùng random toàn cục nhưng kỳ vọng `env.reset(seed=...)` tái lập hoàn toàn. Skeleton gọi `np.random` trong các hàm Exercise 3 thay vì `self.np_random`, nên reproducibility theo Gymnasium seed chưa được bảo đảm đầy đủ.

## 13. Checklist hoàn thành project

- [ ] `get_lemniscate_keypoint` trả đúng shape cho cả scalar và NumPy array.
- [ ] `build_keypoints` trả mảng `(count, 3)` và không lặp điểm cuối.
- [ ] IK dừng theo norm sai số, dùng DLS ổn định và khôi phục state ban đầu.
- [ ] Quintic spline chứa đúng `start`, `end` và có chuyển động đầu/cuối êm.
- [ ] PID xử lý đúng trường hợp chỉ có một error sample.
- [ ] Robot reset có noise; target nằm đúng vùng mong muốn.
- [ ] Action `-1`, `0`, `1` ánh xạ lần lượt tới lower, midpoint, upper joint limits.
- [ ] Reward tăng khi khoảng cách giảm và cộng bonus dưới 5 mm.
- [ ] Observation đúng thứ tự và có 16 phần tử; quaternion có norm gần 1.
- [ ] `env.reset()` và `env.step()` trả đúng API Gymnasium.
- [ ] TensorBoard cho thấy episode reward tăng và tracking error giảm.
- [ ] Checkpoint đánh giá có average final error dưới 5 cm.
- [ ] Video đáp ứng thời lượng và chứa output/theoretical answers theo README.

## 14. Bức tranh tổng kết

Kiến thức cốt lõi của project là việc cùng một bài toán “đưa EE tới target” có thể được giải ở nhiều tầng:

```text
Hình học:       target position → IK → target joint configuration
Điều khiển:     joint error → PID → actuator command → robot dynamics
Robot learning: observation → PPO policy → target joint configuration
```

IK trả lời quan hệ hình học, PID đóng vòng phản hồi để chống sai lệch động lực học, còn PPO học hành vi từ reward. Hiểu rõ robot đang làm việc ở frame nào, action thực sự điều khiển đại lượng nào, và thông tin nào có trong observation là ba điểm quan trọng nhất để đọc, sửa lỗi và cải tiến project này.
