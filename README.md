# [SIGGRAPH Asia 2023] Discovering Fatigued Movements for Virtual Character Animation

<p align="center">
<a href="">Noshaba Cheema</a> | 
<a href="">Rui Xu</a> |
<a href="">Nam Hee Kim</a> |
<a href="">Perttu Hämäläinen</a> |
<a href="">Vladislav Goliyanik</a> </br>
<a href="">Marc Habermann</a> |
<a href="">Christian Theobalt</a> | 
<a href="">Philipp Slusallek</a> </br>
<b>SIGGRAPH Asia 2023 Conference Paper</b>
</p>

<img src ="Media/fatigue_teaser.gif" width="100%">

<p align="center">
<b>
<a href="https://vcai.mpi-inf.mpg.de/projects/FatiguedMovements/">Website</a> | 
<a href="https://dl.acm.org/doi/pdf/10.1145/3610548.3618176">Technical Paper</a> |
<a href="https://www.youtube.com/watch?v=FVOWVz0k1dI">Video</a>
</b>
</p>

------------

<p align="center">
<b>
Abstract
</b>
</br>
<i>Virtual character animation and movement synthesis have advanced rapidly during recent years, especially through a combination of extensive motion capture datasets and machine learning. A remaining challenge is interactively simulating characters that fatigue when performing extended motions, which is indispensable for the realism of generated animations. However, capturing such movements is problematic, as performing movements like backflips with fatigued variations up to exhaustion raises capture cost and risk of injury. Surprisingly, little research has been done on faithful fatigue modeling. To address this, we propose a deep reinforcement learning-based approach, which -- for the first time in literature -- generates control policies for full-body physically simulated agents aware of cumulative fatigue. For this, we first leverage Generative Adversarial Imitation Learning (GAIL) to learn an expert policy for the skill; Second, we learn a fatigue policy by limiting the generated constant torque bounds based on endurance time to non-linear, state- and time-dependent limits in the joint-actuation space using a Three-Compartment Controller (3CC) model. Our results demonstrate that agents can adapt to different fatigue and rest rates interactively, and discover realistic recovery strategies without the need for any captured data of fatigued movement.
</p></i>

------------

### BibTex

If you find this code useful in your research, please cite:

```
@inproceedings{cheema2023fatigue,
  title = {Discovering Fatigued Movements for Virtual Character Animation},
  author = {Cheema, Noshaba and Xu, Rui and Kim, Nam Hee and H{\"a}m{\"a}l{\"a}inen, Perttu and Golyanik, Vladislav and Habermann, Marc and Theobalt, Christian and Slusallek, Philipp},
  booktitle={ACM SIGGRAPH 2023 Conference Proceedings},
  year = {2023}
}
```

### About this repository

This repository contains an implementation of the code for our [SIGGRAPH Asia paper](https://dl.acm.org/doi/pdf/10.1145/3610548.3618176) "Discovering Fatigued Animations for Virtual Character Animation".

### Installation

Download the Isaac Gym Preview 4 release from the [website](https://developer.nvidia.com/isaac-gym), then
follow the installation instructions in the documentation. We highly recommend using a conda environment 
to simplify set up.

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 4
install instructions if you have any trouble running the samples.

Once Isaac Gym is installed and samples work within your current python environment, install this repo:

```bash
pip install -e .
```

------------

### Running pre-trained models

<details>
<summary><b>Run Fatigued Tornado Kick</b></summary>

To run the pre-trained fatigued Tornado kick with (F, R, r)=(2.0, 0.05, 1.0) use the following command:


<pre>
python train.py seed=477 task=HumanoidAMPHandsFatigue train=HumanoidAMPPPOLowGPFatigue headless=False num_envs=1 test=True checkpoint=runs/wushu_fatigue/nn/wushu_fatigue_6000.pth task.env.motion_file=amp_humanoid_wushu_kick.npy task.env.useFatigue=True task.env.fatigueF=2.0 task.env.fatigueR=0.05 task.env.fatigue_r=1.0 task.env.visualizeFatigue=True task.env.showSkyWall=True
</pre>
</details>

<details>
<summary><b>Run Non-Fatigued Tornado Kick Expert</b></summary>

To run the pre-trained Tornado kick expert not yet fine-tuned on fatigue use the following command:


<pre>
python train.py seed=1337 task=HumanoidAMPHandsFatigue train=HumanoidAMPPPOLowGPFatigue headless=False checkpoint=runs/wushu_expert/nn/wushu_expert_4000.pth task.env.motion_file=amp_humanoid_wushu_kick.npy task.env.showSkyWall=True test=True num_envs=1
</pre>
</details>

<details>
<summary><b>Run Goal-Conditioned Fatigued Locomotion:</b></summary>

To run the pre-trained goal-conditioned fatigued locomotion with (F, R, r)=(1.0, 0.01, 1.0) use the command:

<pre>
python train.py seed=537 task=HumanoidAMPHandsFatigue train=HumanoidAMPPPOLowGPFatigue headless=False num_envs=64 test=True checkpoint=runs/run_walk_goal/nn/run_walk_goal_4000.pth task.env.motion_file=amp_humanoid_walk_run.yaml task.env.useFatigue=True task.env.fatigueF=1.0 task.env.fatigueR=0.01 task.env.fatigue_r=1.0 <b>train.params.config.task_reward_w=1.0 train.params.config.disc_grad_penalty=5.0 task.env.TargetObs=True</b> task.env.visualizeFatigue=True task.env.showSkyWall=True
</pre>

**Note:** Since this is a goal-conditioned task where the character runs to a certain target, ```task_reward_w``` should be greater than 0 (we used 1.0), and ```TargetObs``` needs to be set in order for the target be included in the observation space. For locomotion tasks we additionally set ```disc_grad_penalty=5.0``` like in the original Isaac Gym Envs implementation for locomotion and hopping tasks.

</details>

<details>
<summary><b>Run Goal-Conditioned Fatigued Ant</b>
</summary>

To run the pre-trained goal-conditioned fatigued ant with (F, R, r)=(0.4, 0.1, 1.0) use the command:

<pre>
python train.py task=AntFatigue task.env.fatigueF=0.4 task.env.fatigueR=0.1 task.env.fatigue_r=1.0 test=True num_envs=1 headless=False task.env.followCam=True checkpoint=runs/ant_fatigue_0.4_0.1_1.0/nn/ant_fatigue_0.4_0.1_1.0.pth task.env.showSkyWall=True
</pre>
</details>

### Training / Fine-tuning models

<details>
<summary><b>Train Tornado Kick Expert</b></summary>

To train an expert model, which can later be fine-tuned with fatigue run the following command:

<pre>
python train.py seed=476 task=HumanoidAMPHandsFatigue train=HumanoidAMPPPOLowGPFatigue headless=True task.env.motion_file=amp_humanoid_wushu_kick.npy max_iterations=4000 <b>task.env.randomizeFatigueProb=1 task.env.randomizeEveryStep=True</b> experiment=wushu_expert
</pre>

**Note:** For training or fine-tuning we set ```task.env.randomizeFatigueProb=1``` to have a random fatigue initialization at every environment reset for maximizing diversity during training. Additionally, for expert training specifically ```task.env.randomizeEveryStep=True``` needs to be set so that the fatigued motor units ```MF``` in the observation space are random as no "fatigue" exists yet. This is similar to domain randomization.

</details>

<details>
<summary><b>Finetune Tornado-Kick Expert with Fatigue</b></summary>

To train an expert model, which can later be fine-tuned with fatigue run the following command:

<pre>
python train.py seed=477 task=HumanoidAMPHandsFatigue train=HumanoidAMPPPOLowGPFatigue headless=True experiment=finetune_wushu_fatigue checkpoint=runs/wushu_expert/nn/wushu_expert_4000.pth task.env.motion_file=amp_humanoid_wushu_kick.npy max_iterations=6000 task.env.useFatigue=True <b>task.env.randomizeFatigueProb=1</b>
</pre>

**Note:** For training or fine-tuning we set ```task.env.randomizeFatigueProb=1``` to have a random fatigue initialization at every environment reset for maximizing diversity during training. Good (F, R, r)-values for inference for this motion are (2, 0.05, 1).

</details>

<details>s
<summary><b>Finetune Goal-Conditioned Locomotion with Fatigue</b></summary>

To fine-tune an expert model trained on the run motion with fatigue, we use the following command:

<pre>
python train.py seed=537 task=HumanoidAMPHandsFatigue train=HumanoidAMPPPOLowGPFatigue headless=True experiment=finetune_locomotion_fatigue checkpoint=runs/run_goal_expert/nn/run_goal_expert_2000.pth <b>task.env.motion_file=amp_humanoid_walk_run.yaml</b> max_iterations=4000 task.env.useFatigue=True <b>task.env.randomizeFatigueProb=1 train.params.config.task_reward_w=1.0 train.params.config.disc_grad_penalty=5.0 task.env.TargetObs=True</b>
</pre>

**Note:** Since this is a goal-conditioned task where the character runs to a certain target, ```task_reward_w``` should be greater than 0 (we used 1.0), and ```TargetObs``` needs to be set in order for the target be included in the observation space. For locomotion tasks we additionally set ```disc_grad_penalty=5.0``` like in the original Isaac Gym Envs implementation for locomotion and hopping tasks. Futhermore, for training or fine-tuning we set ```task.env.randomizeFatigueProb=1``` to have a random fatigue initialization at every environment reset for maximizing diversity during training. Also note that for fine-tuning we use the yaml file which includes different locomotion modes. It is interesting that to see that while the agent is fatigued, it automatically switches to the walking mode within a single policy.

</details>

<details>
<summary><b>Train Goal-Conditioned Fatigued Ant</b>
</summary>

To train the goal-conditioned fatigued ant use the command:

<pre>
python train.py task=AntFatigue task.env.fatigueF=0.4 task.env.fatigueR=0.1 task.env.fatigue_r=1.0 <b>task.env.randomizeFatigueProb=1</b> headless=True experiment=fatigued_ant
</pre>

**Note:** For training or fine-tuning we set ```task.env.randomizeFatigueProb=1``` to have a random fatigue initialization at every environment reset for maximizing diversity during training.

</details>

### Visualizing the results in Unity

**Stream from Isaac to Unity**

To stream from Isaac to Unity and interactively change the (F, R, r)-values during inference set ```task.env.unity_stream=True``` with the command that you want to run from Isaac. In the unity_vis project use the **"network_vis_isaac_streaming"**-scene and hit play to visualize the Isaac output in real-time in Unity. You can use the sliders to change the (F, R, r)-parameters.

**Record a JSON-file and play it in Unity**

To record JSON-file with the trajectory of the character set ```task.env.record_poses=True```. This will generate the JSON-file, which you can put into the "StreamingAssets" folder in the unity_vis project. Then in the scene **"isaac_load_and_play"** in the "AnimationPlayer"-object enter the json file name under Clipfilename and hit "Replace Animation" and then hit play.

------------

### Acknowledgments

The Unity visualization code is based on the repositories [unity_socket_vis](https://github.com/eherr/unity_socket_vis) and [anim_utils](https://github.com/eherr/anim_utils) by [Erik Herrmann](https://github.com/eherr/).
