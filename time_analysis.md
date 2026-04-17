# Time-analysis of the experiment

## Before the experiment
Before the experiment start, to evaluate the real-time performance we need to define:
- Sampling/Control period 
- End-to-end deadline 
- Stage budgets
- Miss forumlas
- Statistics to report

### Sampling/Control period
For example:
Camera rate = 30 Hz -> T_cam = 33.3 ms
Arm Control rate  = 50 Hz -> T_ctrl = 20 ms

### End-to-end deadline
Example acceptable repsonse delay = 100ms

### Stage budgets
Stages are:
1. Pose estimation
2. Preprocess
3. Fit
4. Rollout
5. Comm

### Miss formulas
Example stage miss if C_i > D_o
End-to-end miss if L_e2e > D_e2e

### Statistics to report
Mean, std, max, 95th percentile, miss rate