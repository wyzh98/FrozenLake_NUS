# The Froze Lake Problem and Variations
Project 1 #ME5406 Deep Learning for Robotics# @NUS

- MC 4x4

| Trail (max=1000) | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| ---------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| First Reach      | 247  | 7    | 172  | /    | 2    | /    | /    | 287  | 113  | 12   |
| Optimal Policy   | 524  | 11   | 240  | /    | 4    | /    | /    | 738  | 194  | 45   |

- SARSA 4x4

| Trail (max=100) | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| --------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| First Reach     | 6    | 4    | 11   | 9    | 17   | 9    | 8    | 7    | 8    | 9    |
| Optimal Policy  | 33   | 20   | 23   | 17   | 65   | 16   | 54   | 29   | 17   | 13   |

- Q-learning 4x4

| Trail (max=100) | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| --------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| First Reach     | 12   | 15   | 12   | 7    | 6    | 8    | 11   | 20   | 7    | 2    |
| Optimal Policy  | 19   | 26   | 24   | 20   | 14   | 14   | 15   | 30   | 15   | 14   |

- SARSA 10x10

| Trail (max=2000)      | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| --------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| First Reach           | 274  | 266  | 379  | 236  | 227  | 188  | 144  | 207  | 173  | 284  |
| First Optimal Policy  | 360  | 274  | 394  | 667  | 295  | 551  | 208  | 415  | 257  | 690  |
| Steady Optimal Policy | 883  | 436  | 611  | 952  | /    | 1020 | 818  | 1048 | 759  | 1383 |

- Q-learning 10x10

| Trail (max=2000)      | 1    | 2    | 3    | 4    | 5    | 6    | 7    | 8    | 9    | 10   |
| --------------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| First Reach           | 632  | 207  | 691  | 750  | 228  | 620  | 735  | 937  | 660  | 1004 |
| First Optimal Policy  | 961  | 773  | 928  | 1001 | 741  | 1034 | 859  | 1328 | 963  | 1407 |
| Steady Optimal Policy | 961  | 773  | 928  | 1001 | 741  | 1060 | 868  | 1337 | 963  | 1407 |

- SARSA with/without penalty 10x10

| Items               | Without penalty | With penalty |
| ------------------- | --------------- | ------------ |
| Average step length | 69.2            | 17.4         |
| Maximum step length | 2080.7          | 135.6        |
| Time spent          | 4.32s           | 1.52s        |

![Learning curve](\fig\lc_10.png)

![Success rate](\fig\sr_10.png)

![Training curve](\fig\sar_10_curve.png)

![Extracted policy and heat map](\fig\p_sar.jpg)