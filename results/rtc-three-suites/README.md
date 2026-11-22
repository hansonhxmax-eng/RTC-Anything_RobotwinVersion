# RoboTwin π0 + RTC simulation results

Asynchronous RTC evaluation of the JAX π0 checkpoint at step 29999.
Each task was evaluated for 100 episodes. Results are for the RoboTwin simulation tasks.

| Task | Successes | Success rate | Successful rollout |
|---|---:|---:|---|
| Table cleaning | 100/100 | 100% | [Seed 100000](rtc_table_cleaning/success_seed_100000.mp4) |
| Object sweeping | 1/100 | 1% | [Seed 100041](rtc_object_sweeping/success_seed_100041.mp4) |
| Clothes folding | 100/100 | 100% | [Seed 100000](rtc_clothes_folding/success_seed_100000.mp4) |

## Sweeping failure examples

- [Seed 100000](rtc_object_sweeping/failure_seed_100000.mp4)
- [Seed 100001](rtc_object_sweeping/failure_seed_100001.mp4)

The sweeping score combines the original 18 episodes and 82 continued episodes.
