```plantuml
@startgantt
scale 1920 width

printscale weekly
[Literature Dive] lasts 10 weeks
[Train Diffusion Model] as [TrainDiff] lasts 1 week
[ILVR] lasts 1 day
[DDIB] lasts 11 days
[EGSDE] lasts 2 week
[Train Diffusion Model 2] as [TrainDiff2] lasts 1 week

Project starts 2023-04-19 
[Literature Dive] starts 2023-04-19
[TrainDiff] starts 2023-04-26
[ILVR] starts at [TrainDiff]'s end
[DDIB] starts at [TrainDiff]'s end
[EGSDE] starts at [DDIB]'s end
[TrainDiff2] starts at [TrainDiff]'s end

@endgantt
```