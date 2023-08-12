```plantuml
@startgantt
scale 1920 width

printscale weekly
[DDIB Test] as [DDIB] lasts 2 weeks
[Improve DDIB] as [DDIB-I] lasts 1 weeks
[DDIB & Simple Diffusion Colorization] as [DDIB-CM] lasts 2 weeks
[Simple Diffusion Colorization] as [GCD] lasts 3 weeks
[Paired Dataset] as [PDS] lasts 3 weeks
[Writing] lasts 6 weeks and ends at 2023-09-18

2023-07-27 to 2023-08-07 is closed

Project starts 2023-06-12
[DDIB Test] starts 2023-06-20 and ends 2023-06-26
[DDIB-I] starts at [DDIB Test]'s end
[DDIB-CM] starts at [DDIB-I]'s end
[GCD] starts at [DDIB Test]'s end
[PDS] starts at [DDIB Test]'s end

@endgantt
```