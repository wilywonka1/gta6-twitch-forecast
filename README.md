[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/wilywonka1/gta6-twitch-forecast/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/wilywonka1/gta6-twitch-forecast/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

# Unlock-cascade quantile forecaster

An entry for make.fun's ["Forecast GTA VI's launch-day Twitch viewers hour by hour"](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) bounty (gta6-twitch-ccu-forecast-01).

The program forecasts how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC), after the first — five quantiles (5%, 25%, 50%, 75%, 95%) for each of hours 2–24, scored by mean pinball loss on the log scale.

How it works: past launches trace a repeatable shape once normalised by their own first hour. For every comparable launch in the case, the program takes the release-day log ratios `ln(viewers_h / viewers_hour1)`, weights each comparable by how close its hour-1 audience is to this case's hour-1 print (a Gaussian kernel on log scale — first-hour scale is a strong proxy for unlock regime), and reads the five quantiles straight off the weighted empirical distribution per hour. A leave-one-out backtest over the comparables scored ≈0.41 against ≈0.93 for the flat persistence baseline.

## Run it

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

`main.py` (Python 3.13, standard library only) reads every `*.in` case file in the input directory and writes one `<id>.out` forecast per case. In the grader it runs as `python3 /bot/main.py` with `/input` and `/output`.
