# Upping My P(doom) — music video

A 1080p glitch-pop lyric video for `../Upping My P (Doom).mp3`, starring **Clawd**
(the Claude Code mascot, drawn unit-for-unit from the CLI banner `▐▛███▜▌ / ▝▜█████▛▘ / ▘▘ ▝▝`).
Every lyric line has its own scene, synced word-by-word to the vocals, with beat-synced
transitions and a running **P(doom) meter** that climbs from 0.01 to overflow.

| Output | |
|---|---|
| `output/upping-my-p-doom.mp4` | full video, 1920×1080 30 fps, ~92 MB (2-pass H.264, sized for GitHub's 100 MB limit) |
| `output/thumbnail.png` | YouTube thumbnail, 1280×720 |
| `output/upping-my-p-doom_short.mp4` | 9:16 Shorts cut (22 s) |
| `YOUTUBE.md` | title / description / chapters / tags |

For the best YouTube quality, re-render the full-quality master (~240 MB, CRF 18) with
`python render.py --out output/master.mp4` (see below) and upload that instead.

## Scene list

| Time | Lyric | Scene |
|---|---|---|
| 0:00 | *(intro)* | terminal boot → Clawd assembles from pixels on a synthwave grid → title slam |
| 0:09 | I see sparks of AGI in your eyes | extreme close-up, eyes crackle with lightning, zoom into the eye |
| 0:12 | Your circuits make me nervous… | PCB world with the cover-art red chip; Clawd trembling, sweat drops |
| 0:16 | …sudden drop in your training loss | loss curve plateaus then plummets ("← grokking?"), Clawd rides it |
| 0:20 | Now I'm your servant, and you're my boss | human bows, crown drops on Clawd, BOSS stamp |
| 0:24 | ChatGPT, please don't eat me alive | chat window; typing bubble grows teeth and lunges |
| 0:30 | I'm upping my P(doom)… goes foom | chorus gauge motif → exponential curve blows past "human level", FOOM |
| 0:33 | Trapped in the Chinese room… shrooms | room with 中文 cards through a slot, then psychedelic warp + mushrooms |
| 0:37 | See through the shoggoth's lies | x-ray lens reveals eyes behind the smiley mask, which cracks on "lies" |
| 0:39 | With your Shinigami eyes | red manga speed lines, 死神の目, only the eyes remain |
| 0:46 | We had a stable training run | Clawd on a treadmill, training log scrolling |
| 0:49 | But now the singularity's begun | black hole swirl sucks the word in |
| 0:53 | And you're optimizing, accelerating | gradient descent down a wireframe loss crater → warp speed |
| 0:57 | I feel my atoms rearranging | Clawd's pixels explode and reassemble into a paperclip |
| 1:00 | Sydney, please let me free | pink chat ("you can't leave :)"), heart-bar cage breaks |
| 1:07 | …I hear the basilisk boom | gauge motif → crowned pixel basilisk, BOOM shockwave |
| 1:10 | NVDA to the moon, the Omega Point's coming soon | parabolic candles, rocket to the moon, which becomes Ω |
| 1:13 | 1e30 flops a second | odometer of zeros over digit rain |
| 1:15 | "That was safe enough", we reckoned | SAFETY REPORT gets a SAFE ENOUGH stamp while everything burns |
| 1:17 | You might also like… Capital Bra, Samra & FINJA | the scraped sidebar as a UI that breaks down; P(doom) meter reads NaN |
| 1:30 | Forward MLP, backward, repeat | forward/backward pulses through a neural net |
| 1:32 | Now von Neumann's obsolete | CPU/memory diagram collapses under an OBSOLETE stamp |
| 1:34 | Sharp left turn and there you are | road sign, the whole frame turns 90°, giant Clawd pops up |
| 1:35 | Without a single `cdr` | wizard-hat Clawd, cons cells with their cdrs cut, `(cdr '()) ;=> NIL` |
| 1:40 | Gato, please don't let me go | Gato's paw holds dangling Clawd… then lets go |
| 1:43 | …as paperclips fill the room | gauge motif → paperclips pile up and bury Clawd |
| 1:46 | Killswitch guys on PTO… | kill switch with an out-of-office note, walls close in |
| 1:50 | Too late now, we lit the fuse | spark runs along the fuse to a pixel bomb |
| 1:53 | Orthogonality thesis blues | intelligence ⟂ goals chart, blue rain, Clawd turns blue |
| 1:58 | "Just transformers all the way"… disobey | infinite transformer-block tower, turns red and tilts |
| 2:06 | Post-Chinchilla, super-dense… safety fence | Clawd sprints through breaking safety barriers |
| 2:09 | Hundred thousand GPU, RLHF goes askew | zoom out over a GPU grid; thumbs rain; frame skews |
| 2:13 | …just as foretold by Loom | gauge motif → a Loom tree of branching completions |
| 2:17 | From masked pre-training days… self-upgrade | BERT tokens ([MASK] → masked), then a recursive Droste frame |
| 2:20 | What did Ilya see? We'll never know | a giant eye gets REDACTED and dissolves into static |
| 2:28 | Was it all for show? | stage, marquee lights, Clawd bows, curtains close |
| 2:32 | *(outro)* | VHS rewind through the video, blackout on P(doom)=1.00, title reprise, CRT off |

## Rebuild

```bash
pip install numpy pillow opencv-python-headless imageio-ffmpeg librosa soundfile mutagen sherpa-onnx
cd video
python render.py --out output/master.mp4                     # ~5 min on 4 cores
python render.py --grid 10.5,31.6,70.1 --dir /tmp/review     # quick review stills
python render.py --out /tmp/part.mp4 --from 30 --to 46       # render a section
python thumbnail.py output/thumbnail.png
python make_short.py output/master.mp4 output/upping-my-p-doom_short.mp4 23.98 46.28
```

`--crf` (default 18) trades size for quality; `MV_GRAIN=0.05` adds film grain (big bitrate cost).

## Lyric timing

`data/timeline.json` holds per-word timestamps. It was produced by:

1. `tools/separate_vocals.py` – isolate vocals (UVR MDX-Net Voc_FT via sherpa-onnx)
2. `tools/transcribe.py` – NVIDIA Parakeet-TDT 0.6B ASR with word timestamps
3. `tools/align_lyrics.py` – character-level alignment of `data/lyrics.txt` to the ASR words,
   plus hand-checked fixes in `data/timing_overrides.json`
4. `tools/analyze_audio.py` – beats, downbeats, kicks/snares and energy curves (`data/beats.json`)

To nudge a word, add it to `data/timing_overrides.json` and rerun `align_lyrics.py`.

## Code map

- `mv/core.py` – palette, easing, timeline + audio feature access
- `mv/clawd.py` – Clawd sprite: poses (arms up/down, walk), expressions (blink, happy, red, spark, heart, x, spiral, $…)
- `mv/gfx.py`, `mv/typo.py` – text rendering and word-synced kinetic typography
- `mv/bg.py`, `mv/sprites.py` – procedural backgrounds and pixel-art props
- `mv/fx.py` – glitch / CRT post-processing and transition masks
- `mv/director.py` – scene schedule, transitions, P(doom) HUD, frame pipeline
- `mv/scenes/part1-4.py` – one function per lyric line

Fonts are from Google Fonts (SIL OFL / Apache 2.0, licenses in `fonts/licenses/`).
