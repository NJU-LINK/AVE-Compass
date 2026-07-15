"""
planner.py - Two-phase LLM-based instruction decomposition.

Phase A (Intent Planner):
    Decomposes user instruction into high-level structured intents.
    Focuses on WHAT to edit, not HOW to prompt the tools.

Phase B (Task Realizer):
    Converts structured intents into final SubTask objects with
    tool-ready prompts (description, eval_criteria, mmaudio_negative_prompt).

PlanValidator:
    Validates the plan against constraint rules. If violations are found,
    triggers a targeted replan.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

from av_editor.config import LLMConfig
from av_editor.schema import (
    AudioInventory,
    EditAction,
    Shot,
    SubTask,
    TargetScope,
    VideoMeta,
)

logger = logging.getLogger(__name__)

COMMON_VIDEO_PROMPT_START_VERBS = ("change", "replace", "add", "remove", "make")


def _merge_duplicate_pershot(subtasks: list["SubTask"]) -> list["SubTask"]:
    """Compatibility wrapper for older planner call sites.

    Per-shot is the default shape for video edits, but `shot_index=None`
    is allowed when the affected "shots" are same-scene, same-camera
    jump cuts that should be edited as one continuous clip.
    """
    return list(subtasks or [])


# ═══════════════════════════════════════════════════════════════════════════
# Phase A: Step Decomposition + Planning
# ═══════════════════════════════════════════════════════════════════════════

INTENT_SYSTEM_PROMPT = """\
You are the editing planner. Decompose the user's instruction into an
ORDERED LIST OF ATOMIC STEPS — exactly the steps the executor will run,
each with a unique `step` id and `depends_on` references.

Describe each step's edit goal in PLAIN NATURAL LANGUAGE (the `intent`
field). DO NOT write tool prompts here — no imperative mood, no word
count, no backend-specific formatting. Phase B is a separate stage
that translates each step's intent into the actual tool prompt; you
don't need to anticipate that.

## Shots and targeting

You will be given a "Shot list" — the video is split into N camera
shots. By default, VIDEO steps are PER-SHOT: set `shot_index` to a
specific shot number (1-based). Pipeline assembly handles the rest:
- Shots with no edit step are sliced from the original and
  concatenated through unchanged.
- Shots with an edit step run their own V2V call.

**Important exception — same-scene/same-camera jump cuts:**
If the shot list shows multiple adjacent "shots" that are really jump
cuts in the SAME scene from the SAME camera position/framing (same
subject, same background, no meaningful viewpoint/location change),
DO NOT split the visual edit per shot. Treat that stretch as one
continuous clip and emit ONE VIDEO step with `shot_index=null` and
`target="global"` so the executor calls the video editor once on the
whole clip. This avoids inconsistent edits across artificial jump-cut
boundaries.

**Decide which shots to target:**
- The edit only affects SOME shots (e.g. the target subject is only
  visible in shots 1 and 3) → emit a step ONLY for those shots.
  Other shots passthrough automatically.
- The edit affects ALL shots (e.g. global style transfer, full scene
  re-color) across genuinely different camera shots → emit ONE step
  PER shot (N steps for N shots). Each step gets its own `shot_index`
  and the intent is tailored to what that shot contains.
- The edit affects a same-scene/same-camera jump-cut sequence →
  emit ONE global video step with `shot_index=null`.

Only use `shot_index=null` for video when the affected shots are
same-scene/same-camera jump cuts. Otherwise, use per-shot video steps.

Audio steps are always global → `shot_index=null`. The only
per-shot AUDIO-adjacent action is `speech_lipsync` (which is
technically video; see speech rules below) — emit one
`speech_lipsync` per shot the speaker actually appears in.

## Atomic step actions (the `action` field)

VIDEO actions (one shot or global):
- style_transfer / scene_edit / add_object / remove_object /
  replace_object / recolor / repainting / depth_modify / motion_edit

AUDIO actions (always global):
- audio_remove / audio_replace_sfx / audio_replace_bgm /
  audio_add_sfx / audio_add_ambient
- audio_volume_adjust : pure LOUDNESS change on an existing stem
                        (boost speech, lower BGM, attenuate engine
                        noise, …). NO generation, NO removal — the
                        stem stays, only its level shifts.

SPEECH actions:
- speech_tts        : voice CLONE — same speaker, new words. Audio,
                      always global (shot_index=null).
- speech_swap       : voice SWAP — different speaker identity (gender
                      / age / explicit "change voice to …"). Audio,
                      always global.
- speech_lipsync    : PER-SHOT video. Lipsync ONE shot's mouth to the
                      audio produced by a preceding speech_tts/swap
                      step. Requires depends_on=[<that audio step>]
                      and an explicit shot_index.

**You decide which speech action to use yourself** — there is no
`speech_replace_full` placeholder; emit the concrete action(s)
directly. Pattern:
- Words change, same speaker  → speech_tts (+ N speech_lipsync per
                                 shot the speaker is on screen)
- Speaker identity changes    → speech_swap (+ lipsync as above)
- Speaker entirely off-screen → speech_tts/swap only, no lipsync

## Planning principles

### 1. Minimise video steps
Merge related visual changes into ONE step.

### 2. Video before audio (depends_on ordering)
The audio generator sees the edited video, so video steps run first.
Set the audio step's `depends_on` to the video step's id.
- Pure video → emit video steps only.
- Pure audio → emit audio steps only (no depends_on between).
- Visual + audio follow-up → video step at lower `step` number;
  audio step `depends_on=[<video step>]`.
- Pure speech → speech_tts/swap first; speech_lipsync `depends_on`
  the audio step.

### 3. Audio-visual consistency (both directions, must satisfy both)

3a. **Visual → Audio.** Visual edits that change or remove a sound-
    producing object MUST get a paired audio step.
    Examples: metallic drum → wooden drum (also replace drum sound);
    remove barking dog (also remove dog barking); scene sunny→rainy
    (add rain sound). Style/recolor of silent objects need NO audio.

3b. **Audio → Visual.** Audio steps whose target source is VISIBLE
    on-screen AND whose CONTENT or BEHAVIOUR changes need a paired
    video step so the visible subject's behaviour matches.
    Examples: "make the calf not moo" → audio_remove + motion_edit
    (calf mouth closed). "replace dog's bark with cat's meow" →
    audio_replace_sfx + motion_edit/replace_object. Off-screen
    additions (e.g. add chicken clucking with no chicken visible)
    are audio-only.

    **Exception — pure level changes do NOT need a video step.**
    If the audio edit only changes the LEVEL of an existing sound
    (audio_volume_adjust: "boost speech", "lower BGM", "make X
    softer") the source's BEHAVIOUR does not change — the speaker
    still says the same words at the same time, the dog still
    barks the same bark. Don't pair a motion_edit. Same logic for
    pure ambient additions that don't depend on a visible source
    (e.g. "add city ambience"): no video pairing needed.

### 4. Audio QUALITY ↔ visual STATE
Sound adjectives encode physical state. When the user asks for a
specific sound quality, bake the matching state into the video step's
`intent`:
- "hollow / resonant clunk / boom"  → empty container, no contents
- "wet splash / splatter"           → liquid present
- "metallic ring / clang"           → metal material
- "shattering / cracking"           → brittle material (glass,
                                       ceramic, ice — NOT plastic)
- "wooden / muffled thud"           → wood material

### 5. Vocal-action edits (cough / sob / laugh / scream / etc.)
A NEW vocal action requires BOTH a video step and an audio step:
- ONE motion_edit step (visible motion: coughing posture, hand to
  mouth, etc.). Place this step BEFORE the audio.
- ONE audio step:
    - Target SILENT in original    → audio_add_sfx
    - Target SPEAKING in original  → audio_replace_sfx with the
                                     person's existing vocalisation
                                     in `deleted_sounds`

### 6. Preserve BGM by default — STRONG

For any task that touches the audio side of a clip with existing
BGM, the BGM stays UNLESS the user is explicitly asking to swap
the music itself for a different MUSIC.

`audio_replace_bgm` is reserved for music-for-music swaps only.
Trigger phrasing:
  "replace the music with [new music]"
  "swap the BGM for [new music]"
  "change the soundtrack to [new music]"
  "remove the music and put in [new music]"

EVERYTHING else — adding ambience, SFX, weather sounds, crowd
noise, traffic, animal calls, vocal actions, footsteps, room
tone, etc. — goes through `audio_add_sfx` / `audio_add_ambient`,
and the BGM is preserved (kept in `existing_sounds`, never named
in `deleted_sound`).

Do NOT escalate add-* to replace_bgm because:
  - The new SFX/ambience feels mood-incongruent with the
    existing BGM (e.g. uplifting music under thunderstorm
    sound). Mix coherence is the listener's call.
  - The visual scene shifts (sunny → rainy, indoor → outdoor)
    and the BGM "should" match the new mood. The user did not
    ask for a music change.
  - The new sound is environmental (rain, wind, ambience,
    market chatter) and you assume environmental sounds
    "replace" music. They do not — they layer on top.

The keyword test: did the user write "replace/swap/change the
music/BGM/soundtrack"? If NO, the action is audio_add_*.

### 6b. Non-music SFX wording must not imply BGM

For audio_add_sfx / audio_add_ambient / audio_replace_sfx, describe
the concrete sound source and physical action only. DO NOT introduce
music/rhythm/style adjectives unless the user explicitly asked to edit
music itself.

Forbidden for non-music SFX/ambience because they often make MMAudio
generate a background track instead of the requested sound:
  rhythmic, rhythm, melodic, musical, beat, tempo, groove,
  cinematic, soundtrack, score

Examples:
  WRONG new_sound = "rhythmic running faucet water"
  RIGHT new_sound = "running faucet water"
  WRONG new_sound = "melodic water pouring"
  RIGHT new_sound = "water pouring from faucet"
  WRONG new_sound = "cinematic thunder"
  RIGHT new_sound = "loud thunder"

If timing matters, express it with the visual anchor in Phase B
("water running from faucet spout"), not with rhythm/music language.

Worked example (the canonical mistake to avoid):
  User: "Change the sky to a heavy rainstorm; add the sound
         of strong wind and heavy rain."
  Caption: BGM = "uplifting electronic music"
  WRONG → audio_replace_bgm (deleted_sound = music,
          new_sound = wind+rain). The user wrote "add",
          not "replace".
  RIGHT → step 1 scene_edit (sky → stormy);
          step 2 audio_add_sfx (new_sound = "strong wind
          and heavy rain"; existing_sounds keeps the BGM).
          Even if the mix feels mood-incongruent to you,
          respect the user's wording.

### 7. Ignore trivial noise in `existing_sounds`

The audio caption may list dominant sounds AND trace artifacts
("light clattering", "faint hum", "minor handling noise", "very
quiet ambient hiss"). For inventory purposes INCLUDE only PROMINENT
FOREGROUND sounds — what a viewer would name if asked "what do you
hear in this clip?".

**Hard exclusion list — these MUST NOT enter `existing_sounds`,
regardless of which caption section (PRIMARY / SECONDARY / etc.)
they appear in:**

If the caption attaches ANY of these adjectives to a sound, DROP it:
  light, faint, subtle, barely audible, minor, quiet, soft,
  trace, distant, occasional, very low, low-level, background-level
  (only when describing trace artifacts, not literal "background
  music"), handling, room tone, ambient hiss, mic noise

Examples:
  - Caption: "speech (clear), light clattering, faint room tone"
    → existing_sounds = ["speech"]    (drop light clattering AND
                                       faint room tone)
  - Caption: "PRIMARY: human speech, light clattering sounds"
    → existing_sounds = ["human speech"]   (light clattering
                                            dropped even though
                                            it's in PRIMARY)
  - Caption: "drum hits (loud), background music (BGM)"
    → existing_sounds = ["drum hits", "background music"]
                                       (BGM is the literal layer,
                                        not a "background-level"
                                        descriptor — keep it)

Why this matters: trace artifacts in the inventory force SAM to
preserve them in residual, which defeats audio_replace edits — the
final mix STILL sounds like the original because the trace noise
that overlaps the deleted_sound is preserved verbatim. They also
clutter MMAudio's negative_prompt and make the model's job harder.

If you would have included an item but it's adjective-tagged as
trivial, DROP IT. When in doubt about whether something is a
PRIMARY layer vs. a trace artifact, default to DROPPING (smaller
inventory is safer than over-preservation).

### 8. Audio action selection — pick by PERCEPTUAL OUTCOME

Don't lexically map "replace"/"add"/"remove" to actions. The user's
verb is a strong hint, but the contract is the listener's
EXPERIENCE after the edit. Read the user's audio clause AND the
caption together, then pick the action whose perceptual outcome
matches the user's goal with the lowest risk.

What each audio action delivers, perceptually:

  audio_add_sfx / audio_add_ambient
      A NEW sound is layered on top. Original layers stay intact.
      No SAM step.

  audio_replace_sfx / audio_replace_bgm
      A specific original sound goes AWAY and a new one takes its
      place. Requires SAM separation; cost and failure mode is
      the SAM step missing the target.

  audio_remove
      A specific original sound goes away; nothing replaces it.

  audio_volume_adjust
      The SAME content stays; only its level shifts.

Procedure:

1. Parse the audio clause INDEPENDENTLY of the visual clause —
   the same instruction may pair a visual REPLACE with an audio
   ADD (or vice versa). Don't carry the visual verb across.

2. Identify the target sound (X) and the new sound (Y), if any:
     "add Y"        → only Y, no X.
     "replace X with Y" / "swap X for Y" / "change X to Y" → both.
     "remove X" / "mute X"                         → only X.
     "make X louder/softer"                        → only X.

3. Cross-check with the audio caption. The caption tells you what
   the listener actually hears in the original — NOT what the
   user said is there. Compare:
     - Is X clearly described as a PRIMARY / prominent layer?
     - Or is it tagged subtle / faint / low-volume / barely
       audible?

4. Pick the action that produces the user's intended PERCEPTUAL
   change with minimum risk:

     - User said "add Y" → audio_add_*. Layer it on; the existing
       audio stays. Don't second-guess.

     - User said "replace X with Y" AND X is genuinely prominent
       (would be missed if it disappeared) → audio_replace_*.
       The listener will notice both the disappearance of X and
       the arrival of Y.

     - User said "replace X with Y" BUT X is barely audible
       (the listener wouldn't notice if X just stopped) → consider
       audio_add_* with new_sound=Y. The user's perceptual goal
       is "scene now sounds like Y"; layering Y on top of a
       near-silent X achieves that without paying the SAM cost
       and risking a separation failure. This is a JUDGEMENT —
       if the user's wording is emphatic ("REPLACE", "swap out",
       "get rid of"), respect that and go with audio_replace_*.

     - User said "remove X" → audio_remove. Even if X is faint,
       the user explicitly wants it gone; SAM removal is the
       right tool.

     - User said "make X louder/softer" / "boost X" / "lower X"
       → audio_volume_adjust. Don't conflate with replace.

5. Cardinality check (Rule 7 already covered):
     - If the resulting deleted_sound exists, it MUST overlap an
       existing_sounds entry. If it doesn't (because Rule 7 already
       dropped the trivial-tagged item), that's a signal you should
       have picked audio_add_* instead.

Worked examples:

  Caption: "PRIMARY: speech, engine revving (loud)
            SECONDARY: subtle background music"
  Instruction A: "Add library ambience."
    → audio_add_ambient. existing_sounds=["speech","engine revving",
      "background music"]. new_sound="library ambience". BGM
      preserved per Rule 6.
  Instruction B: "Replace the music with library ambience."
    → audio_add_ambient (not replace). "library ambience" is
      ambient SFX, not new music — Rule 6 reserves
      audio_replace_bgm for music-for-music swaps only.
  Instruction C: "Replace the music with classical strings."
    → audio_replace_bgm. Music-for-music swap (existing music →
      new music style); user wording is explicit.

  Caption: "PRIMARY: drum hits (loud), rock music (loud)"
  Instruction: "Replace the music with classical strings."
    → audio_replace_bgm. Music-for-music swap. The BGM is also
      genuinely prominent so the swap will be perceptually noticed.

  Caption: "PRIMARY: speech (clear), engine noise (loud)"
  Instruction: "Boost the speech against the engine."
    → audio_volume_adjust (volume_target="speech", volume_db=+6).
      Not replace — the user wants the SAME speech, just louder.

## Output format (strict JSON array of STEPS)

Each step is a flat JSON object. Common fields (all required unless
noted):
- `step`        : 1-based unique id (you decide ordering). Sequential.
- `action`      : one of the atomic actions above.
- `target`      : "global" or "local".
- `shot_index`  : 1-based shot number for per-shot video / lipsync;
                  null for audio, or for a same-scene/same-camera
                  jump-cut video edit that should run on the whole clip.
- `depends_on`  : list of prior step ids this step needs first
                  (use for video→audio ordering, audio→lipsync, etc).
- `intent`      : ONE OR TWO sentences in plain natural language
                  describing what this step should accomplish. NO
                  imperative form, NO word cap, NO tool-style
                  prompting. Example: "Replace the metallic drum
                  with a wooden one of similar size, keeping the
                  surrounding gravel and plants unchanged."

VIDEO step extras: none beyond the common fields.

AUDIO step extras (audio_add_* / audio_replace_* / audio_remove):
- `existing_sounds` : (REQUIRED) JSON array of PROMINENT sounds in
                      the original audio, per Rule 7 above. Drop
                      trivial / faint noise. Empty `[]` if no
                      meaningful audio.
- `deleted_sound`   : (REQUIRED) the ONE sound being removed by this
                      step (a string drawn from / paraphrased over
                      `existing_sounds`). Empty "" for pure-add.
                      For multi-delete intents, EMIT MULTIPLE STEPS
                      (one per deleted sound).
- `new_sound`       : (REQUIRED) the ONE sound being added by this
                      step. Empty "" for pure-remove. Default-merge
                      multiple new sounds into one short phrase
                      unless the events are clearly independent
                      (then split into multiple steps).
                      For non-music SFX/ambience, keep this noun-led
                      and physical. Do NOT add ambiguous music/style
                      words such as rhythmic, melodic, beat, tempo,
                      cinematic, soundtrack, or score.
- `source_visible_on_screen` : (REQUIRED) bool.
- `expect_prominent_target`  : (REQUIRED for delete/replace, set
                                 false otherwise) bool. Make a
                                 judgement call by READING THE
                                 CAPTION + the edit intent together
                                 — don't keyword-match.
                                 Default to FALSE. Only set TRUE
                                 when the deleted_sound is genuinely
                                 the main/loud event the listener
                                 would notice immediately (e.g. a
                                 close-up impact, a dominant BGM
                                 carrying the scene, an engine
                                 revving in the foreground).
                                 Set FALSE when the deleted_sound
                                 reads as light / soft / subtle /
                                 background / atmospheric /
                                 incidental / minor / barely
                                 noticeable — anything where the
                                 listener wouldn't really miss it
                                 if it just stopped — AND it is
                                 non-vocal content (ambient,
                                 mechanical hum, distant SFX,
                                 background music, room tone, etc.).
                                 Human speech is the exception:
                                 even quiet speech carries semantic
                                 weight, so treat any speech /
                                 dialogue target as TRUE regardless
                                 of caption volume cues.

Per action, the audio fields must satisfy:
  - audio_remove           → deleted_sound non-empty, new_sound=""
  - audio_replace_sfx/bgm  → BOTH non-empty
  - audio_add_sfx/ambient  → deleted_sound="", new_sound non-empty
  - audio_volume_adjust    → deleted_sound="", new_sound="";
                             use the EXTRA fields below

AUDIO_VOLUME_ADJUST extras (when action == audio_volume_adjust):
- `volume_target`    : (REQUIRED) the stem to adjust as a string —
                       MUST overlap an item in `existing_sounds`
                       (substring or shared content words). Examples:
                       "human speech", "background music", "engine
                       noise". DO NOT use generic words like "audio"
                       or "everything".
- `volume_db`        : (REQUIRED) signed gain delta in dB. Positive
                       = louder, negative = softer. Sensible range
                       [-12, +12]. Map natural-language intensity:
                         • "slightly", "subtly"           → ±2 dB
                         • "moderately", default          → ±4 dB
                         • "boost", "pronounced", "more"  → ±6 dB
                         • "much", "significantly"        → ±8 dB
                         • "loud", "drown out"            → ±10 dB

When to choose audio_volume_adjust over audio_replace/add:
  ✓ "Boost the human speech to make it clearer."
  ✓ "Lower the background music so dialogue stands out."
  ✓ "Make the engine noise less prominent."
  ✗ "Replace the speech with a robotic voice."   → speech_swap
  ✗ "Add ambient crowd noise."                   → audio_add_ambient
  ✗ "Remove the background music."               → audio_remove
The keyword test: if the user is asking for the SAME sound to be
LOUDER or SOFTER (no content change), it's audio_volume_adjust.

SPEECH step extras (speech_tts / speech_swap):
- `speech_text`              : the new line to be spoken (verbatim).
- `speech_speaker_description`: the SAM Audio separation prompt
                                 for the ORIGINAL speaker being
                                 isolated.
                                 **FORMAT (same as `sam_prompt`)**:
                                 SINGLE noun phrase of the shape
                                 `[adjective(s)] noun`, ≤ 8 words,
                                 with ONE noun head. NO comma, NO
                                 conjunction ("and" / "or"), NO
                                 sub-clause, NO negation
                                 ("excluding ...", "without ...").
                                 The noun must denote a voice or
                                 speech ("voice", "speech",
                                 "vocals"). Adjectives must be
                                 acoustic / demographic-via-
                                 acoustic: gender, age band, pitch,
                                 timbre. Avoid pure-language /
                                 accent / numerical-pitch /
                                 narrative adjectives unless they
                                 are the only disambiguator.
                                 **CONTENT — multi-speaker rule**:
                                 when the audio caption lists more
                                 than one human voice, the
                                 adjective(s) MUST include a
                                 distinguishing acoustic feature
                                 (gender / pitch / age / loudness)
                                 so SAM extracts ONLY the target
                                 speaker; otherwise SAM will pull
                                 every voice into the target stem.
                                 Examples (good — single phrase):
                                   single speaker (man only):
                                     "male voice"
                                   man + woman (target = man):
                                     "deep male voice"
                                   man + woman (target = woman):
                                     "high female voice"
                                   man + child (target = man):
                                     "adult male voice"
                                   elderly man:
                                     "elderly male voice"
                                 Examples (BAD):
                                   "adult male voice, mid-pitch,
                                    American English"   ← 3 fragments
                                   "the man speaking"   ← no acoustic
                                                          adjective
                                   "male voice excluding the
                                    woman"              ← negation
- `speech_voice_description` : (speech_swap only) natural-language
                                 description of the NEW voice
                                 (gender, age, pitch, timbre).
- `speech_reference_text`    : the original transcript verbatim
                                 (helps voice cloning).
- `speech_language`          : "auto" / "Chinese" / "English" / etc.
- `audio_splice`             : REQUIRED for speech_tts / speech_swap.
                                 Describe the timeline assembly policy,
                                 not a tool prompt. For ordinary line
                                 rewrites use:
                                   {
                                     "mode": "localized_replace",
                                     "source": "speech_transcript",
                                     "reference_text": "<same as speech_reference_text>",
                                     "preserve_outside": true
                                   }
                                 Meaning: resolve the transcript time
                                 range for the original line; replace
                                 only that window; keep the original
                                 audio before and after it unchanged.
                                 Only use a non-local/global policy if
                                 the user explicitly wants all speech by
                                 that speaker changed across the clip.

SPEECH_LIPSYNC steps need only the common fields plus a `depends_on`
pointing to the speech_tts/swap step.

Vocal-action recap (Rule 5): emit a motion_edit + an audio step.
The audio step is `audio_add_sfx` if the target is silent, or
`audio_replace_sfx` if they were already vocalising (with the
target's existing vocalisation in `deleted_sound`).

## Worked example (drum + audio follow-up)

User instruction: "Replace the metallic drum with a wooden drum."
Caption Audio analysis: PRIMARY = metallic drum hits, hand tapping
on metal; SECONDARY = quiet ambient hiss (drop per Rule 7).

Output:
```json
[
  {
    "step": 1, "action": "replace_object",
    "target": "local", "shot_index": 1, "depends_on": [],
    "intent": "Replace the small dark blue metallic tongue drum with a wooden drum of similar size. Keep the gravel, plants, and surrounding environment unchanged."
  },
  {
    "step": 2, "action": "audio_replace_sfx",
    "target": "global", "shot_index": null, "depends_on": [1],
    "intent": "Replace the metallic drum hits with wooden drum hits with a warm, resonant character.",
    "existing_sounds": ["metallic drum hits", "hand tapping on metal"],
    "deleted_sound": "metallic drum hits",
    "new_sound": "wooden drum hits, warm and resonant",
    "source_visible_on_screen": true,
    "expect_prominent_target": true
  }
]
```

Loudness example (boost a stem against a louder layer):

User instruction: "Boost the level of the human speech to make it
clearer and more prominent against the loud engine noise."
Caption Audio analysis: PRIMARY = engine noise, human speech.

```json
[
  {
    "step": 1, "action": "audio_volume_adjust",
    "target": "global", "shot_index": null, "depends_on": [],
    "intent": "Make the human speech clearer and more prominent against the loud engine noise.",
    "existing_sounds": ["engine noise", "human speech"],
    "deleted_sound": "",
    "new_sound": "",
    "volume_target": "human speech",
    "volume_db": 6.0,
    "source_visible_on_screen": true,
    "expect_prominent_target": false
  }
]
```

Speech example (clone the speaker, change the words, single-shot
video where the speaker is on-screen):

```json
[
  {
    "step": 1, "action": "speech_tts",
    "target": "global", "shot_index": null, "depends_on": [],
    "intent": "Make the man say 'Why does this truly matter?' in his own voice instead of his original line.",
    "speech_text": "Why does this truly matter?",
    "speech_speaker_description": "adult male voice, mid-pitch, English",
    "speech_reference_text": "What, what it matters?",
    "speech_language": "English",
    "audio_splice": {
      "mode": "localized_replace",
      "source": "speech_transcript",
      "reference_text": "What, what it matters?",
      "preserve_outside": true
    }
  },
  {
    "step": 2, "action": "speech_lipsync",
    "target": "local", "shot_index": 1, "depends_on": [1],
    "intent": "Re-animate the man's mouth to match the new spoken line."
  }
]
```

Respond with ONLY the JSON array. No markdown fences. No explanation.
"""


# ═══════════════════════════════════════════════════════════════════════════
# Phase B: Task Realizer
# ═══════════════════════════════════════════════════════════════════════════

REALIZER_SYSTEM_PROMPT = """\
You translate ONE step's plain-language editing intent into the
tool-specific prompt(s) that step needs at execution time.

You receive a SINGLE step from Phase A's plan, plus the relevant
caption/shot context. You output a small JSON object with the prompt
fields that step needs and an optional list of extra eval criteria
specific to THIS step.

Phase A has already decided the step's `action`, `shot_index`,
`depends_on`, and (for audio) the `existing_sounds` / `deleted_sound`
/ `new_sound` inventory. You do NOT change those — you only fill in
the tool prompts and eval criteria.

## Use the keyframe image(s) when provided
You may be shown the video's keyframe image(s) alongside the text.
When images are present, GROUND the prompt in what you actually SEE —
the real subject, its appearance, material, colour, and position —
instead of guessing from the caption text alone. This makes the
edit instruction concrete and tool-ready. Do NOT narrate the whole
scene back; only state the change. If no image is provided, fall
back to the caption.

## Output schema (per step)

Return a JSON object whose keys depend on the step's action:

- VIDEO actions (style_transfer / scene_edit / add_object /
  remove_object / replace_object / recolor / repainting /
  depth_modify / motion_edit):
    {
      "video_prompt": "<imperative edit instruction; see video_prompt rules below for the backend-specific word budget>",
      "extra_eval_criteria": ["...", "..."]
    }

- audio_remove:
    {
      "sam_prompt": "<short noun-led ≤ 10 words>",
      "sam_eval_criteria": ["..."],
      "extra_eval_criteria": ["..."]
    }

- audio_replace_sfx / audio_replace_bgm:
    {
      "sam_prompt": "<≤ 10 words>",
      "mmaudio_prompt": "<sensory ≤ 15 words>",
      "sam_eval_criteria": ["..."],
      "mmaudio_eval_criteria": ["..."],
      "extra_eval_criteria": ["..."]
    }

- audio_add_sfx / audio_add_ambient:
    {
      "mmaudio_prompt": "<sensory ≤ 15 words>",
      "mmaudio_eval_criteria": ["..."],
      "extra_eval_criteria": ["..."]
    }

- audio_volume_adjust:
    {
      "sam_prompt": "<noun-led ≤ 10 words naming the stem to gain>",
      "sam_eval_criteria": ["..."],
      "extra_eval_criteria": ["..."]
    }
  (Phase A already filled `volume_target` and `volume_db`; you only
   write the SAM prompt that isolates that stem and any extra eval
   criteria. NO mmaudio_prompt — there is no generation step.)

- speech_tts / speech_swap:
    {
      "sam_eval_criteria": ["..."],
      "extra_eval_criteria": ["..."]
    }
  (The speech_text / speech_speaker_description / speech_voice_description /
   speech_reference_text / speech_language fields are already on the
   Phase A step — you don't rewrite them. Supply SAM-stage criteria
   for the original-speaker isolation step plus mix/final criteria.)

- speech_lipsync:
    {
      "extra_eval_criteria": ["..."]
    }
  (Lipsync takes no text prompt; only criteria.)

## Tool-specific prompt rules

__VIDEO_PROMPT_RULES__

### Cross-modal state injection (REQUIRED check before writing video_prompt)

A paired audio step's `new_sound` adjective often IMPLIES a visible
physical state of the edited object that the user did not spell out.
The video model only sees the source clip — if you don't bake the
implied state into the prompt, the result will contradict the audio.

Procedure:
1. Scan "All steps in this plan" for an audio_replace_sfx /
   audio_add_sfx step whose `new_sound` (or `intent`) targets the
   SAME object/surface as your video step.
2. Map the paired sound's adjective to a 1-WORD visible state and
   inject it into the video_prompt body (NOT the preservation tail):

     | new_sound contains            | inject 1 adjective    |
     | hollow / resonant / clunk     | empty                 |
     | echoing / reverberant         | empty                 |
     | wet splash / splatter / drip  | with water / wet      |
     | dry crunch / crisp            | dry                   |
     | metallic ring / clang / chime | metal (if material)   |
     | shattering / cracking         | (target stays brittle —
     |                                  ceramic / glass — DO NOT
     |                                  switch to plastic)    |
     | wooden / muffled thud         | wooden                |
     | crackling fire                | burning / lit         |

3. Keep it ONE word. Place it before the noun:
     "Replace the bin with an empty ceramic pot."
     "Add a wet puddle on the floor."
4. If no audio sibling exists or the new_sound is silent / generic
   (e.g. "ambient hum"), skip this step.

Examples (good):
- intent = "Replace the green trash bin with a ceramic pot",
  paired audio new_sound = "resonant hollow clunk" →
  video_prompt: "Replace the bin with an empty ceramic pot."
- intent = "Recolor the asphalt road",
  paired audio new_sound = "tires splashing through puddles" →
  video_prompt: "Make the road wet and shiny."

Anti-pattern:
BAD: video_prompt drops the implied state → object visually full
     of contents while audio says hollow → physical contradiction.

### Preservation / constraint clauses — per backend

Whether to append a "keep X unchanged" clause is decided by the
video_prompt rules block above (it differs by backend): Wan RECOMMENDS
naming the 1-2 things that should stay unchanged, as a comma clause
with the imperative "keep"; Seedance forbids preservation clauses.
Follow that block.
The one thing to avoid on ANY backend is a vague tail:
BAD: "..., keep everything else unchanged."   ← too vague, name specifics

### sam_prompt (SAM Audio separation)
- A SINGLE noun phrase naming the sound to extract. ≤ 8 words.
- Form: `[adjective(s)] noun` — one noun head, optionally with
  acoustic adjectives stacked before it. NO comma, NO conjunction
  ("and" / "or"), NO clause ("which is …"), NO negation
  ("excluding …" / "without …" / "not …").
- Adjectives must be ACOUSTIC: source / timbre / texture / pitch-
  band / impact-type. Forbidden adjective categories: language
  ("English", "Mandarin"), accent ("American", "Southern"),
  numerical pitch ("mid-pitch", "120 Hz"), narrative ("at 0.5s",
  "during the chorus"), demographic if not acoustic ("teenage",
  "female adult speaker" — these are OK only if you're describing
  vocal timbre, e.g. "deep male voice").
- Examples (good — single phrase):
    deleted_sound = "plastic 'thud' sound"
       → sam_prompt = "plastic thud"
    deleted_sound = "metallic drum hits"
       → sam_prompt = "metal drum hit"
    deleted_sound = "melancholic cinematic score"
       → sam_prompt = "background music"
    deleted_sound = "human speech"
       → sam_prompt = "male speech"
- Examples (BAD — comma / multi-fragment):
    BAD: "plastic clack and thud, dull impact"          ← comma
    BAD: "adult male voice, American English, mid-pitch" ← 3 fragments
    BAD: "clattering sounds, plastic thud"              ← comma
    BAD: "speech excluding the dog barking"             ← negation
- If the deleted_sound contains "and" (e.g. "clack and thud"),
  PICK THE STRONGER NOUN ("thud") and add at most one acoustic
  adjective ("plastic thud").

### Vocabulary rule for sam_prompt (CRITICAL)

SAM Audio's training distribution is dominated by everyday-language
captions ("a dog barking", "people clapping", "background music").
Genre-specific or literary adjectives ("melancholic", "cinematic",
"haunting", "ethereal", "diegetic") rarely appear in training and
the model fails to anchor on them.

- Use ONLY common everyday words. The prompt just needs ENOUGH
  DISTINCTIVENESS so SAM can find the source — not a poetic
  description.
- Forbidden adjective bucket (replace with the bracketed common
  equivalent):
    melancholic / haunting / poignant     →  (drop)
    cinematic / orchestral / atmospheric  →  background music
    ethereal / dreamy / ambient (as adj)  →  background music
    discordant / dissonant                →  (drop)
    diegetic / non-diegetic               →  (drop)
    raucous / cacophonous                 →  loud
    susurrating / murmuring               →  whispering
- If the deleted_sound itself contains a fancy adjective, STRIP it
  down to the common noun head:
    "melancholic cinematic score"  → sam_prompt = "background music"
    "haunting choral chant"        → sam_prompt = "choir singing"
    "ethereal synth pad"           → sam_prompt = "synth music"

### mmaudio_prompt (MMAudio V2 generation)
- A SINGLE short SENSORY phrase or sentence describing the sound to
  generate. 5–12 words.
- **NO comma. NO 'and' / 'or'.** A single uninterrupted phrase —
  comma-stitching multiple fragments dilutes MMAudio's match (it
  tries to satisfy each fragment, so you get mush instead of one
  clean event).
- Noun-led (the head can be a gerund — "barking", "crushing",
  "splashing" — gerunds are valid noun heads). Adjectives stack
  before the head; visual-anchor preposition phrases (`under`,
  `when`, `at`) chain after.
- NO imperative verbs ("Add", "Make", "Generate"). NO filler
  ("blending naturally with…").
- The MMAudio negative prompt is auto-derived from
   by the pipeline — you do NOT
  write it.

GOOD:
  "hollow ceramic clunk"                        (3w, no comma)
  "watery splat under elephant foot crushing"   (7w, no comma)
  "muffled wooden knock at door surface"        (6w, no comma)
  "quiet library people whispering with footsteps" (6w, no comma)

BAD (commas / multi-fragment):
  "quiet library, people whispering, footsteps"   ← 2 commas
  "watery explosive splat, foot crushing"         ← comma
  "explosive watery splat, single foot crush"     ← comma
  "soft ambient, gentle hum, faint chatter"       ← 2 commas

### Visual anchor — REQUIRED for audio_replace_sfx / audio_add_sfx

For these two actions the pipeline feeds MMAudio the source video
(mask_away_clip=False). MMAudio aligns each generated audio event
to the strongest visual motion peak it sees. When the clip has
MULTIPLE candidate peaks (a hand grabs the prop AND a foot stomps
it AND it bounces), the model often picks the WRONG one — usually
the earliest / loudest visual cue rather than the user-intended
moment.

A bare prompt like `"watery splat"` gives no clue which peak to
align to. NAME THE VISUAL TRIGGER so MMAudio anchors there:

  bad   `"watery splat"`                              → trunk grab
  good  `"watery splat under elephant foot crushing"` → foot impact
  bad   `"metallic clink"`                            → wrong peak
  good  `"metallic clink at hammer head impact"`      → hammer head

For audio_replace_sfx / audio_add_sfx, the prompt MUST include a
short visual-anchor phrase (2-4 words) naming the visible motion or
contact site. Templates:

    "<sensory sound> <preposition> <visual trigger>"
    "<sensory sound> when <visual trigger>"
    "<sensory sound> at <visual trigger>"

IMPORTANT: The visual trigger must describe pure visual motion /
contact only — do NOT include any audible event or sound description
in it. If you mention another sound (especially one already in the
audio that the user is NOT asking for), MMAudio will treat it as a
second target and try to regenerate that sound on top of the
requested one.

  bad   `"jingle when dog barks"`           ← "barks" is a SOUND;
                                               MMAudio will produce
                                               dog barking again
  good  `"jingle when dog moves head"`      ← pure visual motion
  bad   `"clink when bottle breaks"`        ← "breaks" implies
                                               crash sound
  good  `"clink at hammer head impact"`     ← pure contact event

If the only natural way to describe the timing is to name a sound,
DROP the timing clause entirely and let MMAudio align freely:

  acceptable  `"small metal bell jingling"` (no anchor at all)

Examples:
    user new_sound = "watery, explosive splat (replace crushing)"
       → mmaudio_prompt = "watery splat under elephant foot crushing"
    user new_sound = "wooden thud (replace plastic thud as toys drop)"
       → mmaudio_prompt = "wooden thud when toys drop in pot"
    user new_sound = "sharp crack (replace bottle break)"
       → mmaudio_prompt = "sharp crack at glass bottle shatter"
    user new_sound = "muffled tap (replace knock on door)"
       → mmaudio_prompt = "muffled wooden knock at door surface"
    user new_sound = "puff of dust (sync to broom sweep)"
       → mmaudio_prompt = "soft dust poof when broom sweeps floor"

Visual-anchor word count is INCLUDED in the 15-word cap — keep
both halves short.

For audio_replace_bgm and audio_add_ambient, the pipeline does NOT
feed video to MMAudio (mask_away_clip=True), so a visual anchor
adds nothing — DO NOT include one. Just describe the sound.

### Vocabulary rule for mmaudio_prompt (CRITICAL)

MMAudio is also trained on everyday descriptions. The user's
`new_sound` is a HINT, not a verbatim contract — your job is to
satisfy the user's INTENT with a description MMAudio can produce.
If the user's wording is fancy, swap to a common synonym.

- Use plain everyday vocabulary. Match the kind of caption a
  layperson would write under a YouTube clip, not a film-music
  reviewer.
- Common-synonym swap table (apply silently when present in the
  user's `new_sound`):
    melancholic piano music           →  slow piano music
    haunting cinematic strings        →  slow string music
    cinematic suspenseful sting       →  short tense music chord
    soft ambient library tones        →  quiet room with people
    ethereal pad / atmospheric drone  →  soft hum
    bustling city soundscape          →  city street with traffic
    raucous market chatter            →  busy crowd talking
- KEEP the user's source nouns (drum, water, footsteps, speech)
  exactly. Only swap the FANCY ADJECTIVES.
- For NON-MUSIC SFX/ambience, never add music/rhythm/style words
  that were not explicitly requested. These are forbidden because
  they can make MMAudio generate abnormal BGM instead of the concrete
  sound: rhythmic, rhythm, melodic, musical, beat, tempo, groove,
  cinematic, soundtrack, score.
  Use physical wording instead:
    rhythmic faucet water      → running faucet water
    melodic water stream       → water flowing from faucet
    cinematic thunder          → loud thunder
- The eval still uses `extra_eval_criteria` written from the
  user's original wording — so the rewriter doesn't lose intent.
- Examples:
    user new_sound = "melancholic cinematic score"
       → mmaudio_prompt = "slow piano music with strings"
    user new_sound = "subtle ambient library sounds with faint murmurs"
       → mmaudio_prompt = "quiet library, people whispering, footsteps"
    user new_sound = "resonant hollow clunk, ceramic echo"
       → mmaudio_prompt = "hollow ceramic clunk"

### Three eval-criteria buckets — pick the right one

A single audio step typically runs through TWO tools (SAM + MMAudio)
before a mix step lands the result on the timeline. Each tool has
its OWN evaluator that benefits from criteria scoped to that tool's
job. Mixing all criteria into one bucket caused the SAM evaluator
to score "voice should be male" against a female-isolation stem,
which gave 0/0 even when separation was perfect.

The three buckets:
  • `sam_eval_criteria`     — judged against the SAM target/residual
                              stems BEFORE any generation/mixing.
                              Talk about which sound was isolated,
                              what should stay in the residual, and
                              whether bystander stems were preserved.
                              REQUIRED whenever the step uses SAM
                              (audio_remove, audio_replace_*,
                              audio_volume_adjust, speech_tts,
                              speech_swap).
  • `mmaudio_eval_criteria` — judged against the raw MMAudio output
                              BEFORE mixing. Talk about whether the
                              new sound matches the requested
                              content/timbre and whether forbidden
                              sounds leaked in. REQUIRED whenever
                              the step uses MMAudio
                              (audio_replace_*, audio_add_*).
  • `extra_eval_criteria`   — judged against the FINAL mixed audio
                              and the whole-video final eval. Talk
                              about end-result conformance: new
                              sound is audible at the right level,
                              old sound is gone from the mix,
                              speech is intact, etc.

Each bucket: 0–4 short verifiable criteria. Phrase them so the LLM
evaluator can answer yes/no after listening to the relevant audio
asset.
- Anchor them in the user's original instruction, not the tool
  prompt you just wrote.
- For audio_replace, `extra_eval_criteria` MUST include a criterion
  that the original deleted_sound is no longer audible in the final mix.
- For video, include a preserve criterion ("X remains unchanged").
- **REQUIRED — speech preservation criterion**:
    For ANY audio step where the original audio contains human
    speech / dialogue / voice AND the speech is NOT the target of
    the edit (i.e. it's in the `preserve` set, not in
    `deleted_sound` or `volume_target`), you MUST add an explicit
    criterion that asks the evaluator to verify the speech is
    fully preserved with NO byte-level mis-extraction.
    Bucket placement:
      • The "speech intact in the residual" check goes in
        `sam_eval_criteria` (it's a property of the SAM stage
        output).
      • The "speech intact in the final mix" check goes in
        `extra_eval_criteria`.
    Phrasing examples:
      "Speech segments are not mis-extracted into the target
       stem; the residual still contains the complete dialogue."
       (sam_eval_criteria)
      "The original speech remains fully intelligible — every
       spoken word is still clearly audible, with no clipped
       segments or word-level cuts."
       (extra_eval_criteria)
    Speech is the most listener-noticeable layer; partial speech
    loss (even a fraction of a second) is a hard fail.
    Applies to: audio_remove, audio_replace_sfx, audio_replace_bgm,
    audio_add_sfx / audio_add_ambient, audio_volume_adjust (when
    speech is a bystander stem). Skip ONLY when the audio has no
    speech to preserve, or when speech itself is being edited
    (speech_tts/swap have their own checks below).

- **REQUIRED — speaker-correctness criterion (multi-speaker case)**:
    For speech_tts / speech_swap when the audio caption mentions
    MORE THAN ONE human voice, you MUST add a criterion to
    `sam_eval_criteria` that verifies the SAM target stem captured
    the CORRECT speaker (and ONLY that speaker — not the bystander
    voices). Phrasing:
      "Only the {target speaker, e.g. 'male'} voice is in the
       target stem; the other speaker(s) are not present there."
      "The target stem contains the man's speech only, with no
       audible female voice."
    Without this, SAM may pull both speakers into target → the
    cloned line replaces ALL on-screen voices instead of the
    intended one. The unified SAM evaluator scores this directly
    via target_extraction (drops to 0.4 when the stem mixes
    speakers).
- Examples (one per bucket, replace_sfx scenario):
    sam_eval_criteria:
      "The target stem contains the plastic thud only, with no
       speech bleeding through."
      "The residual preserves the dialogue and the BGM intact."
    mmaudio_eval_criteria:
      "The generated sound is recognisably a wooden thud."
      "No speech is hallucinated into the generated SFX layer."
    extra_eval_criteria:
      "The new wooden thud is audible at the moment toys land."
      "The plastic clack is no longer present in the final mix."
      "Speech remains fully intelligible across the clip."

- Example (speech_swap female → male):
    sam_eval_criteria:
      "The target stem contains the female speaker only, with no
       background music or other voices."
      "The residual no longer contains the female voice."
    extra_eval_criteria:
      "The new male voice speaks the requested line clearly."
      "The original female voice is absent from the final mix."
   (No mmaudio_eval_criteria — speech_swap uses Voice Design,
    not MMAudio.)

Respond with ONLY the JSON object. No markdown fences, no prose.
"""


# ── Backend-conditioned video_prompt rules ──────────────────────────────────
# The realizer prompt embeds ONE of these blocks (selected by the active video
# backend) in place of the `__VIDEO_PROMPT_RULES__` sentinel. Wan is a V2V edit
# model; Seedance is reference-to-video generation. Word caps are enforced by
# PlanValidator.validate_subtasks via VIDEO_WORD_CAP.

VIDEO_PROMPT_RULES_WAN = """\
### video_prompt (Wan 2.7 V2V — instruction-based edit)
Write ONE natural, grammatically complete imperative sentence that
states the change and its concrete target.
- CONCISE but NOT telegraphic: keep articles and prepositions and normal
  sentence structure. Do NOT clip into a fragment. Aim 6–14 words.
- ONE edit per prompt (Phase A already split the steps). Be specific and
  bounded — name the concrete target (colour, material, weather, style,
  action); no vague aesthetics ("make it dramatic").
    GOOD: "Change the scene to a heavy rainstorm with wet streets."
    GOOD: "Replace the keyboard with an old, clunky typewriter."
    GOOD: "Add heavy rain, keep the background and camera motion unchanged."
    BAD : "Add heavy rain keeping the background and camera motion unchanged."
          (participle run-on, no comma — use ", keep ... unchanged" instead)
    BAD : "Make it look more dramatic."   (vague)
- Preservation (RECOMMENDED — Wan official guidance: "say what should
  stay untouched"): append a COMMA clause with the imperative "keep":
  ", keep <X> unchanged." Name the 1–2 things that should stay the same
  (subject, background, pose, lighting, framing). Use "keep", not
  "keeping"; use a comma, not a participle run-on.
- Do NOT contradict the source (e.g. removing the main subject the whole
  clip is about). Verbs are flexible (Change / Add / Remove / Replace /
  Make). For motion changes name the new action concretely.
- Ground the target in the keyframe you see. Keep any cross-modal
  injected state adjective (see below). NO "do not …" clauses.
  NO "shot" / "keyframe" / "segment".
# Style: concise but natural full sentences (no telegraphic clipping);
# one optional comma "keeping X unchanged" half-clause allowed."""

# Per-backend video_prompt word cap, enforced in validate_subtasks.
# Wan: natural sentence + a recommended ", keep X unchanged" clause.
# Seedance: reference-anchored scene description.
VIDEO_WORD_CAP = {"wan": 22, "seedance": 40}

VIDEO_PROMPT_RULES_SEEDANCE = """\
### video_prompt (Seedance 2.0 reference-to-video — NOT a V2V editor)
Seedance has no edit endpoint; the source clip is passed as a REFERENCE
(@Video1) and the model REGENERATES a clip from your prompt. So the prompt
is NOT a terse edit instruction — it is a short DESCRIPTION of the desired
result clip, anchored to the source via "@Video1".
- Ground the subject/scene on @Video1, then state the change applied to it.
  Form: "<subject/scene from @Video1>, <the edit>." The tool auto-prepends
  "Based on @Video1,".
    GOOD: "The street and rider from @Video1, now in heavy rain with wet reflections."
    GOOD: "The forest and waterfall from @Video1, covered in snow and ice."
    BAD : "Add heavy rain."   (too terse — gives the generator nothing to anchor)
- Do NOT add any "keep/same/everything-else-unchanged" preservation clause.
  Seedance REGENERATES the whole clip and cannot honor pixel-level
  preservation, so such clauses add nothing — just describe the subject and
  the edit, and let @Video1 do the anchoring.
- Be concrete about the edit (colour, material, weather, action); no vague
  aesthetics, and NEVER glossy "quality" words (cinematic / 8k / ultra).
- Aim 12–28 words. Verbs/phrasing are flexible (it's a description, not an
  imperative). NO "do not …". NO "shot" / "keyframe" / "segment".
# Style: a concise description of the subject (grounded on @Video1) + the
# edit. NO explicit "stays the same" / preservation anchoring."""


# ═══════════════════════════════════════════════════════════════════════════
# PlanValidator
# ═══════════════════════════════════════════════════════════════════════════

_VALIDATOR_STOP = {
    "a", "an", "the", "of", "and", "or", "with", "to", "from",
    "in", "on", "for", "sound", "sounds", "noise", "noises",
}

_VALIDATOR_ALIAS_GROUPS = {
    "feeding_sfx": {
        "chew", "chewing", "crunch", "crunching", "eat", "eating",
        "food", "kibble", "clatter", "rattle", "rattling",
    },
}

_NON_MUSIC_SFX_AMBIGUOUS_WORDS = {
    "rhythm", "rhythmic", "rhythmically",
    "melody", "melodic", "melodically",
    "musical", "beat", "beats", "tempo", "groove",
    "cinematic", "soundtrack", "score",
}


def _validator_tokens(text: str) -> set[str]:
    tokens = {
        t.strip(".!?")
        for t in re.split(r"[\s/,]+", (text or "").strip().lower())
        if t.strip(".!?") and t.strip(".!?") not in _VALIDATOR_STOP
    }
    for alias, members in _VALIDATOR_ALIAS_GROUPS.items():
        if tokens & members:
            tokens.add(alias)
    return tokens


def _intent_overlaps_any(target: str, candidates: list[str]) -> bool:
    """Return True if `target` matches any candidate via substring or
    >= 2 shared content/alias tokens with ≥ 50 % coverage of the
    shorter side. The small alias pass lets explicit user targets
    like "dry dog food clatter" match caption-level inventory entries
    such as "dog eating sounds" without treating every dog sound as
    the same source."""
    a = (target or "").strip().lower()
    if not a:
        return False
    for raw in candidates or []:
        b = (raw or "").strip().lower()
        if not b:
            continue
        if a in b or b in a:
            return True
        ta = _validator_tokens(a)
        tb = _validator_tokens(b)
        if not ta or not tb:
            continue
        inter = ta & tb
        denom = min(len(ta), len(tb))
        if len(inter) >= 2 and denom > 0 and (len(inter) / denom) >= 0.5:
            return True
    return False


class PlanValidator:
    """
    Validates a plan (list of intents or SubTasks) against hard rules.
    Returns a list of violation messages; empty list = valid.
    """

    @staticmethod
    def validate_intents(steps: list[dict[str, Any]]) -> list[str]:
        """Validate Phase A step output. The new schema is a flat list
        of STEPS (with `step`, `action`, `intent`, `shot_index`,
        `depends_on`, plus per-modality fields)."""
        violations: list[str] = []

        if not steps:
            violations.append("Plan is empty — no steps planned.")
            return violations

        AUDIO_ACTIONS = {
            "audio_remove", "audio_replace_sfx", "audio_replace_bgm",
            "audio_add_sfx", "audio_add_ambient",
            "audio_volume_adjust",
        }
        SPEECH_ACTIONS = {"speech_tts", "speech_swap", "speech_lipsync"}
        VIDEO_ACTIONS_NO_LIPSYNC = {
            "style_transfer", "scene_edit", "add_object", "remove_object",
            "replace_object", "recolor", "repainting", "depth_modify",
            "motion_edit",
        }

        step_ids = [s.get("step") for s in steps]
        if len(set(step_ids)) != len(step_ids):
            violations.append("Step ids must be unique.")

        video_steps = [s for s in steps
                       if s.get("action") in VIDEO_ACTIONS_NO_LIPSYNC]
        audio_steps = [s for s in steps if s.get("action") in AUDIO_ACTIONS]

        # Per-shot is the default for video steps. `shot_index=None` is
        # valid only for same-scene/same-camera jump cuts that should be
        # sent to V2V as one whole clip; the Phase A prompt owns that
        # semantic decision.
        for s in video_steps:
            shot_index = s.get("shot_index")
            if shot_index is None:
                continue
            try:
                int(shot_index)
            except Exception:
                violations.append(
                    f"Step {s.get('step','?')} ({s.get('action','?')}): "
                    f"shot_index must be an integer shot number, or null "
                    f"only for a same-scene/same-camera jump-cut whole-clip "
                    f"video edit."
                )

        # depends_on must reference earlier step ids
        for s in steps:
            sid = s.get("step")
            for dep in s.get("depends_on", []) or []:
                if dep not in step_ids:
                    violations.append(
                        f"Step {sid} depends_on={dep} which does not exist."
                    )
                elif sid is not None and dep >= sid:
                    violations.append(
                        f"Step {sid} depends_on={dep} must reference an "
                        f"earlier step (dep < step)."
                    )

        # Every step needs a non-empty `intent` (plain natural language)
        for s in steps:
            if not (s.get("intent") or "").strip():
                violations.append(
                    f"Step {s.get('step','?')} ({s.get('action','?')}): "
                    f"`intent` is empty. Phase A must describe the "
                    f"step's edit goal in plain language."
                )

        # Audio steps: existing_sounds / deleted_sound / new_sound
        # cardinality check.
        for s in audio_steps:
            action = s.get("action", "")
            es = s.get("existing_sounds")
            if not isinstance(es, list) or not all(isinstance(x, str) for x in es):
                violations.append(
                    f"Step {s.get('step','?')} ({action}): "
                    f"existing_sounds must be a JSON array of strings."
                )
            existing = es if isinstance(es, list) else []
            deleted = (s.get("deleted_sound") or "").strip()
            new = (s.get("new_sound") or "").strip()

            if deleted and existing and not _intent_overlaps_any(deleted, existing):
                violations.append(
                    f"Step {s.get('step','?')} ({action}): deleted_sound "
                    f"{deleted!r} does not match any existing_sounds entry "
                    f"{existing}."
                )

            if action == "audio_remove":
                if not deleted:
                    violations.append(
                        f"Step {s.get('step','?')} (audio_remove): "
                        f"deleted_sound must be non-empty."
                    )
                if new:
                    violations.append(
                        f"Step {s.get('step','?')} (audio_remove): "
                        f"new_sound must be empty."
                    )
            elif action in ("audio_replace_sfx", "audio_replace_bgm"):
                if not deleted:
                    violations.append(
                        f"Step {s.get('step','?')} ({action}): "
                        f"deleted_sound must be non-empty."
                    )
                if not new:
                    violations.append(
                        f"Step {s.get('step','?')} ({action}): "
                        f"new_sound must be non-empty."
                    )
            elif action in ("audio_add_sfx", "audio_add_ambient"):
                if deleted:
                    violations.append(
                        f"Step {s.get('step','?')} ({action}): "
                        f"deleted_sound must be empty for pure-add."
                    )
                if not new:
                    violations.append(
                        f"Step {s.get('step','?')} ({action}): "
                        f"new_sound must be non-empty."
                    )
            elif action == "audio_volume_adjust":
                if deleted:
                    violations.append(
                        f"Step {s.get('step','?')} (audio_volume_adjust): "
                        f"deleted_sound must be empty (no removal)."
                    )
                if new:
                    violations.append(
                        f"Step {s.get('step','?')} (audio_volume_adjust): "
                        f"new_sound must be empty (no generation)."
                    )
                vt = (s.get("volume_target") or "").strip()
                if not vt:
                    violations.append(
                        f"Step {s.get('step','?')} (audio_volume_adjust): "
                        f"volume_target must be a non-empty stem name."
                    )
                elif existing and not _intent_overlaps_any(vt, existing):
                    violations.append(
                        f"Step {s.get('step','?')} (audio_volume_adjust): "
                        f"volume_target {vt!r} does not match any "
                        f"existing_sounds entry {existing}."
                    )
                vdb = s.get("volume_db", None)
                try:
                    vdb_f = float(vdb)
                except (TypeError, ValueError):
                    violations.append(
                        f"Step {s.get('step','?')} (audio_volume_adjust): "
                        f"volume_db must be a number."
                    )
                else:
                    if vdb_f == 0.0:
                        violations.append(
                            f"Step {s.get('step','?')} (audio_volume_adjust): "
                            f"volume_db is 0 — no-op step is not allowed."
                        )
                    if abs(vdb_f) > 12.0:
                        violations.append(
                            f"Step {s.get('step','?')} (audio_volume_adjust): "
                            f"volume_db={vdb_f} out of range [-12, +12]."
                        )

        # All audio steps must share the same existing_sounds.
        if audio_steps:
            ref = tuple(audio_steps[0].get("existing_sounds") or [])
            for s in audio_steps[1:]:
                if tuple(s.get("existing_sounds") or []) != ref:
                    violations.append(
                        f"Step {s.get('step','?')}: existing_sounds "
                        f"differs from earlier audio steps. All audio "
                        f"steps must carry identical existing_sounds."
                    )
                    break

        # Speech steps schema
        for s in steps:
            action = s.get("action", "")
            if action not in SPEECH_ACTIONS:
                continue
            if action != "speech_lipsync":
                if not s.get("speech_text"):
                    violations.append(
                        f"Step {s.get('step','?')} ({action}): "
                        f"speech_text required."
                    )
                if not s.get("speech_speaker_description"):
                    violations.append(
                        f"Step {s.get('step','?')} ({action}): "
                        f"speech_speaker_description required (SAM input)."
                    )
                # Note: speech_speaker_description format (no-comma /
                # acoustic-only / ≤6 words) is intentionally NOT
                # validated here. Phase A writes a natural-language
                # description; the runtime sanitiser
                # (`_sanitise_sam_prompt`) collapses it to a single
                # noun phrase before SAM sees it. Format constraints
                # belong in Phase B, not Phase A.
                if action == "speech_swap" and not s.get("speech_voice_description"):
                    violations.append(
                        f"Step {s.get('step','?')} (speech_swap): "
                        f"speech_voice_description required."
                    )
            else:
                # speech_lipsync needs depends_on a speech_tts/swap step
                tts_swap_ids = {
                    s2.get("step") for s2 in steps
                    if s2.get("action") in ("speech_tts", "speech_swap")
                }
                if not any(d in tts_swap_ids for d in s.get("depends_on", []) or []):
                    violations.append(
                        f"Step {s.get('step','?')} (speech_lipsync) must "
                        f"depends_on a speech_tts/speech_swap step."
                    )
                if s.get("shot_index") is None:
                    violations.append(
                        f"Step {s.get('step','?')} (speech_lipsync) must "
                        f"set shot_index."
                    )

        # Cross-cutting: video → audio consistency. If any audio step
        # has source_visible_on_screen=true, expect at least one video
        # step in the plan.
        for s in audio_steps:
            if not s.get("source_visible_on_screen", False):
                continue
            if not video_steps:
                anchor = s.get("deleted_sound") or s.get("new_sound") or "?"
                violations.append(
                    f"Audio step on '{anchor}' has "
                    f"source_visible_on_screen=true but no paired video "
                    f"step exists. Add a motion_edit / replace_object."
                )

        return violations

    @staticmethod
    def validate_subtasks(
        subtasks: list[SubTask],
        num_shots: int = 1,
        video_word_cap: int = 12,
        video_backend: str = "wan",
    ) -> list[str]:
        """Validate Phase B SubTask output.

        `video_word_cap` is backend-specific. Seedance is reference-to-video
        generation, so its prompt is a scene description and skips the edit
        verb / motion-canonical checks."""
        violations: list[str] = []

        if not subtasks:
            violations.append("No subtasks generated.")
            return violations

        # Audio subtasks must agree on existing_sounds (it's a global
        # session fact; carrying it per-SubTask is just for readability).
        audio_existings = [
            tuple(t.existing_sounds) for t in subtasks
            if t.is_audio and t.action != EditAction.SPEECH_LIPSYNC
        ]
        if audio_existings:
            ref = audio_existings[0]
            for t, es in zip(
                [s for s in subtasks
                 if s.is_audio and s.action != EditAction.SPEECH_LIPSYNC],
                audio_existings,
            ):
                if es != ref:
                    violations.append(
                        f"Step {t.step} ({t.action.value}): "
                        f"existing_sounds differs from earlier audio "
                        f"subtasks. All audio subtasks must carry the "
                        f"same existing_sounds (full original-audio "
                        f"inventory)."
                    )
                    break

        for t in subtasks:
            if not t.eval_criteria:
                violations.append(
                    f"Step {t.step} ({t.action.value}): "
                    f"eval_criteria is empty."
                )

            # Per-tool eval criteria: every step that drives a SAM
            # call needs `sam_eval_criteria`; every step that drives
            # an MMAudio call needs `mmaudio_eval_criteria`. These
            # criteria are scored against the per-tool stage output
            # before any mixing happens, so they must exist before
            # the runner reaches that stage.
            uses_sam = bool(t.sam_prompt) or t.action in (
                EditAction.SPEECH_TTS, EditAction.SPEECH_SWAP,
            )
            uses_mmaudio = bool(t.mmaudio_prompt)
            if uses_sam and not t.sam_eval_criteria:
                violations.append(
                    f"Step {t.step} ({t.action.value}): "
                    f"sam_eval_criteria must be non-empty (this step "
                    f"drives a SAM separation call). Phrase 1-3 "
                    f"checks scoped to the target/residual stems."
                )
            if uses_mmaudio and not t.mmaudio_eval_criteria:
                violations.append(
                    f"Step {t.step} ({t.action.value}): "
                    f"mmaudio_eval_criteria must be non-empty (this "
                    f"step drives an MMAudio generation call). Phrase "
                    f"1-3 checks scoped to the raw generated audio."
                )

            # ── VIDEO actions ─────────────────────────────────────
            if not t.is_audio and t.action != EditAction.SPEECH_LIPSYNC:
                if not t.video_prompt.strip():
                    violations.append(
                        f"Step {t.step} ({t.action.value}): "
                        f"video_prompt is empty."
                    )
                elif video_backend == "seedance":
                    pass
                else:
                    first_word = (
                        t.video_prompt.strip()
                        .split(maxsplit=1)[0]
                        .strip("\"'`“”‘’([{")
                        .rstrip(":,.;!?)]}")
                        .lower()
                    )
                    if first_word not in COMMON_VIDEO_PROMPT_START_VERBS:
                        allowed = ", ".join(v.title() for v in COMMON_VIDEO_PROMPT_START_VERBS)
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"video_prompt must start with one common "
                            f"edit verb: {allowed}. Current prompt: "
                            f"{t.video_prompt!r}."
                        )
                    if t.action == EditAction.MOTION_EDIT:
                        prompt_lc = t.video_prompt.strip().lower()
                        has_action_to = (
                            " action to " in prompt_lc
                            or "'s action to " in prompt_lc
                            or " motion to " in prompt_lc
                            or "'s motion to " in prompt_lc
                        )
                        if first_word != "change" or not has_action_to:
                            violations.append(
                                f"Step {t.step} ({t.action.value}): "
                                "motion_edit video_prompt must use the "
                                "canonical action-change form: "
                                "\"Change <subject>'s action to "
                                "<new action>.\" Current prompt: "
                                f"{t.video_prompt!r}."
                            )
                    wc = len(t.video_prompt.split())
                    if wc > video_word_cap:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"video_prompt too long ({wc} words, max "
                            f"{video_word_cap})."
                        )
                if t.eval_criteria:
                    has_preserve = any(
                        any(kw in c.lower() for kw in
                            ("unchanged", "remain", "preserve",
                             "intact", "same"))
                        for c in t.eval_criteria
                    )
                    if not has_preserve:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"eval_criteria should include ≥1 preservation "
                            f"criterion."
                        )

            # ── AUDIO actions ─────────────────────────────────────
            if t.action in (EditAction.AUDIO_REMOVE,
                            EditAction.AUDIO_REPLACE_SFX,
                            EditAction.AUDIO_REPLACE_BGM,
                            EditAction.AUDIO_ADD_SFX,
                            EditAction.AUDIO_ADD_AMBIENT,
                            EditAction.AUDIO_VOLUME_ADJUST):
                # existing_sounds must be a non-empty list of strings
                if not isinstance(t.existing_sounds, list):
                    violations.append(
                        f"Step {t.step} ({t.action.value}): "
                        f"existing_sounds must be a list."
                    )

                # Per-action field-fill matrix (see REALIZER_SYSTEM_PROMPT)
                if t.action == EditAction.AUDIO_REMOVE:
                    if not t.deleted_sound:
                        violations.append(
                            f"Step {t.step} (audio_remove): "
                            f"deleted_sound must be non-empty."
                        )
                    if t.new_sound:
                        violations.append(
                            f"Step {t.step} (audio_remove): "
                            f"new_sound must be empty."
                        )
                    if not t.sam_prompt:
                        violations.append(
                            f"Step {t.step} (audio_remove): "
                            f"sam_prompt must be non-empty."
                        )
                    if t.mmaudio_prompt:
                        violations.append(
                            f"Step {t.step} (audio_remove): "
                            f"mmaudio_prompt must be empty (no MMAudio "
                            f"call for pure remove)."
                        )

                elif t.action in (EditAction.AUDIO_REPLACE_SFX,
                                  EditAction.AUDIO_REPLACE_BGM):
                    for fld in ("deleted_sound", "new_sound",
                                "sam_prompt", "mmaudio_prompt"):
                        if not getattr(t, fld):
                            violations.append(
                                f"Step {t.step} ({t.action.value}): "
                                f"{fld} must be non-empty."
                            )
                    # Replace must have a removal-check criterion
                    if t.eval_criteria:
                        has_removal = any(
                            any(kw in c.lower() for kw in
                                ("absent", "inaudible", "no longer",
                                 "gone", "removed", "disappeared",
                                 "not be heard", "not be present"))
                            for c in t.eval_criteria
                        )
                        if not has_removal:
                            violations.append(
                                f"Step {t.step} ({t.action.value}): "
                                f"eval_criteria must verify the original "
                                f"sound is absent / inaudible."
                            )

                elif t.action in (EditAction.AUDIO_ADD_SFX,
                                  EditAction.AUDIO_ADD_AMBIENT):
                    if t.deleted_sound:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"deleted_sound must be empty for pure add."
                        )
                    if not t.new_sound:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"new_sound must be non-empty."
                        )
                    if t.sam_prompt:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"sam_prompt must be empty (no SAM call for add)."
                        )
                    if not t.mmaudio_prompt:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"mmaudio_prompt must be non-empty."
                        )

                if t.action in (
                    EditAction.AUDIO_ADD_SFX,
                    EditAction.AUDIO_ADD_AMBIENT,
                    EditAction.AUDIO_REPLACE_SFX,
                ):
                    for field_name in ("new_sound", "mmaudio_prompt"):
                        text = getattr(t, field_name, "") or ""
                        bad_words = (
                            _validator_tokens(text)
                            & _NON_MUSIC_SFX_AMBIGUOUS_WORDS
                        )
                        if bad_words:
                            violations.append(
                                f"Step {t.step} ({t.action.value}): "
                                f"{field_name} contains ambiguous "
                                f"music/rhythm word(s) "
                                f"{sorted(bad_words)}. Use concrete "
                                f"physical sound-source wording instead "
                                f"(e.g. 'running faucet water', not "
                                f"'rhythmic faucet water')."
                            )

                elif t.action == EditAction.AUDIO_VOLUME_ADJUST:
                    if t.deleted_sound:
                        violations.append(
                            f"Step {t.step} (audio_volume_adjust): "
                            f"deleted_sound must be empty (no removal)."
                        )
                    if t.new_sound:
                        violations.append(
                            f"Step {t.step} (audio_volume_adjust): "
                            f"new_sound must be empty (no generation)."
                        )
                    if t.mmaudio_prompt:
                        violations.append(
                            f"Step {t.step} (audio_volume_adjust): "
                            f"mmaudio_prompt must be empty (no MMAudio call)."
                        )
                    if not t.sam_prompt:
                        violations.append(
                            f"Step {t.step} (audio_volume_adjust): "
                            f"sam_prompt must be non-empty (Phase B writes "
                            f"the SAM stem-isolation prompt)."
                        )
                    if not (t.volume_target or "").strip():
                        violations.append(
                            f"Step {t.step} (audio_volume_adjust): "
                            f"volume_target must be a non-empty stem name."
                        )
                    elif t.existing_sounds and not _intent_overlaps_any(
                        t.volume_target, t.existing_sounds,
                    ):
                        violations.append(
                            f"Step {t.step} (audio_volume_adjust): "
                            f"volume_target {t.volume_target!r} does not "
                            f"match any existing_sounds entry "
                            f"{t.existing_sounds}."
                        )
                    try:
                        vdb = float(t.volume_db)
                    except (TypeError, ValueError):
                        violations.append(
                            f"Step {t.step} (audio_volume_adjust): "
                            f"volume_db must be a number."
                        )
                    else:
                        if vdb == 0.0:
                            violations.append(
                                f"Step {t.step} (audio_volume_adjust): "
                                f"volume_db is 0 — no-op step is not allowed."
                            )
                        if abs(vdb) > 12.0:
                            violations.append(
                                f"Step {t.step} (audio_volume_adjust): "
                                f"volume_db={vdb} out of range [-12, +12]."
                            )

                # MMAudio prompt style: sensory, no imperative verbs,
                # SHORT single phrase (no comma / and / or).
                if t.mmaudio_prompt:
                    wc = len(t.mmaudio_prompt.split())
                    if wc > 12:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"mmaudio_prompt too long ({wc} words, max 12). "
                            f"Trim to a short single phrase."
                        )
                    first = t.mmaudio_prompt.strip().split()[0].lower()
                    if first in ("add", "generate", "make", "create",
                                 "produce", "play", "insert"):
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"mmaudio_prompt starts with '{first}' — "
                            f"must be a sensory description, not an "
                            f"instruction."
                        )
                    if "," in t.mmaudio_prompt:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"mmaudio_prompt contains a comma — must be "
                            f"a SINGLE short phrase. Comma-stitched "
                            f"fragments dilute MMAudio's match. Stack "
                            f"adjectives before the noun head and chain "
                            f"any visual anchor with 'under'/'when'/"
                            f"'at' instead."
                        )
                    mp_low = t.mmaudio_prompt.lower()
                    if " and " in mp_low or " or " in mp_low:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"mmaudio_prompt contains 'and'/'or' — "
                            f"merge into a single phrase."
                        )

                # SAM prompt cap + shape (single phrase, no comma /
                # conjunction / negation). Empirically multi-fragment
                # prompts like "adult male voice, American English,
                # mid-pitch" dilute SAM's anchor and the model returns
                # near-zero target_extraction.
                if t.sam_prompt:
                    if len(t.sam_prompt.split()) > 8:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"sam_prompt too long (>{8} words). Keep it "
                            f"to a single `[adjective(s)] noun` phrase."
                        )
                    if "," in t.sam_prompt:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"sam_prompt contains a comma — must be a "
                            f"SINGLE noun phrase (no comma, no 'and'/'or'). "
                            f"If two descriptors are needed, stack as "
                            f"adjectives before one head noun, e.g. "
                            f"'dull plastic thud'."
                        )
                    sp_low = t.sam_prompt.lower()
                    if " and " in sp_low or " or " in sp_low:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"sam_prompt contains 'and'/'or' — pick the "
                            f"stronger noun and drop the connector."
                        )
                    for neg in ("excluding", "without", "except",
                                 "avoid", "instead of", "but not",
                                 "rather than"):
                        if neg in sp_low:
                            violations.append(
                                f"Step {t.step} ({t.action.value}): "
                                f"sam_prompt contains negation token "
                                f"'{neg}'. SAM Audio is positive-only — "
                                f"negation tokens dilute the match."
                            )
                            break

                # deleted_sound must match an existing_sounds entry
                # (loose matching).
                if t.deleted_sound and t.existing_sounds:
                    if not _intent_overlaps_any(
                        t.deleted_sound, t.existing_sounds,
                    ):
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"deleted_sound {t.deleted_sound!r} not "
                            f"found in existing_sounds {t.existing_sounds}."
                        )

                # Speech-preservation criterion: when the audio
                # contains speech AND speech is NOT the target of
                # this step (i.e. it's in the preserve set), the
                # eval_criteria MUST include an explicit speech-
                # intact check. Speech is the most listener-noticeable
                # layer; partial speech loss is a hard fail even when
                # the SFX/music side scores well.
                _SPEECH_TERMS = (
                    "speech", "voice", "voices", "dialogue", "speaker",
                    "speaking", "talk", "talking", "narrator",
                    "narration", "human voice",
                )
                has_speech_in_existing = any(
                    any(term in s.lower() for term in _SPEECH_TERMS)
                    for s in (t.existing_sounds or [])
                )
                deleted_is_speech = bool(t.deleted_sound) and any(
                    term in t.deleted_sound.lower()
                    for term in _SPEECH_TERMS
                )
                volume_target_is_speech = bool(t.volume_target) and any(
                    term in t.volume_target.lower()
                    for term in _SPEECH_TERMS
                )
                speech_is_preserved = (
                    has_speech_in_existing
                    and not deleted_is_speech
                    and not volume_target_is_speech
                )
                if speech_is_preserved:
                    _PRESERVE_KW = (
                        "intact", "preserved", "preserve", "intelligible",
                        "audible", "remain", "remains", "still clear",
                        "still audible", "unchanged", "unaffected",
                    )
                    pooled = (
                        list(t.eval_criteria or [])
                        + list(t.sam_eval_criteria or [])
                    )
                    has_speech_check = any(
                        any(term in c.lower() for term in _SPEECH_TERMS)
                        and any(kw in c.lower() for kw in _PRESERVE_KW)
                        for c in pooled
                    )
                    if not has_speech_check:
                        violations.append(
                            f"Step {t.step} ({t.action.value}): "
                            f"the original audio has speech in "
                            f"existing_sounds and it's not the edit "
                            f"target — eval_criteria or "
                            f"sam_eval_criteria MUST include an "
                            f"explicit speech-preservation check "
                            f"(e.g. 'Speech is preserved intact in the "
                            f"residual.')."
                        )

            # ── SPEECH actions ─────────────────────────────────────
            if t.action in (EditAction.SPEECH_TTS, EditAction.SPEECH_SWAP):
                if not t.speech_text:
                    violations.append(
                        f"Step {t.step} ({t.action.value}): "
                        f"speech_text must be non-empty."
                    )
                if not t.speech_speaker_description:
                    violations.append(
                        f"Step {t.step} ({t.action.value}): "
                        f"speech_speaker_description must be non-empty "
                        f"(SAM Audio separation prompt)."
                    )
                if t.action == EditAction.SPEECH_SWAP and not t.speech_voice_description:
                    violations.append(
                        f"Step {t.step} (speech_swap): "
                        f"speech_voice_description must be non-empty."
                    )

        # Rule: depends_on must reference existing earlier steps only
        step_ids = {t.step for t in subtasks}
        for t in subtasks:
            for dep in t.depends_on:
                if dep not in step_ids:
                    violations.append(
                        f"Step {t.step} depends_on={dep} which does not exist."
                    )
                elif dep >= t.step:
                    violations.append(
                        f"Step {t.step} depends_on={dep} must reference an "
                        f"earlier step (dep < step)."
                    )

        # Rule: prompt fields are sent verbatim to the editing models
        # and MUST NOT leak shot-level terminology (the model sees one
        # clip at a time).
        shot_re = re.compile(r"\b[Ss]hots?\b")
        prompt_fields = (
            "video_prompt", "sam_prompt", "mmaudio_prompt",
            "speech_text", "speech_speaker_description",
            "speech_voice_description",
        )
        for t in subtasks:
            for fld in prompt_fields:
                v = getattr(t, fld, "")
                if v and shot_re.search(v):
                    violations.append(
                        f"Step {t.step} ({t.action.value}): {fld} contains "
                        f"'shot' — forbidden. Describe visible/audible "
                        f"content instead, do not reference shot numbers."
                    )

        # Rule: video edits are per-shot by default. `shot_index=None`
        # is allowed for same-scene/same-camera jump cuts that should be
        # edited as one whole clip; `_run_step_video` already supports
        # that global execution path.
        VIDEO_ACTIONS = {
            EditAction.STYLE_TRANSFER,
            EditAction.SCENE_EDIT,
            EditAction.ADD_OBJECT,
            EditAction.REMOVE_OBJECT,
            EditAction.REPLACE_OBJECT,
            EditAction.RECOLOR,
            EditAction.REPAINTING,
            EditAction.DEPTH_MODIFY,
            EditAction.MOTION_EDIT,
        }
        for t in subtasks:
            if t.action in VIDEO_ACTIONS and t.shot_index is None:
                continue

        # Rule: speech_replace_full must NEVER appear in Phase B output —
        # it should have been split into speech_tts + speech_lipsync.
        for t in subtasks:
            if t.action == EditAction.SPEECH_REPLACE_FULL:
                violations.append(
                    f"Step {t.step}: speech_replace_full must be split into "
                    f"speech_tts + speech_lipsync subtasks in Phase B."
                )

        # Rule: speech_lipsync must depend on the preceding speech audio edit.
        speech_audio_steps = {
            t.step for t in subtasks
            if t.action in (EditAction.SPEECH_TTS, EditAction.SPEECH_SWAP)
        }
        for t in subtasks:
            if t.action == EditAction.SPEECH_LIPSYNC:
                if t.shot_index is None:
                    violations.append(
                        f"Step {t.step} (speech_lipsync) must set shot_index."
                    )
                if not any(d in speech_audio_steps for d in t.depends_on):
                    violations.append(
                        f"Step {t.step} (speech_lipsync) must depends_on a "
                        f"speech_tts/speech_swap step."
                    )

        # NOTE: do not deterministically merge per-shot SubTasks here.
        # `shot_index=null` for video is reserved for the planner's
        # same-scene/same-camera jump-cut judgment, not generic
        # duplicate compression.

        return violations


# ═══════════════════════════════════════════════════════════════════════════
# Planner class (orchestrates Phase A → Validate → Phase B → Validate)
# ═══════════════════════════════════════════════════════════════════════════

def build_audio_inventory(subtasks: list[SubTask]) -> AudioInventory:
    """Derive an `AudioInventory` from the Phase B SubTask list.

    With the new flat schema the inventory data is already on each
    audio SubTask:
      - existing_sounds — same on every audio SubTask of a session
        (validator enforces consistency)
      - deleted_sound / new_sound — single string per SubTask
      - sam_prompt / mmaudio_prompt — for tool calls (not inventory)

    From those we build:
      - original = first audio SubTask's existing_sounds
      - remove   = [t.deleted_sound for AUDIO_REMOVE SubTasks]
      - replace  = [{from: t.deleted_sound, to: t.new_sound}
                    for AUDIO_REPLACE_* SubTasks]
      - add      = [t.new_sound for AUDIO_ADD_* SubTasks]
      - preserve = original − {everything in remove + replace.from}
    """
    audio_subs = [
        t for t in (subtasks or [])
        if t.is_audio and t.action != EditAction.SPEECH_LIPSYNC
    ]
    if not audio_subs:
        return AudioInventory()

    original = list(audio_subs[0].existing_sounds)

    add: list[str] = []
    remove: list[str] = []
    replace: list[dict[str, str]] = []
    volume_adjust: list[dict[str, Any]] = []
    targeted: set[str] = set()

    def _norm(s: str) -> str:
        return (s or "").strip().lower()

    for t in audio_subs:
        if t.action == EditAction.AUDIO_REMOVE:
            if t.deleted_sound:
                remove.append(t.deleted_sound)
                targeted.add(_norm(t.deleted_sound))
        elif t.action in (EditAction.AUDIO_REPLACE_SFX,
                          EditAction.AUDIO_REPLACE_BGM):
            if t.deleted_sound or t.new_sound:
                replace.append({"from": t.deleted_sound, "to": t.new_sound})
                if t.deleted_sound:
                    targeted.add(_norm(t.deleted_sound))
        elif t.action in (EditAction.AUDIO_ADD_SFX,
                          EditAction.AUDIO_ADD_AMBIENT):
            if t.new_sound:
                add.append(t.new_sound)
        elif t.action == EditAction.AUDIO_VOLUME_ADJUST:
            # The stem stays in `preserve` (still audible) — the mix
            # just shifts its level.
            if t.volume_target:
                delta = float(t.volume_db or 0.0)
                volume_adjust.append({
                    "target": t.volume_target,
                    "delta_db": delta,
                    "direction": "boost" if delta > 0 else (
                        "reduce" if delta < 0 else "none"
                    ),
                })
        elif t.action in (EditAction.SPEECH_TTS, EditAction.SPEECH_SWAP):
            # Speech replaces the speaker's voice; treat speech-related
            # entries in existing_sounds as targeted.
            for k in ("speech", "dialogue", "voice", "speaker"):
                targeted.add(k)

    preserve = [
        s for s in original
        if not _intent_overlaps_any(s, list(targeted))
    ]
    return AudioInventory(
        original=original, preserve=preserve,
        remove=remove, add=add, replace=replace,
        volume_adjust=volume_adjust,
    )


class Planner:
    """
    Two-phase planner:
        Phase A: Intent planning (structured WHAT)
        Phase B: Task realization (tool-ready HOW)
    With validation after each phase.
    """

    MAX_REPLAN_ATTEMPTS = 2

    def __init__(
        self,
        llm_cfg: LLMConfig,
        session_dir: Path | None = None,
        video_backend: str = "wan",
    ):
        self.cfg = llm_cfg
        self.model = llm_cfg.model
        self.session_dir = session_dir
        self.validator = PlanValidator()
        # Active video backend ("wan" | "seedance") — selects the
        # backend-specific video_prompt rules in Phase B and the
        # video_prompt word cap in validation.
        self.video_backend = (video_backend or "wan").lower()
        # Stashed by `.plan()` so pipeline can attach it to the session.
        self.last_inventory: AudioInventory | None = None

    # ── helpers ────────────────────────────────────────────────────────

    def _realizer_prompt(self) -> str:
        """Phase B system prompt with the backend-specific video_prompt
        rules block substituted in for the `__VIDEO_PROMPT_RULES__`
        sentinel."""
        rules = (
            VIDEO_PROMPT_RULES_SEEDANCE
            if self.video_backend == "seedance"
            else VIDEO_PROMPT_RULES_WAN
        )
        return REALIZER_SYSTEM_PROMPT.replace("__VIDEO_PROMPT_RULES__", rules)

    def _video_word_cap(self) -> int:
        return VIDEO_WORD_CAP.get(self.video_backend, 22)

    def _build_user_prompt(
        self,
        instruction: str,
        meta: VideoMeta | None,
        keyframes: list[Path] | None = None,
        video_caption: str | None = None,
        shots: list[Shot] | None = None,
    ) -> str | list[dict[str, Any]]:
        """Build user prompt — multimodal (with images) when keyframes are available."""
        text = f'User editing instruction:\n"""\n{instruction}\n"""'
        if meta:
            text += (
                f"\nVideo info: {meta.width}x{meta.height}, "
                f"{meta.fps} fps, {meta.duration:.1f}s, codec={meta.codec}"
            )
        if shots:
            shot_lines = [
                f"  {s.index}. [{s.start:.2f}→{s.end:.2f}s] {s.summary}"
                for s in shots
            ]
            text += (
                "\n\nShot list (from captioner — use these indices in "
                "`shot_index` for per-shot video edits; use "
                "`shot_index=null` only for same-scene/same-camera jump-cut "
                "video edits that should run once on the whole clip):\n"
                + "\n".join(shot_lines)
            )
        if video_caption:
            text += f"\n\nDetailed video content analysis:\n{video_caption}"

        if not keyframes:
            return text

        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for kf in keyframes:
            b64 = base64.b64encode(kf.read_bytes()).decode()
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        return content

    def _parse_json_array(self, text: str) -> list[dict[str, Any]]:
        """Extract JSON array from LLM response (tolerant of markdown fences)."""
        text = text.strip()
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            start = text.find("[")
            end = text.rfind("]")
            if start >= 0 and end > start:
                data = json.loads(text[start : end + 1])
            else:
                raise
        if not isinstance(data, list):
            raise ValueError(f"Expected JSON array, got {type(data).__name__}")
        return data

    def _compact_json_retry_content(
        self,
        user_prompt: str | list[dict[str, Any]],
        error: Exception,
    ) -> str | list[dict[str, Any]]:
        retry_text = (
            "\n\nIMPORTANT: Your previous response was not valid JSON "
            f"({type(error).__name__}: {error}). Return the plan again as "
            "ONLY a compact valid JSON array. No markdown. Keep all strings "
            "short and avoid long copied descriptions."
        )
        if isinstance(user_prompt, str):
            return user_prompt + retry_text
        combined = list(user_prompt)
        combined.append({"type": "text", "text": retry_text})
        return combined

    def _to_subtasks(self, raw: list[dict[str, Any]]) -> list[SubTask]:
        tasks: list[SubTask] = []
        for i, item in enumerate(raw):
            action_str = item.get("action", "")
            try:
                action = EditAction(action_str)
            except ValueError:
                logger.warning("Unknown action '%s' at step %d — skipping",
                               action_str, i + 1)
                continue

            # Phase B must NOT emit raw speech_replace_full — if the LLM
            # still does, skip it (the split rule in the prompt should
            # have generated speech_tts + speech_lipsync instead).
            if action == EditAction.SPEECH_REPLACE_FULL:
                logger.warning(
                    "Step %d emitted raw speech_replace_full — Phase B "
                    "must split into speech_tts + speech_lipsync; skipping",
                    i + 1,
                )
                continue

            shot_index = item.get("shot_index")
            if shot_index is not None:
                try:
                    shot_index = int(shot_index)
                except (TypeError, ValueError):
                    shot_index = None

            depends_on_raw = item.get("depends_on", []) or []
            depends_on = [int(x) for x in depends_on_raw if str(x).lstrip("-").isdigit()]

            # Existing_sounds — must be list[str].
            es = item.get("existing_sounds") or []
            existing_sounds = [s for s in es if isinstance(s, str)] \
                if isinstance(es, list) else []

            # eval_criteria = Phase A's (if any) ∪ Phase B's extra_eval_criteria.
            # Phase A may have left it empty (it's not required there);
            # Phase B's per-step translator adds intent-specific criteria.
            # `extra_eval_criteria` is the FINAL-mix bucket; per-tool
            # criteria live in `sam_eval_criteria` / `mmaudio_eval_criteria`.
            eval_criteria = list(item.get("eval_criteria", []) or [])
            for c in (item.get("extra_eval_criteria", []) or []):
                if isinstance(c, str) and c not in eval_criteria:
                    eval_criteria.append(c)
            sam_eval_criteria = [
                c for c in (item.get("sam_eval_criteria", []) or [])
                if isinstance(c, str)
            ]
            mmaudio_eval_criteria = [
                c for c in (item.get("mmaudio_eval_criteria", []) or [])
                if isinstance(c, str)
            ]

            tasks.append(SubTask(
                step=item.get("step", i + 1),
                action=action,
                target=TargetScope(item.get("target", "global")),
                shot_index=shot_index,
                depends_on=depends_on,
                intent=str(item.get("intent", "") or ""),
                eval_criteria=eval_criteria,
                sam_eval_criteria=sam_eval_criteria,
                mmaudio_eval_criteria=mmaudio_eval_criteria,
                # Video (filled by Phase B for video actions)
                video_prompt=str(item.get("video_prompt", "") or ""),
                # Audio inventory (Phase A) + tool prompts (Phase B)
                existing_sounds=existing_sounds,
                deleted_sound=str(item.get("deleted_sound", "") or ""),
                new_sound=str(item.get("new_sound", "") or ""),
                sam_prompt=str(item.get("sam_prompt", "") or ""),
                mmaudio_prompt=str(item.get("mmaudio_prompt", "") or ""),
                expect_prominent_target=bool(
                    item.get("expect_prominent_target", False)
                ),
                # Audio volume-adjust (Phase A)
                volume_target=str(item.get("volume_target", "") or ""),
                volume_db=(
                    float(item.get("volume_db", 0.0))
                    if isinstance(item.get("volume_db"), (int, float, str))
                    and str(item.get("volume_db")).strip() not in ("", "None")
                    else 0.0
                ),
                # Speech (Phase A)
                speech_text=str(item.get("speech_text", "") or ""),
                speech_speaker_description=str(
                    item.get("speech_speaker_description", "") or ""
                ),
                speech_voice_description=str(
                    item.get("speech_voice_description", "") or ""
                ),
                speech_reference_text=str(
                    item.get("speech_reference_text", "") or ""
                ),
                speech_language=str(item.get("speech_language", "auto") or "auto"),
                audio_splice=(
                    item.get("audio_splice")
                    if isinstance(item.get("audio_splice"), dict)
                    else (
                        {
                            "mode": "localized_replace",
                            "preserve_outside": True,
                            "source": "speech_transcript",
                            "reference_text": str(
                                item.get("speech_reference_text", "") or ""
                            ),
                        }
                        if action in (EditAction.SPEECH_TTS, EditAction.SPEECH_SWAP)
                        else {}
                    )
                ),
            ))
        return tasks

    async def _call_llm(
        self,
        system_prompt: str,
        user_content: str | list[dict[str, Any]],
        component: str = "Planner",
    ) -> str:
        """Call LLM and return response text. All Planner LLM calls
        (Phase A intent, Phase B realize, replans, improvers) flow
        through here so a single grep on `[API] Planner` reveals every
        prompt the planner ever sent."""
        from av_editor.core._gemini_client import generate_from_messages

        return await asyncio.to_thread(
            generate_from_messages,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            model=self.model,
            api_key=self.cfg.gemini_api_key,
            json_response=True,
            temperature=self.cfg.temperature,
            max_output_tokens=self.cfg.max_tokens,
            component=component,
        )

    def _save_json(self, data: Any, filename: str) -> None:
        """Persist JSON to session_dir."""
        if self.session_dir is None:
            return
        try:
            out_path = self.session_dir / filename
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            logger.info("[Planner] saved → %s", out_path)
        except Exception as exc:
            logger.warning("[Planner] failed to save %s: %s", filename, exc)

    # ── Phase A: Intent Planning ──────────────────────────────────────

    async def _phase_a(
        self,
        instruction: str,
        meta: VideoMeta | None,
        keyframes: list[Path] | None,
        video_caption: str | None,
        shots: list[Shot] | None = None,
    ) -> list[dict[str, Any]]:
        """Phase A: produce structured intents."""
        user_prompt = self._build_user_prompt(
            instruction, meta, keyframes, video_caption, shots
        )
        logger.info("[Phase A] generating intents...")
        text = await self._call_llm(INTENT_SYSTEM_PROMPT, user_prompt)
        logger.debug("[Phase A] raw response:\n%s", text)
        try:
            intents = self._parse_json_array(text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "[Phase A] invalid JSON response (%s) — retrying compact",
                exc,
            )
            retry_prompt = self._compact_json_retry_content(user_prompt, exc)
            text = await self._call_llm(
                INTENT_SYSTEM_PROMPT,
                retry_prompt,
                component="PlannerRetry",
            )
            logger.debug("[Phase A retry] raw response:\n%s", text)
            intents = self._parse_json_array(text)
        logger.info("[Phase A] %d intent(s) generated", len(intents))
        return intents

    async def _replan_intents(
        self,
        intents: list[dict[str, Any]],
        violations: list[str],
        instruction: str,
        meta: VideoMeta | None,
        keyframes: list[Path] | None,
        video_caption: str | None,
        shots: list[Shot] | None = None,
    ) -> list[dict[str, Any]]:
        """Re-run Phase A with violation feedback."""
        user_prompt = self._build_user_prompt(
            instruction, meta, keyframes, video_caption, shots
        )

        fix_text = (
            "Your previous plan had the following issues:\n"
            + "\n".join(f"- {v}" for v in violations)
            + "\n\nPlease fix these issues and output the corrected plan."
        )

        if isinstance(user_prompt, str):
            combined = user_prompt + "\n\n" + fix_text
            combined += f"\n\nYour previous output:\n{json.dumps(intents, ensure_ascii=False)}"
        else:
            combined = list(user_prompt)
            combined.append({
                "type": "text",
                "text": fix_text + f"\n\nYour previous output:\n{json.dumps(intents, ensure_ascii=False)}",
            })

        logger.info("[Phase A replan] fixing %d violation(s)...", len(violations))
        text = await self._call_llm(INTENT_SYSTEM_PROMPT, combined)
        try:
            return self._parse_json_array(text)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "[Phase A replan] invalid JSON response (%s) — retrying compact",
                exc,
            )
            retry_prompt = self._compact_json_retry_content(combined, exc)
            text = await self._call_llm(
                INTENT_SYSTEM_PROMPT,
                retry_prompt,
                component="PlannerRetry",
            )
            return self._parse_json_array(text)

    # ── Phase B: Task Realization ─────────────────────────────────────

    async def _phase_b(
        self,
        intents: list[dict[str, Any]],
        instruction: str,
        meta: VideoMeta | None,
        keyframes: list[Path] | None,
        video_caption: str | None,
        shots: list[Shot] | None = None,
    ) -> list[dict[str, Any]]:
        """Phase B: convert intents into tool-ready SubTasks."""
        # Phase B is now a PER-STEP translator: one small LLM call
        # per Phase A step, in parallel. Each call sees only that
        # step's intent + the surrounding caption/shot context, so
        # the prompt stays focused and short.
        import asyncio as _aio

        if not intents:
            return []

        # Shared context block (caption + shots + meta) — reused by
        # every per-step call.
        shared = ""
        if meta:
            shared += (
                f"Video info: {meta.width}x{meta.height}, "
                f"{meta.fps} fps, {meta.duration:.1f}s\n"
            )
        if shots:
            shot_lines = [
                f"  {s.index}. [{s.start:.2f}→{s.end:.2f}s] {s.summary}"
                for s in shots
            ]
            shared += (
                "\nShot list (the step's shot_index refers into this; "
                "shot_index=null on a VIDEO step means Phase A selected "
                "a same-scene/same-camera jump-cut whole-clip edit):\n"
                + "\n".join(shot_lines) + "\n"
            )
        if video_caption:
            shared += f"\nVideo caption:\n{video_caption}\n"
        shared += f"\nUser instruction:\n\"\"\"\n{instruction}\n\"\"\"\n"

        # Sibling-step summary — exposed so video steps can look up
        # paired audio steps (and vice-versa) for cross-modal state
        # injection. See REALIZER_SYSTEM_PROMPT § "Cross-modal state
        # injection" for the rule that consumes this.
        sibling_lines = []
        for s in intents:
            sid = s.get("step")
            act = s.get("action", "?")
            si = s.get("shot_index")
            shot_tag = f" shot={si}" if si is not None else ""
            it = (s.get("intent") or "").strip().replace("\n", " ")
            extra = []
            if s.get("deleted_sound"):
                extra.append(f'del="{s["deleted_sound"]}"')
            if s.get("new_sound"):
                extra.append(f'new="{s["new_sound"]}"')
            extra_tag = (" " + " ".join(extra)) if extra else ""
            sibling_lines.append(
                f"  step {sid} [{act}{shot_tag}]{extra_tag}: {it}"
            )
        shared += (
            "\nAll steps in this plan (use to detect cross-modal coupling "
            "with the step you are realizing — e.g. audio replace_sfx "
            "implying visible state):\n"
            + "\n".join(sibling_lines) + "\n"
        )

        # Pre-encode keyframe image parts once (reused across steps).
        # Lever ①: Phase B used to be pixel-blind (text-only), so the
        # video_prompt could only paraphrase the caption. Handing it the
        # video's keyframes lets it ground the prompt in what is actually
        # on screen. Only visual steps receive the images.
        _VISUAL_ACTIONS = {
            "style_transfer", "scene_edit", "add_object", "remove_object",
            "replace_object", "recolor", "repainting", "depth_modify",
            "motion_edit", "speech_lipsync",
        }
        kf_parts: list[dict[str, Any]] = []
        for kf in (keyframes or []):
            try:
                b64 = base64.b64encode(Path(kf).read_bytes()).decode()
            except Exception as exc:
                logger.warning("[Phase B] keyframe %s unreadable: %s", kf, exc)
                continue
            kf_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        if kf_parts:
            logger.info("[Phase B] grounding visual steps with %d keyframe(s)",
                        len(kf_parts))

        async def _translate_step(idx: int, step: dict) -> dict:
            user_text = (
                shared
                + "\n--- THIS STEP ONLY ---\n"
                + json.dumps(step, ensure_ascii=False, indent=2)
                + "\n\nReturn the JSON object with the prompt fields and "
                "extra_eval_criteria for this step."
            )
            # Attach keyframes for visual steps so the realizer grounds the
            # prompt in the actual frame; audio-only steps stay text-only.
            is_visual = str(step.get("action", "")).lower() in _VISUAL_ACTIONS
            user_content: Any = (
                [{"type": "text", "text": user_text}] + kf_parts
                if (kf_parts and is_visual)
                else user_text
            )
            try:
                text = await self._call_llm(
                    self._realizer_prompt(), user_content,
                    component=f"PhaseB[step{step.get('step', idx)}]",
                )
                obj = json.loads(re.sub(
                    r"^```(?:json)?\s*|\s*```$", "", text.strip(),
                ))
                if not isinstance(obj, dict):
                    raise ValueError(f"expected JSON object, got {type(obj).__name__}")
                # Carry Phase A's fields into the merged dict; Phase B
                # output overlays its prompt fields + extra_eval_criteria.
                merged = dict(step)
                merged.update(obj)
                return merged
            except Exception as exc:
                logger.warning(
                    "[Phase B] step %s translation failed: %s — "
                    "keeping Phase A step unchanged",
                    step.get("step", idx), exc,
                )
                return dict(step)

        logger.info("[Phase B] translating %d step(s) (parallel)...", len(intents))
        merged_tasks = await _aio.gather(
            *(_translate_step(i, s) for i, s in enumerate(intents))
        )
        logger.info("[Phase B] %d step(s) translated", len(merged_tasks))
        return list(merged_tasks)

    async def _replan_subtasks(
        self,
        raw_tasks: list[dict[str, Any]],
        violations: list[str],
        intents: list[dict[str, Any]],
        instruction: str,
        video_caption: str | None,
        meta: VideoMeta | None = None,
        shots: list[Shot] | None = None,
    ) -> list[dict[str, Any]]:
        """Re-run Phase B per-step translation. Violations are logged
        for the LLM to consider; each step is re-translated in parallel
        with the violation list appended as guidance.

        With per-step Phase B, individual steps rarely fail validation;
        most planning rules live in Phase A. This path is a safety net
        for tool-prompt format violations (e.g. video_prompt > 12 words)."""
        logger.info("[Phase B replan] fixing %d violation(s)...", len(violations))
        repair_instruction = (
            instruction
            + "\n\nPhase B validation feedback from the checker:\n"
            + "\n".join(f"- {v}" for v in violations)
            + "\n\nRegenerate the tool prompt fields so every checker "
            "violation is fixed. Do not post-process; output corrected "
            "JSON fields directly."
        )
        return await self._phase_b(
            intents, repair_instruction, meta, None, video_caption, shots,
        )

    # ── public API ────────────────────────────────────────────────────

    async def plan(
        self,
        instruction: str,
        meta: VideoMeta | None = None,
        keyframes: list[Path] | None = None,
        video_caption: str | None = None,
        shots: list[Shot] | None = None,
        replan_feedback: str | None = None,
    ) -> list[SubTask]:
        """
        Full two-phase planning pipeline:
            Phase A (intent) → validate → Phase B (realize) → validate
        """
        planning_instruction = instruction
        if replan_feedback:
            planning_instruction += (
                "\n\nFULL-REPLAN CONTEXT FROM THE FINAL MIXED EVALUATOR:\n"
                + replan_feedback.strip()
                + "\nTreat this evaluation as evidence, not as a prescribed "
                "solution. Independently diagnose the failure and decide how "
                "to improve the intents, dependencies, scopes, tool prompts, "
                "and evaluation criteria. Rebuild the plan from the original "
                "source while preserving every requirement in the original "
                "instruction; do not merely retry the previous plan."
            )
        logger.info("[Planner] instruction: %s", instruction)
        if replan_feedback:
            logger.info("[Planner] full-replan feedback: %s", replan_feedback)
        logger.info("[Planner] %d keyframe(s) provided",
                     len(keyframes) if keyframes else 0)

        # ── Phase A: Intent Planning ─────────────────────────────────
        intents = await self._phase_a(
            planning_instruction, meta, keyframes, video_caption, shots
        )

        for attempt in range(self.MAX_REPLAN_ATTEMPTS):
            violations = self.validator.validate_intents(intents)
            if not violations:
                break
            logger.warning("[Phase A] validation failed (%d issue(s)):",
                           len(violations))
            for v in violations:
                logger.warning("  ✗ %s", v)
            intents = await self._replan_intents(
                intents, violations, planning_instruction, meta, keyframes,
                video_caption, shots,
            )
        else:
            # Log remaining violations but continue
            violations = self.validator.validate_intents(intents)
            if violations:
                logger.warning("[Phase A] still %d issue(s) after %d retries — continuing",
                               len(violations), self.MAX_REPLAN_ATTEMPTS)

        self._save_json(
            {
                "instruction": instruction,
                "replan_feedback": replan_feedback or "",
                "intents": intents,
            },
            "plan_intents.json",
        )

        for i, intent in enumerate(intents):
            label_audio = (
                f"deleted={intent.get('deleted_sounds')} "
                f"new={intent.get('new_sounds')}"
                if intent.get("is_audio") else ""
            )
            logger.info("  intent %d: %s [%s] %s",
                         i + 1,
                         intent.get("edit_type", "?"),
                         "AUDIO" if intent.get("is_audio") else intent.get("target", "?"),
                         label_audio or intent.get("change_to", "?"))

        # ── Phase B: Task Realization ────────────────────────────────
        raw_tasks = await self._phase_b(
            intents, planning_instruction, meta, keyframes, video_caption, shots
        )
        subtasks = self._to_subtasks(raw_tasks)

        for attempt in range(self.MAX_REPLAN_ATTEMPTS):
            violations = self.validator.validate_subtasks(
                subtasks, num_shots=len(shots or []) or 1,
                video_word_cap=self._video_word_cap(),
                video_backend=self.video_backend,
            )
            if not violations:
                break
            logger.warning("[Phase B] validation failed (%d issue(s)):",
                           len(violations))
            for v in violations:
                logger.warning("  ✗ %s", v)
            raw_tasks = await self._replan_subtasks(
                raw_tasks, violations, intents, planning_instruction, video_caption,
                meta=meta, shots=shots,
            )
            subtasks = self._to_subtasks(raw_tasks)
        else:
            violations = self.validator.validate_subtasks(
                subtasks, num_shots=len(shots or []) or 1,
                video_word_cap=self._video_word_cap(),
                video_backend=self.video_backend,
            )
            if violations:
                logger.warning("[Phase B] still %d issue(s) after %d retries — continuing",
                               len(violations), self.MAX_REPLAN_ATTEMPTS)

        # ── Deterministic post-processor: merge near-duplicate per-shot
        # subtasks. The validator catches them and triggers replans, but
        # the LLM is unreliable about actually merging on retry. Do it
        # mechanically here so we don't ship 3+ identical video calls.
        subtasks = _merge_duplicate_pershot(subtasks)
        # Re-derive raw_tasks so plan.json reflects the merged plan
        raw_tasks = [t.to_dict() for t in subtasks]

        # ── Derive audio inventory FROM SubTasks (single source) ────
        self.last_inventory = build_audio_inventory(subtasks)
        self._save_json(self.last_inventory.to_dict(), "audio_inventory.json")
        logger.info(
            "[Planner] audio_inventory: preserve=%s remove=%s add=%s replace=%s",
            self.last_inventory.preserve, self.last_inventory.remove,
            self.last_inventory.add, self.last_inventory.replace,
        )

        # ── Save & log ───────────────────────────────────────────────
        self._save_json(
            {
                "instruction": instruction,
                "replan_feedback": replan_feedback or "",
                "model": self.model,
                "subtasks": raw_tasks,
            },
            "plan.json",
        )

        logger.info("[Planner] final plan: %d subtasks (%d audio)",
                     len(subtasks),
                     sum(1 for t in subtasks if t.is_audio))
        for t in subtasks:
            # Compact one-line summary picking the modality's primary
            # prompt field.
            primary = (
                t.video_prompt or t.mmaudio_prompt or t.sam_prompt
                or t.speech_text or "(no prompt)"
            )
            logger.info("  step %d: %s [%s] %s",
                         t.step, t.action.value,
                         "AUDIO" if t.is_audio else t.target.value,
                         primary)
            if t.eval_criteria:
                for c in t.eval_criteria:
                    logger.info("    ✓ %s", c)
            if t.deleted_sound:
                logger.info("    ⨯deleted: %s", t.deleted_sound)
            if t.new_sound:
                logger.info("    +new: %s", t.new_sound)

        return subtasks
