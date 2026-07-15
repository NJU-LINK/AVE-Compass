"""
schema.py - Data models for the AVE Agent pipeline.

All intermediate data structures are defined here as dataclasses/TypedDicts
so that every module shares the same contract.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EditAction(str, Enum):
    """Atomic editing action types (video + audio)."""
    # ── Video actions ──
    STYLE_TRANSFER = "style_transfer"
    SCENE_EDIT = "scene_edit"
    ADD_OBJECT = "add_object"
    REMOVE_OBJECT = "remove_object"
    REPLACE_OBJECT = "replace_object"
    RECOLOR = "recolor"
    REPAINTING = "repainting"
    DEPTH_MODIFY = "depth_modify"
    # Changes to how a subject MOVES or behaves over time — covers
    # human actions (cough, sob, wave, stand up), object motion (a ball
    # rolling faster, a car stopping) and action-intensity tweaks
    # (gentle sob → violent crying). Visual-only; pair with an audio
    # step if the motion implies a new sound.
    MOTION_EDIT = "motion_edit"
    # ── Audio actions ──
    AUDIO_ADD_SFX = "audio_add_sfx"
    AUDIO_ADD_AMBIENT = "audio_add_ambient"
    AUDIO_REPLACE_SFX = "audio_replace_sfx"
    AUDIO_REPLACE_BGM = "audio_replace_bgm"
    AUDIO_REMOVE = "audio_remove"
    # Pure loudness/dynamics — adjust the level of an existing stem
    # (speech, BGM, an SFX layer) without generating new audio. SAM
    # separates the stem; ffmpeg `volume=` re-mixes at the new level.
    AUDIO_VOLUME_ADJUST = "audio_volume_adjust"
    # ── Speech actions ──
    # High-level Phase A intent — Phase B splits this into the two
    # atomic actions below.
    SPEECH_REPLACE_FULL = "speech_replace_full"
    # Atomic speech actions emitted by Phase B:
    SPEECH_TTS = "speech_tts"              # audio: voice-clone + mix → edited_audio
    SPEECH_SWAP = "speech_swap"            # audio: voice-design (identity change) + mix
    SPEECH_LIPSYNC = "speech_lipsync"      # video: lipsync one shot to edited_audio slice

    # ── helpers ──

    @property
    def is_audio(self) -> bool:
        """True for actions whose primary output is the audio track."""
        if self.value.startswith("audio_"):
            return True
        # speech_tts (voice-clone) and speech_swap (voice-design) both
        # produce audio (mix of cloned/designed voice + residual).
        # speech_lipsync produces video.
        return self.value in ("speech_tts", "speech_swap")

    @property
    def is_video(self) -> bool:
        return not self.is_audio

    @property
    def is_speech(self) -> bool:
        return self.value.startswith("speech_")


class TargetScope(str, Enum):
    GLOBAL = "global"     # edit applies to the whole frame
    LOCAL = "local"       # edit applies to a region / object


class StepStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvalVerdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"


# ---------------------------------------------------------------------------
# Planner output
# ---------------------------------------------------------------------------

@dataclass
class SubTask:
    """One atomic editing step produced by the Planner.

    Schema is FLAT and EXPLICIT — fields are named per modality so
    pipeline branches read them directly without dict-key archaeology.
    Phase B is responsible for filling exactly the fields its action
    type needs; the validator rejects mismatches.

    Field groups by modality:
      • Common:  step, action, target, shot_index, depends_on,
                 eval_criteria
      • Video:   video_prompt
      • Audio:   existing_sounds, deleted_sound, new_sound,
                 sam_prompt, mmaudio_prompt, expect_prominent_target
      • Speech:  speech_text, speech_speaker_description,
                 speech_voice_description, speech_reference_text,
                 speech_language, audio_splice

    Audio splitting rules (enforced by Phase B + validator):
      - ≤1 deletion per SubTask → multi-delete intent expands into
        multiple SubTasks.
      - new_sound is a single sensory prompt; multiple new_sounds
        from one intent are merged by Phase B unless the scenario is
        complex (then split into multiple SubTasks).
      - existing_sounds is the FULL original-audio inventory and is
        carried verbatim on every audio SubTask of a session.

    Pipeline derives `preserve` and MMAudio negative_prompt from
    `existing_sounds − {deleted_sound}` deterministically — no
    per-SubTask `mmaudio_negative_prompt` field.
    """
    step: int
    action: EditAction
    target: TargetScope = TargetScope.GLOBAL
    shot_index: Optional[int] = None
    depends_on: list[int] = field(default_factory=list)
    # Phase A's plain-language description of THIS step's edit intent.
    # No tool-format constraints (no imperative requirement, no word
    # cap). Phase B reads this + the step's modality fields to write
    # the actual tool prompts. Also surfaced to evaluators so the
    # check-points stay grounded in user intent.
    intent: str = ""
    # Audio-side criteria for the post-branch mix and the final
    # whole-video eval. Per-tool stages (SAM / MMAudio) have their
    # own dedicated criteria below.
    eval_criteria: list[str] = field(default_factory=list)
    # Per-tool eval criteria, scoped to the call those tools make
    # before any mixing happens. `sam_eval_criteria` is consumed by
    # the SAM separation evaluator (target stem + residual stem
    # checks). `mmaudio_eval_criteria` is consumed by the MMAudio
    # generation evaluator (raw generated audio score).
    sam_eval_criteria: list[str] = field(default_factory=list)
    mmaudio_eval_criteria: list[str] = field(default_factory=list)

    # ── Video-specific ────────────────────────────────────────────
    video_prompt: str = ""

    # ── Audio-specific ────────────────────────────────────────────
    existing_sounds: list[str] = field(default_factory=list)
    deleted_sound: str = ""
    new_sound: str = ""
    sam_prompt: str = ""
    mmaudio_prompt: str = ""
    expect_prominent_target: bool = False

    # ── Audio volume-adjust-specific (action == AUDIO_VOLUME_ADJUST)
    # `volume_target` is the stem to adjust (must overlap an item in
    # `existing_sounds`). `volume_db` is a SIGNED gain delta in dB —
    # positive = louder, negative = softer. Hard-clamped to ±12 dB at
    # the validator level.
    volume_target: str = ""
    volume_db: float = 0.0

    # ── Speech-specific ───────────────────────────────────────────
    speech_text: str = ""
    speech_speaker_description: str = ""
    speech_voice_description: str = ""
    speech_reference_text: str = ""
    speech_language: str = "auto"
    # Speech timeline policy. For local line rewrites this tells the
    # runner to preserve the original audio outside the matched
    # transcript window and only replace the target line region.
    # Example:
    # {"mode": "localized_replace", "preserve_outside": true,
    #  "source": "speech_transcript", "reference_text": "..."}
    audio_splice: dict[str, Any] = field(default_factory=dict)

    @property
    def is_audio(self) -> bool:
        return self.action.is_audio

    def to_dict(self) -> dict:
        d: dict[str, Any] = {
            "step": self.step,
            "action": self.action.value,
            "target": self.target.value,
            "is_audio": self.is_audio,
            "shot_index": self.shot_index,
            "depends_on": self.depends_on,
            "intent": self.intent,
            "eval_criteria": self.eval_criteria,
        }
        if self.video_prompt:
            d["video_prompt"] = self.video_prompt
        if self.action.is_audio or self.action == EditAction.AUDIO_REMOVE:
            d["existing_sounds"] = self.existing_sounds
            d["deleted_sound"] = self.deleted_sound
            d["new_sound"] = self.new_sound
            d["sam_prompt"] = self.sam_prompt
            d["mmaudio_prompt"] = self.mmaudio_prompt
            if self.sam_eval_criteria:
                d["sam_eval_criteria"] = self.sam_eval_criteria
            if self.mmaudio_eval_criteria:
                d["mmaudio_eval_criteria"] = self.mmaudio_eval_criteria
            if self.expect_prominent_target:
                d["expect_prominent_target"] = True
            if self.action == EditAction.AUDIO_VOLUME_ADJUST:
                d["volume_target"] = self.volume_target
                d["volume_db"] = self.volume_db
        if self.action.is_speech:
            for k in ("speech_text", "speech_speaker_description",
                     "speech_voice_description", "speech_reference_text"):
                v = getattr(self, k)
                if v:
                    d[k] = v
            if self.speech_language != "auto":
                d["speech_language"] = self.speech_language
            if self.audio_splice:
                d["audio_splice"] = self.audio_splice
        return d


# ---------------------------------------------------------------------------
# Video metadata extracted by Preprocessor
# ---------------------------------------------------------------------------

@dataclass
class VideoMeta:
    width: int
    height: int
    fps: float
    duration: float          # seconds
    codec: str
    total_frames: int


@dataclass
class Shot:
    """One camera shot detected by the captioner.

    Indices are 1-based and contiguous; start/end are seconds relative
    to the source video.
    """
    index: int
    start: float
    end: float
    summary: str = ""

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "summary": self.summary,
        }


# ---------------------------------------------------------------------------
# Preprocessor output
# ---------------------------------------------------------------------------

@dataclass
class PreprocessResult:
    video_path: Path          # video-only file (no audio), for editing tools
    audio_path: Optional[Path]  # extracted audio (None if silent)
    meta: VideoMeta
    has_audio: bool
    original_video: Optional[Path] = None  # original video with audio, for captioning


# ---------------------------------------------------------------------------
# Tool call record
# ---------------------------------------------------------------------------

@dataclass
class ToolCall:
    tool_name: str
    input_params: dict[str, Any]
    output_path: Optional[Path] = None
    raw_response: dict[str, Any] = field(default_factory=dict)
    success: bool = False
    error_msg: str = ""


# ---------------------------------------------------------------------------
# Evaluation records
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    verdict: EvalVerdict
    quality_score: float = 0.0        # 0-1, how well the edit matches intent
    consistency_score: float = 0.0    # 0-1, how little the non-target changed
    reason: str = ""
    checklist_scores: dict[str, float] = field(default_factory=dict)  # per-criterion scores


@dataclass
class AudioEvalResult:
    """Evaluation result for the audio editing stage."""
    instruction_score: float = 0.0           # 0-1: requested sounds added/removed/replaced correctly
    sync_score: float = 0.0                  # 0-1: audio-visual synchronization quality
    fidelity_score: float = 0.0             # 0-1: original audio (BGM/ambient) preserved clearly
    overall_score: float = 0.0              # weighted average
    passed: bool = False
    reason: str = ""
    checklist_scores: dict[str, float] = field(default_factory=dict)
    generated_audio_too_quiet: bool = False  # generated sound present but too low in mix
    original_audio_too_quiet: bool = False   # original audio present but overpowered
    generated_audio_contamination: bool = False  # unwanted voice/content leaked into SFX


# ---------------------------------------------------------------------------
# V2 — stage-specific audio evaluators
# ---------------------------------------------------------------------------

@dataclass
class AudioInventory:
    """Ground-truth audio plan derived from the caption + intents.

    `original` lists the sounds the captioner reported in the source.
    `preserve` / `remove` / `add` / `replace` describe what each audio
    edit is supposed to do to that inventory — they are the check-points
    for every downstream audio evaluator.
    """
    original: list[str] = field(default_factory=list)     # sounds already present
    preserve: list[str] = field(default_factory=list)     # must remain audible
    remove: list[str] = field(default_factory=list)       # must disappear
    add: list[str] = field(default_factory=list)          # must appear
    replace: list[dict[str, str]] = field(default_factory=list)  # [{"from":..,"to":..}]
    # Pure loudness adjustments. Each entry: {target: <stem name>,
    # delta_db: <signed float>, direction: "boost"|"reduce"}.
    # The stem itself stays in `preserve` — the mix just rebalances.
    volume_adjust: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "original": self.original,
            "preserve": self.preserve,
            "remove": self.remove,
            "add": self.add,
            "replace": self.replace,
            "volume_adjust": self.volume_adjust,
        }


@dataclass
class SeparationEvalResult:
    """Evaluator output for one SAM Audio separation call."""
    passed: bool = False
    # residual should contain preserve[] and must NOT contain target.
    residual_fidelity: float = 0.0       # 0-1: preserve items still in residual
    target_extraction: float = 0.0       # 0-1: target fully moved into the target stem
    reason: str = ""
    # Human-readable suggestion for the SAM prompt rewriter when passed=False.
    rewrite_hint: str = ""


@dataclass
class GenerationEvalResult:
    """Evaluator output for one MMAudio / Qwen TTS generation call."""
    passed: bool = False
    content_score: float = 0.0           # 0-1: requested sound(s) present
    sync_score: float = 0.0              # 0-1: events aligned with video
    hallucination: bool = False          # unexpected content (e.g. speech in rain SFX)
    duplicated_from_negative: bool = False  # re-generated sounds we explicitly excluded
    reason: str = ""
    rewrite_hint: str = ""


@dataclass
class VolumeAdjustment:
    """Concrete numeric knobs for ffmpeg amix on the next mix attempt."""
    original_volume: float = 1.0
    generated_volume: float = 1.0

    def to_dict(self) -> dict:
        return {
            "original_volume": self.original_volume,
            "generated_volume": self.generated_volume,
        }


@dataclass
class MixEvalResult:
    """Evaluator output for the final muxed video.

    When `passed=False`, inspect the actionable flags and pick the cheapest
    remediation:
      - volume_adjustment set → re-run mix_audio_tracks only (ffmpeg, ~0 cost).
      - needs_regenerate=True → bubble back up to the generation / separation
        retry loop; content itself is broken.
      - needs_replan=True → return evaluator feedback to the planner and rerun
        the complete plan from the original inputs.
    """
    passed: bool = False
    overall_score: float = 0.0
    instruction_score: float = 0.0       # requested edit and plan-criteria conformance
    fidelity_score: float = 0.0          # preservation of non-target source content
    quality_score: float = 0.0           # perceptual and cross-modal coherence
    volume_balance: float = 0.0          # 0-1: original vs generated loudness balance
    reason: str = ""
    volume_adjustment: Optional[VolumeAdjustment] = None
    needs_regenerate: bool = False       # content broken, not fixable by remix
    needs_replan: bool = False           # plan/dependency failure, not locally repairable
    replan_confidence: float = 0.0        # evaluator confidence after trigger gating
    replan_feedback: str = ""            # actionable context for the next planning pass


# ---------------------------------------------------------------------------
# State snapshot (one per step)
# ---------------------------------------------------------------------------

@dataclass
class StateSnapshot:
    step_id: int
    video_path: Path                        # video file AFTER this step
    description: str                        # semantic description of current state
    tool_call: Optional[ToolCall] = None
    eval_result: Optional[EvalResult] = None
    status: StepStatus = StepStatus.PENDING


# ---------------------------------------------------------------------------
# Pipeline-level session
# ---------------------------------------------------------------------------

@dataclass
class EditSession:
    """Top-level session that tracks the full editing pipeline."""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    original_video: Path = Path()
    instruction: str = ""
    subtasks: list[SubTask] = field(default_factory=list)
    preprocess: Optional[PreprocessResult] = None
    shots: list["Shot"] = field(default_factory=list)
    history: list[StateSnapshot] = field(default_factory=list)
    final_output: Optional[Path] = None
    audio_inventory: Optional[AudioInventory] = None

    @property
    def current_video(self) -> Path:
        """Return the latest video path in the editing chain."""
        if self.history:
            return self.history[-1].video_path
        if self.preprocess:
            return self.preprocess.video_path
        return self.original_video
