# Realism Evaluation Prompt

# Task
Rate the Realism of the TARGET edited audio-video on 5 sub-dimensions (each 1-5).
Realism = whether the target is natural, well-formed and coherent, NOT whether it follows the instruction.

# Rules
- Judge the TARGET only. The edit instruction and source caption are context, not requirements.
- Do not penalize an edit that was not performed. If an attempted change is absent, ignore it.
- For audio dimensions, output "NA" if there is no audio or no clear sound source.
- Motion blur is scene-adaptive: blur from fast motion or camera movement is acceptable; blur on static content is a defect.

# Shared 1-5 scale
5 = fully natural and coherent, with no issues.
4 = minor warp, distortion, or blur that does not affect recognition or perception.
3 = noticeable AI-generated appearance, illogical behavior, or a physics violation, but no glaring error.
2 = clear deformation, abnormal or extra object, sudden appearance or disappearance, clipping, or an unblended edit region.
1 = severe or broken.

# Sub-dimensions
VISUAL
- object_integrity: subject form and existence. Penalize structural errors, texture morphing, flicker, identity changes, and unexplained objects appearing or disappearing. Reasonable occlusion and exits are acceptable.
- interaction_physics: physical and commonsense logic. Penalize violations of gravity, contact, collision, liquids, slicing, breaking, pouring, support, and action-object correspondence.
- naturalness: whether the result looks like a real, coherently styled video. Penalize mismatched lighting, shadows, colors, seams, inconsistent style, warping, flicker, and jitter.

AUDIO
- AAS: signal purity, including clipping, static, dropout, and robotic or distorted sound. Use "NA" if there is no audio.
- MTC: whether sound timbre matches visible materials and room acoustics. Use "NA" if there is no clear sound source.

# Context
Edit instruction: {edit_prompt}
Source caption: {caption}

# Output
Return only this JSON:
{
  "object_integrity": <integer 1-5>,
  "interaction_physics": <integer 1-5>,
  "naturalness": <integer 1-5>,
  "AAS": <integer 1-5 or "NA">,
  "MTC": <integer 1-5 or "NA">,
  "reason": "one concise sentence naming the dominant issue, or none"
}
