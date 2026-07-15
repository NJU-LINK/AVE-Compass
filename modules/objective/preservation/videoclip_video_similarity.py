from pathlib import Path
from typing import Any, Dict

from modules.objective.preservation.video_similarity_adapter import compute_visual_video_similarity


def compute_videoclip_video_similarity(sample: Dict[str, Any], config: Dict[str, Any], work_dir: Path) -> Dict[str, Any]:
    return compute_visual_video_similarity("videoclip_video_similarity", sample, config, work_dir)
