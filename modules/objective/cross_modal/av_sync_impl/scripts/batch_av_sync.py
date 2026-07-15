#!/usr/bin/env python3
import argparse
import json
import os
import sys
import torch
import torchvision
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
from omegaconf import OmegaConf
from tqdm import tqdm

try:
    import torchvision.transforms.functional_tensor as _ft
except ImportError:
    import torchvision.transforms.functional as _ft_module
    sys.modules['torchvision.transforms.functional_tensor'] = _ft_module

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}


def reencode_video(path, vfps=25, afps=16000, in_size=256):
    """重新编码视频到指定格式"""
    import subprocess
    new_path = Path.cwd() / 'vis' / f'{Path(path).stem}_{vfps}fps_{in_size}side_{afps}hz.mp4'
    new_path.parent.mkdir(exist_ok=True)
    new_path = str(new_path)
    cmd = 'ffmpeg'
    cmd += ' -hide_banner -loglevel panic'
    cmd += f' -y -i {path}'
    cmd += f" -vf fps={vfps},scale=iw*{in_size}/'min(iw,ih)':ih*{in_size}/'min(iw,ih)',crop='trunc(iw/2)'*2:'trunc(ih/2)'*2"
    cmd += f" -ar {afps}"
    cmd += f' {new_path}'
    subprocess.call(cmd.split())
    return new_path


def patch_config(cfg):
    """修复配置"""
    cfg.model.params.afeat_extractor.params.ckpt_path = None
    cfg.model.params.vfeat_extractor.params.ckpt_path = None
    cfg.model.params.transformer.target = cfg.model.params.transformer.target\
                                             .replace('.modules.feature_selector.', '.sync_model.')
    return cfg


def process_single_video(vid_path: str, model, transforms, grid, device, cfg, 
                         offset_sec: float = 0.0, v_start_i_sec: float = 0.0) -> Dict[str, Any]:
    """处理单个视频文件"""
    vfps = 25
    afps = 16000
    in_size = 256
    
    try:
        v, _, info = torchvision.io.read_video(vid_path, pts_unit='sec')
        _, H, W, _ = v.shape
        if info['video_fps'] != vfps or info['audio_fps'] != afps or min(H, W) != in_size:
            vid_path = reencode_video(vid_path, vfps, afps, in_size)
        
        from dataset.dataset_utils import get_video_and_audio
        from scripts.train_utils import prepare_inputs
        
        rgb, audio, meta = get_video_and_audio(vid_path, get_meta=True)

        # Pad short videos to at least crop_len_sec so transforms don't fail
        # The actual crop_len_sec is computed by TemporalCropAndOffsetForSyncabilityTraining:
        #   seg_size_sec = segment_size_vframes / vfps
        #   trim_size_in_seg = n_segments - (1 - step_size_seg) * (n_segments - 1)
        #   crop_len_sec = trim_size_in_seg * seg_size_sec
        # Use cfg.data.crop_len_sec (5s) as upper bound to be safe
        crop_len_sec = cfg.data.crop_len_sec
        a_fps_actual = meta['audio']['framerate'][0]
        v_fps_actual = meta['video']['fps'][0]
        min_vframes = int(v_fps_actual * crop_len_sec)
        min_aframes = int(a_fps_actual * crop_len_sec)
        if rgb.shape[0] < min_vframes:
            pad_v = min_vframes - rgb.shape[0]
            rgb = torch.cat([rgb, rgb[-1:].repeat(pad_v, 1, 1, 1)], dim=0)
        if audio.shape[0] < min_aframes:
            pad_a = min_aframes - audio.shape[0]
            audio = torch.cat([audio, torch.zeros(pad_a, dtype=audio.dtype)], dim=0)

        item = dict(
            video=rgb,
            audio=audio,
            meta=meta,
            path=vid_path,
            split='test',
            targets={
                'v_start_i_sec': v_start_i_sec,
                'offset_sec': offset_sec,
            },
        )
        
        item = transforms['test'](item)
        batch = torch.utils.data.default_collate([item])
        aud, vid, targets = prepare_inputs(batch, device)
        
        with torch.set_grad_enabled(False):
            with torch.autocast('cuda', enabled=cfg.training.use_half_precision):
                _, logits = model(vid, aud)
        
        off_probs = torch.softmax(logits, dim=-1)
        k = min(off_probs.shape[-1], 5)
        topk_logits, topk_preds = torch.topk(logits, k)
        
        pred_class = int(topk_preds[0, 0].item())
        pred_offset_sec = float(grid[pred_class].item())
        pred_prob = float(off_probs[0, pred_class].item())
        
        return {
            'video_path': str(vid_path),
            'predicted_offset_sec': pred_offset_sec,
            'predicted_class': pred_class,
            'predicted_probability': pred_prob,
            'success': True,
            'error': None
        }
        
    except Exception as e:
        return {
            'video_path': str(vid_path),
            'predicted_offset_sec': None,
            'predicted_class': None,
            'predicted_probability': None,
            'success': False,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description='音视频同步评估')
    parser.add_argument('--video_dir', required=True, help='视频目录')
    parser.add_argument('--output_file', required=True, help='输出JSON文件')
    parser.add_argument('--device', default='cuda:0', help='设备')
    parser.add_argument('--exp_name', default='24-01-04T16-39-21', help='模型实验名称')
    parser.add_argument('--model_base', default=None, help='模型基础目录(默认自动查找)')
    args = parser.parse_args()
    
    video_dir = Path(args.video_dir)
    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    video_files = sorted([p for p in video_dir.iterdir() if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS])
    
    if not video_files:
        payload = {
            'metric': 'av_sync',
            'summary': {'mean_offset': 0.0, 'mean_confidence': 0.0, 'total_samples': 0},
            'results': []
        }
        output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
        return
    
    synchformer_dir = Path(__file__).resolve().parents[1] / 'Objective' / 'Similarity' / 'Synchformer-main'
    project_dir = Path(__file__).resolve().parents[5]
    models_sync_dir = project_dir / 'models' / 'cross_modal' / 'av_sync' / 'synchformer'
    
    sys.path.insert(0, str(synchformer_dir))
    sys.path.insert(0, str(synchformer_dir / 'model' / 'modules' / 'feat_extractors' / 'visual'))
    
    from dataset.transforms import make_class_grid
    from scripts.train_utils import get_model, get_transforms
    from utils.utils import check_if_file_exists_else_download
    
    print(f"加载Synchformer模型... (设备: {args.device})")
    device = args.device if args.device.startswith('cuda') and torch.cuda.is_available() else 'cpu'
    
    if args.model_base:
        model_base = Path(args.model_base)
    else:
        model_base = None
    
    cfg_path = None
    ckpt_path = None
    
    search_dirs = []
    if model_base:
        search_dirs.append(model_base)
    search_dirs.extend([models_sync_dir, synchformer_dir])
    
    for base in search_dirs:
        candidate_cfg = base / 'logs' / 'sync_models' / args.exp_name / f'cfg-{args.exp_name}.yaml'
        candidate_ckpt = base / 'logs' / 'sync_models' / args.exp_name / f'{args.exp_name}.pt'
        if candidate_cfg.exists() and candidate_ckpt.exists():
            cfg_path = candidate_cfg
            ckpt_path = candidate_ckpt
            print(f"找到模型文件: {base}")
            break
    
    if cfg_path is None or ckpt_path is None:
        print(f"模型文件未找到，尝试下载...")
        cfg_path = synchformer_dir / 'logs' / 'sync_models' / args.exp_name / f'cfg-{args.exp_name}.yaml'
        ckpt_path = synchformer_dir / 'logs' / 'sync_models' / args.exp_name / f'{args.exp_name}.pt'
        try:
            check_if_file_exists_else_download(str(cfg_path))
            check_if_file_exists_else_download(str(ckpt_path))
        except Exception as e:
            print(f"模型文件不存在: {e}")
            payload = {
                'metric': 'av_sync',
                'summary': {'mean_offset': 0.0, 'mean_confidence': 0.0, 'total_samples': len(video_files)},
                'results': [{'file': str(v), 'filename': v.name, 'predicted_offset_sec': None, 
                            'predicted_probability': None, 'sync_quality': None, 'error': '模型文件不存在'} 
                           for v in video_files]
            }
            output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
            return
    
    cfg = OmegaConf.load(str(cfg_path))
    cfg = patch_config(cfg)
    
    _, model = get_model(cfg, device)
    ckpt = torch.load(str(ckpt_path), map_location=torch.device('cpu'), weights_only=False)
    model.load_state_dict(ckpt['model'])
    model.eval()
    
    transforms = get_transforms(cfg, ['test'])
    max_off_sec = cfg.data.max_off_sec
    num_cls = cfg.model.params.transformer.params.off_head_cfg.params.out_features
    grid = make_class_grid(-max_off_sec, max_off_sec, num_cls)
    
    print("模型加载完成!")
    
    results = []
    offsets = []
    confidences = []
    
    for video_path in tqdm(video_files, desc='Processing videos'):
        result = process_single_video(str(video_path), model, transforms, grid, device, cfg)
        
        offset = result.get('predicted_offset_sec')
        conf = result.get('predicted_probability')
        
        if offset is not None:
            offsets.append(float(offset))
        if conf is not None:
            confidences.append(float(conf))
        
        sync_quality = 'unknown'
        if offset is not None:
            abs_offset = abs(offset)
            if abs_offset <= 0.1:
                sync_quality = 'excellent'
            elif abs_offset <= 0.2:
                sync_quality = 'good'
            elif abs_offset <= 0.5:
                sync_quality = 'acceptable'
            else:
                sync_quality = 'poor'
        
        results.append({
            'file': str(video_path),
            'filename': video_path.name,
            'predicted_offset_sec': offset,
            'predicted_probability': conf,
            'sync_quality': sync_quality,
            'error': result.get('error')
        })
        print(f"处理: {video_path.name} -> 偏移: {offset if offset else 'N/A'}, 质量: {sync_quality}")
    
    payload = {
        'metric': 'av_sync',
        'summary': {
            'mean_offset': float(sum(offsets) / len(offsets)) if offsets else 0.0,
            'mean_confidence': float(sum(confidences) / len(confidences)) if confidences else 0.0,
            'total_samples': len(results)
        },
        'results': results
    }
    
    output_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"结果已保存到: {output_path}")


if __name__ == '__main__':
    main()
