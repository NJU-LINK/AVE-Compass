# Adapted from https://github.com/bytedance/LatentSync/blob/a229c394/eval/syncnet/syncnet_eval.py
"""SyncNet-based lip-sync evaluation.

Evaluates audio-visual lip synchronization by computing the SyncNet confidence
score between lip movements and audio. Returns (offset, min_dist, confidence).
"""
import glob
import math
import os
import subprocess
from shutil import rmtree

import cv2
import numpy
import python_speech_features
import torch
from scipy import signal
from scipy.io import wavfile

import sys
from pathlib import Path

# Support both package import and direct script execution
sys.path.insert(0, str(Path(__file__).resolve().parent))

from syncnet import S


def calc_pdist(feat1: torch.Tensor, feat2: torch.Tensor, vshift: int = 10) -> list:
    """Compute pairwise distances between audio and visual features across temporal shifts."""
    win_size = vshift * 2 + 1
    feat2p = torch.nn.functional.pad(feat2, (0, 0, vshift, vshift))
    dists = []
    for i in range(len(feat1)):
        dists.append(
            torch.nn.functional.pairwise_distance(
                feat1[[i], :].repeat(win_size, 1),
                feat2p[i : i + win_size, :],
            )
        )
    return dists


class SyncNetEval(torch.nn.Module):
    """SyncNet evaluator for lip-sync confidence scoring."""

    def __init__(self, dropout: float = 0, num_layers_in_fc_layers: int = 1024, device: str = "cpu") -> None:
        super().__init__()
        self.__S__ = S(num_layers_in_fc_layers=num_layers_in_fc_layers).to(device)
        self.device = device

    def evaluate(self, video_path: str, temp_dir: str = "temp", batch_size: int = 20, vshift: int = 15):
        """Evaluate lip-sync for a single video.

        Returns:
            Tuple of (av_offset, min_dist, confidence).
            - av_offset: detected audio-visual offset in frames
            - min_dist: minimum distance between audio and visual features
            - confidence: sync confidence (median_dist - min_dist), higher is better
        """
        self.__S__.eval()

        # ========== Extract frames and audio ==========
        if os.path.exists(temp_dir):
            rmtree(temp_dir)
        os.makedirs(temp_dir)

        command = f"ffmpeg -loglevel error -nostdin -y -i {video_path} -f image2 {os.path.join(temp_dir, '%06d.jpg')}"
        subprocess.call(command, shell=True, stdout=None)
        command = (
            f"ffmpeg -loglevel error -nostdin -y -i {video_path} -async 1 -ac 1 -vn "
            f"-acodec pcm_s16le -ar 16000 {os.path.join(temp_dir, 'audio.wav')}"
        )
        subprocess.call(command, shell=True, stdout=None)

        # ========== Load video frames ==========
        images = []
        flist = glob.glob(os.path.join(temp_dir, "*.jpg"))
        flist.sort()
        for fname in flist:
            img_input = cv2.imread(fname)
            img_input = cv2.resize(img_input, (224, 224))
            images.append(img_input)
        im = numpy.stack(images, axis=3)
        im = numpy.expand_dims(im, axis=0)
        im = numpy.transpose(im, (0, 3, 4, 1, 2))
        imtv = torch.autograd.Variable(torch.from_numpy(im.astype(float)).float())

        # ========== Load audio ==========
        sample_rate, audio = wavfile.read(os.path.join(temp_dir, "audio.wav"))
        mfcc = zip(*python_speech_features.mfcc(audio, sample_rate))
        mfcc = numpy.stack([numpy.array(i) for i in mfcc])
        cc = numpy.expand_dims(numpy.expand_dims(mfcc, axis=0), axis=0)
        cct = torch.autograd.Variable(torch.from_numpy(cc.astype(float)).float())

        # ========== Check audio and video input length ==========
        min_length = min(len(images), math.floor(len(audio) / 640))

        # ========== Generate video and audio feats ==========
        lastframe = min_length - 5
        if lastframe <= 0:
            rmtree(temp_dir)
            raise ValueError(f"Video too short for lip-sync evaluation: {video_path} (min_length={min_length})")

        im_feat = []
        cc_feat = []
        for i in range(0, lastframe, batch_size):
            im_batch = [
                imtv[:, :, vframe : vframe + 5, :, :]
                for vframe in range(i, min(lastframe, i + batch_size))
            ]
            im_in = torch.cat(im_batch, 0)
            im_out = self.__S__.forward_lip(im_in.to(self.device))
            im_feat.append(im_out.data.cpu())

            cc_batch = [
                cct[:, :, :, vframe * 4 : vframe * 4 + 20]
                for vframe in range(i, min(lastframe, i + batch_size))
            ]
            cc_in = torch.cat(cc_batch, 0)
            cc_out = self.__S__.forward_aud(cc_in.to(self.device))
            cc_feat.append(cc_out.data.cpu())

        im_feat = torch.cat(im_feat, 0)
        cc_feat = torch.cat(cc_feat, 0)

        # ========== Compute offset and confidence ==========
        dists = calc_pdist(im_feat, cc_feat, vshift=vshift)
        mean_dists = torch.mean(torch.stack(dists, 1), 1)
        min_dist, minidx = torch.min(mean_dists, 0)
        av_offset = vshift - minidx
        conf = torch.median(mean_dists) - min_dist

        rmtree(temp_dir)
        return av_offset.item(), min_dist.item(), conf.item()

    def loadParameters(self, path: str) -> None:
        """Load SyncNet model weights."""
        loaded_state = torch.load(path, map_location=lambda storage, loc: storage, weights_only=True)
        self_state = self.__S__.state_dict()
        for name, param in loaded_state.items():
            self_state[name].copy_(param)
