"""
Evaluation metrics for shadow generation.

Implements IoU, RMSE, S-RMSE, and ZNCC metrics as described in the paper
for evaluating shadow prediction quality.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim


class ShadowMetrics:
    """
    Evaluation metrics for shadow generation quality.
    
    Implements the four main metrics used in the paper:
    - IoU (Intersection over Union)
    - RMSE (Root Mean Square Error)
    - S-RMSE (Scaled Root Mean Square Error)
    - ZNCC (Zero-normalized Cross Correlation)
    """
    
    def __init__(self, epsilon: float = 1e-8):
        self.epsilon = epsilon
        
    def compute_iou(self, pred_mask: torch.Tensor, gt_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute Intersection over Union between predicted and ground truth masks.
        
        Args:
            pred_mask: Predicted shadow mask (binary)
            gt_mask: Ground truth shadow mask (binary)
            
        Returns:
            IoU score
        """
        pred_mask = pred_mask.float()
        gt_mask = gt_mask.float()
        
        intersection = (pred_mask * gt_mask).sum()
        union = pred_mask.sum() + gt_mask.sum() - intersection
        
        iou = intersection / (union + self.epsilon)
        
        return iou
    
    def compute_rmse(self, pred_shadow: torch.Tensor, gt_shadow: torch.Tensor) -> torch.Tensor:
        """
        Compute Root Mean Square Error between predicted and ground truth shadows.
        
        Args:
            pred_shadow: Predicted shadow map
            gt_shadow: Ground truth shadow map
            
        Returns:
            RMSE score
        """
        mse = F.mse_loss(pred_shadow, gt_shadow)
        rmse = torch.sqrt(mse)
        
        return rmse
    
    def compute_s_rmse(self, pred_shadow: torch.Tensor, gt_shadow: torch.Tensor) -> torch.Tensor:
        """
        Compute scaled Root Mean Square Error as defined in [48].
        
        Args:
            pred_shadow: Predicted shadow map
            gt_shadow: Ground truth shadow map
            
        Returns:
            S-RMSE score
        """
        # Ensure tensors are in [0, 1] range
        pred_norm = torch.clamp(pred_shadow, 0, 1)
        gt_norm = torch.clamp(gt_shadow, 0, 1)
        
        # Compute mean squared error
        mse = F.mse_loss(pred_norm, gt_norm)
        
        # Add regularization term
        reg_term = 0.01 / ((1 - gt_norm.mean()) + self.epsilon)
        
        s_rmse = torch.sqrt(mse + reg_term)
        
        return s_rmse
    
    def compute_zncc(self, pred_shadow: torch.Tensor, gt_shadow: torch.Tensor) -> torch.Tensor:
        """
        Compute Zero-normalized Cross Correlation as defined in [18].
        
        Args:
            pred_shadow: Predicted shadow map
            gt_shadow: Ground truth shadow map
            
        Returns:
            ZNCC score
        """
        # Flatten tensors
        pred_flat = pred_shadow.flatten()
        gt_flat = gt_shadow.flatten()
        
        # Zero-center
        pred_centered = pred_flat - pred_flat.mean()
        gt_centered = gt_flat - gt_flat.mean()
        
        # Compute normalized cross-correlation
        nominator = (pred_centered * gt_centered).sum()
        denominator = torch.sqrt(
            (pred_centered ** 2).sum() * (gt_centered ** 2).sum()
        ) + self.epsilon
        
        zncc = nominator / denominator
        
        return zncc
    
    def compute_all_metrics(self, 
                           pred_shadow: torch.Tensor,
                           gt_shadow: torch.Tensor,
                           pred_mask: Optional[torch.Tensor] = None,
                           gt_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute all evaluation metrics.
        
        Args:
            pred_shadow: Predicted shadow map
            gt_shadow: Ground truth shadow map
            pred_mask: Predicted binary mask (optional)
            gt_mask: Ground truth binary mask (optional)
            
        Returns:
            Dictionary containing all metric scores
        """
        # Prepare inputs
        pred_shadow = pred_shadow.float()
        gt_shadow = gt_shadow.float()
        
        # Compute metrics
        metrics = {
            'rmse': self.compute_rmse(pred_shadow, gt_shadow),
            's_rmse': self.compute_s_rmse(pred_shadow, gt_shadow),
            'zncc': self.compute_zncc(pred_shadow, gt_shadow)
        }
        
        # Compute IoU if masks are provided
        if pred_mask is not None and gt_mask is not None:
            # Convert to binary if needed
            if pred_mask.max() <= 1.0:
                pred_binary = (pred_mask > 0.5).float()
            else:
                pred_binary = pred_mask.float()
                
            if gt_mask.max() <= 1.0:
                gt_binary = (gt_mask > 0.5).float()
            else:
                gt_binary = gt_mask.float()
                
            metrics['iou'] = self.compute_iou(pred_binary, gt_binary)
        
        return metrics
    
    @torch.no_grad()
    def batch_compute_metrics(self, 
                             pred_shadows: torch.Tensor,
                             gt_shadows: torch.Tensor,
                             pred_masks: Optional[torch.Tensor] = None,
                             gt_masks: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute metrics for batch of predictions.
        
        Args:
            pred_shadows: Batch of predicted shadows (B, C, H, W)
            gt_shadows: Batch of ground truth shadows (B, C, H, W)
            pred_masks: Batch of predicted masks (B, 1, H, W) (optional)
            gt_masks: Batch of ground truth masks (B, 1, H, W) (optional)
            
        Returns:
            Dictionary with mean metrics across batch
        """
        batch_size = pred_shadows.shape[0]
        
        # Compute metrics for each sample
        batch_metrics = []
        
        for i in range(batch_size):
            metrics = self.compute_all_metrics(
                pred_shadows[i],
                gt_shadows[i],
                pred_masks[i] if pred_masks is not None else None,
                gt_masks[i] if gt_masks is not None else None
            )
            batch_metrics.append(metrics)
        
        # Average across batch
        avg_metrics = {}
        for key in batch_metrics[0].keys():
            avg_metrics[key] = torch.stack([m[key] for m in batch_metrics]).mean()
            
        return avg_metrics
    
    def evaluate_shadow_quality(self, pred_shadow: torch.Tensor,
                               gt_shadow: torch.Tensor) -> Dict[str, float]:
        """
        Comprehensive shadow quality evaluation.
        
        Args:
            pred_shadow: Predicted shadow map
            gt_shadow: Ground truth shadow map
            
        Returns:
            Dictionary with quality scores
        """
        # Basic metrics
        basic_metrics = self.compute_all_metrics(pred_shadow, gt_shadow)
        
        # Additional quality metrics
        quality_metrics = {
            'ssim': self._compute_ssim(pred_shadow, gt_shadow),
            'psnr': self._compute_psnr(pred_shadow, gt_shadow),
            'shadow_coverage': self._compute_shadow_coverage(pred_shadow),
            'edge_alignment': self._compute_edge_alignment(pred_shadow, gt_shadow)
        }
        
        # Combine all metrics
        all_metrics = {**basic_metrics, **quality_metrics}
        
        # Convert to float for easier use
        return {k: v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in all_metrics.items()}
    
    def _compute_ssim(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute Structural Similarity Index."""
        pred_np = pred.squeeze().cpu().numpy()
        gt_np = gt.squeeze().cpu().numpy()
        
        # Ensure same shape
        if pred_np.shape != gt_np.shape:
            gt_np = cv2.resize(gt_np, pred_np.shape[::-1])
        
        ssim_score = ssim(pred_np, gt_np, data_range=1.0)
        
        return torch.tensor(ssim_score)
    
    def _compute_psnr(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """Compute Peak Signal-to-Noise Ratio."""
        mse = F.mse_loss(pred, gt)
        psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
        
        return psnr
    
    def _compute_shadow_coverage(self, shadow: torch.Tensor) -> torch.Tensor:
        """Compute percentage of image covered by shadow."""
        coverage = shadow.mean()
        return coverage
    
    def _compute_edge_alignment(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        """
        Compute how well shadow edges align with ground truth.
        
        Uses edge detection to measure alignment quality.
        """
        # Convert to numpy for OpenCV operations
        pred_np = pred.squeeze().cpu().numpy()
        gt_np = gt.squeeze().cpu().numpy()
        
        # Detect edges
        pred_edges = cv2.Canny((pred_np * 255).astype(np.uint8), 50, 150)
        gt_edges = cv2.Canny((gt_np * 255).astype(np.uint8), 50, 150)
        
        # Compute edge alignment
        intersection = np.logical_and(pred_edges > 0, gt_edges > 0).sum()
        union = np.logical_or(pred_edges > 0, gt_edges > 0).sum()
        
        alignment_score = intersection / (union + self.epsilon)
        
        return torch.tensor(alignment_score)


class ShadowDirectionMetrics:
    """
    Specialized metrics for evaluating shadow direction control.
    """
    
    def __init__(self):
        self.main_metrics = ShadowMetrics()
    
    def compute_direction_error(self, pred_shadow: torch.Tensor,
                             gt_shadow: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute direction-specific errors.
        
        Args:
            pred_shadow: Predicted shadow map
            gt_shadow: Ground truth shadow map
            
        Returns:
            Direction error metrics
        """
        errors = {}
        
        # Shadow centroid displacement
        pred_centroid = self._compute_centroid(pred_shadow)
        gt_centroid = self._compute_centroid(gt_shadow)
        
        centroid_error = torch.norm(pred_centroid - gt_centroid)
        errors['centroid_displacement'] = centroid_error
        
        # Shadow elongation error
        pred_elongation = self._compute_elongation(pred_shadow)
        gt_elongation = self._compute_elongation(gt_shadow)
        
        elongation_error = torch.abs(pred_elongation - gt_elongation)
        errors['elongation_error'] = elongation_error
        
        # Shadow orientation error
        pred_orientation = self._compute_orientation(pred_shadow)
        gt_orientation = self._compute_orientation(gt_shadow)
        
        orientation_error = torch.min(
            torch.abs(pred_orientation - gt_orientation),
            torch.abs(pred_orientation - gt_orientation - 2 * np.pi)
        )
        errors['orientation_error'] = orientation_error
        
        return errors
    
    def _compute_centroid(self, shadow: torch.Tensor) -> torch.Tensor:
        """Compute shadow centroid."""
        shadow = shadow.squeeze().cpu().numpy()
        
        # Compute moments
        M = cv2.moments(shadow)
        
        if M["m00"] == 0:
            # Fallback to image center
            h, w = shadow.shape
            return torch.tensor([w/2, h/2])
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        return torch.tensor([cx, cy])
    
    def _compute_elongation(self, shadow: torch.Tensor) -> torch.Tensor:
        """Compute shadow elongation ratio."""
        shadow_binary = (shadow.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        
        contours, _ = cv2.findContours(shadow_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return torch.tensor(1.0)
        
        # Get largest contour
        contour = max(contours, key=cv2.contourArea)
        
        # Compute ellipse fitting
        ellipse = cv2.fitEllipse(contour)
        (cx, cy), (width, height), angle = ellipse
        
        elongation = max(width, height) / (min(width, height) + 1e-8)
        
        return torch.tensor(attenuation)
    
    def _compute_orientation(self, shadow: torch.Tensor) -> torch.Tensor:
        """Compute shadow orientation angle."""
        shadow_binary = (shadow.squeeze().cpu().numpy() > 0.5).astype(np.uint8)
        
        # Compute image moments
        moments = cv2.moments(shadow_binary)
        
        if moments["mu20"] == 0:
            return torch.tensor(0.0)
        
        # Compute orientation
        orientation = 0.5 * np.arctan2(2 * moments["mu11"], 
                                      moments["mu20"] - moments["mu02"])
        
        return torch.tensor(orientation)


def create_benchmark_evaluation(model_results: Dict[str, torch.Tensor],
                               test_tracks: List[str]) -> Dict[str, Dict[str, float]]:
    """
    Create benchmark evaluation across test tracks.
    
    Args:
        model_results: Model predictions organized by track
        test_tracks: List of track names (softness, horizontal, vertical)
        
    Returns:
        Comprehensive evaluation results
    """
    base_metrics = ShadowMetrics()
    direction_metrics = ShadowDirectionMetrics()
    
    evaluation_results = {}
    
    for track in test_tracks:
        track_results = model_results[track]
        
        # Basic quality metrics
        quality_scores = base_metrics.batch_compute_metrics(
            track_results['predictions'],
            track_results['ground_truth']
        )
        
        # Direction-specific metrics
        if 'direction' in track:
            direction_scores = {}
            for i in range(len(track_results['predictions'])):
                individual_scores = direction_metrics.compute_direction_error(
                    track_results['predictions'][i],
                    track_results['ground_truth'][i]
                )
                for key, value in individual_scores.items():
                    if key not in direction_scores:
                        direction_scores[key] = []
                    direction_scores[key].append(value.item())
            
            # Average direction scores
            for key in direction_scores:
                direction_scores[key] = np.mean(direction_scores[key])
        
        # Combine results
        evaluation_results[track] = {
            **{k: v.item() for k, v in quality_scores.items()},
            **direction_scores if 'direction' in track else {}
        }
    
    return evaluation_results
