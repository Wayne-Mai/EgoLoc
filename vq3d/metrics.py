import os
import sys
import numpy as np
from vq3d.bounding_box import BoundingBox

class distL2():
    def compute(self, v1:np.ndarray, v2:np.ndarray) -> float:
        d = np.linalg.norm(v1-v2)
        return d

class angularError():
    def compute(self, v1:np.ndarray, v2:np.ndarray) -> float:
        angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
        return angle

class accuracy():
    def compute(self, t:np.ndarray, box1:BoundingBox, box2:BoundingBox) -> bool:

        c = (box1.center + box2.center) / 2.0

        d = np.linalg.norm(c-t)
        d_gt = np.linalg.norm(box1.center - box2.center)

        diag1 = np.sqrt(np.sum(box1.sizes**2))
        diag2 = np.sqrt(np.sum(box2.sizes**2))

        m = np.mean([diag1, diag2])
        delta = np.exp(-m)
        success=d < 6*(d_gt + delta)
        print(f"\t Success:{success}, {d:.3f} < {6*(d_gt + delta):.3f} ? pred:{t} gt:{c}. ERROR : {np.abs(t-c)}")
        return d < 6*(d_gt + delta)

    # deal with rotated bbox center, the logic is unchanged from ego4d
    def mp_compute(self, t:np.ndarray, box1:BoundingBox, box2:BoundingBox) -> bool:
        
        c = (box1.mp_center() + box2.mp_center()) / 2.0

        d = np.linalg.norm(c-t)
        d_gt = np.linalg.norm(box1.center - box2.center)

        diag1 = np.sqrt(np.sum(box1.sizes**2))
        diag2 = np.sqrt(np.sum(box2.sizes**2))

        m = np.mean([diag1, diag2])
        delta = np.exp(-m)
        success=d < 6*(d_gt + delta)
        print(f"\t Success:{success}, {d:.3f} < {6*(d_gt + delta):.3f} ? pred:{t} gt:{c}. ERROR : {np.abs(t-c)}")
        return d < 6*(d_gt + delta)
