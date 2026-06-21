"""Detection/pose head + eval for the D1 (gate-pose) pivot.

The OFA supernet is searched as a *backbone* (``supernet.pose_backbone.PoseBackbone``); this
package grafts a YOLO11-pose neck/head onto it (``pose_model``) via 1x1 channel adapters
(``adapter``) and scores candidates with Ultralytics' pose validator (``evaluate``). The
adapter layer is torch-only and ``.venv``-testable; the head graft + eval import ultralytics
lazily and run under ``.venv-nas`` / GPU.
"""
