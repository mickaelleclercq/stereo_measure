import sys
import os
import csv
import math
import logging
import numpy as np
import cv2

# Configure Qt environment BEFORE importing PyQt5
# Avoid OpenCV's embedded Qt plugin path overriding PyQt's plugins
os.environ.pop("QT_PLUGIN_PATH", None)
os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

# Prefer X11/xcb when DISPLAY is available (often more stable under WSLg); otherwise Wayland if present
if not os.environ.get("QT_QPA_PLATFORM"):
    if os.environ.get("DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "xcb"
    elif os.environ.get("WAYLAND_DISPLAY"):
        os.environ["QT_QPA_PLATFORM"] = "wayland"

# NOW import PyQt5 after environment configuration
from PyQt5 import QtCore, QtGui, QtWidgets
from typing import Any, Optional, cast

# ---------- Default Video Paths ----------
#DEFAULT_LEFT_VIDEO = r"C:\\Users\\mykae\\OneDrive - Centre de recherche du CHU de Québec\\cloud\\Projects\\damien\\stereo_taille\\atimaono_stereo_1\\left\\GX010257.MP4"
#DEFAULT_RIGHT_VIDEO = r"C:\\Users\\mykae\\OneDrive - Centre de recherche du CHU de Québec\\cloud\\Projects\\damien\\stereo_taille\\atimaono_stereo_1\\right\\GX010107.MP4"

DEFAULT_LEFT_VIDEO = r"C:\\Users\\mykae\\OneDrive - Centre de recherche du CHU de Québec\\cloud\\Projects\\damien\\stereo_taille\\meridien_test_support_8\\left\\GX010301.MP4"
DEFAULT_RIGHT_VIDEO = r"C:\\Users\\mykae\\OneDrive - Centre de recherche du CHU de Québec\\cloud\\Projects\\damien\\stereo_taille\\meridien_test_support_8\\right\\GX010125.MP4"

# ---------- Logging ----------
logger = logging.getLogger("video_matcher")

def setup_logging(level: int = logging.DEBUG) -> None:
    logger.setLevel(level)
    fmt = logging.Formatter(fmt="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    # Ensure a console handler exists
    if not any(isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler) for h in logger.handlers):
        sh = logging.StreamHandler(stream=sys.stdout)
        sh.setLevel(level)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
    # Ensure a file handler exists
    if not any(isinstance(h, logging.FileHandler) for h in logger.handlers):
        try:
            log_path = os.path.join(os.getcwd(), "video_matcher.log")
            fh = logging.FileHandler(log_path, mode='w', encoding='utf-8')
            fh.setLevel(level)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
        except Exception as e:
            # If file handler cannot be created, continue with console only, but log why
            try:
                logger.error("Failed to open log file: %s", e)
            except Exception:
                pass
    logger.propagate = False

def install_global_exception_hook() -> None:
    """Install a global excepthook to avoid silent crashes."""
    def _hook(exc_type, exc_value, exc_traceback):
        try:
            logger.exception("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
        except Exception:
            pass
        try:
            if QtWidgets.QApplication.instance() is not None:
                QtWidgets.QMessageBox.critical(None, "Erreur", f"{exc_type.__name__}: {exc_value}")
        except Exception:
            pass
    try:
        sys.excepthook = _hook
    except Exception as e:
        try:
            logger.error("Failed to install excepthook: %s", e)
        except Exception:
            pass

def install_qt_message_logging() -> None:
    """Route Qt's internal messages (qDebug/qWarning/…) into Python logging."""
    try:
        def handler(msg_type, context, message):  # type: ignore[no-redef]
            try:
                msg = str(message)
                if msg_type == QtCore.QtDebugMsg:  # type: ignore[attr-defined]
                    logger.debug("[Qt] %s", msg)
                elif msg_type == QtCore.QtInfoMsg:  # type: ignore[attr-defined]
                    logger.info("[Qt] %s", msg)
                elif msg_type == QtCore.QtWarningMsg:  # type: ignore[attr-defined]
                    logger.warning("[Qt] %s", msg)
                elif msg_type == QtCore.QtCriticalMsg:  # type: ignore[attr-defined]
                    logger.error("[Qt] %s", msg)
                elif msg_type == QtCore.QtFatalMsg:  # type: ignore[attr-defined]
                    logger.critical("[Qt] %s", msg)
                else:
                    logger.info("[Qt:%s] %s", str(msg_type), msg)
            except Exception as e:
                _log_ex(e, "qt_message_handler")
        try:
            QtCore.qInstallMessageHandler(handler)  # type: ignore[attr-defined]
        except Exception as e:
            _log_ex(e, "qInstallMessageHandler")
    except Exception as e:
        _log_ex(e, "install_qt_message_logging")

def _log_ex(ex: Exception, where: str) -> None:
    logger.exception("Exception in %s: %s", where, ex)

# ---------- Utils couleur, morpho, géométrie ----------

def windows_to_wsl_path(path: str) -> str:
    # Convert a Windows path like C:\Users\... to WSL /mnt/c/Users/...
    if not path:
        return path
    p = path.replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        drive = p[0].lower()
        p = f"/mnt/{drive}{p[2:]}"
    return p

def bgr_to_lab(img_bgr):
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

def lab_distance_map(lab_img, seed_xy):
    x, y = seed_xy
    h, w = lab_img.shape[:2]
    x = np.clip(x, 0, w - 1)
    y = np.clip(y, 0, h - 1)
    seed = lab_img[int(y), int(x)].astype(np.float32)
    diff = lab_img.astype(np.float32) - seed
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    return dist

def largest_component(mask):
    mask_u8 = (mask.astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    idx = 1 + np.argmax(areas)
    return (labels == idx)

def clean_mask(mask, k=3, it=1):
    kernel = np.ones((k, k), np.uint8)
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel, iterations=it)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel, iterations=it)
    return (m > 0)

def dilate_mask(mask, k=15, it=1):
    kernel = np.ones((k, k), np.uint8)
    m = cv2.dilate(mask.astype(np.uint8), kernel, iterations=it)
    return (m > 0)

def mask_iou(a, b) -> float:
    try:
        A = a.astype(np.bool_)
        B = b.astype(np.bool_)
        inter = int(np.logical_and(A, B).sum())
        union = int(np.logical_or(A, B).sum())
        return float(inter) / float(union) if union > 0 else 0.0
    except Exception:
        return 0.0

def seed_local_segmentation(lab_img, seed_xy, tol, radius=200, patch=5):
    """Segment around the seed using a Lab color threshold on the full image
    (no hard rectangular ROI clipping), then keep only the component connected
    to the seed. Returns a full-size boolean mask.

    The 'radius' parameter is kept for API compatibility but no longer limits
    the segmentation extent, avoiding visible "cadre" artifacts.
    """
    h, w = lab_img.shape[:2]
    sx, sy = int(seed_xy[0]), int(seed_xy[1])
    if not (0 <= sx < w and 0 <= sy < h):
        return np.zeros((h, w), dtype=bool)
    # Seed patch mean for robustness
    px0 = max(0, sx - patch//2); px1 = min(w, sx + patch//2 + 1)
    py0 = max(0, sy - patch//2); py1 = min(h, sy + patch//2 + 1)
    seed_patch = lab_img[py0:py1, px0:px1].astype(np.float32)
    if seed_patch.size == 0:
        seed_vec = lab_img[sy, sx].astype(np.float32)
    else:
        seed_vec = seed_patch.reshape(-1, 3).mean(axis=0)
    # Global distance map (avoid rectangular ROI borders)
    diff = lab_img.astype(np.float32) - seed_vec
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    local = (dist < float(tol))
    # Keep component containing the seed on the full image
    mask_u8 = (local.astype(np.uint8) * 255)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        keep = local
    else:
        seed_label = labels[sy, sx]
        keep = (labels == seed_label)
    keep = clean_mask(keep, k=3, it=1)
    try:
        logger.debug("Segmentation: seed=%s tol=%.2f pixels=%d", tuple(seed_xy), float(tol), int(keep.sum()))
    except Exception:
        pass
    return keep

def mask_contour(mask):
    m = (mask.astype(np.uint8) * 255)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # Pour éviter les contours trop complexes, simplifier si nécessaire
    cnt = max(cnts, key=cv2.contourArea)
    # Si le contour est très complexe (beaucoup de points), l'approximer
    if len(cnt) > 1000:
        epsilon = 0.005 * cv2.arcLength(cnt, True)
        cnt = cv2.approxPolyDP(cnt, epsilon, True)
    return cnt

def create_feature_detector(prefer_sift=True):
    detector = None
    norm = None
    try:
        if prefer_sift and hasattr(cv2, "SIFT_create"):
            detector = cv2.SIFT_create(nfeatures=2000)  # type: ignore[attr-defined]
            norm = cv2.NORM_L2
        else:
            raise Exception("Force ORB")
    except Exception:
        if hasattr(cv2, "ORB_create"):
            detector = cv2.ORB_create(nfeatures=2000)  # type: ignore[attr-defined]
            norm = cv2.NORM_HAMMING
        else:
            if hasattr(cv2, "BRISK_create"):
                detector = cv2.BRISK_create()  # type: ignore[attr-defined]
                norm = cv2.NORM_HAMMING
            else:
                raise RuntimeError("Aucun détecteur de features disponible (SIFT/ORB/BRISK)")
    return detector, norm

def match_homography(imgL, maskL, imgR, prefer_sift=True):
    detector, norm = create_feature_detector(prefer_sift=prefer_sift)
    grayL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

    # Étendre la zone de détection avec un masque dilaté pour plus de features
    dilated_mask = dilate_mask(maskL, k=20, it=2)  # Zone étendue autour de l'objet
    
    kpsL = detector.detect(grayL, None)
    kpsL = points_in_mask(kpsL, dilated_mask)  # Utiliser le masque étendu
    logger.debug("Features dans masque étendu: %d", len(kpsL))
    
    if len(kpsL) < 4:  # Réduire le seuil minimum
        logger.warning("Pas assez de features (%d < 4) dans masque étendu", len(kpsL))
        return None, None

    kpsL, desL = detector.compute(grayL, kpsL)
    kpsR, desR = detector.detectAndCompute(grayR, None)
    if desL is None or desR is None:
        logger.warning("Échec calcul descripteurs (L=%s R=%s)", desL is None, desR is None)
        return None, None

    # Choose BFMatcher norm by descriptor dtype
    try:
        use_l2 = bool(desL.dtype == np.float32)
    except Exception:
        use_l2 = True
    norm_type = cv2.NORM_L2 if use_l2 else cv2.NORM_HAMMING
    bf = cv2.BFMatcher(int(norm_type), crossCheck=False)
    matches = bf.knnMatch(desL, desR, k=2)
    good = []
    for match_pair in matches:
        if len(match_pair) >= 2:
            m, n = match_pair
            if m.distance < 0.8 * n.distance:  # Relaxer le seuil
                good.append(m)
    
    logger.debug("Matches trouvés: %d", len(good))
    if len(good) < 4:  # Réduire seuil minimum
        logger.warning("Pas assez de bons matches (%d < 4)", len(good))
        return None, None

    ptsL = np.asarray([kpsL[m.queryIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)
    ptsR = np.asarray([kpsR[m.trainIdx].pt for m in good], dtype=np.float32).reshape(-1, 1, 2)

    H, mask_inl = cv2.findHomography(ptsL, ptsR, cv2.RANSAC, 5.0)  # Relaxer seuil RANSAC
    if H is not None:
        inliers = np.sum(mask_inl) if mask_inl is not None else 0
        logger.debug("Homographie trouvée avec %d inliers sur %d matches", inliers, len(good))
    else:
        logger.warning("Échec findHomography")
    return H, mask_inl

def template_fallback(imgL, boxL, imgR):
    # Secours simple si H indisponible: NCC template matching
    tpl, (x0, y0) = crop_by_box(imgL, boxL)
    if tpl.size == 0: 
        return None
    grayT = cv2.cvtColor(tpl, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)
    if grayR.shape[0] < grayT.shape[0] or grayR.shape[1] < grayT.shape[1]:
        return None
    res = cv2.matchTemplate(grayR, grayT, cv2.TM_CCOEFF_NORMED)
    _, maxv, _, maxloc = cv2.minMaxLoc(res)
    x, y = maxloc
    h, w = grayT.shape
    boxR = np.array([[x, y],
                     [x + w, y],
                     [x + w, y + h],
                     [x, y + h]], dtype=np.int32)
    return boxR

# ---------- Suivi multi-frames ----------

def points_in_mask(kps, mask):
    h, w = mask.shape
    keep = []
    for kp in kps:
        x, y = int(round(kp.pt[0])), int(round(kp.pt[1]))
        if 0 <= x < w and 0 <= y < h and mask[y, x]:
            keep.append(kp)
    return keep

def crop_by_box(img, box):
    x_min = int(max(0, int(np.min(box[:, 0]))))
    y_min = int(max(0, int(np.min(box[:, 1]))))
    x_max = int(min(img.shape[1] - 1, int(np.max(box[:, 0]))))
    y_max = int(min(img.shape[0] - 1, int(np.max(box[:, 1]))))
    return img[y_min:y_max+1, x_min:x_max+1].copy(), (x_min, y_min)

def to_qimage_bgr(img_bgr):
    h, w = img_bgr.shape[:2]
    try:
        logger.debug("to_qimage_bgr: src shape=%sx%s", w, h)
    except Exception:
        pass
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    # Use the actual row stride to build the QImage, then copy so it owns the data
    stride = int(rgb.strides[0]) if hasattr(rgb, 'strides') else (3 * w)
    qimg = QtGui.QImage(rgb.data, w, h, stride, QtGui.QImage.Format_RGB888).copy()
    try:
        logger.debug("to_qimage_bgr: qimg w=%d h=%d bytesPerLine=%d isNull=%s", qimg.width(), qimg.height(), qimg.bytesPerLine(), str(qimg.isNull()))
    except Exception:
        pass
    return qimg

def oriented_box_points_from_mask(mask):
    cnt = mask_contour(mask)
    if cnt is None or len(cnt) < 3:
        return None
    try:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        return box
    except Exception:
        # Fallback: utiliser bounding box simple si minAreaRect échoue
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return None
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        # Rectangle axis-aligned comme fallback
        box = np.array([[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]], dtype=np.int32)
        return box

def order_box_points(box):
    """Return box points ordered as [top-left, top-right, bottom-right, bottom-left]."""
    if box is None:
        return None
    pts = np.asarray(box, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    diff = pts[:, 0] - pts[:, 1]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    ordered = np.array([tl, tr, br, bl], dtype=np.int32)
    return ordered

def is_valid_box(box, img_shape=None, forbid_zero=True):
    try:
        arr = np.asarray(box)
        if arr is None or arr.shape != (4, 2):
            return False
        rect = cv2.minAreaRect(arr.astype(np.float32).reshape(-1, 1, 2))
        (_, _), (w, h), _ = rect
        if w < 3 or h < 3:
            return False
        if img_shape is not None:
            H, W = img_shape[:2]
            if np.any(arr[:, 0] < 0) or np.any(arr[:, 1] < 0) or np.any(arr[:, 0] >= W) or np.any(arr[:, 1] >= H):
                return False
        if forbid_zero and np.any(arr == 0):
            return False
        return True
    except Exception:
        return False

def box_near_image_edge(box, img_shape, margin: int = 4) -> bool:
    """Return True if any point in box is within 'margin' pixels of the image border."""
    try:
        H, W = img_shape[:2]
        arr = np.asarray(box).reshape(4, 2)
        x = arr[:, 0]; y = arr[:, 1]
        return bool(
            np.any(x <= margin) or np.any(y <= margin) or
            np.any(x >= (W - 1 - margin)) or np.any(y >= (H - 1 - margin))
        )
    except Exception:
        return False

def init_track_points(mask, max_pts=500):
    m = (mask.astype(np.uint8) * 255)
    # Utiliser des paramètres plus robustes pour la détection de points
    pts = cv2.goodFeaturesToTrack(m, maxCorners=max_pts, qualityLevel=0.02, minDistance=8, blockSize=5)
    
    # Si pas assez de points, essayer avec des paramètres moins stricts
    if pts is None or len(pts) < 20:
        pts = cv2.goodFeaturesToTrack(m, maxCorners=max_pts, qualityLevel=0.01, minDistance=5, blockSize=3)
    
    return pts

def track_next_affine(prev_img, next_img, prev_pts):
    if prev_pts is None or prev_pts.shape[0] < 6:
        return None, None, None
    prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_BGR2GRAY)
    next_gray = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)
    
    # Paramètres plus robustes pour le tracking
    # Cast None for nextPts to satisfy some static analyzers; OpenCV accepts None here
    next_pts, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, cast(Any, None),
                                                 winSize=(15, 15), maxLevel=2,  # Fenêtre plus petite, moins de niveaux
                                                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03))
    
    # Filtrer les points avec un seuil d'erreur plus strict
    good_prev = prev_pts[st.flatten()==1]
    good_next = next_pts[st.flatten()==1]
    good_err = err[st.flatten()==1]
    
    # Éliminer les points avec une erreur trop élevée
    error_threshold = np.percentile(good_err, 75)  # Garder les 75% meilleurs
    good_mask = good_err.flatten() <= error_threshold
    good_prev = good_prev[good_mask]
    good_next = good_next[good_mask]
    
    if good_prev.shape[0] < 6:
        return None, None, None
    
    # Utiliser RANSAC avec des paramètres plus stricts
    M, inl = cv2.estimateAffinePartial2D(good_prev, good_next, method=cv2.RANSAC, 
                                         ransacReprojThreshold=2.0, maxIters=2000, confidence=0.99)
    if M is None:
        return None, None, None
    return M, good_prev[inl.flatten()==1], good_next[inl.flatten()==1]

def warp_box_affine(box, M):
    box_h = np.hstack([box.astype(np.float32), np.ones((4,1), np.float32)])
    warped = (M @ box_h.T).T
    return warped.astype(np.int32)

def warp_mask_affine(mask, M, shape):
    return cv2.warpAffine((mask.astype(np.uint8)*255), M, (shape[1], shape[0])) > 0

def _affine_is_reasonable(M: np.ndarray, img_shape) -> bool:
    """Sanity-check an affine transform M (2x3) for scale/shear/translation bounds."""
    try:
        H, W = img_shape[:2]
        # Decompose
        A = M[:, :2].astype(np.float32)
        t = M[:, 2].astype(np.float32)
        # SVD to get scales
        U, S, Vt = np.linalg.svd(A)
        smin, smax = float(min(S)), float(max(S))
        # Limit scaling and anisotropy
        if not (0.6 <= smin <= 1.8 and 0.6 <= smax <= 1.8):
            return False
        if smax / max(smin, 1e-6) > 2.0:
            return False
        # Limit translation to a fraction of image diagonal
        diag = float(np.hypot(W, H))
        if float(np.hypot(float(t[0]), float(t[1]))) > 0.25 * diag:
            return False
        return True
    except Exception:
        return False

# ---------- Widgets d’affichage ----------

class FrameView(QtWidgets.QGraphicsView):
    seedClicked = QtCore.pyqtSignal(int, int)  # x, y

    def __init__(self, parent=None):
        super().__init__(parent)
        # Optional name for logging context (e.g., "left"/"right")
        self.name = "view"
        self.setScene(QtWidgets.QGraphicsScene(self))
        try:
            logger.debug("FrameView[%s]: created QGraphicsScene=%s", self.name, str(self.scene()))
        except Exception:
            pass
        # Improve painting reliability
        try:
            self.setViewportUpdateMode(QtWidgets.QGraphicsView.FullViewportUpdate)
        except Exception:
            pass
        try:
            self.setBackgroundBrush(QtGui.QBrush(QtGui.QColor(30, 30, 30)))
        except Exception:
            pass
        # Scene items
        self.pix = QtWidgets.QGraphicsPixmapItem()
        # Two overlays: one for filled segmentation, one for outline box
        self.overlayMask = QtWidgets.QGraphicsPathItem()
        self.overlayBox = QtWidgets.QGraphicsPathItem()
        # Simple debug text to validate painting path
        self.debugText = QtWidgets.QGraphicsSimpleTextItem("")

        # Configure overlay styles
        pen_mask = QtGui.QPen(QtGui.QColor(255, 50, 50, 160))
        pen_mask.setWidth(1)
        self.overlayMask.setPen(pen_mask)
        self.overlayMask.setBrush(QtGui.QBrush(QtGui.QColor(255, 50, 50, 80)))
        # Box overlay: visible outline, no fill
        pen_box = QtGui.QPen(QtGui.QColor(255, 220, 50, 255))
        pen_box.setWidth(3)
        self.overlayBox.setPen(pen_box)
        self.overlayBox.setBrush(QtGui.QBrush(QtCore.Qt.BrushStyle.NoBrush))

        # Add items to scene
        scn = self.scene()
        if scn is not None:
            scn.addItem(self.pix)
            scn.addItem(self.overlayMask)
            scn.addItem(self.overlayBox)
            scn.addItem(self.debugText)

        # Z-order
        self.pix.setZValue(0)
        self.overlayMask.setZValue(10)
        self.overlayBox.setZValue(11)
        self.debugText.setZValue(20)

        # Debug text style
        try:
            self.debugText.setBrush(QtGui.QBrush(QtGui.QColor(255, 255, 255)))
            f = QtGui.QFont()
            f.setPointSize(10)
            self.debugText.setFont(f)
            self.debugText.setPos(8, 8)
        except Exception:
            pass

        # Interaction/flags
        self._zoom = 0
        self._did_autofit = False
        self.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
        # Normal arrow cursor, no drag-by-hand
        self.setDragMode(QtWidgets.QGraphicsView.NoDrag)
        try:
            vp = self.viewport()
            if vp is not None:
                vp.setCursor(QtCore.Qt.ArrowCursor)  # type: ignore[attr-defined]
        except Exception:
            pass

    def set_frame(self, img_bgr, box=None, contour=None):
        if img_bgr is None:
            # Clear pixmap and overlays when no image, then update viewport
            self.pix.setPixmap(QtGui.QPixmap())
            self.overlayMask.setPath(QtGui.QPainterPath())
            self.overlayBox.setPath(QtGui.QPainterPath())
            try:
                self.debugText.setText("")
            except Exception:
                pass
            try:
                logger.debug("FrameView[%s].set_frame: cleared (no image)", self.name)
            except Exception:
                pass
            try:
                vp = self.viewport()
                if vp is not None:
                    vp.update()
            except Exception:
                pass
            return
        try:
            logger.debug("FrameView[%s].set_frame: input shape=%s", self.name, None if img_bgr is None else tuple(img_bgr.shape))
        except Exception:
            pass
        qimg = to_qimage_bgr(img_bgr)
        self.pix.setPixmap(QtGui.QPixmap.fromImage(qimg))
        try:
            pm = self.pix.pixmap()
            logger.debug("FrameView[%s].set_frame: pixmap isNull=%s size=%sx%s", self.name, str(pm.isNull()), pm.width(), pm.height())
        except Exception:
            pass
        scn = self.scene()
        if scn is not None:
            scn.setSceneRect(QtCore.QRectF(0, 0, qimg.width(), qimg.height()))
            try:
                r = scn.sceneRect()
                logger.debug("FrameView[%s].sceneRect: x=%.1f y=%.1f w=%.1f h=%.1f", self.name, r.x(), r.y(), r.width(), r.height())
            except Exception:
                pass
        # Build separate paths for mask (filled) and box (outline)
        path_mask = QtGui.QPainterPath()
        if contour is not None and len(contour) >= 3:
            pts = [QtCore.QPointF(float(p[0][0]), float(p[0][1])) for p in contour]
            path_mask.addPolygon(QtGui.QPolygonF(pts))
        self.overlayMask.setPath(path_mask)

        path_box = QtGui.QPainterPath()
        if box is not None:
            pts = [QtCore.QPointF(float(p[0]), float(p[1])) for p in box]
            if len(pts) >= 3:
                pts.append(pts[0])  # close polygon
            path_box.addPolygon(QtGui.QPolygonF(pts))
        self.overlayBox.setPath(path_box)
        # Auto-fit on first image
        if not self._did_autofit and self.pix.pixmap() and not self.pix.pixmap().isNull():
            try:
                self.setTransform(QtGui.QTransform())
                self.fitInView(self.pix, QtCore.Qt.KeepAspectRatio)  # type: ignore[attr-defined]
                self._did_autofit = True
                try:
                    vp = self.viewport()
                    tr = self.transform()
                    if vp is not None:
                        vr = vp.rect()
                        logger.debug("FrameView[%s]: autofit applied viewport=%sx%s m11=%.3f m22=%.3f", self.name, vr.width(), vr.height(), tr.m11(), tr.m22())
                except Exception:
                    pass
            except Exception:
                pass
        try:
            vp = self.viewport()
            if vp is not None:
                vp.update()
        except Exception:
            pass
        # Update debug text to confirm rendering
        try:
            self.debugText.setText(f"{qimg.width()}x{qimg.height()}")
        except Exception:
            pass

    def wheelEvent(self, event) -> None:
        # Zoom with mouse wheel; auto-fit when zoom returns to baseline
        if event is None or not hasattr(event, 'angleDelta'):
            return
        try:
            delta = event.angleDelta().y()
        except Exception:
            delta = 0
        if delta == 0:
            return
        factor = 1.25 if delta > 0 else 0.8
        self.scale(factor, factor)
        self._zoom += 1 if delta > 0 else -1
        if self._zoom <= 0:
            self._zoom = 0
            try:
                self.setTransform(QtGui.QTransform())
                self.fitInView(self.pix, QtCore.Qt.KeepAspectRatio)  # type: ignore[attr-defined]
            except Exception:
                pass

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:  # type: ignore[name-defined]
        try:
            if event.button() == QtCore.Qt.LeftButton and self.pix.pixmap() and not self.pix.pixmap().isNull():  # type: ignore[attr-defined]
                sp = self.mapToScene(event.pos())
                x = int(max(0, min(self.pix.pixmap().width() - 1, int(round(sp.x())))))
                y = int(max(0, min(self.pix.pixmap().height() - 1, int(round(sp.y())))))
                # Log any click on the view with coordinates
                try:
                    btn_name = "LeftButton"
                except Exception:
                    btn_name = str(getattr(event, 'button', lambda: 'unknown')())
                logger.info("UI: %s click at (%d,%d) button=%s", self.name, x, y, btn_name)
                self.seedClicked.emit(x, y)
        except Exception:
            pass
        super().mousePressEvent(event)

class VideoStream:
    def __init__(self, path: Optional[str] = None):
        self.cap: Optional[cv2.VideoCapture] = None
        if path:
            self.open(path)

    def open(self, path: str) -> None:
        # Release previous capture if any
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        cap = cv2.VideoCapture(path)
        if not cap or not cap.isOpened():
            raise RuntimeError("Impossible d'ouvrir " + str(path))
        self.cap = cap
        try:
            logger.info("VideoStream.open: opened '%s' frames=%d size=%dx%d fps=%.2f", path,
                        int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
                        int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
                        int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
                        float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0))
        except Exception:
            pass

    def frame_count(self) -> int:
        if self.cap is None:
            return 0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return max(n, 0)

    def get_frame(self, idx: int):
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = self.cap.read()
        try:
            if ok and frame is not None:
                logger.debug("VideoStream.get_frame: idx=%d ok=True shape=%s", int(idx), tuple(frame.shape))
            else:
                logger.warning("VideoStream.get_frame: idx=%d ok=False", int(idx))
        except Exception:
            pass
        return frame if ok else None

# ---------- Fenêtre principale ----------

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Bi-Viewer - ORB/SIFT - Segmentation - Tracking")
        try:
            logger.debug("MainWindow: init start")
        except Exception:
            pass

        # Etat
        self.left = VideoStream()
        self.right = VideoStream()
        self.idx = 0
        self.max_idx = 0
        self.left_frame = None
        self.right_frame = None
        self.left_lab = None
        self.seed = None
        self.tol = 18.0
        self.left_mask = None
        self.right_mask = None
        self.left_box = None
        self.right_box = None
        self.prefer_sift = True
        # Keyframe reference for robust tracking
        self.keyframe_L_img = None
        self.keyframe_L_mask = None
        self.keyframe_L_kps = None
        self.keyframe_L_des = None
        self.keyframe_detector = None
        self.keyframe_norm = None
        # Playback & history
        self.history = {}  # idx -> { 'Lbox': np.ndarray|None, 'Rbox': np.ndarray|None, 'Lmask': np.ndarray|None, 'Rmask': np.ndarray|None }
        self.history_end = -1

        # Tracking
        self.left_pts = None
        self.right_pts = None

        # UI
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        hl = QtWidgets.QHBoxLayout(central)

        self.viewL = FrameView(); self.viewL.name = "left_view"
        self.viewR = FrameView(); self.viewR.name = "right_view"
        self.viewL.seedClicked.connect(self.on_seed)
        hl.addWidget(self.viewL, 1)
        hl.addWidget(self.viewR, 1)

        side = QtWidgets.QVBoxLayout()
        hl.addLayout(side)

        # Chargement
        self.btnOpenL = QtWidgets.QPushButton("Ouvrir vidéo gauche")
        self.btnOpenR = QtWidgets.QPushButton("Ouvrir vidéo droite")
        self.btnOpenL.clicked.connect(self.on_open_left)
        self.btnOpenR.clicked.connect(self.on_open_right)
        side.addWidget(self.btnOpenL)
        side.addWidget(self.btnOpenR)
        # Goto frame juste sous le bouton droite
        self.btnGoto = QtWidgets.QPushButton("Aller à frame…")
        self.btnGoto.clicked.connect(self.goto_frame)
        side.addWidget(self.btnGoto)

        # Slider frames + label
        self.lblFrame = QtWidgets.QLabel(f"Frame: {self.idx}/{self.max_idx}")
        self.sld = QtWidgets.QSlider(QtCore.Qt.Horizontal)  # type: ignore[attr-defined]
        self.sld.setMinimum(0)
        self.sld.valueChanged.connect(self.on_seek)
        self.sld.sliderReleased.connect(self.on_slider_released)
        side.addWidget(self.lblFrame)
        side.addWidget(self.sld)

        # Tolérance
        self.lblTol = QtWidgets.QLabel(f"Tolérance couleur: {int(self.tol)}")
        self.sTol = QtWidgets.QSlider(QtCore.Qt.Horizontal)  # type: ignore[attr-defined]
        self.sTol.setRange(1, 80)
        self.sTol.setValue(int(self.tol))
        self.sTol.valueChanged.connect(self.on_tol_change)
        self.sTol.sliderReleased.connect(self.on_tol_slider_released)
        side.addWidget(self.lblTol)
        side.addWidget(self.sTol)

        # SIFT-ORB
        self.cmbFeat = QtWidgets.QComboBox()
        self.cmbFeat.addItems(["SIFT", "ORB"])
        self.cmbFeat.currentIndexChanged.connect(self.on_feat_change)
        side.addWidget(QtWidgets.QLabel("Descripteur"))
        side.addWidget(self.cmbFeat)

        # Step 1 hint
        side.addWidget(QtWidgets.QLabel("1. Cliquer l'objet à gauche (étapes 2-4 automatiques)"))

        # Actions essentielles seulement
        self.btnRunAll = QtWidgets.QPushButton("Lancer sur les frames suivantes")
        self.btnExport = QtWidgets.QPushButton("Exporter CSV")
        self.btnReset = QtWidgets.QPushButton("Reset")
        for b in [self.btnRunAll, self.btnExport, self.btnReset]:
            side.addWidget(b)
        # Wire actions via logging wrappers
        self.btnRunAll.clicked.connect(self.on_click_run_all)
        self.btnExport.clicked.connect(self.on_click_export)
        self.btnReset.clicked.connect(self.on_click_reset)

        # Tableau résultats
        self.table = QtWidgets.QTableWidget(0, 1 + 8 + 8)
        # Police plus petite pour la table
        table_font = QtGui.QFont()
        table_font.setPointSize(8)  # Police plus petite
        self.table.setFont(table_font)
        headers = ["Frame"] + [f"L_x{i}" for i in range(1,5)] + [f"L_y{i}" for i in range(1,5)] + \
                  [f"R_x{i}" for i in range(1,5)] + [f"R_y{i}" for i in range(1,5)]
        self.table.setHorizontalHeaderLabels(headers)
        header = self.table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        # Select full rows and disable editing
        try:
            self.table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
            self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        except Exception:
            pass
        side.addWidget(self.table, 1)
        # Connect table selection to frame navigation
        try:
            self.table.cellClicked.connect(self.on_table_row_clicked)
            self.table.itemSelectionChanged.connect(self.on_table_selection_changed)
        except Exception:
            pass

        # Status bar and initial window size
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.resize(1400, 800)

        # Integrated progress widgets (embedded progress instead of popup)
        self.lblProgress = QtWidgets.QLabel("")
        self.progressBar = QtWidgets.QProgressBar()
        self.progressBar.setMinimum(0)
        self.progressBar.setValue(0)
        self.progressBar.setVisible(False)
        self.btnCancelProg = QtWidgets.QPushButton("Stop")
        self.btnCancelProg.setVisible(False)
        self.btnCancelProg.clicked.connect(self.on_cancel_processing)
        progLayout = QtWidgets.QHBoxLayout()
        progLayout.addWidget(self.progressBar, 1)
        progLayout.addWidget(self.btnCancelProg)
        side.addLayout(progLayout)
        side.addWidget(self.lblProgress)
        self.cancelProcessing = False

        try:
            logger.debug("MainWindow: UI built; attempting auto-open defaults")
        except Exception:
            pass

        # Open default videos if present (convert Windows path to WSL)
        try:
            left_path = windows_to_wsl_path(DEFAULT_LEFT_VIDEO)
            right_path = windows_to_wsl_path(DEFAULT_RIGHT_VIDEO)
            if os.path.exists(left_path):
                self.left.open(left_path)
            if os.path.exists(right_path):
                self.right.open(right_path)
            if self.left.cap or self.right.cap:
                self.update_limits()
                self.refresh_frames()
        except Exception as e:
            _log_ex(e, "auto_open_default_videos")
        try:
            logger.debug("MainWindow: init done")
        except Exception:
            pass
    def on_table_row_clicked(self, row, col):
        try:
            frame_item = self.table.item(row, 0)
            if frame_item is not None:
                idx = int(frame_item.text())
                logger.info(f"UI: table row clicked, going to frame {idx}")
                self.goto_frame_idx(idx)
        except Exception as e:
            _log_ex(e, "on_table_row_clicked")

    def on_table_selection_changed(self):
        try:
            sel = self.table.selectionModel()
            if sel is None:
                return
            rows = sel.selectedRows()
            if not rows:
                return
            row = rows[0].row()
            it = self.table.item(row, 0)
            if it is None:
                return
            idx = int(it.text())
            logger.info(f"UI: table selection changed, going to frame {idx}")
            self.goto_frame_idx(idx)
        except Exception as e:
            _log_ex(e, "on_table_selection_changed")

    def goto_frame_idx(self, idx):
        # Clamp idx to valid range
        idx = int(np.clip(int(idx), 0, self.max_idx))
        # Sync the slider position without emitting signals
        try:
            self.sld.blockSignals(True)
            self.sld.setValue(idx)
        finally:
            try:
                self.sld.blockSignals(False)
            except Exception:
                pass
        # Display from history if available to restore overlays
        if idx in self.history:
            self.show_from_history(idx)
        else:
            self.idx = idx
            self.refresh_frames()
        if hasattr(self, 'lblFrame') and self.lblFrame is not None:
            self.lblFrame.setText(f"Frame: {self.idx}/{self.max_idx}")

    # ----- Handlers -----

    # UI wrappers for logging
    def on_open_left(self):
        logger.info("UI: clicked Open Left")
        self.open_video(True)

    def on_open_right(self):
        logger.info("UI: clicked Open Right")
        self.open_video(False)

    def on_slider_released(self):
        try:
            logger.info("UI: slider released at %d", int(self.sld.value()))
        except Exception:
            pass

    def on_tol_slider_released(self):
        try:
            logger.info("UI: tol slider released at %d", int(self.sTol.value()))
        except Exception:
            pass

    def on_click_run_all(self):
        logger.info("UI: clicked 'Lancer sur les frames suivantes'")
        self.process_sequence()

    def on_click_export(self):
        logger.info("UI: clicked 'Exporter CSV'")
        self.export_csv()

    def on_click_reset(self):
        logger.info("UI: clicked 'Reset'")
        self.reset_selection()

    def on_cancel_processing(self):
        logger.info("UI: clicked Stop (cancel processing)")
        self.cancelProcessing = True

    def open_video(self, left=True):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Choisir une vidéo", "", "Vidéos (*.mp4 *.avi *.mov *.mkv)")
        if not path:
            return
        try:
            if left:
                self.left.open(path)
            else:
                self.right.open(path)
            self.update_limits()
            self.refresh_frames()
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Erreur", str(e))

    def update_limits(self):
        nL = self.left.frame_count()
        nR = self.right.frame_count()
        if nL > 0 and nR > 0:
            self.max_idx = min(nL, nR) - 1
        else:
            self.max_idx = max(nL, nR) - 1
        self.max_idx = max(self.max_idx, 0)
        self.sld.setMaximum(self.max_idx)
        logger.info("Limits updated: max_idx=%d (left=%d right=%d)", self.max_idx, nL, nR)
        # Update frame label with new max
        try:
            if hasattr(self, 'lblFrame') and self.lblFrame is not None:
                self.lblFrame.setText(f"Frame: {self.idx}/{self.max_idx}")
        except Exception:
            pass

    def on_seek(self, v):
        self.idx = v
        logger.debug("Seek to frame %d", self.idx)
        self.refresh_frames()

    def goto_frame(self):
        if self.max_idx <= 0:
            self.status.showMessage("Aucune vidéo chargée.")
            return
        v, ok = QtWidgets.QInputDialog.getInt(self, "Aller à la frame", "Numéro de frame:", value=self.idx, min=0, max=self.max_idx)
        if ok:
            self.sld.setValue(int(v))

    def on_tol_change(self, v):
        self.tol = float(v)
        logger.debug("Tolerance changed to %.2f", self.tol)
        # Reflect tolerance value in label
        try:
            if hasattr(self, 'lblTol') and self.lblTol is not None:
                self.lblTol.setText(f"Tolérance couleur: {int(self.tol)}")
        except Exception:
            pass
        if self.left_frame is not None and self.seed is not None:
            self.segment_current()

    def on_feat_change(self, idx):
        self.prefer_sift = (idx == 0)

    def on_seed(self, x, y):
        if self.left_frame is None:
            return
        self.seed = (x, y)
        logger.info("Seed set at left=(%d,%d)", x, y)
        # Automatiser les étapes 1-4
        self.segment_current()  # Étape 1: Segmentation gauche
        self.find_in_right()    # Étape 2: Chercher et segmenter à droite  
        self.draw_box_step()    # Étape 3: Dessiner les boîtes
        self.measure_and_log()  # Étape 4: Mesurer 4 coins

    def refresh_frames(self):
        self.left_frame = self.left.get_frame(self.idx) if self.left.cap else None
        self.right_frame = self.right.get_frame(self.idx) if self.right.cap else None
        try:
            logger.debug("refresh_frames: idx=%d L=%s R=%s", self.idx,
                         None if self.left_frame is None else tuple(self.left_frame.shape),
                         None if self.right_frame is None else tuple(self.right_frame.shape))
        except Exception:
            pass
        if self.left_frame is not None:
            self.left_lab = bgr_to_lab(self.left_frame)
        self.viewL.set_frame(self.left_frame, box=self.left_box,
                             contour=mask_contour(self.left_mask) if self.left_mask is not None else None)
        self.viewR.set_frame(self.right_frame, box=self.right_box,
                             contour=mask_contour(self.right_mask) if self.right_mask is not None else None)
        try:
            self.status.showMessage(f"Frame {self.idx}/{self.max_idx}")
        except Exception:
            pass
        logger.debug("Refreshed frames at idx=%d", self.idx)
        # Also update the frame label
        try:
            if hasattr(self, 'lblFrame') and self.lblFrame is not None:
                self.lblFrame.setText(f"Frame: {self.idx}/{self.max_idx}")
        except Exception:
            pass

    # ----- History & Playback -----

    def save_history(self, idx):
        self.history[idx] = {
            'Lbox': None if self.left_box is None else self.left_box.copy(),
            'Rbox': None if self.right_box is None else self.right_box.copy(),
            'Lmask': None if self.left_mask is None else self.left_mask.copy(),
            'Rmask': None if self.right_mask is None else self.right_mask.copy(),
            'Lshape': None if self.left_frame is None else self.left_frame.shape,
            'Rshape': None if self.right_frame is None else self.right_frame.shape,
        }
        self.history_end = max(self.history_end, idx)
        logger.debug("Saved history for frame %d (history_end=%d)", idx, self.history_end)

    def show_from_history(self, idx):
        # Load frames and overlay from saved history
        self.idx = int(np.clip(idx, 0, self.max_idx))
        self.left_frame = self.left.get_frame(self.idx) if self.left.cap else None
        self.right_frame = self.right.get_frame(self.idx) if self.right.cap else None
        rec = self.history.get(self.idx, None)
        self.left_box = rec['Lbox'] if rec is not None else None
        self.right_box = rec['Rbox'] if rec is not None else None
        self.left_mask = rec['Lmask'] if rec is not None else None
        self.right_mask = rec['Rmask'] if rec is not None else None
        self.viewL.set_frame(self.left_frame, box=self.left_box,
                             contour=mask_contour(self.left_mask) if self.left_mask is not None else None)
        self.viewR.set_frame(self.right_frame, box=self.right_box,
                             contour=mask_contour(self.right_mask) if self.right_mask is not None else None)
        logger.debug("Show from history: idx=%d", self.idx)
        # Update frame label when jumping in history
        try:
            if hasattr(self, 'lblFrame') and self.lblFrame is not None:
                self.lblFrame.setText(f"Frame: {self.idx}/{self.max_idx}")
        except Exception:
            pass

    # ----- Segmentation -----

    def segment_current(self):
        if self.left_frame is None or self.seed is None:
            self.status.showMessage("Clique dans l’image gauche pour définir la graine.")
            return
        # Local color segmentation around the seed and keep only the connected component
        logger.info("Segment current at idx=%d with tol=%.2f", self.idx, self.tol)
        mask = seed_local_segmentation(self.left_lab, self.seed, tol=self.tol, radius=220, patch=7)
        self.left_mask = mask
        # Reset any previous box; user can draw it explicitly via the button
        self.left_box = None
        cnt = mask_contour(mask)
        # Show segmentation contour only
        self.viewL.set_frame(self.left_frame, box=None, contour=cnt)
        
        # Set this as the keyframe for future tracking
        try:
            if self.left_frame is not None and self.left_mask is not None:
                self.keyframe_detector, self.keyframe_norm = create_feature_detector(self.prefer_sift)
                self.keyframe_L_img = self.left_frame.copy()
                self.keyframe_L_mask = self.left_mask.copy()
                gray = cv2.cvtColor(self.keyframe_L_img, cv2.COLOR_BGR2GRAY)
                kps = self.keyframe_detector.detect(gray, None)
                kps = points_in_mask(kps, self.keyframe_L_mask)
                self.keyframe_L_kps, self.keyframe_L_des = self.keyframe_detector.compute(gray, kps)
                logger.info("Set keyframe at idx=%d with %d features", self.idx, len(self.keyframe_L_kps))
        except Exception as e:
            _log_ex(e, "set_keyframe")

    def draw_box_step(self):
        """Compute and show bounding boxes around the segmented objects on both videos."""
        if self.left_frame is None:
            self.status.showMessage("Charge une vidéo gauche.")
            return
        # Ensure left segmentation exists
        if self.left_mask is None:
            if self.seed is None:
                self.status.showMessage("Clique dans l’image gauche pour segmenter.")
                return
            self.segment_current()
            if self.left_mask is None:
                self.status.showMessage("Impossible de segmenter l’objet gauche.")
                return
        # Left box
        self.left_box = oriented_box_points_from_mask(self.left_mask)
        logger.info("Draw left box at idx=%d: %s", self.idx, None if self.left_box is None else self.left_box.tolist())
        cntL = mask_contour(self.left_mask)
        self.viewL.set_frame(self.left_frame, box=self.left_box, contour=cntL)
        # Right box (if right segmentation present)
        if self.right_frame is not None and self.right_mask is not None:
            self.right_box = oriented_box_points_from_mask(self.right_mask)
            logger.info("Draw right box at idx=%d: %s", self.idx, None if self.right_box is None else self.right_box.tolist())
            cntR = mask_contour(self.right_mask)
            self.viewR.set_frame(self.right_frame, box=self.right_box, contour=cntR)
        self.status.showMessage("Boîtes tracées (gauche" + (" et droite" if self.right_mask is not None else "") + ").")

    def reset_selection(self):
        """Clear all selections, masks, boxes, seeds and tracking on both videos."""
        # State reset
        self.seed = None
        self.left_mask = None
        self.right_mask = None
        self.left_box = None
        self.right_box = None
        self.left_pts = None
        self.right_pts = None
        # Keyframe reset
        self.keyframe_L_img = None
        self.keyframe_L_mask = None
        self.keyframe_L_kps = None
        self.keyframe_L_des = None
        # Clear results table
        if hasattr(self, 'table') and self.table is not None:
            try:
                self.table.setRowCount(0)
            except Exception:
                pass
        # Refresh views to remove overlays
        self.viewL.set_frame(self.left_frame, box=None, contour=None)
        self.viewR.set_frame(self.right_frame, box=None, contour=None)
        self.status.showMessage("Sélections et tableau réinitialisés.")
        logger.info("Selections reset at idx=%d; table cleared", self.idx)

    # ----- Matching droite -----

    def find_in_right(self):
        if self.left_frame is None or self.right_frame is None or self.left_mask is None:
            self.status.showMessage("Il faut une segmentation sur la gauche et deux vidéos chargées.")
            return
        logger.info("Find in right at idx=%d", self.idx)
        # Ensure left box (for template fallback center); compute if missing
        if self.left_box is None and self.left_mask is not None:
            self.left_box = oriented_box_points_from_mask(self.left_mask)
        seedR = None
        # Try homography for right seed from left seed
        H, _ = match_homography(self.left_frame, self.left_mask, self.right_frame, prefer_sift=self.prefer_sift)
        if H is not None and self.seed is not None:
            pt = np.array([[float(self.seed[0]), float(self.seed[1]), 1.0]], dtype=np.float32).T  # 3x1
            Hp = H @ pt
            if Hp[2,0] != 0:
                rx = float(Hp[0,0] / Hp[2,0]); ry = float(Hp[1,0] / Hp[2,0])
                seedR = (int(round(rx)), int(round(ry)))
        # If no homography seed, use template center
        if seedR is None and self.left_box is not None:
            boxR = template_fallback(self.left_frame, self.left_box, self.right_frame)
            if boxR is not None:
                cx = int(round(np.mean(boxR[:,0]))); cy = int(round(np.mean(boxR[:,1])))
                seedR = (cx, cy)
        logger.info("Right seed: %s", None if seedR is None else str(seedR))
        if seedR is None:
            self.status.showMessage("Impossible de localiser l’objet à droite.")
            return
        # Segment locally on the right around the found seed
        labR = bgr_to_lab(self.right_frame)
        maskR = seed_local_segmentation(labR, seedR, tol=self.tol, radius=220, patch=7)
        self.right_mask = maskR
        try:
            logger.debug("Right segmentation size=%d", int(maskR.sum()))
        except Exception:
            pass
        self.right_box = None  # boxes are drawn in step 3
        cntR = mask_contour(maskR)
        self.viewR.set_frame(self.right_frame, box=None, contour=cntR)

        # Init points pour suivi
        self.left_pts = init_track_points(self.left_mask)
        if self.right_mask is not None and self.right_mask.shape[:2] == self.right_frame.shape[:2]:
            self.right_pts = init_track_points(self.right_mask)
        else:
            self.right_pts = None

    # ----- Mesure et logging -----

    def measure_and_log(self):
        logger.debug("Measure at idx=%d", self.idx)
        Lbox = oriented_box_points_from_mask(self.left_mask) if self.left_mask is not None else None
        Rbox = oriented_box_points_from_mask(self.right_mask) if self.right_mask is not None else None
        if Lbox is None and self.left_box is not None:
            Lbox = self.left_box
        if Rbox is None and self.right_box is not None:
            Rbox = self.right_box
        # Validate boxes to avoid zeros or degenerate values and near-edge artifacts
        if Lbox is not None:
            if not is_valid_box(Lbox, img_shape=self.left_frame.shape if self.left_frame is not None else None, forbid_zero=False):
                Lbox = None
            elif self.left_frame is not None and box_near_image_edge(Lbox, self.left_frame.shape, margin=2):
                logger.warning("Left box near edge at idx=%d; discarding measurement", self.idx)
                Lbox = None
        if Rbox is not None:
            if not is_valid_box(Rbox, img_shape=self.right_frame.shape if self.right_frame is not None else None, forbid_zero=False):
                Rbox = None
            elif self.right_frame is not None and box_near_image_edge(Rbox, self.right_frame.shape, margin=2):
                logger.warning("Right box near edge at idx=%d; discarding right measurement", self.idx)
                Rbox = None
        if Lbox is None:
            self.status.showMessage("Rien à mesurer à gauche (boîte invalide).")
            logger.warning("No valid left box at idx=%d; skip logging", self.idx)
            return
        # Additional sanity: area should not explode from previous logged left box
        try:
            prev_idx = self.idx - 1
            prev_rec = self.history.get(prev_idx)
            if prev_rec and prev_rec.get('Lbox') is not None and self.left_frame is not None:
                prev_mask = np.zeros(self.left_frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(prev_mask, [prev_rec['Lbox'].astype(np.int32)], -1, 255, thickness=-1)
                curr_mask = np.zeros(self.left_frame.shape[:2], dtype=np.uint8)
                cv2.drawContours(curr_mask, [Lbox.astype(np.int32)], -1, 255, thickness=-1)
                iouL = mask_iou(prev_mask > 0, curr_mask > 0)
                area_prev = float(prev_mask.sum())
                area_curr = float(curr_mask.sum())
                ratio = area_curr / max(area_prev, 1.0)
                if iouL < 0.1 or ratio < 0.2 or ratio > 6.0:
                    logger.warning("Left measure suspicious (IoU=%.3f ratio=%.2f) at idx=%d; skipping row", iouL, ratio, self.idx)
                    return
        except Exception:
            pass
        # Enforce consistent corner order for CSV stability
        Lbox = order_box_points(Lbox)
        if Rbox is not None:
            Rbox = order_box_points(Rbox)

        row = self.table.rowCount()
        self.table.insertRow(row)
        def setnum(c, v):
            try:
                vi = int(round(float(v)))
            except Exception:
                vi = None
            item = QtWidgets.QTableWidgetItem("" if (vi is None or vi == 0) else str(vi))
            self.table.setItem(row, c, item)

        self.table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(self.idx)))
        # L (always write numbers for valid Lbox)
        if Lbox is not None:
            for i in range(4):
                setnum(1+i, Lbox[i,0])
            for i in range(4):
                setnum(1+4+i, Lbox[i,1])
        # R (write blanks if invalid/absent), include basic sanity check vs previous
        if Rbox is not None:
            try:
                prev_idx = self.idx - 1
                prev_rec = self.history.get(prev_idx)
                if prev_rec and prev_rec.get('Rbox') is not None and self.right_frame is not None:
                    prev_maskR = np.zeros(self.right_frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(prev_maskR, [prev_rec['Rbox'].astype(np.int32)], -1, 255, thickness=-1)
                    curr_maskR = np.zeros(self.right_frame.shape[:2], dtype=np.uint8)
                    cv2.drawContours(curr_maskR, [Rbox.astype(np.int32)], -1, 255, thickness=-1)
                    iouR = mask_iou(prev_maskR > 0, curr_maskR > 0)
                    area_prevR = float(prev_maskR.sum()); area_currR = float(curr_maskR.sum())
                    ratioR = area_currR / max(area_prevR, 1.0)
                    if iouR < 0.08 or ratioR < 0.15 or ratioR > 7.0:
                        logger.warning("Right measure suspicious (IoU=%.3f ratio=%.2f) at idx=%d; blanking R columns", iouR, ratioR, self.idx)
                        Rbox = None
            except Exception:
                pass
        if Rbox is not None:
            for i in range(4):
                setnum(1+8+i, Rbox[i,0])
            for i in range(4):
                setnum(1+8+4+i, Rbox[i,1])
        else:
            for i in range(8):
                self.table.setItem(row, 1+8+i, QtWidgets.QTableWidgetItem(""))
        logger.info("Row appended for idx=%d (L=%s R=%s)", self.idx,
                    None if Lbox is None else Lbox.tolist(),
                    None if Rbox is None else Rbox.tolist())

    # ----- Traitement séquence -----

    def process_sequence(self):
        """Version simplifiée - appelle la méthode SIFT simple."""
        return self.process_sequence_simple_sift()

    def process_sequence_complex_original(self):
        # Preconditions
        if self.left.cap is None or self.right.cap is None:
            self.status.showMessage("Charge deux vidéos d’abord.")
            return
        if self.left_mask is None:
            self.status.showMessage("Fais d’abord la segmentation gauche (clic).")
            return
        if self.right_box is None and self.right_mask is None:
            self.status.showMessage("Fais d’abord la recherche droite et dessine la boîte (étapes 2 et 3).")
            return

        # Ensure current frames are available
        if self.left_frame is None or self.right_frame is None:
            self.refresh_frames()
        if self.left_frame is None or self.right_frame is None:
            self.status.showMessage("Frames indisponibles.")
            return

        # Ask for end frame
        end_frame, ok = QtWidgets.QInputDialog.getInt(
            self, "Fin de séquence", "Aller jusqu'à la frame:", value=self.max_idx,
            min=min(self.idx + 1, self.max_idx), max=self.max_idx
        )
        if not ok:
            return
        # Ask for frame step (stride)
        max_step = max(1, end_frame - self.idx)
        step, ok2 = QtWidgets.QInputDialog.getInt(
            self, "Pas de frames", "Traiter 1 frame sur:", value=1, min=1, max=max_step
        )
        if not ok2:
            return
        # Compute iteration count for progress (integrated UI instead of popup)
        total = (end_frame - self.idx + step - 1) // step
        logger.info("Process sequence: start=%d end=%d step=%d total=%d", self.idx, end_frame, step, total)
        # Setup integrated progress
        try:
            self.cancelProcessing = False
            self.progressBar.setMaximum(int(total))
            self.progressBar.setValue(0)
            self.progressBar.setVisible(True)
            self.btnCancelProg.setVisible(True)
            self.lblProgress.setText("Traitement des frames…")
        except Exception as e:
            _log_ex(e, "init_progress_ui")

        # Initialize previous frames and Lab caches
        left_prev = self.left_frame.copy()
        right_prev = self.right_frame.copy()
        left_prev_lab = bgr_to_lab(left_prev)
        right_prev_lab = bgr_to_lab(right_prev)

        Lbox = self.left_box.copy() if self.left_box is not None else oriented_box_points_from_mask(self.left_mask)
        # Determine starting right box
        if self.right_box is not None:
            Rbox = self.right_box.copy()
        elif self.right_mask is not None:
            Rbox = oriented_box_points_from_mask(self.right_mask)
        else:
            Rbox = None
        if Lbox is None or Rbox is None:
            try:
                self.progressBar.setVisible(False)
                self.btnCancelProg.setVisible(False)
                self.lblProgress.setText("")
            except Exception:
                pass
            self.status.showMessage("Boîtes initiales non valides.")
            return

        # Init points si besoin
        if self.left_pts is None:
            self.left_pts = init_track_points(self.left_mask)
        if self.right_pts is None and self.right_mask is not None:
            self.right_pts = init_track_points(self.right_mask)

        # Save current state into history
        self.save_history(self.idx)

        # Variables pour l'adaptation de la tolérance
        consecutive_failures = 0
        adaptive_tol = self.tol

        iter_idx = 0
        for k in range(self.idx + step, end_frame + 1, step):
            if self.cancelProcessing:
                break
            try:
                currL = self.left.get_frame(k)
                currR = self.right.get_frame(k)
                if currL is None or currR is None:
                    break
            except Exception as e:
                self.status.showMessage(f"Erreur lors de la récupération des frames à l'index {k}: {e}")
                _log_ex(e, "get_frame")
                break
            logger.debug("Iter k=%d: tracking...", k)

            # Safe default warped mask (always initialize)
            Hc, Wc = currL.shape[:2]
            Lmask_warp = np.zeros((Hc, Wc), dtype=bool)
            if isinstance(self.left_mask, np.ndarray) and self.left_mask.shape[:2] == (Hc, Wc):
                Lmask_warp = self.left_mask.copy()

            # Keep a copy of previous left mask/box for validation
            prev_left_mask = None if self.left_mask is None else self.left_mask.copy()
            prev_left_box = None if Lbox is None else Lbox.copy()

            # Suivi gauche: tenter un recalage sur la keyframe si le suivi optique échoue
            tracking_failed = False
            M_L, pL0, pL1 = track_next_affine(left_prev, currL, self.left_pts)
            
            # Si le suivi optique est faible ou a échoué, tenter un recalage sur la keyframe
            recalage_keyframe = False
            if M_L is None or (pL1 is not None and len(pL1) < 10):
                logger.warning("Optical flow weak/failed at k=%d; attempting keyframe match", k)
                try:
                    if (self.keyframe_L_img is not None and
                        self.keyframe_detector is not None and
                        self.keyframe_L_des is not None and
                        self.keyframe_L_kps is not None):
                        gray_curr = cv2.cvtColor(currL, cv2.COLOR_BGR2GRAY)
                        kps_curr, des_curr = self.keyframe_detector.detectAndCompute(gray_curr, None)
                        if des_curr is not None and kps_curr is not None:
                            # Choose matcher by descriptor dtype
                            try:
                                use_l2 = bool(self.keyframe_L_des.dtype == np.float32)
                            except Exception:
                                use_l2 = True
                            norm_type = cv2.NORM_L2 if use_l2 else cv2.NORM_HAMMING
                            bf = cv2.BFMatcher(int(norm_type), crossCheck=False)
                            matches = bf.knnMatch(self.keyframe_L_des, des_curr, k=2)
                            good = [m for m, n in matches if n is not None and m.distance < 0.75 * n.distance]
                            if len(good) >= 10:
                                kps_kf = list(self.keyframe_L_kps)
                                pts_kf = np.asarray([kps_kf[m.queryIdx].pt for m in good], dtype=np.float32)
                                pts_curr = np.asarray([kps_curr[m.trainIdx].pt for m in good], dtype=np.float32)
                                M_L_kf, inl = cv2.estimateAffinePartial2D(pts_kf, pts_curr, method=cv2.RANSAC, ransacReprojThreshold=3.0)
                                if M_L_kf is not None and inl is not None and np.sum(inl) > 8:
                                    # Le recalage a réussi, on l'utilise
                                    M_L = M_L_kf
                                    # On transforme le masque de la keyframe, pas le masque précédent
                                    Lmask_warp = warp_mask_affine(self.keyframe_L_mask, M_L, currL.shape)
                                    Lbox = warp_box_affine(oriented_box_points_from_mask(self.keyframe_L_mask), M_L)
                                    # Réinitialiser les points de tracking depuis le nouveau masque
                                    self.left_pts = init_track_points(Lmask_warp)
                                    logger.info("Successfully re-localized with keyframe at k=%d", k)
                                    recalage_keyframe = True
                                else:
                                    tracking_failed = True
                            else:
                                tracking_failed = True
                        else:
                            tracking_failed = True
                except Exception as e:
                    _log_ex(e, "keyframe_matching")
                    tracking_failed = True
            
            if tracking_failed:
                # Si tout a échoué, on fait le fallback couleur comme avant
                self.status.showMessage("Suivi gauche perdu: fallback couleur")
                logger.warning("All tracking methods failed at k=%d; using color fallback", k)
                lab = bgr_to_lab(currL)
                if self.left_mask is not None:
                    core_prev = cv2.erode(self.left_mask.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1) > 0
                    seed_lab = np.nanmedian(left_prev_lab[core_prev].astype(np.float32).reshape(-1, 3), axis=0) if np.any(core_prev) else np.nanmedian(lab.astype(np.float32).reshape(-1, 3), axis=0)
                else:
                    seed_lab = np.nanmedian(lab.astype(np.float32).reshape(-1, 3), axis=0)
                diff = np.sqrt(np.sum((lab.astype(np.float32) - seed_lab)**2, axis=2))
                Lmask = clean_mask(largest_component(diff < adaptive_tol), 3, 1)
                self.left_mask = Lmask
                Lbox = oriented_box_points_from_mask(Lmask)
                self.left_pts = init_track_points(self.left_mask)
                M_L = None # Pas de transformation géométrique
            
            # Adapter la tolérance selon le succès du tracking
            if tracking_failed:
                consecutive_failures += 1
                adaptive_tol = min(80.0, self.tol * (1.0 + consecutive_failures * 0.2))
            else:
                consecutive_failures = max(0, consecutive_failures - 1)
                adaptive_tol = max(self.tol, adaptive_tol * 0.95)  # Réduction graduelle
            
            if not recalage_keyframe:
                if M_L is not None and _affine_is_reasonable(M_L, currL.shape):
                    if Lbox is not None:
                        Lbox = warp_box_affine(Lbox, M_L)
                    if self.left_mask is not None:
                        Lmask_warp = warp_mask_affine(self.left_mask, M_L, currL.shape)
            
            lab = bgr_to_lab(currL)
            # Couleur de référence: noyau (erosion) du masque précédent
            core_prev = cv2.erode(Lmask_warp.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1) > 0
            if np.any(core_prev):
                seed_lab = np.nanmedian(left_prev_lab[core_prev].astype(np.float32).reshape(-1, 3), axis=0)
            else:
                # À défaut, utiliser la médiane globale courante
                seed_lab = np.nanmedian(lab.astype(np.float32).reshape(-1, 3), axis=0)
            # Distance couleur sur les canaux a,b (moins sensible à la luminance)
            lab_ab = lab.astype(np.float32)[:, :, 1:3]
            seed_ab = np.array(seed_lab, dtype=np.float32)[1:3]
            diff = np.sqrt(np.sum((lab_ab - seed_ab) ** 2, axis=2))
            # Pas de découpage rectangulaire: pas de ROI, uniquement la contrainte du masque précédent
            Lcand = (diff < adaptive_tol)
            # Gating Mahalanobis sur (a,b) basé sur les stats du noyau précédent
            try:
                ab_prev = left_prev_lab.astype(np.float32)[:, :, 1:3]
                pts_prev = ab_prev[core_prev]
                if pts_prev is not None and pts_prev.shape[0] > 80:
                    mu = np.nanmean(pts_prev, axis=0)
                    cov = np.cov(pts_prev.T)
                    # régularisation
                    cov = cov + np.eye(2, dtype=np.float32) * 4.0
                    inv = np.linalg.inv(cov)
                    d = lab_ab - mu.reshape(1,1,2)
                    # Mahalanobis^2
                    md2 = d[:, :, 0]* (inv[0,0]*d[:, :, 0] + inv[0,1]*d[:, :, 1]) + d[:, :, 1]* (inv[1,0]*d[:, :, 0] + inv[1,1]*d[:, :, 1])
                    # seuil khi-2 à 95% pour 2 ddl ≈ 5.99 (assoupli légèrement)
                    chi_thr = 7.0
                    Lcand = Lcand & (md2 < chi_thr)
            except Exception:
                pass
            
            # Si la segmentation par canaux a,b est trop restrictive, essayer avec tous les canaux
            if np.sum(Lcand & Lmask_warp) < 0.3 * np.sum(Lmask_warp):
                logger.debug("ab-only segmentation too restrictive at k=%d, trying full Lab", k)
                diff_full = lab.astype(np.float32) - seed_lab
                dist_full = np.sqrt(np.sum(diff_full * diff_full, axis=2))
                Lcand = (dist_full < adaptive_tol * 1.1)  # Utiliser la tolérance adaptative
            
            # Contrainte: rester dans une zone de recherche limitée (masque précédent dilaté)
            try:
                search_area = dilate_mask(Lmask_warp, k=9, it=1)
            except Exception:
                search_area = Lmask_warp
            Lcand_constrained = (Lcand & search_area)
            # Choisir la composante qui recouvre le mieux le masque warpé précédent (plutôt que la plus grande)
            stats_cc = None
            try:
                lbls_cnt, lbls, stats_cc, _ = cv2.connectedComponentsWithStats((Lcand_constrained.astype(np.uint8) * 255), connectivity=8)
            except Exception:
                lbls_cnt = 0
                lbls = None
            if lbls_cnt and lbls_cnt > 1 and lbls is not None:
                try:
                    ref_core = cv2.erode(Lmask_warp.astype(np.uint8), np.ones((3,3), np.uint8), iterations=1) > 0
                except Exception:
                    ref_core = Lmask_warp
                best_lbl = -1
                best_inter = -1
                best_area = -1
                for li in range(1, int(lbls_cnt)):
                    comp = (lbls == li)
                    inter = int((comp & ref_core).sum())
                    area = int(comp.sum())
                    if inter > best_inter or (inter == best_inter and area > best_area):
                        best_inter = inter
                        best_area = area
                        best_lbl = li
                if best_lbl > 0:
                    Lmask = (lbls == best_lbl)
                else:
                    Lmask = Lmask_warp.copy()
            else:
                Lmask = Lcand_constrained
            Lmask = clean_mask(largest_component(Lmask), 3, 1)
            left_forced_warp = False
            # Exiger un chevauchement minimal avec Lmask_warp
            try:
                iou_step = mask_iou(Lmask, Lmask_warp)
                if iou_step < 0.6:
                    logger.warning("Left per-frame IoU too low (%.3f < 0.60); forcing warped mask", iou_step)
                    Lmask = Lmask_warp.copy(); left_forced_warp = True
            except Exception:
                pass
            # Vérifier la distance des centroïdes
            try:
                def centroid_of(mask_bool):
                    m = cv2.moments(mask_bool.astype(np.uint8))
                    if m['m00'] > 0:
                        return (float(m['m10']/m['m00']), float(m['m01']/m['m00']))
                    return None
                c_prev = centroid_of(Lmask_warp)
                c_curr = centroid_of(Lmask)
                if c_prev is not None and c_curr is not None:
                    dx = c_curr[0] - c_prev[0]; dy = c_curr[1] - c_prev[1]
                    dist = math.hypot(dx, dy)
                    thr = max(12.0, 0.05 * math.hypot(Wc, Hc))
                    if dist > thr:
                        logger.warning("Left centroid jump %.1fpx > %.1fpx; forcing warped mask", dist, thr)
                        Lmask = Lmask_warp.copy(); left_forced_warp = True
            except Exception:
                pass
            # Vérifier la dérive de couleur (ab) via histogrammes 2D
            try:
                prev_core = cv2.erode(Lmask_warp.astype(np.uint8), np.ones((5,5), np.uint8), iterations=1) > 0
                if np.any(prev_core) and np.any(Lmask):
                    lab_curr = lab.astype(np.float32)
                    ab_prev = left_prev_lab.astype(np.float32)[:, :, 1:3]
                    ab_curr = lab_curr[:, :, 1:3]
                    prev_pts = ab_prev[prev_core]
                    curr_pts = ab_curr[Lmask]
                    if prev_pts.shape[0] > 100 and curr_pts.shape[0] > 100:
                        bins = 24
                        hist_prev, _, _ = np.histogram2d(prev_pts[:,0], prev_pts[:,1], bins=bins, range=[[0,255],[0,255]])
                        hist_curr, _, _ = np.histogram2d(curr_pts[:,0], curr_pts[:,1], bins=bins, range=[[0,255],[0,255]])
                        hp = hist_prev.flatten().astype(np.float64); hc = hist_curr.flatten().astype(np.float64)
                        if hp.sum() > 0: hp /= hp.sum()
                        if hc.sum() > 0: hc /= hc.sum()
                        bc = float(np.sum(np.sqrt(hp * hc)))
                        d_bhat = math.sqrt(max(0.0, 1.0 - bc))
                        if d_bhat > 0.35:  # seuil strict
                            logger.warning("Left color hist drift too high (Bhattacharyya=%.2f); forcing warped mask", d_bhat)
                            Lmask = Lmask_warp.copy(); left_forced_warp = True
            except Exception:
                pass
            self.left_mask = Lmask
            # Recalculer la boîte gauche à partir du nouveau masque
            Lbox_new = oriented_box_points_from_mask(Lmask)
            if Lbox_new is not None:
                Lbox = Lbox_new
            if Lbox is None:
                # Si échec, garder la boîte transformée géométriquement
                pass
            
            if not recalage_keyframe:
                self.left_pts = pL1.reshape(-1, 1, 2) if pL1 is not None else None
            
            # Réinitialiser périodiquement les points de tracking pour éviter la dérive
            if iter_idx > 0 and iter_idx % 10 == 0 and self.left_mask is not None:  # Tous les 10 frames
                try:
                    new_pts = init_track_points(self.left_mask, max_pts=300)
                    if new_pts is not None and len(new_pts) > 20:
                        self.left_pts = new_pts
                        logger.debug("Refreshed left tracking points at k=%d", k)
                except Exception as e:
                    _log_ex(e, "refresh_left_points")

            # Mettre à jour la keyframe si la segmentation est stable et de bonne qualité
            if iter_idx > 0 and iter_idx % 15 == 0: # Tous les 15 frames
                try:
                    if self.left_mask is not None and np.sum(self.left_mask) > 100:
                        # Always (re)create detector to avoid None states
                        det_ref, norm_ref = create_feature_detector(self.prefer_sift)
                        if det_ref is not None:
                            self.keyframe_detector, self.keyframe_norm = det_ref, norm_ref
                            self.keyframe_L_img = currL.copy()
                            self.keyframe_L_mask = self.left_mask.copy()
                            gray = cv2.cvtColor(self.keyframe_L_img, cv2.COLOR_BGR2GRAY)
                            kps = det_ref.detect(gray, None)
                            kps = points_in_mask(kps, self.keyframe_L_mask)
                            self.keyframe_L_kps, self.keyframe_L_des = det_ref.compute(gray, kps)
                            nfeat = 0 if self.keyframe_L_kps is None else len(self.keyframe_L_kps)
                            logger.info("Updated keyframe at idx=%d with %d features", k, nfeat)
                except Exception as e:
                    _log_ex(e, "update_keyframe")

            # Validate left segmentation vs previous (avoid snowballing); skip if we already forced warp
            try:
                if left_forced_warp:
                    raise RuntimeError("left_forced_warp - skip validation")
                if prev_left_mask is not None and Lmask is not None and Lmask_warp is not None:
                    iouL = mask_iou(Lmask, Lmask_warp)
                    area_prevL = float(Lmask_warp.sum())
                    area_currL = float(Lmask.sum())
                    ratioL = (area_currL / max(area_prevL, 1.0)) if area_prevL > 0 else 1.0
                    # Assouplir les seuils de validation pour éviter les rejets excessifs
                    suspiciousL = (iouL < 0.3) or (ratioL < 0.5) or (ratioL > 2.0)
                    if suspiciousL:
                        logger.warning("Left segmentation suspicious at k=%d (IoU=%.3f, ratio=%.2f); trying alternative seeds", k, iouL, ratioL)
                        alt_seedsL = []
                        # Use current (possibly warped) box corners and midpoints as seeds
                        src_box = Lbox if Lbox is not None else prev_left_box
                        if src_box is not None:
                            pts = [src_box[i] for i in range(4)] + [((src_box[i]+src_box[(i+1)%4])//2) for i in range(4)]
                            for p in pts[:6]:
                                sx, sy = int(p[0]), int(p[1])
                                if 0 <= sx < currL.shape[1] and 0 <= sy < currL.shape[0]:
                                    alt_seedsL.append((sx, sy))
                        # Add centroid of previous warped mask
                        try:
                            m = cv2.moments(Lmask_warp.astype(np.uint8))
                            if m['m00'] > 0:
                                cx = int(round(m['m10']/m['m00']))
                                cy = int(round(m['m01']/m['m00']))
                                if 0 <= cx < currL.shape[1] and 0 <= cy < currL.shape[0]:
                                    alt_seedsL.append((cx, cy))
                        except Exception:
                            pass
                        # Evaluate candidates at current tol, then a lowered tol if needed
                        best_maskL = Lmask
                        best_boxL = Lbox
                        best_scoreL = iouL - 0.05*abs(np.log(max(ratioL, 1e-6)))
                        def try_seeds_with_tol(tol_try: float):
                            nonlocal best_maskL, best_boxL, best_scoreL
                            for sx, sy in alt_seedsL:
                                try:
                                    ml = seed_local_segmentation(lab, (sx, sy), tol=tol_try, radius=220, patch=7)
                                    ml = clean_mask(largest_component(ml), 3, 1)
                                    iou_t = mask_iou(ml, Lmask_warp)
                                    area_t = float(ml.sum())
                                    ratio_t = area_t / max(area_prevL, 1.0)
                                    score = iou_t - 0.03*abs(np.log(max(ratio_t, 1e-6)))  # Moins pénalisant
                                    if score > best_scoreL:
                                        best_scoreL = score
                                        best_maskL = ml
                                        best_boxL = oriented_box_points_from_mask(ml)
                                except Exception as e:
                                    _log_ex(e, "left_alt_seed_try")
                        area_prevL = float(Lmask_warp.sum())
                        try_seeds_with_tol(self.tol)
                        # Essayer avec une tolérance plus élevée si nécessaire
                        if best_scoreL < 0.15:
                            try_seeds_with_tol(min(80.0, float(self.tol) * 1.5))
                        # Essayer avec une tolérance plus basse pour être plus précis
                        if best_scoreL < 0.1:
                            try_seeds_with_tol(max(3.0, float(self.tol) * 0.6))
                        # Si toujours très mauvais, réinitialiser les points de tracking
                        if best_scoreL < 0.08:
                            logger.warning("Left alternatives failed at k=%d; reverting to previous warped mask and reinit tracking", k)
                            self.left_mask = Lmask_warp
                            # Réinitialiser les points de tracking pour éviter l'accumulation d'erreurs
                            self.left_pts = init_track_points(self.left_mask)
                            # Keep Lbox as geometric warp of previous (already applied)
                        else:
                            self.left_mask = best_maskL
                            if best_boxL is not None:
                                Lbox = best_boxL
                            # Réinitialiser périodiquement les points pour éviter la dérive
                            if iter_idx % 3 == 0:  # Tous les 3 frames
                                self.left_pts = init_track_points(self.left_mask)
            except Exception as e:
                _log_ex(e, "left_consistency")

            # Suivi droite
            if self.right_pts is not None:
                M_R, pR0, pR1 = track_next_affine(right_prev, currR, self.right_pts)
            else:
                M_R, pR0, pR1 = (None, None, None)
            # Ensure labR is defined in either branch
            labR = None
            if M_R is not None and _affine_is_reasonable(M_R, currR.shape):
                Rbox = warp_box_affine(Rbox, M_R)
                # Segmentation droite locale autour de la boîte, avec référence couleur côté droit
                labR = bgr_to_lab(currR)
                # Référence couleur: noyau (erosion) du masque précédent à droite si dispo
                if self.right_mask is not None:
                    core_prev_R = cv2.erode((self.right_mask.astype(np.uint8)), np.ones((5,5), np.uint8), iterations=1) > 0
                    if np.any(core_prev_R):
                        seed_lab_R = np.nanmedian(right_prev_lab[core_prev_R].astype(np.float32).reshape(-1,3), axis=0)
                    else:
                        seed_lab_R = np.nanmedian(labR.astype(np.float32).reshape(-1,3), axis=0)
                else:
                    seed_lab_R = np.nanmedian(labR.astype(np.float32).reshape(-1, 3), axis=0)
                # Distance couleur sur a,b
                labR_ab = labR.astype(np.float32)[:, :, 1:3]
                seedR_ab = np.array(seed_lab_R, dtype=np.float32)[1:3]
                diffR = np.sqrt(np.sum((labR_ab - seedR_ab) ** 2, axis=2))
                cand = (diffR < self.tol)
                # Contraindre par le masque précédent warpé si dispo
                Rmask_warp = warp_mask_affine(self.right_mask, M_R, currR.shape) if self.right_mask is not None else None
                if Rmask_warp is not None:
                    cand = cand & Rmask_warp
                Rmask_base = clean_mask(largest_component(cand), 3, 1)
                # Exiger un chevauchement minimal avec le masque warpé si dispo
                if Rmask_warp is not None:
                    try:
                        if mask_iou(Rmask_base, Rmask_warp) < 0.2:
                            logger.warning("Right per-frame IoU too low; forcing warped constraint")
                            Rmask_base = Rmask_warp.copy()
                    except Exception:
                        pass

                # Validate against previous mask: IoU and area change
                iou_prev = mask_iou(Rmask_base, self.right_mask) if (self.right_mask is not None) else 1.0
                area_prev = float(self.right_mask.sum()) if (self.right_mask is not None) else 1.0
                area_curr = float(Rmask_base.sum()) if Rmask_base is not None else 0.0
                ratio = (area_curr / max(area_prev, 1.0)) if area_prev > 0 else 1.0
                # Assouplir les seuils pour le côté droit aussi
                suspicious = (iou_prev < 0.15) or (ratio < 0.2) or (ratio > 5.0)
                best_mask = Rmask_base
                best_box = oriented_box_points_from_mask(Rmask_base)
                best_score = iou_prev

                if suspicious:
                    logger.warning("Right segmentation suspicious at k=%d (IoU=%.3f, ratio=%.2f); trying alternative seeds", k, iou_prev, ratio)
                    # Try alternative seeds from projected left points (if available) or template center
                    alt_seeds = []
                    try:
                        H_alt, _ = match_homography(currL, self.left_mask, currR, prefer_sift=self.prefer_sift)
                    except Exception as e:
                        H_alt = None
                        _log_ex(e, "match_homography_alt")
                    if H_alt is not None:
                        # Use up to 6 corners: left box corners and their midpoints
                        if Lbox is not None:
                            ptsL = [Lbox[i] for i in range(4)] + [((Lbox[i]+Lbox[(i+1)%4])//2) for i in range(4)]
                        else:
                            ptsL = []
                        for p in ptsL[:6]:
                            pt = np.array([[float(p[0]), float(p[1]), 1.0]], dtype=np.float32).T
                            Hp = H_alt @ pt
                            if Hp[2,0] != 0:
                                rx = int(round(float(Hp[0,0]/Hp[2,0]))); ry = int(round(float(Hp[1,0]/Hp[2,0])))
                                if 0 <= rx < currR.shape[1] and 0 <= ry < currR.shape[0]:
                                    alt_seeds.append((rx, ry))
                    if not alt_seeds and best_box is not None:
                        cx = int(round(np.mean(best_box[:,0]))); cy = int(round(np.mean(best_box[:,1])))
                        alt_seeds.append((cx, cy))

                    # Evaluate candidates; pick best by IoU with previous mask (tie-break by area closeness)
                    for sx, sy in alt_seeds:
                        try:
                            labR2 = labR if labR is not None else bgr_to_lab(currR)
                            mask_try = seed_local_segmentation(labR2, (sx, sy), tol=self.tol, radius=220, patch=7)
                            mask_try = clean_mask(largest_component(mask_try), 3, 1)
                            iou_t = mask_iou(mask_try, self.right_mask) if (self.right_mask is not None) else 1.0
                            area_t = float(mask_try.sum())
                            ratio_t = area_t / max(area_prev, 1.0)
                            score = iou_t - 0.03*abs(np.log(max(ratio_t, 1e-6)))  # Moins pénalisant
                            if score > best_score:
                                best_score = score
                                best_mask = mask_try
                                best_box = oriented_box_points_from_mask(mask_try)
                        except Exception as e:
                            _log_ex(e, "alt_seed_try")

                # Apply best result
                self.right_mask = best_mask
                if best_box is not None:
                    Rbox = best_box
                self.right_pts = pR1.reshape(-1, 1, 2) if pR1 is not None else None
                
                # Réinitialiser périodiquement les points de tracking pour éviter la dérive (côté droit)
                if iter_idx % 5 == 0 and self.right_mask is not None:  # Tous les 5 frames
                    try:
                        new_pts = init_track_points(self.right_mask, max_pts=300)
                        if new_pts is not None and len(new_pts) > 20:
                            self.right_pts = new_pts
                            logger.debug("Refreshed right tracking points at k=%d", k)
                    except Exception as e:
                        _log_ex(e, "refresh_right_points")
            else:
                # Recalage ponctuel si le suivi a lâché: tentative rapide par template sur cette frame
                Rbox_try = template_fallback(currL, Lbox, currR)
                if Rbox_try is not None:
                    Rbox = Rbox_try
                    labR = bgr_to_lab(currR)
                    # Référence couleur côté droit (noyau du masque précédent si dispo)
                    if self.right_mask is not None:
                        core_prev_R = cv2.erode((self.right_mask.astype(np.uint8)), np.ones((5,5), np.uint8), iterations=1) > 0
                        if np.any(core_prev_R):
                            seed_lab_R = np.nanmedian(right_prev_lab[core_prev_R].astype(np.float32).reshape(-1,3), axis=0)
                        else:
                            seed_lab_R = np.nanmedian(labR.astype(np.float32).reshape(-1,3), axis=0)
                    else:
                        seed_lab_R = np.nanmedian(labR.astype(np.float32).reshape(-1, 3), axis=0)
                    labR_ab = labR.astype(np.float32)[:, :, 1:3]
                    seedR_ab = np.array(seed_lab_R, dtype=np.float32)[1:3]
                    diffR = np.sqrt(np.sum((labR_ab - seedR_ab) ** 2, axis=2))
                    
                    # Essayer d'abord avec les canaux a,b seulement
                    Rmask_ab = (diffR < self.tol)
                    Rmask_ab = clean_mask(largest_component(Rmask_ab), 3, 1)
                    
                    # Si trop petit, essayer avec tous les canaux Lab
                    if np.sum(Rmask_ab) < 50:  # Moins de 50 pixels
                        logger.debug("Right ab-only segmentation too small at k=%d, trying full Lab", k)
                        diffR_full = labR.astype(np.float32) - seed_lab_R
                        distR_full = np.sqrt(np.sum(diffR_full * diffR_full, axis=2))
                        Rmask = clean_mask(largest_component(distR_full < self.tol * 1.2), 3, 1)
                    else:
                        Rmask = Rmask_ab
                    
                    self.right_mask = Rmask
                    # Recalculer la boîte droite à partir du nouveau masque (fallback)
                    Rbox_new = oriented_box_points_from_mask(self.right_mask)
                    if Rbox_new is not None:
                        Rbox = Rbox_new
                    self.right_pts = init_track_points(self.right_mask)
                else:
                    self.right_pts = None
                    logger.warning("Right tracking lost and template failed at k=%d", k)

            # Mémoriser résultats dans le tableau & history
            self.idx = k
            self.left_frame, self.right_frame = currL, currR
            self.left_lab = bgr_to_lab(currL)
            self.left_box, self.right_box = Lbox, Rbox
            try:
                self.measure_and_log()
            except Exception as e:
                _log_ex(e, "measure_and_log")
                self.status.showMessage(f"Erreur lors de la mesure/log à l'index {k}: {e}")
                break
            self.save_history(k)

            # Afficher en direct (respecte le zoom courant)
            self.viewL.set_frame(self.left_frame, box=self.left_box,
                                 contour=mask_contour(self.left_mask) if self.left_mask is not None else None)
            self.viewR.set_frame(self.right_frame, box=self.right_box,
                                 contour=mask_contour(self.right_mask) if self.right_mask is not None else None)

            # Update live frame label and progress text (integrated)
            try:
                if hasattr(self, 'lblFrame') and self.lblFrame is not None:
                    self.lblFrame.setText(f"Frame: {self.idx}/{self.max_idx}")
                if hasattr(self, 'lblProgress') and self.lblProgress is not None:
                    self.lblProgress.setText(f"Traitement des frames… ({self.idx}/{end_frame})")
            except Exception as e:
                _log_ex(e, "progress_label")

            left_prev = currL
            right_prev = currR
            # Mettre à jour les images Lab précédentes
            left_prev_lab = lab
            # labR peut être None si aucune segmentation droite n'a été faite sur cette itération
            right_prev_lab = labR if labR is not None else bgr_to_lab(right_prev)

            iter_idx += 1
            try:
                if hasattr(self, 'progressBar') and self.progressBar is not None:
                    self.progressBar.setValue(int(iter_idx))
            except Exception as e:
                _log_ex(e, "progress_bar_set")
            QtWidgets.QApplication.processEvents()

        self.refresh_frames()
        # Hide integrated progress UI when done/canceled
        try:
            self.progressBar.setVisible(False)
            self.btnCancelProg.setVisible(False)
            self.lblProgress.setText("")
            self.cancelProcessing = False
        except Exception as e:
            _log_ex(e, "hide_progress_ui")

    def process_sequence_simple_sift(self):
        """Traitement séquence avec approche SIFT simple frame par frame."""
        # Preconditions
        if self.left.cap is None or self.right.cap is None:
            self.status.showMessage("Charge deux vidéos d'abord.")
            return
        if self.left_mask is None:
            self.status.showMessage("Fais d'abord la segmentation gauche (clic).")
            return
        if self.right_box is None and self.right_mask is None:
            self.status.showMessage("Fais d'abord la recherche droite et dessine la boîte (étapes 2 et 3).")
            return

        # Ensure current frames are available
        if self.left_frame is None or self.right_frame is None:
            self.refresh_frames()
        if self.left_frame is None or self.right_frame is None:
            self.status.showMessage("Frames indisponibles.")
            return

        # Ask for end frame
        end_frame, ok = QtWidgets.QInputDialog.getInt(
            self, "Fin de séquence", "Aller jusqu'à la frame:", value=self.max_idx,
            min=min(self.idx + 1, self.max_idx), max=self.max_idx
        )
        if not ok:
            return
        # Ask for frame step (stride)
        max_step = max(1, end_frame - self.idx)
        step, ok2 = QtWidgets.QInputDialog.getInt(
            self, "Pas de frames", "Traiter 1 frame sur:", value=1, min=1, max=max_step
        )
        if not ok2:
            return
        # Compute iteration count for progress
        total = (end_frame - self.idx + step - 1) // step
        logger.info("Process sequence SIFT simple: start=%d end=%d step=%d total=%d", self.idx, end_frame, step, total)
        
        # Setup integrated progress
        try:
            self.cancelProcessing = False
            self.progressBar.setMaximum(int(total))
            self.progressBar.setValue(0)
            self.progressBar.setVisible(True)
            self.btnCancelProg.setVisible(True)
            self.lblProgress.setText("Traitement SIFT simple…")
        except Exception as e:
            _log_ex(e, "init_progress_ui")

        # Initialize SIFT detector
        try:
            detector, norm = create_feature_detector(prefer_sift=getattr(self, 'prefer_sift', True))
        except Exception as e:
            _log_ex(e, "create_detector")
            try:
                self.progressBar.setVisible(False)
                self.btnCancelProg.setVisible(False)
                self.lblProgress.setText("")
            except Exception:
                pass
            self.status.showMessage("Impossible de créer le détecteur de features.")
            return

        # Boîtes initiales
        Lbox = self.left_box.copy() if self.left_box is not None else oriented_box_points_from_mask(self.left_mask)
        if self.right_box is not None:
            Rbox = self.right_box.copy()
        elif self.right_mask is not None:
            Rbox = oriented_box_points_from_mask(self.right_mask)
        else:
            Rbox = None
        
        if Lbox is None:
            try:
                self.progressBar.setVisible(False)
                self.btnCancelProg.setVisible(False)
                self.lblProgress.setText("")
            except Exception:
                pass
            self.status.showMessage("Boîte gauche initiale non valide.")
            return

        # Save current state into history
        self.save_history(self.idx)

        # Variables de suivi
        prev_idx = self.idx
        prev_left_mask = self.left_mask.copy() if self.left_mask is not None else None  # Stocker le masque précédent
        iter_idx = 0
        
        for k in range(self.idx + step, end_frame + 1, step):
            if self.cancelProcessing:
                break
                
            try:
                currL = self.left.get_frame(k)
                currR = self.right.get_frame(k)
                if currL is None or currR is None:
                    logger.warning("Frame %d non disponible", k)
                    break
            except Exception as e:
                _log_ex(e, f"get_frame_{k}")
                break
                
            logger.info("Processing frame %d avec SIFT simple", k)

            # 1. Multi-fallback Localization: SIFT → ORB → Template → Centroïde
            seed_pos = None
            try:
                # Charger la frame précédente
                prevL = self.left.get_frame(prev_idx)
                if prevL is not None and prev_left_mask is not None:
                    
                    # MÉTHODE 1: SIFT matching entre frames
                    try:
                        prev_gray = cv2.cvtColor(prevL, cv2.COLOR_BGR2GRAY)
                        curr_gray = cv2.cvtColor(currL, cv2.COLOR_BGR2GRAY)
                        
                        # Features dans le masque précédent
                        kps_prev = detector.detect(prev_gray, None)
                        kps_prev = [kp for kp in kps_prev 
                                   if (0 <= int(kp.pt[0]) < prev_left_mask.shape[1] and 
                                       0 <= int(kp.pt[1]) < prev_left_mask.shape[0] and 
                                       prev_left_mask[int(kp.pt[1]), int(kp.pt[0])])]
                        
                        if len(kps_prev) >= 6:
                            kps_prev, des_prev = detector.compute(prev_gray, kps_prev)
                            kps_curr, des_curr = detector.detectAndCompute(curr_gray, None)
                            
                            if des_prev is not None and des_curr is not None and len(kps_curr) > 6:
                                # Matching
                                use_l2 = bool(des_prev.dtype == np.float32)
                                norm_type = cv2.NORM_L2 if use_l2 else cv2.NORM_HAMMING
                                bf = cv2.BFMatcher(int(norm_type), crossCheck=False)
                                matches = bf.knnMatch(des_prev, des_curr, k=2)
                                
                                good = []
                                for match_pair in matches:
                                    if len(match_pair) == 2:
                                        m, n = match_pair
                                        if m.distance < 0.75 * n.distance:
                                            good.append(m)
                                
                                if len(good) >= 6:
                                    pts_prev = np.asarray([kps_prev[m.queryIdx].pt for m in good], dtype=np.float32)
                                    pts_curr = np.asarray([kps_curr[m.trainIdx].pt for m in good], dtype=np.float32)
                                    
                                    M, inliers = cv2.estimateAffinePartial2D(
                                        pts_prev, pts_curr, method=cv2.RANSAC, 
                                        ransacReprojThreshold=3.0, confidence=0.99
                                    )
                                    
                                    if M is not None and inliers is not None and np.sum(inliers) >= 4:
                                        # Transformer le centroïde du masque précédent
                                        m_prev = cv2.moments(prev_left_mask.astype(np.uint8))
                                        if m_prev['m00'] > 0:
                                            cx_prev = m_prev['m10']/m_prev['m00']
                                            cy_prev = m_prev['m01']/m_prev['m00']
                                            pt_prev = np.array([[cx_prev, cy_prev, 1.0]], dtype=np.float32).T
                                            pt_curr = np.vstack([M, [0, 0, 1]]) @ pt_prev
                                            seed_pos = (int(round(pt_curr[0,0])), int(round(pt_curr[1,0])))
                                            logger.info("Frame %d: localisation SIFT gauche réussie → (%d,%d)", k, seed_pos[0], seed_pos[1])
                    except Exception as e:
                        _log_ex(e, f"sift_left_localization_{k}")

                    # MÉTHODE 2: ORB fallback si SIFT a échoué
                    if seed_pos is None and getattr(self, 'prefer_sift', True):
                        logger.info("Frame %d: fallback ORB gauche après échec SIFT", k)
                        try:
                            detector_orb, _ = create_feature_detector(prefer_sift=False)
                            prev_gray = cv2.cvtColor(prevL, cv2.COLOR_BGR2GRAY)
                            curr_gray = cv2.cvtColor(currL, cv2.COLOR_BGR2GRAY)
                            
                            kps_prev = detector_orb.detect(prev_gray, None)
                            kps_prev = [kp for kp in kps_prev 
                                       if (0 <= int(kp.pt[0]) < prev_left_mask.shape[1] and 
                                           0 <= int(kp.pt[1]) < prev_left_mask.shape[0] and 
                                           prev_left_mask[int(kp.pt[1]), int(kp.pt[0])])]
                            
                            if len(kps_prev) >= 4:
                                kps_prev, des_prev = detector_orb.compute(prev_gray, kps_prev)
                                kps_curr, des_curr = detector_orb.detectAndCompute(curr_gray, None)
                                
                                if des_prev is not None and des_curr is not None:
                                    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
                                    matches = bf.knnMatch(des_prev, des_curr, k=2)
                                    
                                    good = []
                                    for match_pair in matches:
                                        if len(match_pair) == 2:
                                            m, n = match_pair
                                            if m.distance < 0.8 * n.distance:
                                                good.append(m)
                                    
                                    if len(good) >= 4:
                                        pts_prev = np.asarray([kps_prev[m.queryIdx].pt for m in good], dtype=np.float32)
                                        pts_curr = np.asarray([kps_curr[m.trainIdx].pt for m in good], dtype=np.float32)
                                        
                                        M, inliers = cv2.estimateAffinePartial2D(
                                            pts_prev, pts_curr, method=cv2.RANSAC, 
                                            ransacReprojThreshold=4.0, confidence=0.95
                                        )
                                        
                                        if M is not None and inliers is not None and np.sum(inliers) >= 3:
                                            m_prev = cv2.moments(prev_left_mask.astype(np.uint8))
                                            if m_prev['m00'] > 0:
                                                cx_prev = m_prev['m10']/m_prev['m00']
                                                cy_prev = m_prev['m01']/m_prev['m00']
                                                pt_prev = np.array([[cx_prev, cy_prev, 1.0]], dtype=np.float32).T
                                                pt_curr = np.vstack([M, [0, 0, 1]]) @ pt_prev
                                                seed_pos = (int(round(pt_curr[0,0])), int(round(pt_curr[1,0])))
                                                logger.info("Frame %d: localisation ORB gauche réussie → (%d,%d)", k, seed_pos[0], seed_pos[1])
                        except Exception as e:
                            _log_ex(e, f"orb_left_localization_{k}")

                    # MÉTHODE 3: Template matching fallback
                    if seed_pos is None and hasattr(self, 'left_box') and self.left_box is not None:
                        logger.info("Frame %d: fallback template matching gauche", k)
                        try:
                            boxL_fallback = template_fallback(prevL, self.left_box, currL)
                            if boxL_fallback is not None and is_valid_box(boxL_fallback, currL.shape):
                                cx = int(np.mean(boxL_fallback[:,0]))
                                cy = int(np.mean(boxL_fallback[:,1]))
                                if 0 <= cx < currL.shape[1] and 0 <= cy < currL.shape[0]:
                                    seed_pos = (cx, cy)
                                    logger.info("Frame %d: localisation template gauche réussie → (%d,%d)", k, seed_pos[0], seed_pos[1])
                        except Exception as e:
                            _log_ex(e, f"template_left_localization_{k}")

                # MÉTHODE 4: Centroïde fallback (dernier recours)
                if seed_pos is None:
                    if prev_left_mask is not None:
                        m_prev = cv2.moments(prev_left_mask.astype(np.uint8))
                        if m_prev['m00'] > 0:
                            seed_pos = (int(round(m_prev['m10']/m_prev['m00'])), int(round(m_prev['m01']/m_prev['m00'])))
                            logger.warning("Frame %d: fallback centroïde gauche → (%d,%d)", k, seed_pos[0], seed_pos[1])
                        else:
                            seed_pos = (currL.shape[1]//2, currL.shape[0]//2)
                            logger.warning("Frame %d: fallback centre image gauche", k)
                    else:
                        seed_pos = (currL.shape[1]//2, currL.shape[0]//2)
                        logger.warning("Frame %d: fallback centre image gauche (pas de masque)", k)

            except Exception as e:
                _log_ex(e, f"left_localization_{k}")
                # Dernier fallback en cas d'exception
                if prev_left_mask is not None:
                    m_prev = cv2.moments(prev_left_mask.astype(np.uint8))
                    if m_prev['m00'] > 0:
                        seed_pos = (int(round(m_prev['m10']/m_prev['m00'])), int(round(m_prev['m01']/m_prev['m00'])))
                    else:
                        seed_pos = (currL.shape[1]//2, currL.shape[0]//2)
                else:
                    seed_pos = (currL.shape[1]//2, currL.shape[0]//2)

            # 2. Segmentation gauche comme un clic automatique avec validation de taille
            try:
                if not (0 <= seed_pos[0] < currL.shape[1] and 0 <= seed_pos[1] < currL.shape[0]):
                    seed_pos = (currL.shape[1]//2, currL.shape[0]//2)
                
                # Référence de taille du masque précédent pour éviter l'explosion
                prev_area = np.sum(prev_left_mask) if prev_left_mask is not None else 5000
                max_area = max(50000, int(prev_area) * 3)  # Maximum 3x la taille précédente
                min_area = max(100, int(prev_area) // 3)   # Minimum 1/3 de la taille précédente
                
                lab_curr = bgr_to_lab(currL)
                Lmask_new = seed_local_segmentation(lab_curr, seed_pos, tol=self.tol, radius=220, patch=7)
                Lmask_new = clean_mask(largest_component(Lmask_new), 3, 1)
                
                # Validation de taille critique
                new_area = np.sum(Lmask_new)
                if new_area > max_area:
                    logger.warning("Frame %d: segmentation trop grande (%d > %d), réduction tolérance", k, new_area, max_area)
                    # Essayer avec une tolérance réduite
                    reduced_tol = self.tol * 0.7
                    Lmask_new = seed_local_segmentation(lab_curr, seed_pos, tol=reduced_tol, radius=220, patch=7)
                    Lmask_new = clean_mask(largest_component(Lmask_new), 3, 1)
                    new_area = np.sum(Lmask_new)
                    if new_area > max_area:
                        logger.warning("Frame %d: segmentation encore trop grande, utiliser masque précédent", k)
                        Lmask_new = prev_left_mask.copy() if prev_left_mask is not None else np.zeros(currL.shape[:2], dtype=bool)
                        new_area = np.sum(Lmask_new)
                
                if new_area < min_area:
                    logger.warning("Frame %d: segmentation trop petite (%d < %d), fallback couleur globale", k, new_area, min_area)
                    seed_lab = lab_curr[seed_pos[1], seed_pos[0]].astype(np.float32)
                    diff = np.sqrt(np.sum((lab_curr.astype(np.float32) - seed_lab)**2, axis=2))
                    Lmask_new = clean_mask(largest_component(diff < self.tol), 3, 1)
                    new_area = np.sum(Lmask_new)
                    # Double vérification après fallback
                    if new_area > max_area or new_area < 50:
                        logger.warning("Frame %d: fallback échoué, garder masque précédent", k)
                        Lmask_new = prev_left_mask.copy() if prev_left_mask is not None else np.zeros(currL.shape[:2], dtype=bool)
                
                # Mettre à jour le seed pour la frame suivante
                if np.sum(Lmask_new) > 0:
                    m_new = cv2.moments(Lmask_new.astype(np.uint8))
                    if m_new['m00'] > 0:
                        new_seed = (int(round(m_new['m10']/m_new['m00'])), int(round(m_new['m01']/m_new['m00'])))
                        logger.info("Frame %d: seed mis à jour vers (%d,%d)", k, new_seed[0], new_seed[1])
                        # Note: on ne met pas self.seed ici car c'est utilisé seulement pour le GUI
                
                Lbox_new = oriented_box_points_from_mask(Lmask_new)
                logger.info("Frame %d: segmentation gauche OK, aire=%d (prev=%d)", k, int(np.sum(Lmask_new)), int(prev_area))
                
            except Exception as e:
                _log_ex(e, f"segmentation_left_{k}")
                # Garder le masque précédent en cas d'échec
                Lmask_new = self.left_mask.copy() if self.left_mask is not None else np.zeros(currL.shape[:2], dtype=bool)
                Lbox_new = Lbox

            # 3. Segmentation droite via homographie
            Rmask_new = None
            Rbox_new = None
            try:
                if Lmask_new is not None and np.sum(Lmask_new) > 10:
                    logger.info("Frame %d: tentative homographie L->R", k)
                    H, _ = match_homography(currL, Lmask_new, currR, prefer_sift=getattr(self, 'prefer_sift', True))
                    if H is not None:
                        logger.info("Frame %d: homographie trouvée", k)
                        # Utiliser la position seed (comme le clic manuel) au lieu du centroïde du masque
                        if seed_pos is not None and 0 <= seed_pos[0] < currL.shape[1] and 0 <= seed_pos[1] < currL.shape[0]:
                            cx_left = float(seed_pos[0])
                            cy_left = float(seed_pos[1])
                        else:
                            # Fallback au centroïde si seed_pos invalide
                            m_left = cv2.moments(Lmask_new.astype(np.uint8))
                            if m_left['m00'] > 0:
                                cx_left = m_left['m10']/m_left['m00']
                                cy_left = m_left['m01']/m_left['m00']
                            else:
                                cx_left = currL.shape[1] / 2
                                cy_left = currL.shape[0] / 2
                        
                        pt_left = np.array([[cx_left, cy_left, 1.0]], dtype=np.float32).T
                        pt_right = H @ pt_left
                        if pt_right[2,0] != 0:
                            rx = int(round(pt_right[0,0]/pt_right[2,0]))
                            ry = int(round(pt_right[1,0]/pt_right[2,0]))
                            logger.info("Frame %d: point projeté à droite (%d,%d)", k, rx, ry)
                            if 0 <= rx < currR.shape[1] and 0 <= ry < currR.shape[0]:
                                lab_right = bgr_to_lab(currR)
                                Rmask_new = seed_local_segmentation(lab_right, (rx, ry), tol=self.tol, radius=220, patch=7)
                                if Rmask_new is not None:
                                    Rmask_new = clean_mask(largest_component(Rmask_new), 3, 1)
                                    if np.sum(Rmask_new) > 10:
                                        Rbox_new = oriented_box_points_from_mask(Rmask_new)
                                        logger.info("Frame %d: segmentation droite OK via homographie, aire=%d", k, int(np.sum(Rmask_new)))
                                    else:
                                        logger.warning("Frame %d: segmentation droite trop petite", k)
                                        Rmask_new = None
                                else:
                                    logger.warning("Frame %d: échec segmentation droite locale", k)
                            else:
                                logger.warning("Frame %d: point projeté hors image droite (%d,%d)", k, rx, ry)
                        else:
                            logger.warning("Frame %d: projection homographie invalide", k)
                    else:
                        logger.warning("Frame %d: échec homographie L->R", k)
                        
                        # Fallback 1: Essayer ORB si SIFT a échoué
                        if getattr(self, 'prefer_sift', True):  # Si on était en SIFT
                            logger.info("Frame %d: fallback ORB après échec SIFT", k)
                            try:
                                H_orb, _ = match_homography(currL, Lmask_new, currR, prefer_sift=False)
                                if H_orb is not None:
                                    logger.info("Frame %d: homographie ORB trouvée", k)
                                    # Utiliser la position seed (comme le clic manuel) au lieu du centroïde du masque
                                    if seed_pos is not None and 0 <= seed_pos[0] < currL.shape[1] and 0 <= seed_pos[1] < currL.shape[0]:
                                        cx_left = float(seed_pos[0])
                                        cy_left = float(seed_pos[1])
                                    else:
                                        # Fallback au centroïde si seed_pos invalide
                                        m_left = cv2.moments(Lmask_new.astype(np.uint8))
                                        if m_left['m00'] > 0:
                                            cx_left = m_left['m10']/m_left['m00']
                                            cy_left = m_left['m01']/m_left['m00']
                                        else:
                                            cx_left = currL.shape[1] / 2
                                            cy_left = currL.shape[0] / 2
                                    
                                    pt_left = np.array([[cx_left, cy_left, 1.0]], dtype=np.float32).T
                                    pt_right = H_orb @ pt_left
                                    if pt_right[2,0] != 0:
                                        rx = int(round(pt_right[0,0]/pt_right[2,0]))
                                        ry = int(round(pt_right[1,0]/pt_right[2,0]))
                                        logger.info("Frame %d: point ORB projeté à droite (%d,%d)", k, rx, ry)
                                        if 0 <= rx < currR.shape[1] and 0 <= ry < currR.shape[0]:
                                            # Validation de taille pour éviter l'explosion
                                            prev_right_area = np.sum(self.right_mask) if hasattr(self, 'right_mask') and self.right_mask is not None else 5000
                                            max_right_area = max(50000, int(prev_right_area) * 3)
                                            min_right_area = max(100, int(prev_right_area) // 3)
                                            
                                            lab_right = bgr_to_lab(currR)
                                            Rmask_new = seed_local_segmentation(lab_right, (rx, ry), tol=self.tol, radius=220, patch=7)
                                            if Rmask_new is not None:
                                                Rmask_new = clean_mask(largest_component(Rmask_new), 3, 1)
                                                new_right_area = np.sum(Rmask_new)
                                                
                                                # Validation critique de taille
                                                if new_right_area > max_right_area:
                                                    logger.warning("Frame %d: segmentation ORB droite trop grande (%d > %d), réduction tolérance", k, new_right_area, max_right_area)
                                                    # Essayer avec une tolérance réduite
                                                    reduced_tol = self.tol * 0.7
                                                    Rmask_new = seed_local_segmentation(lab_right, (rx, ry), tol=reduced_tol, radius=220, patch=7)
                                                    if Rmask_new is not None:
                                                        Rmask_new = clean_mask(largest_component(Rmask_new), 3, 1)
                                                        new_right_area = np.sum(Rmask_new)
                                                        if new_right_area > max_right_area:
                                                            logger.warning("Frame %d: segmentation ORB droite encore trop grande, annuler", k)
                                                            Rmask_new = None
                                                
                                                if Rmask_new is not None and new_right_area < min_right_area:
                                                    logger.warning("Frame %d: segmentation ORB droite trop petite (%d < %d)", k, new_right_area, min_right_area)
                                                    Rmask_new = None
                                                
                                                if Rmask_new is not None and np.sum(Rmask_new) > 10:
                                                    Rbox_new = oriented_box_points_from_mask(Rmask_new)
                                                    logger.info("Frame %d: segmentation droite ORB OK, aire=%d (prev=%d)", k, int(np.sum(Rmask_new)), int(prev_right_area))
                                                else:
                                                    logger.warning("Frame %d: segmentation ORB finale trop petite", k)
                                                    Rmask_new = None
                                            else:
                                                logger.warning("Frame %d: échec segmentation ORB", k)
                                        else:
                                            logger.warning("Frame %d: point ORB hors image (%d,%d)", k, rx, ry)
                                    else:
                                        logger.warning("Frame %d: projection ORB invalide", k)
                                else:
                                    logger.warning("Frame %d: échec homographie ORB", k)
                            except Exception as e:
                                _log_ex(e, f"orb_fallback_{k}")                        # Fallback 2: Template matching si ORB a aussi échoué
                        if Rmask_new is None and hasattr(self, 'right_mask') and self.right_mask is not None and hasattr(self, 'right_box') and self.right_box is not None:
                            logger.info("Frame %d: fallback - template matching depuis frame précédente", k)
                            try:
                                # Obtenir frame droite précédente
                                prevR = self.right.get_frame(prev_idx)
                                if prevR is not None:
                                    # Template matching simple
                                    boxR_fallback = template_fallback(prevR, self.right_box, currR)
                                    if boxR_fallback is not None and is_valid_box(boxR_fallback, currR.shape):
                                        logger.info("Frame %d: template matching réussi", k)
                                        # Segmenter autour du centre de la nouvelle boîte
                                        cx = int(np.mean(boxR_fallback[:,0]))
                                        cy = int(np.mean(boxR_fallback[:,1]))
                                        if 0 <= cx < currR.shape[1] and 0 <= cy < currR.shape[0]:
                                            # Validation de taille pour template matching droite  
                                            prev_right_area = np.sum(self.right_mask) if hasattr(self, 'right_mask') and self.right_mask is not None else 5000
                                            max_right_area = max(50000, int(prev_right_area) * 3)
                                            min_right_area = max(100, int(prev_right_area) // 3)
                                            
                                            lab_right = bgr_to_lab(currR)
                                            Rmask_new = seed_local_segmentation(lab_right, (cx, cy), tol=self.tol, radius=220, patch=7)
                                            if Rmask_new is not None:
                                                Rmask_new = clean_mask(largest_component(Rmask_new), 3, 1)
                                                new_right_area = np.sum(Rmask_new)
                                                
                                                # Validation critique de taille
                                                if new_right_area > max_right_area:
                                                    logger.warning("Frame %d: segmentation template droite trop grande (%d > %d), réduction tolérance", k, new_right_area, max_right_area)
                                                    # Essayer avec une tolérance réduite
                                                    reduced_tol = self.tol * 0.7
                                                    Rmask_new = seed_local_segmentation(lab_right, (cx, cy), tol=reduced_tol, radius=220, patch=7)
                                                    if Rmask_new is not None:
                                                        Rmask_new = clean_mask(largest_component(Rmask_new), 3, 1)
                                                        new_right_area = np.sum(Rmask_new)
                                                        if new_right_area > max_right_area:
                                                            logger.warning("Frame %d: segmentation template droite encore trop grande, annuler", k)
                                                            Rmask_new = None
                                                
                                                if Rmask_new is not None and new_right_area < min_right_area:
                                                    logger.warning("Frame %d: segmentation template droite trop petite (%d < %d)", k, new_right_area, min_right_area)
                                                    Rmask_new = None
                                                
                                                if Rmask_new is not None and np.sum(Rmask_new) > 10:
                                                    Rbox_new = oriented_box_points_from_mask(Rmask_new)
                                                    logger.info("Frame %d: fallback template + segmentation réussi, aire=%d (prev=%d)", k, int(np.sum(Rmask_new)), int(prev_right_area))
                                                else:
                                                    logger.warning("Frame %d: échec segmentation fallback template", k)
                                            else:
                                                logger.warning("Frame %d: échec segmentation fallback template", k)
                                        else:
                                            logger.warning("Frame %d: centre template hors image (%d,%d)", k, cx, cy)
                                    else:
                                        logger.warning("Frame %d: échec template matching", k)
                                else:
                                    logger.warning("Frame %d: frame droite précédente indisponible", k)
                            except Exception as e:
                                _log_ex(e, f"template_fallback_{k}")
                        
                        # Si template matching a échoué, utiliser l'ancien masque comme dernier recours
                        if Rmask_new is None and hasattr(self, 'right_mask') and self.right_mask is not None:
                            logger.info("Frame %d: fallback final - copie masque droite précédent", k)
                            try:
                                # Utiliser le masque précédent comme approximation
                                Rmask_new = self.right_mask.copy()
                                Rbox_new = oriented_box_points_from_mask(Rmask_new)
                                if Rbox_new is not None:
                                    logger.info("Frame %d: copie masque précédent réussie", k)
                            except Exception as e:
                                _log_ex(e, f"fallback_right_{k}")
                else:
                    logger.warning("Frame %d: masque gauche invalide pour homographie", k)
                        
                if Rmask_new is None or np.sum(Rmask_new) < 10:
                    logger.warning("Frame %d: ÉCHEC FINAL segmentation droite", k)
                    
            except Exception as e:
                _log_ex(e, f"segmentation_right_{k}")

            # 4. Update state - CRUCIAL: mettre à jour le masque de référence pour la frame suivante
            self.left_frame = currL
            self.right_frame = currR
            # Garder le nouveau masque pour la frame suivante SEULEMENT s'il est valide
            if Lmask_new is not None and np.sum(Lmask_new) > 10:
                self.left_mask = Lmask_new  # Mise à jour critique pour le SIFT suivant
                logger.info("Frame %d: masque gauche mis à jour pour frame suivante", k)
            else:
                logger.warning("Frame %d: garder ancien masque (nouveau invalide)", k)
            # Mise à jour masque et boîte droits
            if Rmask_new is not None and np.sum(Rmask_new) > 10:
                self.right_mask = Rmask_new
                self.right_box = Rbox_new
                logger.info("Frame %d: masque et boîte droite mis à jour", k)
            else:
                logger.warning("Frame %d: garder ancien masque/boîte droite", k)
            self.left_box = Lbox_new
            Lbox = Lbox_new
            Rbox = Rbox_new

            # 5. Update visual display in real-time
            try:
                # Update left view with new segmentation and box
                left_contour = mask_contour(Lmask_new) if Lmask_new is not None else None
                self.viewL.set_frame(currL, box=Lbox_new, contour=left_contour)
                logger.debug("Frame %d: mise à jour vue gauche avec segmentation", k)
                
                # Update right view - toujours afficher la frame même sans segmentation
                right_contour = mask_contour(Rmask_new) if Rmask_new is not None else None
                self.viewR.set_frame(currR, box=Rbox_new, contour=right_contour)
                if Rbox_new is not None:
                    logger.debug("Frame %d: mise à jour vue droite avec boîte et segmentation", k)
                else:
                    logger.debug("Frame %d: mise à jour vue droite sans boîte", k)
                
                # Force GUI update
                QtWidgets.QApplication.processEvents()
                
            except Exception as e:
                _log_ex(e, f"visual_update_{k}")

            # 6. Save measurements - temporarily set idx to k for measurement
            if Lbox_new is not None or Rbox_new is not None:
                old_idx = self.idx
                self.idx = k  # Temporarily set to current frame
                try:
                    self.measure_and_log()
                    # Force table update
                    QtWidgets.QApplication.processEvents()
                except Exception as e:
                    _log_ex(e, f"measure_and_log_{k}")
                finally:
                    self.idx = old_idx  # Restore original idx
                
            # Update progress et préparer pour frame suivante
            iter_idx += 1
            prev_idx = k
            # CRUCIAL: mettre à jour le masque précédent pour la prochaine itération
            if Lmask_new is not None and np.sum(Lmask_new) > 10:
                prev_left_mask = Lmask_new.copy()
                logger.info("Frame %d: masque précédent mis à jour pour itération suivante", k)
            
            try:
                self.progressBar.setValue(iter_idx)
                self.lblProgress.setText(f"SIFT simple: frame {k}")
                QtWidgets.QApplication.processEvents()
                
            except Exception as e:
                _log_ex(e, f"update_progress_{k}")

            # Save history periodically
            if iter_idx % 10 == 0:
                self.save_history(k)

        # Final update
        self.idx = min(end_frame, self.max_idx)
        self.refresh_frames()
        
        # Hide integrated progress UI when done/canceled
        try:
            self.progressBar.setVisible(False)
            self.btnCancelProg.setVisible(False)
            self.lblProgress.setText("")
            self.cancelProcessing = False
        except Exception as e:
            _log_ex(e, "hide_progress_ui")

    # ----- Export CSV -----

    def export_csv(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Exporter CSV", "mesures.csv", "CSV (*.csv)")
        if not path:
            return
        cols = self.table.columnCount()
        rows = self.table.rowCount()
        headers_list = []
        for i in range(cols):
            hitem = self.table.horizontalHeaderItem(i)
            headers_list.append(hitem.text() if hitem is not None else "")
        with open(path, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(headers_list)
            for r in range(rows):
                row_vals = []
                for c in range(cols):
                    it = self.table.item(r, c)
                    txt = it.text().strip() if it is not None and it.text() is not None else ""
                    # Remove zeros on export for coordinate columns only (not the frame index at col 0)
                    if c > 0 and txt in ("0", "0.0", "+0", "-0"):
                        txt = ""
                    row_vals.append(txt)
                wr.writerow(row_vals)
        self.status.showMessage(f"CSV écrit: {path}")

# ---------- Lancement ----------

def main():
    # Try to mitigate common Qt issues on Linux/WSL by forcing software OpenGL
    try:
        # PyQt5 exposes AA_UseSoftwareOpenGL directly on Qt
        QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL, True)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Warn if no display server is available
    if sys.platform.startswith("linux") and not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        print("[Erreur] Aucun serveur d'affichage détecté (DISPLAY/WAYLAND_DISPLAY manquant).\n"
              "Cette application Qt nécessite un environnement graphique (X11/Wayland).\n"
              "Conseils: activez WSLg (Windows 11), ou utilisez un serveur X (VcXsrv) et exportez DISPLAY.\n"
              "En l'absence d'affichage, passage en mode 'offscreen' (pas d'interface).",
              file=sys.stderr)
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

    # Prefer PyQt5's plugin path over OpenCV's bundled one to avoid xcb loader conflicts
    try:
        import PyQt5 as _pyqt5
        pyqt_plugins = os.path.join(os.path.dirname(_pyqt5.__file__), 'Qt', 'plugins')
        if os.path.isdir(pyqt_plugins):
            os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = pyqt_plugins
            os.environ['QT_PLUGIN_PATH'] = pyqt_plugins
    except Exception:
        pass

    # Log environment relevant to Qt platform selection
    setup_logging()
    install_global_exception_hook()
    logger.info("ENV: DISPLAY=%s WAYLAND_DISPLAY=%s QT_QPA_PLATFORM=%s",
                os.environ.get("DISPLAY"), os.environ.get("WAYLAND_DISPLAY"), os.environ.get("QT_QPA_PLATFORM"))
    logger.info("ENV: QT_PLUGIN_PATH=%s QT_QPA_PLATFORM_PLUGIN_PATH=%s",
                os.environ.get("QT_PLUGIN_PATH"), os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH"))
    # Warn for misconfigured runtime dir (Wayland)
    xdg = os.environ.get("XDG_RUNTIME_DIR") or "/run/user/1000"
    try:
        st = os.stat(xdg)
        perm = oct(st.st_mode & 0o777)
        if perm != '0o700':
            logger.warning("XDG_RUNTIME_DIR permissions are %s, expected 0o700: %s", perm, xdg)
    except Exception as e:
        _log_ex(e, "xdg_runtime_dir_check")

    try:
        app = QtWidgets.QApplication(sys.argv)
    except Exception as e:
        print("[Qt] Échec d'initialisation de l'interface graphique:", e, file=sys.stderr)
        print("Essayez d'installer les dépendances Qt/XCB côté Linux (ex.: libxcb-xinerama0, libxkbcommon-x11-0)\n"
              "ou lancez sous WSLg avec un DISPLAY valide.", file=sys.stderr)
        return
    # Setup logging and route Qt messages
    install_qt_message_logging()
    # Ensure Qt uses PyQt5 plugin path at runtime as well
    try:
        paths = [QtCore.QLibraryInfo.location(QtCore.QLibraryInfo.PluginsPath)]  # type: ignore[attr-defined]
        if paths and all(isinstance(p, str) for p in paths):
            QtCore.QCoreApplication.setLibraryPaths(paths)  # type: ignore[attr-defined]
            try:
                logger.info("Qt plugin path: %s", paths)
            except Exception:
                pass
    except Exception:
        pass
    # If Wayland platform with bad runtime perms, try switching to xcb dynamically
    try:
        plat_probe = QtGui.QGuiApplication.platformName()
        if plat_probe.lower().startswith('wayland'):
            xdg = os.environ.get("XDG_RUNTIME_DIR") or "/run/user/1000"
            try:
                st = os.stat(xdg)
                perm = oct(st.st_mode & 0o777)
                if perm != '0o700' and os.environ.get("DISPLAY"):
                    logger.warning("Switching Qt platform to xcb due to XDG_RUNTIME_DIR perms=%s", perm)
                    QtCore.QCoreApplication.setAttribute(QtCore.Qt.AA_UseSoftwareOpenGL, True)  # type: ignore[attr-defined]
                    os.environ['QT_QPA_PLATFORM'] = 'xcb'
            except Exception:
                pass
    except Exception:
        pass

    # Log platform and library paths
    try:
        plat = QtGui.QGuiApplication.platformName()
        libpaths = list(QtCore.QCoreApplication.libraryPaths())  # type: ignore[attr-defined]
        logger.info("Qt platform: %s", plat)
        logger.info("Qt libraryPaths: %s", libpaths)
    except Exception:
        pass

    w = MainWindow()
    w.show()
    try:
        geo = w.geometry()
        logger.info("MainWindow shown: x=%d y=%d w=%d h=%d", geo.x(), geo.y(), geo.width(), geo.height())
    except Exception:
        pass
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
