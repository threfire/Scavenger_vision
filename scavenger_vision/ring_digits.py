from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


MASK_SIZE = 24
DIGIT_SIZE = 20
TEMPLATE_DIGITS = (1, 2, 3, 4, 5, 7)


def normalized_mask(mask):
    x, y, width, height = cv2.boundingRect(mask)
    scale = DIGIT_SIZE / max(width, height)
    scaled_width = max(1, round(width * scale))
    scaled_height = max(1, round(height * scale))
    glyph = mask[y:y + height, x:x + width].astype(np.float32)
    resized = cv2.resize(glyph, (scaled_width, scaled_height),
                         interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.zeros((MASK_SIZE, MASK_SIZE), dtype=np.float32)
    left = (MASK_SIZE - scaled_width) // 2
    top = (MASK_SIZE - scaled_height) // 2
    canvas[top:top + scaled_height, left:left + scaled_width] = resized
    vector = canvas.ravel()
    return vector / np.linalg.norm(vector)


def footless_ones(printed_glyphs):
    # 常见的无底座数字 1 只有竖笔和短斜笔。
    variants = []
    for glyph in printed_glyphs:
        _, top, _, height = cv2.boundingRect(glyph)
        stem = np.flatnonzero(glyph[top + height // 2])
        bottom = top + height * 3 // 4
        no_base = glyph.copy()
        no_base[bottom:, :stem[0]] = 0
        no_base[bottom:, stem[-1] + 1:] = 0
        if not np.array_equal(no_base, glyph):
            variants.append(no_base)
    for width in (4, 6, 8):
        for flag_width, flag_height in ((0, 0), (10, 8), (12, 12), (10, 5)):
            glyph = np.zeros((72, 72), dtype=np.uint8)
            cv2.rectangle(glyph, (42, 18), (41 + width, 53), 255, -1)
            if flag_width:
                flag = np.array([(42, 18), (42, 17 + width),
                                 (42 - flag_width, 17 + flag_height + width),
                                 (42 - flag_width, 18 + flag_height)], dtype=np.int32)
                cv2.fillConvexPoly(glyph, flag, 255)
            variants.append(glyph)
    return variants


@lru_cache(maxsize=1)
def digit_templates():
    # 多种字形、笔画宽度和方向，允许相机转向后继续读取编号。
    fonts = (cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX,
             cv2.FONT_HERSHEY_COMPLEX, cv2.FONT_HERSHEY_PLAIN)
    rotations = [cv2.getRotationMatrix2D((36, 36), angle, 1)
                 for angle in range(0, 360, 10)]
    with np.load(Path(__file__).with_name('ring_glyphs.npz')) as archive:
        printed_masks = archive['masks']

    templates = []
    for number in TEMPLATE_DIGITS:
        digit = str(number)
        glyphs = list(printed_masks[number])
        if number == 1:
            glyphs.extend(footless_ones(printed_masks[number]))
        for font in fonts:
            for thickness in (2, 5):
                scale = cv2.getFontScaleFromHeight(font, 34, thickness)
                (width, height), _ = cv2.getTextSize(digit, font, scale, thickness)
                canvas = np.zeros((72, 72), dtype=np.uint8)
                cv2.putText(canvas, digit, ((72 - width) // 2, (72 + height) // 2),
                            font, scale, 255, thickness)
                glyphs.append(canvas)
        variants = [normalized_mask(cv2.warpAffine(glyph, rotation, (72, 72),
                                                    flags=cv2.INTER_NEAREST))
                    for glyph in glyphs for rotation in rotations]
        templates.append(np.asarray(variants, dtype=np.float32))
    return tuple(templates)


def read_ring_digit(gray, hole):
    center, axes, angle = hole
    if min(axes) < 12:
        return 0
    x, y, width, height = cv2.boundingRect(cv2.boxPoints(hole))
    left, top = max(0, x), max(0, y)
    patch = gray[top:y + height, left:x + width]
    if patch.size == 0:
        return 0

    mask = np.zeros_like(patch)
    opening = ((center[0] - left, center[1] - top),
               (axes[0] * 0.88, axes[1] * 0.88), angle)
    cv2.ellipse(mask, opening, 255, -1)
    inside = mask != 0
    pixels = patch[inside]
    if pixels.size < 30:
        return 0
    threshold, _ = cv2.threshold(pixels, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    dark_background = np.count_nonzero(pixels <= threshold) > pixels.size / 2
    foreground = patch > threshold if dark_background else patch <= threshold
    ink = (inside & foreground).astype(np.uint8)
    count, labels, stats, centers = cv2.connectedComponentsWithStats(ink, connectivity=8)
    candidates = []
    for index in range(1, count):
        _, _, width, height, area = stats[index]
        dx, dy = centers[index] - opening[0]
        if (area >= max(7, pixels.size * 0.012)
                and max(width, height) >= min(axes) * 0.22
                and np.hypot(dx, dy) <= min(axes) * 0.30):
            candidates.append(index)
    if not candidates:
        return 0
    candidates.sort(key=lambda index: stats[index, cv2.CC_STAT_AREA], reverse=True)
    primary = candidates[0]
    if (len(candidates) > 1
            and stats[candidates[1], cv2.CC_STAT_AREA] >= stats[primary, cv2.CC_STAT_AREA] * 0.45):
        return 0
    component = labels == primary
    interior = cv2.erode(mask, np.ones((3, 3), dtype=np.uint8))
    contrast = abs(float(np.median(pixels)) - float(np.mean(patch[component])))
    if contrast < 22 or np.any(component & (interior == 0)):
        return 0

    glyph = component.astype(np.uint8)
    contours, hierarchy = cv2.findContours(glyph, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # 印刷的 1、2、3 没有闭合孔，明显的内孔通常来自其他数字。
    inner_area_limit = max(2, stats[primary, cv2.CC_STAT_AREA] * 0.025)
    if hierarchy is not None and any(
            parent >= 0 and cv2.contourArea(contours[index]) >= inner_area_limit
            for index, parent in enumerate(hierarchy[0, :, 3])):
        return 0

    vector = normalized_mask(glyph)
    scores = [np.einsum('ti,i->t', templates, vector).max() for templates in digit_templates()]
    best, second, *_ = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)
    digit = TEMPLATE_DIGITS[best]
    if digit > 3 or scores[best] < 0.74 or scores[best] - scores[second] < 0.045:
        return 0
    return digit
