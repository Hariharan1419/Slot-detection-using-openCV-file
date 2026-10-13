"""
Smart Vehicle Parking System - OpenCV bounding-box prototype
-------------------------------------------------------------
* Phone camera (or any video file / webcam) is the live feed.
* Each parking slot is a rectangle you draw once (mark mode).
* A YOLO model detects vehicles; a slot is OCCUPIED when a vehicle box
  overlaps it enough for long enough (debounced, so passing cars don't flicker it).
* A virtual gate line counts vehicles ENTERING / EXITING using tracked IDs.
* Integrity check: expected cars inside (baseline + in - out) must match the
  number of occupied slots, otherwise an ALERT is raised and logged.
* Optional: send the free-slot count to the Arduino LCD over serial.

Usage
-----
  (Using toy cars / boxes / any object?  add:  --mode objects   and start with all slots EMPTY)

  1) Mark the slots and the gate line (once per camera position):
       python parking_system.py mark --source http://192.168.1.5:8080/video

  2) Run the system:
       python parking_system.py run --source http://192.168.1.5:8080/video
       python parking_system.py run --source http://192.168.1.5:8080/video --model my_model.pt

  Test without a phone:  --source parking_video.mp4   or   --source 0 (webcam)
"""
import argparse
import csv
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

CONFIG_FILE = "parking_config.json"
EVENT_LOG = "parking_events.csv"
FRAME_WIDTH = 960  # every frame is resized to this width so saved coordinates always match

# COCO names treated as vehicles (works with the default yolov8n.pt).
# If your own model has different class names, edit this set.
VEHICLE_NAMES = {"car", "motorcycle", "bus", "truck", "vehicle"}


# --------------------------------------------------------------------------- #
# Video input
# --------------------------------------------------------------------------- #
class LatestFrameReader:
    """Reads a stream in a background thread and always returns the newest frame.
    Phone streams buffer frames, which makes the display lag; this avoids it."""

    def __init__(self, source):
        self.source = int(source) if str(source).isdigit() else source
        self.is_http = isinstance(self.source, str) and self.source.lower().startswith(("http://", "https://"))
        self.is_file = isinstance(self.source, str) and Path(self.source).is_file()
        self.cap = None
        self.frame = None
        self.ok = True
        self.lock = threading.Lock()
        self.new = threading.Event()
        if self.is_http:
            # Phone streams (IP Webcam etc.) are read directly as MJPEG; this avoids
            # the FFmpeg decoder crash ("overread 8 / Invalid stream index").
            threading.Thread(target=self._mjpeg_loop, daemon=True).start()
            if not self.new.wait(8):
                self.ok = False
                raise RuntimeError(
                    f"No video received from {source}\n"
                    "Check: phone app server is running, same Wi-Fi, and the URL opens in your browser.")
        else:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open video source: {source}")
            threading.Thread(target=self._loop, daemon=True).start()

    def _push(self, frame):
        with self.lock:
            self.frame = frame
        self.new.set()

    def _mjpeg_loop(self):
        """Parse a multipart MJPEG (or single-image) HTTP stream by JPEG start/end markers."""
        url = self.source
        single = url.lower().split("?")[0].endswith((".jpg", ".jpeg"))  # e.g. /shot.jpg
        while self.ok:
            try:
                if single:  # snapshot URL: poll it
                    data = urllib.request.urlopen(url, timeout=5).read()
                    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        self._push(img)
                    time.sleep(0.05)
                    continue
                stream = urllib.request.urlopen(url, timeout=10)
                buf = b""
                while self.ok:
                    chunk = stream.read(4096)
                    if not chunk:
                        break
                    buf += chunk
                    a = buf.find(b"\xff\xd8")
                    b = buf.find(b"\xff\xd9", a + 2) if a != -1 else -1
                    if a != -1 and b != -1:
                        jpg, buf = buf[a:b + 2], buf[b + 2:]
                        img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            self._push(img)
                    elif a == -1 and len(buf) > 1_000_000:
                        buf = b""  # no JPEG found, drop junk
            except Exception as e:
                if self.ok:
                    print(f"[stream] {e} - retrying in 2s")
                    time.sleep(2)

    def _loop(self):
        fail = 0
        while self.ok:
            good, frame = self.cap.read()
            if not good:
                if self.is_file:  # video file ended
                    self.ok = False
                    break
                fail += 1
                if fail > 30:  # stream dropped -> try reconnecting
                    self.cap.release()
                    time.sleep(1)
                    self.cap = cv2.VideoCapture(self.source)
                    fail = 0
                continue
            fail = 0
            self._push(frame)
            if self.is_file:
                time.sleep(1 / 30)  # play files at roughly real time

    def read(self, timeout=2.0):
        self.new.wait(timeout)
        self.new.clear()
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.ok = False
        if self.cap is not None:
            self.cap.release()


def resize(frame):
    h, w = frame.shape[:2]
    return cv2.resize(frame, (FRAME_WIDTH, int(h * FRAME_WIDTH / w)))


# --------------------------------------------------------------------------- #
# Mark mode: draw slots + gate line
# --------------------------------------------------------------------------- #
def mark(args):
    reader = LatestFrameReader(args.source)
    print("Live preview: press 's' to freeze a frame for marking, 'q' to quit.")
    snap = None
    while True:
        f = reader.read()
        if f is None:
            if not reader.ok:
                break
            continue
        f = resize(f)
        cv2.imshow("preview", f)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("s"):
            snap = f
            break
        if k == ord("q"):
            break
    reader.release()
    cv2.destroyAllWindows()
    if snap is None:
        sys.exit("No frame captured.")

    slots = []
    while True:
        view = snap.copy()
        for i, (x, y, w, h) in enumerate(slots):
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(view, str(i + 1), (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(view, f"Slots: {len(slots)}  |  drag a box + ENTER, ESC when finished",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        x, y, w, h = cv2.selectROI("mark slots", view, showCrosshair=False)
        if w == 0 or h == 0:
            break
        slots.append([int(x), int(y), int(w), int(h)])
    cv2.destroyAllWindows()

    # gate line: two clicks
    pts = []

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 2:
            pts.append([x, y])

    cv2.namedWindow("mark gate line")
    cv2.setMouseCallback("mark gate line", on_mouse)
    while len(pts) < 2:
        view = snap.copy()
        for (x, y, w, h) in slots:
            cv2.rectangle(view, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for p in pts:
            cv2.circle(view, tuple(p), 6, (0, 0, 255), -1)
        cv2.putText(view, "Click 2 points across the gate/lane (cars must cross this line)",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.imshow("mark gate line", view)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()

    cfg = {"frame_width": FRAME_WIDTH, "slots": slots, "gate_line": pts if len(pts) == 2 else None}
    Path(CONFIG_FILE).write_text(json.dumps(cfg, indent=2))
    print(f"Saved {len(slots)} slots and gate line to {CONFIG_FILE}")


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def side(a, b, p):
    """Which side of line a->b the point p is on (sign only matters)."""
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossed_line(a, b, p, q):
    """Return +1 / -1 if segment p->q crosses segment a->b (direction by side change), else 0."""
    s1, s2 = side(a, b, p), side(a, b, q)
    s3, s4 = side(p, q, a), side(p, q, b)
    if s1 * s2 < 0 and s3 * s4 < 0:
        return 1 if s1 < 0 < s2 else -1
    return 0


def slot_overlap(slot, box):
    """Fraction of the SLOT area covered by the vehicle box."""
    sx, sy, sw, sh = slot
    x1, y1, x2, y2 = box
    ix = max(0, min(sx + sw, x2) - max(sx, x1))
    iy = max(0, min(sy + sh, y2) - max(sy, y1))
    return (ix * iy) / float(sw * sh)


# --------------------------------------------------------------------------- #
# Run mode
# --------------------------------------------------------------------------- #
class CentroidTracker:
    """Gives moving blobs stable ids by matching each blob to the nearest previous one."""

    def __init__(self, max_dist=90, max_missing=20):
        self.next_id = 1
        self.objs = {}  # id -> [cx, cy, missing]
        self.max_dist = max_dist
        self.max_missing = max_missing

    def update(self, boxes):
        cents = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in boxes]
        used, ids = set(), [None] * len(boxes)
        for oid, (ox, oy, _) in list(self.objs.items()):
            best, bd = None, self.max_dist
            for j, (cx, cy) in enumerate(cents):
                d = ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5
                if j not in used and d < bd:
                    best, bd = j, d
            if best is None:
                self.objs[oid][2] += 1
                if self.objs[oid][2] > self.max_missing:
                    del self.objs[oid]
            else:
                used.add(best)
                ids[best] = oid
                self.objs[oid] = [cents[best][0], cents[best][1], 0]
        for j, c in enumerate(cents):
            if ids[j] is None:
                ids[j] = self.next_id
                self.objs[self.next_id] = [c[0], c[1], 0]
                self.next_id += 1
        return ids


def prep_gray(frame):
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (7, 7), 0)


def slot_filled(gray, ref, slot, diff_thr, fill_frac):
    """True if enough pixels inside the slot differ from the empty-lot reference picture."""
    x, y, w, h = slot
    d = cv2.absdiff(gray[y:y + h, x:x + w], ref[y:y + h, x:x + w])
    return (d > diff_thr).mean() >= fill_frac


class EventLog:
    def __init__(self, path):
        new = not Path(path).exists()
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(["time", "event", "detail"])

    def log(self, event, detail=""):
        self.w.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), event, detail])
        self.f.flush()
        print(f"[{time.strftime('%H:%M:%S')}] {event} {detail}")


def run(args):
    objects_mode = args.mode == "objects"

    if not Path(CONFIG_FILE).exists():
        sys.exit(f"{CONFIG_FILE} not found. Run:  python parking_system.py mark --source <url>")
    cfg = json.loads(Path(CONFIG_FILE).read_text())
    slots = cfg["slots"]
    gate = cfg.get("gate_line")
    n_slots = len(slots)
    if not n_slots:
        sys.exit("No slots saved. Re-run mark mode.")

    model = None
    vehicle_ids = []
    if not objects_mode:
        is_world = "world" in Path(args.model).name.lower()
        if is_world:
            # YOLO-World: describe the object in words, no training needed.
            from ultralytics import YOLOWorld
            model = YOLOWorld(args.model)
            prompts = [c.strip() for c in (args.classes or "toy car").split(",") if c.strip()]
            model.set_classes(prompts)
            print(f"YOLO-World is looking for: {prompts}")
            vehicle_ids = list(range(len(prompts)))
        else:
            from ultralytics import YOLO  # imported here so 'mark' / objects mode work without it
            model = YOLO(args.model)
            names = model.names
            wanted = {c.strip().lower() for c in args.classes.split(",")} if args.classes else VEHICLE_NAMES
            vehicle_ids = [i for i, n in names.items() if n.lower() in wanted]
            print(f"Model classes: {list(names.values())}")
            if not vehicle_ids:
                # custom models often use names like "toy_car"; treat every class as a vehicle
                vehicle_ids = list(names.keys())
                print("Note: no class matched the car names, so ALL classes of this model count as vehicles. "
                      "Use --classes name1,name2 to pick specific ones.")
    else:
        print("OBJECTS MODE: keep ALL slots EMPTY and the lane clear while the empty reference is captured "
              f"(first {args.warmup_frames} frames). Press 'b' any time to recapture it.")
        bgsub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=False)
        tracker = CentroidTracker()
        ref_sum, ref_n, ref = None, 0, None

    ser = None
    if args.serial:
        import serial  # pip install pyserial
        ser = serial.Serial(args.serial, 9600, timeout=1)
        time.sleep(2)

    log = EventLog(EVENT_LOG)
    reader = LatestFrameReader(args.source)

    occupied = [False] * n_slots       # debounced state
    streak = [0] * n_slots             # frames the raw state has disagreed with the debounced one
    need_frames = args.hold_frames

    last_pos = {}                      # track id -> last centre
    last_counted = {}                  # track id -> time of last count (avoid double counting)
    entries = exits = 0
    baseline = None                    # cars already inside when the system starts
    warmup = 0
    mismatch_since = None
    alert = False
    last_sent = None
    prev_free = None
    log.log("START", f"slots={n_slots}")

    while True:
        frame = reader.read()
        if frame is None:
            if not reader.ok:
                break
            continue
        frame = resize(frame)

        dets = []  # (x1,y1,x2,y2,id)
        if objects_mode:
            gray = prep_gray(frame)
            if ref is None:  # still building the empty-lot reference
                ref_sum = gray.astype(np.float32) if ref_sum is None else ref_sum + gray
                ref_n += 1
                if ref_n >= args.warmup_frames:
                    ref = (ref_sum / ref_n).astype(np.uint8)
                    log.log("REFERENCE", "empty-lot picture captured")
            mask = bgsub.apply(frame)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            mask = cv2.dilate(mask, np.ones((9, 9), np.uint8), iterations=2)
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            boxes = []
            for c in cnts:
                if cv2.contourArea(c) >= args.min_area:
                    x, y, w, h = cv2.boundingRect(c)
                    boxes.append((x, y, x + w, y + h))
            for b, tid in zip(boxes, tracker.update(boxes)):
                dets.append((*b, int(tid)))
        else:
            res = model.track(frame, persist=True, conf=args.conf, classes=vehicle_ids,
                              tracker="bytetrack.yaml", verbose=False)[0]
            if res.boxes is not None and len(res.boxes):
                xyxy = res.boxes.xyxy.cpu().numpy()
                ids = res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None else [-1] * len(xyxy)
                for b, tid in zip(xyxy, ids):
                    dets.append((*b.tolist(), int(tid)))

        # ---- slot occupancy (debounced) ----
        for i, slot in enumerate(slots):
            if objects_mode:
                raw = ref is not None and slot_filled(gray, ref, slot, args.diff, args.fill)
            else:
                raw = any(slot_overlap(slot, d[:4]) >= args.overlap for d in dets)
            if raw != occupied[i]:
                streak[i] += 1
                if streak[i] >= need_frames:
                    occupied[i] = raw
                    streak[i] = 0
                    log.log("SLOT_OCCUPIED" if raw else "SLOT_FREED", f"slot={i + 1}")
            else:
                streak[i] = 0

        # ---- gate counting ----
        if gate:
            a, b = tuple(gate[0]), tuple(gate[1])
            now = time.time()
            for (x1, y1, x2, y2, tid) in dets:
                if tid < 0:
                    continue
                c = ((x1 + x2) / 2, (y1 + y2) / 2)
                if tid in last_pos and now - last_counted.get(tid, 0) > 3:
                    d = crossed_line(a, b, last_pos[tid], c)
                    if args.flip:
                        d = -d
                    if d > 0:
                        entries += 1
                        last_counted[tid] = now
                        log.log("ENTRY", f"id={tid} total_in={entries}")
                    elif d < 0:
                        exits += 1
                        last_counted[tid] = now
                        log.log("EXIT", f"id={tid} total_out={exits}")
                last_pos[tid] = c

        # ---- baseline + integrity check ----
        warmup += 1
        if baseline is None and warmup >= args.warmup_frames + (8 if objects_mode else 0):
            baseline = 0 if objects_mode else sum(occupied)
            log.log("BASELINE", f"cars_already_inside={baseline}")
        occ_count = sum(occupied)
        free = n_slots - occ_count
        if baseline is not None:
            expected = max(0, baseline + entries - exits)
            if expected != occ_count:
                mismatch_since = mismatch_since or time.time()
                if not alert and time.time() - mismatch_since > args.mismatch_secs:
                    alert = True
                    log.log("ALERT", f"expected_inside={expected} occupied_slots={occ_count}")
            else:
                if alert:
                    log.log("ALERT_CLEARED", "")
                mismatch_since, alert = None, False
        else:
            expected = None

        if free != prev_free:
            log.log("FREE_SLOTS", str(free))
            prev_free = free
        if ser and free != last_sent:
            ser.write(f"FREE:{free}\n".encode())
            last_sent = free

        # ---- drawing ----
        for i, (x, y, w, h) in enumerate(slots):
            col = (0, 0, 255) if occupied[i] else (0, 200, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), col, 2)
            cv2.putText(frame, f"{i + 1}:{'FULL' if occupied[i] else 'FREE'}", (x + 4, y + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        for (x1, y1, x2, y2, tid) in dets:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 200, 0), 1)
            cv2.putText(frame, f"id{tid}", (int(x1), int(y1) - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 0), 1)
        if gate:
            cv2.line(frame, tuple(gate[0]), tuple(gate[1]), (0, 255, 255), 2)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 34), (30, 30, 30), -1)
        txt = f"FREE {free}/{n_slots}   IN {entries}   OUT {exits}   INSIDE(expected) {expected if expected is not None else '...'}"
        cv2.putText(frame, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        if alert:
            cv2.putText(frame, "ALERT: gate count != occupied slots", (10, 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Smart Parking", frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("b") and objects_mode:  # recapture the empty-lot reference
            ref, ref_sum, ref_n = None, None, 0
            occupied = [False] * n_slots
            entries = exits = 0
            baseline, warmup = None, 0
            log.log("RECAPTURE", "keep slots empty")
        if k == ord("r"):  # re-baseline after fixing a miscount
            entries = exits = 0
            baseline = sum(occupied)
            log.log("RESET", f"baseline={baseline}")

    log.log("STOP", f"in={entries} out={exits}")
    reader.release()
    cv2.destroyAllWindows()


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="Smart parking: OpenCV slot detection + entry/exit counting")
    sub = p.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--source", required=True,
                        help="phone stream URL (e.g. http://192.168.1.5:8080/video), video file, or webcam index")

    sub.add_parser("mark", parents=[common], help="draw slot boxes and the gate line")

    r = sub.add_parser("run", parents=[common], help="run detection + counting")
    r.add_argument("--mode", choices=["yolo", "objects"], default="yolo",
                   help="yolo = detect real vehicles; objects = detect ANY object (toy cars, boxes, blocks)")
    r.add_argument("--diff", type=int, default=35, help="objects mode: pixel change needed to count as different")
    r.add_argument("--fill", type=float, default=0.25, help="objects mode: fraction of slot that must change to be FULL")
    r.add_argument("--min-area", type=int, default=600, help="objects mode: smallest moving blob that counts at the gate")
    r.add_argument("--model", default="yolov8n.pt",
                   help="path to your trained YOLO .pt model, or yolov8s-worldv2.pt to find objects by description")
    r.add_argument("--classes", default=None, help="class names to count, e.g. toy_car. With a YOLO-World model: what to look for in words, e.g. \"toy car\"")
    r.add_argument("--conf", type=float, default=0.35, help="detection confidence")
    r.add_argument("--overlap", type=float, default=0.40, help="fraction of slot covered to count as occupied")
    r.add_argument("--hold-frames", type=int, default=15, help="frames a change must persist (debounce)")
    r.add_argument("--warmup-frames", type=int, default=45, help="frames before the starting car count is fixed")
    r.add_argument("--mismatch-secs", type=float, default=15, help="seconds of count mismatch before ALERT")
    r.add_argument("--flip", action="store_true", help="swap IN/OUT direction")
    r.add_argument("--serial", default=None, help="Arduino port, e.g. COM3 or /dev/ttyUSB0")

    args = p.parse_args()
    mark(args) if args.cmd == "mark" else run(args)


if __name__ == "__main__":
    main()
