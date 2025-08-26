import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque, Counter

# Optional voice (wrap in try to avoid hard dependency)
try:
    import pyttsx3
    VOICE_AVAILABLE = True
except Exception:
    VOICE_AVAILABLE = False

# ---------------------------
# Configuration (tweakable)
# ---------------------------
MAX_HANDS = 2
SMOOTH_WINDOW = 7           # number of frames to vote on a gesture
PINCH_THRESH = 0.22         # normalized distance thumb-index for pinch
FIST_OPEN_THRESH = 0.35     # threshold for detecting closed fist (avg tip distances)
FIST_HOLD_SEC = 0.8         # seconds to hold fist to trigger erase
BOTH_OPEN_HOLD_SEC = 1.0    # seconds to hold both-open to clear all
SWIPE_VEL_THRESH = 0.9      # px/frame magnitude to consider a swipe
SWIPE_FRAMES = 5            # frames window to compute swipe velocity
COOLDOWN_SEC = 0.8          # time between accepting tokens to avoid duplicates

# Colors for HUD
COLOR_BG = (8, 12, 20)
COLOR_PANEL = (12, 30, 42)
COLOR_ACCENT = (36, 220, 200)
COLOR_TEXT = (220, 230, 240)
COLOR_WARN = (220, 100, 100)


# ---------------------------
# Utils & Gesture Detection
# ---------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

def distance(a, b):
    return np.hypot(a[0]-b[0], a[1]-b[1])

def normalized_landmarks_to_px(landmarks, w, h):
    return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]

def fingers_up_from_landmarks(lm_px):
    """
    Returns list [thumb, index, middle, ring, pinky] -> 1 if finger considered 'up'
    Uses simple heuristic: compare tip y with pip y (lower y means up on image).
    Thumb uses x-direction heuristic.
    """
    tips = [4, 8, 12, 16, 20]
    up = []
    # Thumb: compare tip x vs ip x (works for front-facing)
    try:
        up.append(1 if lm_px[4][0] < lm_px[3][0] else 0)  # thumb left of ip (mirror view)
    except:
        up.append(0)
    for t in tips[1:]:
        try:
            up.append(1 if lm_px[t][1] < lm_px[t-2][1] else 0)
        except:
            up.append(0)
    return up

def is_pinch(lm_norm):
    """Normalized landmarks (0..1): detect pinch between thumb tip (4) and index tip (8)."""
    a = (lm_norm[4].x, lm_norm[4].y)
    b = (lm_norm[8].x, lm_norm[8].y)
    d = np.hypot(a[0]-b[0], a[1]-b[1])
    return d < PINCH_THRESH, d

def is_fist_closed(lm_px):
    """Rough check: average distance between finger tips and wrist; closed fist has small distances."""
    tips = [4,8,12,16,20]
    wrist = lm_px[0]
    ds = [distance(wrist, lm_px[t]) for t in tips if t < len(lm_px)]
    if not ds:
        return False, 999
    return (np.mean(ds) < (FIST_OPEN_THRESH * 1000)), np.mean(ds)

def hands_open_count(fingers_list):
    """Return how many hands appear open (>=4 fingers up)"""
    count = 0
    for f in fingers_list:
        if sum(f) >= 4:
            count += 1
    return count

def detect_number_from_fingers(fingers):
    """Map simple finger counts to digits 0-5 (extend as needed)."""
    cnt = sum(fingers)
    # Map thumb-only pattern to '1' ambiguity handled elsewhere
    mapping = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5'}
    return mapping.get(cnt, None)


# ---------------------------
# Main GestureCalculator Class
# ---------------------------
class GestureCalculator:
    def __init__(self, use_voice=False):
        self.hands = mp_hands.Hands(max_num_hands=MAX_HANDS, min_detection_confidence=0.75, min_tracking_confidence=0.75)
        self.cap = cv2.VideoCapture(0)
        self.pending_votes = deque(maxlen=SMOOTH_WINDOW)   # store recent token votes (strings)
        self.pending_pinches = deque(maxlen=SMOOTH_WINDOW)
        self.frame_positions = deque(maxlen=SWIPE_FRAMES)  # store recent centroid positions for swipe detection
        self.last_accept_time = 0.0
        self.expression_tokens = []   # list of strings/tokens forming the expression
        self.current_pending_token = ""  # current recognized token waiting for confirmation
        self.fist_start_time = None
        self.both_open_start_time = None
        self.use_voice = use_voice and VOICE_AVAILABLE
        if self.use_voice:
            self.voice = pyttsx3.init()
            self.voice.setProperty('rate', 160)

    def speak(self, txt):
        if self.use_voice:
            self.voice.say(txt)
            self.voice.runAndWait()

    def draw_ui(self, frame, pending, expression, status_hint):
        h, w = frame.shape[:2]
        panel_h = 140
        overlay = frame.copy()
        # Draw bottom HUD panel
        cv2.rectangle(overlay, (12, h - panel_h - 12), (w - 12, h - 12), COLOR_PANEL, -1)
        cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

        # Title
        cv2.putText(frame, "Gesture Calculator (Advanced)".upper(), (28, h - panel_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_ACCENT, 2, cv2.LINE_AA)

        # Expression
        cv2.putText(frame, "Expr: " + (" ".join(expression) if expression else " "), (28, h - panel_h + 52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2, cv2.LINE_AA)

        # Pending
        cv2.putText(frame, f"Pending: {pending}", (28, h - panel_h + 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_ACCENT, 2, cv2.LINE_AA)

        # Helper / status
        cv2.putText(frame, status_hint, (28, h - panel_h + 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1, cv2.LINE_AA)

        # small legend
        legend_x = w - 320
        cv2.putText(frame, "Controls:", (legend_x, h - panel_h + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.55, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, "Pinch -> Confirm", (legend_x, h - panel_h + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, "Fist (hold) -> Erase last", (legend_x, h - panel_h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)
        cv2.putText(frame, "Both-open (hold) -> Clear all", (legend_x, h - panel_h + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_TEXT, 1, cv2.LINE_AA)

    def compute_swipe(self):
        """Compute average velocity from stored positions and decide direction or None."""
        if len(self.frame_positions) < 2:
            return None
        pts = np.array(self.frame_positions)
        # velocity = last - first over frames
        dx = (pts[-1][0] - pts[0][0]) / max(1, len(pts))
        dy = (pts[-1][1] - pts[0][1]) / max(1, len(pts))
        mag = np.hypot(dx, dy)
        if mag < SWIPE_VEL_THRESH:
            return None
        # determine dominant direction
        if abs(dx) > abs(dy):
            return 'SWIPE_RIGHT' if dx > 0 else 'SWIPE_LEFT'
        else:
            return 'SWIPE_DOWN' if dy > 0 else 'SWIPE_UP'

    def accept_token(self, token):
        """Add recognized token to expression with cooldown."""
        now = time.time()
        if now - self.last_accept_time < COOLDOWN_SEC:
            return False
        # Basic validation: avoid consecutive operators
        ops = set(['+', '-', '*', '/'])
        if token in ops and (not self.expression_tokens or self.expression_tokens[-1] in ops):
            # invalid operator entry
            return False
        self.expression_tokens.append(token)
        self.last_accept_time = now
        self.speak(token if self.use_voice else "")
        return True

    def erase_last(self):
        if self.expression_tokens:
            popped = self.expression_tokens.pop()
            self.speak("Erased " + str(popped) if self.use_voice else "")
            return popped
        return None

    def clear_all(self):
        self.expression_tokens = []
        self.speak("Cleared" if self.use_voice else "")

    def evaluate_expression(self):
        try:
            expr = "".join(self.expression_tokens)
            if not expr:
                return None
            # safe evaluate: use eval but restrict globals (still not 100% sandboxed but okay for local)
            res = eval(expr, {"__builtins__": {}}, {})
            return str(res)
        except Exception:
            return "ERR"

    def run(self):
        status_hint = "Show number by fingers, pinch to confirm"
        last_result = None

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Camera read failed")
                break
            frame = cv2.flip(frame, 1)  # mirror for user
            h, w = frame.shape[:2]
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)

            # default detected values
            detected_tokens = []
            pinch_votes = []
            fingers_list = []
            centroids = []

            # Process hands
            if results.multi_hand_landmarks:
                for hand_landmarks, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                    # draw
                    mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # normalized already
                    lm_norm = hand_landmarks.landmark
                    # pixel coords
                    lm_px = normalized_landmarks_to_px(lm_norm, w, h)
                    fingers = fingers_up_from_landmarks(lm_px)
                    fingers_list.append(fingers)

                    # pinch detection (normalized)
                    pinched, pin_dist = is_pinch(lm_norm)
                    pinch_votes.append(pinched)

                    # detect basic number from finger count
                    num_token = detect_number_from_fingers(fingers)
                    if num_token is not None:
                        detected_tokens.append(num_token)

                    # compute centroid for swipe detection
                    cx = int(np.mean([p[0] for p in lm_px]))
                    cy = int(np.mean([p[1] for p in lm_px]))
                    centroids.append((cx, cy))

            # aggregate: find mode token among detected_tokens
            token_vote = None
            if detected_tokens:
                token_vote = Counter(detected_tokens).most_common(1)[0][0]

            # push to vote history for smoothing
            self.pending_votes.append(token_vote)
            self.pending_pinches.append(1 if any(pinch_votes) else 0)
            # compute smoothed token as mode ignoring None
            votes = [v for v in list(self.pending_votes) if v is not None]
            smoothed_token = Counter(votes).most_common(1)[0][0] if votes else ""

            # swipe detection (use centroid history)
            if centroids:
                # if multiple hands, use first hand centroid for swipe detection
                self.frame_positions.append(centroids[0])
            else:
                # slowly decay (append last position if available)
                if self.frame_positions:
                    self.frame_positions.append(self.frame_positions[-1])

            swipe = self.compute_swipe()

            # Fist & both-open detection for erase/clear
            fist_closed = False
            fist_mean_dist = 999
            if results.multi_hand_landmarks and len(results.multi_hand_landmarks) >= 1:
                # use first hand lm_px to detect fist closure
                lm_px0 = normalized_landmarks_to_px(results.multi_hand_landmarks[0].landmark, w, h)
                fist_closed, fist_mean_dist = is_fist_closed(lm_px0)

            hands_open = hands_open_count(fingers_list)

            now = time.time()

            # handle both-open hold => clear all
            if hands_open >= 2:
                if self.both_open_start_time is None:
                    self.both_open_start_time = now
                elif now - self.both_open_start_time > BOTH_OPEN_HOLD_SEC:
                    self.clear_all()
                    status_hint = "Cleared all"
                    self.both_open_start_time = None
                    # reset votes to avoid immediate re-trigger
                    self.pending_votes.clear()
                    self.pending_pinches.clear()
            else:
                self.both_open_start_time = None

            # handle fist hold => erase last token
            if fist_closed:
                if self.fist_start_time is None:
                    self.fist_start_time = now
                elif now - self.fist_start_time > FIST_HOLD_SEC:
                    erased = self.erase_last()
                    status_hint = f"Erased {erased}" if erased else "Nothing to erase"
                    self.fist_start_time = None
                    self.pending_votes.clear()
                    self.pending_pinches.clear()
            else:
                self.fist_start_time = None

            # If pinch consensus and have smoothed token -> accept it
            pinch_consensus = (sum(self.pending_pinches) >= (SMOOTH_WINDOW // 2 + 1))
            if pinch_consensus and smoothed_token:
                accepted = self.accept_token(smoothed_token)
                if accepted:
                    status_hint = f"Accepted {smoothed_token}"
                    # clear pending votes to avoid repeats
                    self.pending_votes.clear()
                    self.pending_pinches.clear()
                else:
                    status_hint = "Invalid token or cooldown"

            # Swipe-based operators (require a decisive swipe)
            if swipe is not None:
                op_map = {
                    'SWIPE_LEFT': '-',
                    'SWIPE_RIGHT': '+',
                    'SWIPE_UP': '*',
                    'SWIPE_DOWN': '/'
                }
                op_token = op_map.get(swipe)
                if op_token:
                    accepted = self.accept_token(op_token)
                    if accepted:
                        status_hint = f"Operator {op_token} added"
                        # reset frame_positions to avoid multiple registers
                        self.frame_positions.clear()
                # else ignore small swipe / no-op

            # If user shows two hands open -> act as equals (evaluate)
            if hands_open >= 2 and not (self.both_open_start_time):
                # quick two-hand open (not hold) triggers evaluate
                res = self.evaluate_expression()
                last_result = res
                if res is not None:
                    # show & also set expression to result (so user can continue)
                    if res != "ERR":
                        self.expression_tokens = [res]
                        status_hint = "Result: " + str(res)
                        self.speak("Result " + str(res))
                    else:
                        status_hint = "Evaluation error"
                        self.speak("Error")
                    # cooldown
                    time.sleep(0.4)

            # Render UI
            display_pending = smoothed_token if smoothed_token else ""
            self.draw_ui(frame, display_pending, self.expression_tokens, status_hint)

            # show last result if present near top
            if last_result is not None:
                cv2.putText(frame, f"Last: {last_result}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_ACCENT, 2, cv2.LINE_AA)

            # show pinch debug (optional)
            #cv2.putText(frame, f"PinchVotes: {sum(self.pending_pinches)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_TEXT, 1)

            # show on screen
            cv2.imshow("Gesture Calculator - Advanced", frame)
            k = cv2.waitKey(1) & 0xFF
            if k == ord('q'):
                break
            elif k == ord('='):
                # manual evaluate
                res = self.evaluate_expression()
                last_result = res
                if res is not None:
                    if res != "ERR":
                        self.expression_tokens = [res]
                        self.speak("Result " + str(res))
                    else:
                        self.speak("Error")
            elif k == ord('c'):
                self.clear_all()
            elif k == ord('e'):
                self.erase_last()

        self.cap.release()
        cv2.destroyAllWindows()


# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--voice", action="store_true", help="Enable voice feedback (requires pyttsx3)")
    args = p.parse_args()
    gc = GestureCalculator(use_voice=args.voice)
    gc.run()
