"""PV selection policy shared by all milestones and live tracking."""
import math

TARGET_SELECTION = 'highest_confidence'


def fresh(received, now, maximum_age=.5):
    """Accept only recent observations, excluding future timestamps."""
    return 0 <= now-received <= maximum_age


def target_from_boxes(boxes, class_id, confidence, size):
    """Select highest-confidence valid PV; equal scores keep detector order."""
    w,h=size
    candidates=[]
    for x1,y1,x2,y2,score,cls in boxes:
        if not all(math.isfinite(float(x)) for x in (x1,y1,x2,y2,score,cls)):
            continue
        if int(cls)!=class_id or score < confidence or score > 1:
            continue
        if 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h:
            candidates.append(dict(box=[x1,y1,x2,y2],confidence=score,
                                   center=[(x1+x2)/2,(y1+y2)/2]))
    return max(candidates,key=lambda target:target['confidence']) if candidates else None, len(candidates)



def selected_observation_valid(sample):
    """Candidate count is diagnostic; validate the selected PV coordinates."""
    return (sample.get('status') == 'detected' and sample.get('count', 0) >= 1
            and all(isinstance(sample.get(axis), (int, float))
                    and math.isfinite(sample[axis]) for axis in ('u', 'v')))
