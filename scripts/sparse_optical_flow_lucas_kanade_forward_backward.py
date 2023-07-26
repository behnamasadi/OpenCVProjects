import numpy as np
import cv2

lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict(maxCorners=500,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)


class App:
    def __init__(self, video_src):
        self.track_len = 10
        self.detect_interval = 1
        self.tracks = []
        self.cam = cv2.VideoCapture(video_src)

        self.cam.set(cv2.CAP_PROP_FOURCC,
                     cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

        self.width = 1920
        self.height = 1080

        self.width = 320
        self.height = 240

        self.width = 320*2
        self.height = 240*2

        # self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        # self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        self.frame_idx = 0

    def run(self):
        while True:
            ret, frame = self.cam.read()
            frame = cv2.resize(frame, (self.width, self.height))
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()

            if len(self.tracks) > 0:
                img0, img1 = self.prev_gray, frame_gray
                p0 = np.float32([tr[-1]
                                for tr in self.tracks]).reshape(-1, 1, 2)

                p1, st, err = cv2.calcOpticalFlowPyrLK(
                    img0, img1, p0, None, **lk_params)
                p0r, st, err = cv2.calcOpticalFlowPyrLK(
                    img1, img0, p1, None, **lk_params)
                d = abs(p0-p0r).reshape(-1, 2).max(-1)
                good = d < 1
                new_tracks = []
                for tr, (x, y), good_flag in zip(self.tracks, p1.reshape(-1, 2), good):
                    if not good_flag:
                        continue
                    tr.append((x, y))
                    if len(tr) > self.track_len:
                        del tr[0]
                    new_tracks.append(tr)
                    cv2.circle(vis, (int(x), int(y)), 2, (0, 255, 0), -1)
                self.tracks = new_tracks
                cv2.polylines(vis, [np.int32(tr)
                              for tr in self.tracks], False, (0, 255, 0))
                cv2.putText(vis, 'track count: %d' % len(
                    self.tracks), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            if self.frame_idx % self.detect_interval == 0:
                mask = np.zeros_like(frame_gray)
                mask[:] = 255
                for x, y in [np.int32(tr[-1]) for tr in self.tracks]:
                    cv2.circle(mask, (x, y), 5, 0, -1)
                p = cv2.goodFeaturesToTrack(
                    frame_gray, mask=mask, **feature_params)
                if p is not None:
                    for x, y in np.float32(p).reshape(-1, 2):
                        self.tracks.append([(x, y)])

            self.frame_idx += 1
            self.prev_gray = frame_gray
            cv2.imshow('lk_track', vis)

            ch = 0xFF & cv2.waitKey(1)
            if ch == 27:
                break


def main():
    import sys
    # argv[1] should be /dev/video0 or video_path

    try:
        video_src = sys.argv[1]
    except:
        video_src = "/dev/video0"

    App(video_src).run()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
