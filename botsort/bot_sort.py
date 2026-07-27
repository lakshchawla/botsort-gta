import numpy as np
from collections import deque
from typing import Optional

from . import matching

from .basetrack import BaseTrack, TrackState
from .kalman_filter import KalmanFilter
import math

class ID_Assigner:
    def __init__(self, init_id=0):
        self.cur_id = init_id

    def next_id(self):
        self.cur_id += 1
        return self.cur_id

class STrack(BaseTrack):
    shared_kalman = KalmanFilter()

    # Stream key used when `feat` is passed as a single array rather than a
    # dict of {model_name: embedding} - the common single-ReID-model case.
    _DEFAULT_STREAM = "primary"

    def __init__(self, tlwh, score, feat=None, obj_meta = None, pose=None, num_kpts=0, img_path=None, feat_history=50):

        # wait activate
        self._tlwh = np.asarray(tlwh, dtype=np.float64)
        self.is_touching_edge = obj_meta
        self.kalman_filter = None
        self.mean, self.covariance = None, None
        self.is_activated = False

        self.score = score
        self.tracklet_len = 0

        # One EMA-smoothed / one raw-latest embedding per named ReID stream
        # (e.g. {"reidnet": ..., "clipreid": ...} when multiple SGIEs feed
        # this tracklet - see multi_cam_reid.py). Whichever stream name is
        # inserted first becomes the "primary" stream: self.smooth_feat/
        # self.curr_feat below stay plain-ndarray aliases for it, so BoT-
        # SORT's own short-term IOU+embedding association (matching.py's
        # embedding_distance) keeps reading a single vector, unaffected by a
        # second model being carried through. The cross-camera tracklet-
        # level engine (global_tracklet_association.py) is the one consumer
        # that reads the full smooth_feats dict.
        self.smooth_feats: dict = {}
        self.curr_feats: dict = {}
        self._primary_stream: Optional[str] = None
        self.pose = pose
        self.num_kpts = num_kpts
        self.img_path = img_path
        self.alpha = 0.9
        if feat is not None:
            self.update_features(feat)

        self.centroid = np.asarray(self._tlwh[:2] + self._tlwh[2:] / 2, dtype=np.float64)
        self.t_global_id = 0
        self.global_id = 0
        # DS shared frame_id at which the current t_global_id was resolved - set
        # by GTA (botsort/global_tracklet_association.py), unused by GlobalRegistry.
        # Lets a display probe show "how long has this identity been resolved"
        # straight off the STrack, with no second call into the registry.
        self.t_identity_since_frame = 0

        # adding object meta for global_id_assignment
        self.curr_obj_meta_ref = obj_meta

        self.matched_dist = None

    def update_features(self, feat, det_confidence=1.0):
        """Update this tracklet's per-stream EMA feature(s).

        `feat` is either a single embedding (np.ndarray) - stored under the
        implicit `_DEFAULT_STREAM` key - or a dict of {stream_name: embedding}
        when multiple ReID models feed this tracklet. Each stream gets its own
        independent EMA, identical in shape to the original single-model
        logic (confidence-weighted alpha, unit-renormalized after blending).
        """
        streams = feat if isinstance(feat, dict) else {self._DEFAULT_STREAM: feat}
        for name, vec in streams.items():
            if vec is None:
                continue
            vec = np.asarray(vec, dtype=np.float32)
            n = np.linalg.norm(vec)
            vec = (vec / n) if n > 1e-9 else vec
            self.curr_feats[name] = vec
            prev = self.smooth_feats.get(name)
            if prev is None:
                self.smooth_feats[name] = vec
            else:
                # Low detection confidence (partial occlusion, blur) → scale down
                # the new-observation weight so occluded frames drift the EMA less.
                # effective_alpha approaches 1.0 (freeze) as det_confidence → 0.
                effective_alpha = self.alpha + (1.0 - self.alpha) * (1.0 - float(det_confidence))
                blended = effective_alpha * prev + (1.0 - effective_alpha) * vec
                bn = np.linalg.norm(blended)
                self.smooth_feats[name] = (blended / bn) if bn > 1e-9 else blended
            if self._primary_stream is None:
                self._primary_stream = name

    @property
    def smooth_feat(self):
        return self.smooth_feats.get(self._primary_stream) if self._primary_stream is not None else None

    @property
    def curr_feat(self):
        return self.curr_feats.get(self._primary_stream) if self._primary_stream is not None else None

    def predict(self):
        mean_state = self.mean.copy()
        if self.state != TrackState.Tracked:
            mean_state[6] = 0
            mean_state[7] = 0

        self.mean, self.covariance = self.kalman_filter.predict(mean_state, self.covariance)

    @staticmethod
    def multi_predict(stracks):
        if len(stracks) > 0:
            multi_mean = np.asarray([st.mean.copy() for st in stracks])
            multi_covariance = np.asarray([st.covariance for st in stracks])
            for i, st in enumerate(stracks):
                if st.state != TrackState.Tracked:
                    multi_mean[i][6] = 0
                    multi_mean[i][7] = 0
            multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(multi_mean, multi_covariance)
            for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
                stracks[i].mean = mean
                stracks[i].covariance = cov

    def activate(self, kalman_filter, frame_id, id_assigner=None):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        if not id_assigner:
            self.track_id = self.next_id()
        else:
            self.track_id = id_assigner.next_id()

        self.mean, self.covariance = self.kalman_filter.initiate(self.tlwh_to_xywh(self._tlwh))

        self.tracklet_len = 0
        self.state = TrackState.Tracked
        if frame_id == 1:
            self.is_activated = True
        self.frame_id = frame_id
        self.start_frame = frame_id

    def re_activate(self, new_track, frame_id, new_id=False, id_assigner=None):
        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_track.tlwh))
        if new_track.curr_feats:
            # Forward every stream the new detection carries (not just the
            # primary-stream curr_feat alias) so a second ReID model's
            # embedding keeps flowing into this tracklet's EMA after the
            # first frame too.
            self.update_features(new_track.curr_feats, det_confidence=new_track.score)
        self.tracklet_len = 0
        self.state = TrackState.Tracked
        self.is_activated = True
        self.frame_id = frame_id
        if new_id:
            if not id_assigner:
                self.track_id = self.next_id()
            else:
                self.track_id = id_assigner.next_id()
        self.score = new_track.score
        self.pose = new_track.pose
        self.num_kpts = new_track.num_kpts
        self.img_path = new_track.img_path
        # Only update if the new detection actually carries edge info.
        # Second-association detections are created without obj_meta (no
        # bbox-edge check), so is_touching_edge=None there - overwriting with
        # None would silently disable the edge guard in GTA.step().
        if new_track.is_touching_edge is not None:
            self.is_touching_edge = new_track.is_touching_edge

        self.centroid = self.tlwh_to_xywh(new_track.tlwh)[:2]

    def update(self, new_track, frame_id):
        """
        Update a matched track
        :type new_track: STrack
        :type frame_id: int
        :type update_feature: bool
        :return:
        """
        self.frame_id = frame_id
        self.tracklet_len += 1

        new_tlwh = new_track.tlwh

        self.mean, self.covariance = self.kalman_filter.update(self.mean, self.covariance, self.tlwh_to_xywh(new_tlwh))

        ''' only suitable for reid wrapper '''
        # if new_track.curr_feat is not None and not new_track.is_touching_edge:

        if new_track.curr_feats:
            # See re_activate()'s comment - forward every stream, not just
            # the primary-stream alias.
            self.update_features(new_track.curr_feats, det_confidence=new_track.score)

        self.state = TrackState.Tracked
        self.is_activated = True

        self.score = new_track.score
        self.pose = new_track.pose
        self.num_kpts = new_track.num_kpts
        self.img_path = new_track.img_path
        if new_track.is_touching_edge is not None:
            self.is_touching_edge = new_track.is_touching_edge

        self.centroid = self.tlwh_to_xywh(new_track.tlwh)[:2]

    @property
    def tlwh(self):
        """Get current position in bounding box format `(top left x, top left y,
                width, height)`.
        """
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[:2] -= ret[2:] / 2
        return ret

    @property
    def tlbr(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    @property
    def xywh(self):
        """Convert bounding box to format `(min x, min y, max x, max y)`, i.e.,
        `(top left, bottom right)`.
        """
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2.0
        return ret

    @staticmethod
    def tlwh_to_xyah(tlwh):
        """Convert bounding box to format `(center x, center y, aspect ratio,
        height)`, where the aspect ratio is `width / height`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def tlwh_to_xywh(tlwh):
        """Convert bounding box to format `(center x, center y, width,
        height)`.
        """
        ret = np.asarray(tlwh).copy()
        ret[:2] += ret[2:] / 2
        return ret

    def to_xywh(self):
        return self.tlwh_to_xywh(self.tlwh)

    @staticmethod
    def tlbr_to_tlwh(tlbr):
        '''
        dummy function according to reference script, 
        ds already pushing TLWH format bbox
        '''
        # ret = np.asarray(tlbr).copy()
        # ret[2:] -= ret[:2]
        return tlbr

    @staticmethod
    def tlwh_to_tlbr(tlwh):
        ret = np.asarray(tlwh).copy()
        ret[2:] += ret[:2]
        return ret

    def __repr__(self):
        return 'OT_{}_({}-{})'.format(self.track_id, self.start_frame, self.end_frame)


class BoTSORT(object):
    def __init__(self, cam_source = 1, track_high_thresh=0.6, track_low_thresh=0.1, new_track_thresh=0.7, track_buffer=30,
                match_thresh=0.5, with_reid=True, proximity_thresh=0.2, appearance_thresh=0.4, appearance_veto_thresh=None, euc_thresh=0.1,
                fuse_score=True, frame_rate=30, max_batch_size=8, map_len=None, real_data=True, registry = None, frame_size =  (1920, 1080), roi_padding = (0,0)):
        self.tracked_stracks = []  
        self.lost_stracks = []  # type: list[STrack]

        self.cam_source = cam_source
        
        # check viability of removed stracks + filter for removed tracks
        self.removed_stracks = []  # type: list[STrack]
        
        BaseTrack.clear_count()

        self.frame_id = 0

        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.new_track_thresh = new_track_thresh

        self.buffer_size = int(frame_rate / 30.0 * track_buffer)
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()

        self.match_thresh = match_thresh
        self.fuse_score = fuse_score

        # ReID module
        self.with_reid = with_reid
        self.real_data = real_data
        self.proximity_thresh = proximity_thresh
        self.appearance_thresh = appearance_thresh
        # Distance above which an embedding confidently means "different
        # person" and should block an otherwise-IOU-valid match. This must
        # stay looser than appearance_thresh (the "assist" bar) - appearance
        # routinely degrades under partial occlusion/motion blur without
        # actually being a different person, so re-using appearance_thresh
        # here vetoes legitimate occlusion recoveries and forces new IDs.
        self.appearance_veto_thresh = (
            appearance_veto_thresh if appearance_veto_thresh is not None
            else min(1.0, appearance_thresh + 0.3)
        )
        self.euc_thresh = euc_thresh
        self.max_batch_size = max_batch_size

        self.max_len = map_len if map_len else np.sqrt(1920**2 + 1080**2)

        self.id_assigner = ID_Assigner(init_id=cam_source * 1_000)
        self.registry = registry

        self.frame_size = frame_size
        self.roi_padding = roi_padding
        
        # self.encoder = FastReIDInterface('./reid/configs/AIC24/sbs_R50-ibn.yml', './pretrained/market_aic_sbs_R50-ibn.pth', 'cuda')
        # self.id_assigner = None
        # self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])

    def update(self, output_results):
        self.frame_id += 1
        activated_starcks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        
        
        # print(f"[BS] Tracked Stracks\t{[(track.t_global_id, track.track_id) for track in self.tracked_stracks]}")
        # print(f"[BS] Lost Stracks\t{[(track.t_global_id, track.track_id) for track in self.lost_stracks]}")
        # print(f"[BS] Removed Strack\t{[(track.t_global_id, track.track_id) for track in self.removed_stracks]}")

        if len(output_results):
            scores = np.array([d['det_confidence'] for d in output_results])
            bboxes = np.array([d['bbox'] for d in output_results])
            classes = np.array([1 for d in output_results])
            # 'reid_vectors' (dict[str, np.ndarray], one entry per ReID model
            # - see multi_cam_reid.py's probe) takes priority over the
            # legacy single-model 'reid_vector' key so a caller feeding
            # multiple SGIEs' embeddings straight through to STrack(feat=...)
            # - which accepts either shape, see update_features - just
            # works. dtype=object: elements are ndarrays/dicts/None, not a
            # numeric matrix numpy could stack.
            features = np.array(
                [d.get('reid_vectors', d.get('reid_vector')) for d in output_results],
                dtype=object,
            )
            obj_meta = np.array([d['obj_meta'] for d in output_results])

            # Remove bad detections
            lowest_inds = scores > self.track_low_thresh
            bboxes = bboxes[lowest_inds]
            scores = scores[lowest_inds]
            classes = classes[lowest_inds]
            features = features[lowest_inds]
            obj_meta = obj_meta[lowest_inds]
            
            # Find high threshold detections
            
            remain_inds = scores > self.track_high_thresh
            dets = bboxes[remain_inds]
            scores_keep = scores[remain_inds]
            classes_keep = classes[remain_inds]
            features_keep = features[remain_inds]
            obj_meta_keep = obj_meta[remain_inds]

            # pose_input = [{"bbox": det} for det in dets]
            # pose_input = dets
        else:
            bboxes = []
            scores = []
            classes = []
            dets = []
            scores_keep = []
            obj_meta = []
            # pose_input = []

        if len(dets) > 0:
            if self.with_reid:
                # features_keep = self.encoder.inference(img, dets)

                # pose_result = inference_topdown(pose, img, pose_input, bbox_format='xyxy')
                # pose_result = np.array([np.concatenate([p.pred_instances.keypoints[0], np.expand_dims(p.pred_instances.keypoint_scores[0], axis=1)], axis=1) for p in pose_result])
                # num_kpts_per_bbox = count_kpts_per_bbox(pose_input, pose_result)
                # new_ratio = all_good_pose_bbox(pose_input, pose_result)
                detections = [STrack(
                    tlwh = STrack.tlbr_to_tlwh(tlbr), 
                    score = s, 
                    feat=f, 
                    obj_meta = oma,
                ) for (tlbr, s, f, oma) in zip(dets, scores_keep, features_keep, obj_meta_keep)]
            else:
                # dont focus (no - reid case)
                detections = [STrack(STrack.tlbr_to_tlwh(tlbr), s) for
                              (tlbr, s) in zip(dets, scores_keep)]
        else:
            new_ratio = None
            detections = []

        ''' Add newly detected tracklets to tracked_stracks'''
        unconfirmed = []
        tracked_stracks = []  # type: list[STrack]
        for track in self.tracked_stracks:
            if not track.is_activated:
                unconfirmed.append(track)
            else:
                tracked_stracks.append(track)

        ''' Step 2: First association, with high score detection boxes'''
        strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)

        # Predict the current location with KF
        STrack.multi_predict(strack_pool)

        # Fix camera motion
        # warp = self.gmc.apply(img, dets)
        # STrack.multi_gmc(strack_pool, warp)
        # STrack.multi_gmc(unconfirmed, warp)

        # Associate with high score detection boxes

        # if self.fuse_score:
            # ious_dists = matching.fuse_score(ious_dists, detections)
        
        # centroid_dists = matching.centroid_distance(strack_pool, detections)
        # centroid_dists /= self.max_len
        # centroid_dists_mask = (centroid_dists > self.proximity_thresh)


        ious_dists = matching.iou_distance(strack_pool, detections)

        if self.with_reid:
            emb_dists = matching.embedding_distance(strack_pool, detections)
            valid_mask = np.logical_and(
                emb_dists < self.appearance_thresh,
                ious_dists < self.proximity_thresh
            )

            hat_emb_dists = np.ones_like(emb_dists)
            hat_emb_dists[valid_mask] = emb_dists[valid_mask]
            dists = np.minimum(ious_dists, hat_emb_dists)

            # Veto: dists = min(iou, emb) can never be *worse* than the IOU
            # distance alone, so a confidently bad appearance match never
            # actually blocked an IOU-only match - it just failed to help it.
            # That's what causes ID swaps between two different, overlapping
            # people (e.g. crossing pedestrians). If a pair is IOU-matchable
            # but appearance *confidently* disagrees, refuse the match
            # outright. Uses appearance_veto_thresh (looser than
            # appearance_thresh) so merely noisy appearance - e.g. during a
            # brief occlusion - doesn't itself kill a legitimate track.
            appearance_mismatch = np.logical_and(
                ious_dists < self.match_thresh,
                emb_dists >= self.appearance_veto_thresh,
            )
            dists[appearance_mismatch] = 1.0
        else:
            dists = ious_dists

        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.match_thresh)
    

        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = detections[idet]
            if track.state == TrackState.Tracked:
                track.update(detections[idet], self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, id_assigner=self.id_assigner)
                refind_stracks.append(track)

        ''' Step 3: Second association, with low score detection boxes'''
        if len(scores):
            inds_high = scores < self.track_high_thresh
            inds_low = scores > self.track_low_thresh
            inds_second = np.logical_and(inds_low, inds_high)
            dets_second = bboxes[inds_second]
            scores_second = scores[inds_second]
            classes_second = classes[inds_second]
        else:
            dets_second = []
            scores_second = []
            classes_second = []

        # association the untrack to the low score detections
        if len(dets_second) > 0:
            '''Detections'''
            detections_second = [STrack(tlwh=STrack.tlbr_to_tlwh(tlbr), score=s) for
                                 (tlbr, s) in zip(dets_second, scores_second)]
        else:
            detections_second = []

        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = detections_second[idet]
            if track.state == TrackState.Tracked:
                track.update(det, self.frame_id)
                activated_starcks.append(track)
            else:
                track.re_activate(det, self.frame_id, new_id=False, id_assigner=self.id_assigner)
                refind_stracks.append(track)

        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == TrackState.Lost:
                track.mark_lost()
                lost_stracks.append(track)

        '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''
        detections = [detections[i] for i in u_detection]
        # if self.fuse_score:
        #     ious_dists = matching.fuse_score(ious_dists, detections)
        # dists = matching.iou_distance(r_tracked_stracks, detections_second)
        ious_dists = matching.iou_distance(unconfirmed, detections)

        if self.with_reid:
            emb_dists = matching.embedding_distance(unconfirmed, detections)
            valid_mask = np.logical_and(
                emb_dists < self.appearance_thresh,
                ious_dists < self.proximity_thresh
            )

            hat_emb_dists = np.ones_like(emb_dists)
            hat_emb_dists[valid_mask] = 0.5 * emb_dists[valid_mask]
            dists = np.minimum(ious_dists, hat_emb_dists)

            # Same appearance veto as the first association round (see
            # above) - an unconfirmed track is a brand-new tracklet, so
            # letting a confidently-bad-appearance/good-IOU pair through
            # here is just as likely to hand it someone else's identity.
            appearance_mismatch = np.logical_and(
                ious_dists < 0.7,
                emb_dists >= self.appearance_veto_thresh,
            )
            dists[appearance_mismatch] = 1.0
        else:
            dists = ious_dists

        matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
        for itracked, idet in matches:
            unconfirmed[itracked].update(detections[idet], self.frame_id)
            activated_starcks.append(unconfirmed[itracked])
        for it in u_unconfirmed:
            track = unconfirmed[it]
            track.mark_removed()
            
            if track.t_global_id != 0:
                removed_stracks.append(track)

        """ Step 4: Init new stracks"""
        for inew in u_detection:
            track = detections[inew]
            if track.score < self.new_track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id, id_assigner=self.id_assigner)
            activated_starcks.append(track)

        """ Step 5: Update state"""
        for track in self.lost_stracks:
            if self.frame_id - track.end_frame > self.max_time_lost:
                track.mark_removed()
                removed_stracks.append(track)
                # ← ADD THESE 2 LINES:
                if self.registry is not None:
                    self.registry.deactivate_track(track.track_id)

        """ Merge """
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
        self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
        self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        # self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.removed_stracks.extend(removed_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        # output_stracks = [track for track in self.tracked_stracks if track.is_activated]
        output_stracks = [track for track in self.tracked_stracks]
        
        # disp_curr = np.array([t.t_global_id for t in activated_starcks])
        # print(f"[CURR] {disp_curr}")
        
        # disp_lost = np.array([t.t_global_id for t in self.lost_stracks])
        # print(f"[LOST] {disp_lost}")
        
        if not len(unconfirmed): 
            disp_uncm = np.array([t.t_global_id for t in activated_starcks])
            # print(f"[UNCM] {disp_uncm}")
        
        # return output_stracks
        return output_stracks


def joint_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res


def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())


def remove_duplicate_stracks(stracksa, stracksb):
    pdist = matching.iou_distance(stracksa, stracksb)
    pairs = np.where(pdist < 0.15)
    dupa, dupb = list(), list()
    for p, q in zip(*pairs):
        timep = stracksa[p].frame_id - stracksa[p].start_frame
        timeq = stracksb[q].frame_id - stracksb[q].start_frame
        if timep > timeq:
            dupb.append(q)
        else:
            dupa.append(p)
    resa = [t for i, t in enumerate(stracksa) if not i in dupa]
    resb = [t for i, t in enumerate(stracksb) if not i in dupb]
    return resa, resb


def count_kpts_per_bbox(pose_input, pose_result):
    num_kpts_per_bbox = []

    for bbox in pose_input:
        x1, y1, x2, y2 = bbox
        num = 0
        for kpts in pose_result:
            keypoints_inside_bbox = kpts[
            (kpts[:, 0] >= x1) & (kpts[:, 0] <= x2) &
            (kpts[:, 1] >= y1) & (kpts[:, 1] <= y2)
            ]
            num += len(keypoints_inside_bbox)
        num_kpts_per_bbox.append(num)
    
    return np.array(num_kpts_per_bbox)

def all_good_pose_bbox(pose_input, pose_result):
    new_ratio = [0] * 7

    num = 0
    for bbox, kpts in zip(pose_input, pose_result):
        if sum(kpts[:,2] > 0.8) == 14:
            num += 1
            x1, y1, x2, y2 = bbox
            w, h = x2 - x1, y2 - y1
            new_ratio[0] += h / w
            new_ratio[1] += h / (kpts[13, 1] - kpts[12, 1])
            new_ratio[2] += h / np.mean(kpts[4:8, 1] - kpts[12, 1])
            new_ratio[3] += h / np.mean(kpts[[8,9], 1] - kpts[12, 1])
            new_ratio[4] += h / (kpts[13, 1] - kpts[12, 1])
            new_ratio[5] += h / np.mean(kpts[[0,1], 1] - kpts[12, 1])
            new_ratio[6] += h / np.mean(kpts[[2,3], 1] - kpts[12, 1])

    contains_inf = any(math.isinf(x) or x < 0 for x in new_ratio)
    if num > 0 and not contains_inf:
        return np.array(new_ratio)/num
    else:
        return None